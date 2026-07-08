"""Dense healing trainer (Lever-1 Gate-1): self-distill the SERIAL dense teacher
into the window-parallel ``parallel-attn:g2`` function (eval/dense_parallel.py).

No tracks, no SyncBoundary, no vocab-parallel — a plain dense student (layers
trainable; embed/lm_head/final-norm frozen so the residual-stream geometry the
head reads stays anchored) against a frozen HookedTeacher. Two recipe-proven
loss terms per step:

1. **Teacher-forced per-window MSE** — for each window, run the rewired window
   step from the TEACHER's window-entry hidden and match the teacher's
   window-exit hidden: normalized block-MSE clamped per-window (the
   ``--normalize-block-mse --block-mse-clamp`` recipe). Backward runs
   immediately per window and the input is detached, so gradient lives entirely
   in that window's layers and activation memory is bounded by one window.
2. **Free-running centered logit-MSE** — full rewired forward from the
   embeddings with per-window activation checkpointing, matched to the teacher's
   final logits (``logit_mse(center=True)``). Covers compounding/exposure and
   supervises the directions ``lm_head`` reads. No forward-KL (diverges, see
   the objective-saturation record); no student-forcing curriculum in v1.

The teacher capture indices are the window-END layers (``windows[i][-1]``):
window ``i``'s TF input is capture[end(window i−1)] (embeddings for the first)
and its target is capture[end(window i)].
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import torch


def _mem(tag: str) -> None:
    """Phase-level memory attribution, enabled by HEAL_MEM_DEBUG=1 (rank-0 style:
    caller controls which ranks have the env set)."""
    if os.environ.get("HEAL_MEM_DEBUG"):
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        print(f"[mem] {tag}: alloc={a:.2f}GB reserved={r:.2f}GB", flush=True)

from pt_converter.eval.dense_parallel import MODES, WindowScaffold, window_step
from pt_converter.train.losses import block_mse


@dataclass
class HealConfig:
    windows: "list[list[int]]"
    mode: str = "parallel-attn"
    normalize_block_mse: bool = True
    block_mse_clamp: "float | None" = 10.0
    lambda_logit_mse: float = 0.05
    # Seq-chunk for the FR logit term's full-vocab fp32 expansion (the (B,T,V)
    # float() transients OOM at 40GB when un-chunked; 0 disables chunking).
    logit_chunk: int = 512

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")
        self.capture_indices: "list[int]" = [w[-1] for w in self.windows]


def tf_window_losses(
    text_model,
    cfg: HealConfig,
    scaf: WindowScaffold,
    emb: torch.Tensor,
    captures: "dict[int, torch.Tensor]",
    attention_mask: "torch.Tensor | None",
    *,
    do_backward: bool = True,
) -> "dict[str, float]":
    """Term 1: per-window teacher-forced normalized MSE, backward-per-window.

    Every window's TF input and target come from the frozen teacher, so the
    windows are independent; the per-window ``backward()`` frees each window's
    activations before the next one runs (mirrors ``distill_step``'s block loop).
    """
    n = len(cfg.windows)
    total = 0.0
    worst = 0.0
    prev_end: "int | None" = None
    for window in cfg.windows:
        h_in = (emb if prev_end is None else captures[prev_end]).detach()
        target = captures[window[-1]].detach()
        out, _ = window_step(scaf, window, h_in, h_in, cfg.mode)
        loss = block_mse(
            out, target, attention_mask,
            normalize=cfg.normalize_block_mse, clamp_max=cfg.block_mse_clamp,
        )
        if do_backward and loss.requires_grad:
            (loss / n).backward()
        total += float(loss.detach())
        worst = max(worst, float(loss.detach()))
        prev_end = window[-1]
    return {"tf_window_mse": total / n, "tf_window_mse_max": worst}


def fr_logit_losses(
    text_model,
    lm_head,
    cfg: HealConfig,
    scaf: WindowScaffold,
    emb: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: "torch.Tensor | None",
    *,
    do_backward: bool = True,
) -> "dict[str, float]":
    """Term 2: free-running centered logit-MSE (weighted by ``lambda_logit_mse``).
    Also reports the (loss-free) forward KL as a diagnostic reader.

    Manual gradient checkpointing, one window live at a time (torch.utils
    checkpoint + FSDP2 backward prefetch kept ~all 17 recomputed window graphs
    resident and OOM'd at 40GB):

    1. no-grad FR forward caching the 18 window-entry states (~17MB each);
    2. loss stage: per-seq-chunk centered logit-MSE backwards into a detached
       final-state leaf (caps the (B, T, V) fp32 transients at chunk size;
       ``lm_head``/``norm`` frozen so only the hidden grad matters);
    3. reverse sweep: re-run each window WITH grad from its cached entry leaf
       and backward its exit grad — FSDP sees a plain fwd+bwd per window.

    The training modes' ``window_step`` never READS ``attn_base`` (dropseam
    does — unsupported here), so each window is a pure function of its entry
    stream and the manual chain is exact.
    """
    if do_backward and cfg.mode == "parallel-attn-dropseam":
        raise NotImplementedError("FR training path does not support dropseam (attn_base chain)")
    # ---- stage 1: no-grad FR forward, caching window-entry states ----
    with torch.no_grad():
        states = [emb]
        for window in cfg.windows:
            nxt, _ = window_step(scaf, window, states[-1], states[-1], cfg.mode)
            states.append(nxt)
    _mem("after FR fwd")

    grad_to_stack = do_backward and torch.is_grad_enabled()
    # ---- stage 2: chunked loss on the final state. RMSNorm is per-position, so
    # the frozen final norm is applied per chunk (norm(x)[:, s:e] == norm(x[:, s:e]))
    # and each chunk's graph (slice → norm → lm_head → loss) is independent —
    # no retain_graph, no accumulated fp32 buffers. ----
    leaf = states[-1].detach().requires_grad_(True) if grad_to_stack else states[-1]
    T = leaf.shape[1]
    chunk = cfg.logit_chunk if cfg.logit_chunk and cfg.logit_chunk > 0 else T
    if attention_mask is None:
        n_pos = float(leaf.shape[0] * T)
    else:
        n_pos = float(attention_mask.sum().clamp(min=1))
    mse_sum = 0.0
    kl_sum = 0.0
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        mask_c = attention_mask[:, s:e] if attention_mask is not None else None
        logits_c = lm_head(text_model.norm(leaf[:, s:e]))
        t_c = teacher_logits[:, s:e].float()
        d = logits_c.float() - t_c
        d = d - d.mean(dim=-1, keepdim=True)
        per_token = d.pow(2).mean(dim=-1)  # (B, Tc)
        if mask_c is not None:
            per_token = per_token * mask_c
        loss_c = per_token.sum() / n_pos
        if grad_to_stack:
            (cfg.lambda_logit_mse * loss_c).backward()
        mse_sum += float(loss_c.detach())
        with torch.no_grad():
            s_logp = torch.log_softmax(logits_c.detach().float(), dim=-1)
            t_logp = torch.log_softmax(t_c, dim=-1)
            per = (t_logp.exp() * (t_logp - s_logp)).sum(-1)  # (B, Tc)
            if mask_c is not None:
                per = per * mask_c
            kl_sum += float(per.sum())
    _mem("after logit chunks")

    # ---- stage 3: manual reverse sweep, one window's fwd+bwd at a time ----
    if grad_to_stack:
        g = leaf.grad
        del leaf
        for idx in range(len(cfg.windows) - 1, -1, -1):
            src = states[idx].detach().requires_grad_(True)
            out, _ = window_step(scaf, cfg.windows[idx], src, src, cfg.mode)
            out.backward(gradient=g)
            g = src.grad
            states[idx + 1] = None  # free as we go
    return {"fr_logit_mse": mse_sum, "fr_kl": kl_sum / n_pos}


def heal_step(
    text_model,
    lm_head,
    teacher,
    cfg: HealConfig,
    input_ids: torch.Tensor,
    attention_mask: "torch.Tensor | None",
    *,
    do_backward: bool = True,
) -> "dict[str, float]":
    """One healing step (no optimizer logic): teacher forward → TF window losses
    (backward per window) → FR logit-MSE (one backward). Returns metrics.

    ``teacher`` is a HookedTeacher whose hook indices == ``cfg.capture_indices``.
    With ``do_backward=False`` this is the validation reader (call under no_grad).
    """
    need_logits = cfg.lambda_logit_mse > 0 or not do_backward
    _mem("step start")
    teacher_logits, captures = teacher.forward(
        input_ids, attention_mask=attention_mask, need_logits=need_logits
    )
    _mem("after teacher fwd")
    emb = text_model.embed_tokens(input_ids)
    scaf = WindowScaffold(text_model, emb, attention_mask)
    metrics = tf_window_losses(
        text_model, cfg, scaf, emb, captures, attention_mask, do_backward=do_backward
    )
    _mem("after TF windows")
    if need_logits:
        metrics.update(fr_logit_losses(
            text_model, lm_head, cfg, scaf, emb, teacher_logits, attention_mask,
            do_backward=do_backward,
        ))
        _mem("after FR")
    return metrics
