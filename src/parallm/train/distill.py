"""The d1b-heal distillation step (lever-B / post-attn schedule).

Rebuilt from the pre-parallm-pivot trainer for the 27B N=8 pool-free program,
keeping the rail-validated recipe semantics and dropping the refuted-lever
surface (vocab-parallel, cross-head estimators, window-parallel, free-running
MSE). Three deliberate deviations from the 9B-record trainer, all forced by
27B memory or strictly signal-improving:

1. **Teacher = the frozen original slices on the EXACT schedule** (2 syncs per
   layer — bit-identical to the dense model by construction, the engine's
   dense rail) instead of a resident dense HF model: a dense 27B teacher is
   54 GB/rank and cannot exist next to the student on a 40 GB device, while
   the frozen-slice teacher costs one extra track slice per rank and shares
   the frozen embed/lm_head storage with the student.
2. **Replicated output loss** instead of vocab-parallel: every rank holds a
   frozen lm_head copy and computes the identical full-vocab KL/CE/logit-MSE,
   so each rank's (collective-free) backward carries the FULL loss signal into
   its own track — vocab-parallel delivered only the rank's shard slice of it.
3. **embed_tokens + lm_head are FROZEN** (they are exact dense copies; the
   heal targets slice function).  # ponytail: also saves ~7.6 GB of optimizer
   state on rank 0 — unfreeze via --train-embeddings if the Stage-1 gate fails.

Step structure (the record recipe):
  1. Teacher forward (no_grad): boundary targets (post-attn residual at
     boundaries, post-MLP at the final layer / mid-window taps) + the final
     post-norm hidden for the chunked KL.
  2. Teacher-forced post-attn block loop: per boundary segment, sync, tap
     block-MSE (normalized, clamped), backward, free the graph; with
     probability ``student_forcing_prob`` the carried state is the student's
     own synced hidden instead of the teacher's (sf=1.0 chain ≡ the deployed
     d1b forward bit-for-bit).
  3. Full free-running student forward (per-sublayer checkpointing) + chunked
     KL/CE/centered-logit-MSE; ONE backward with the accumulated hidden grad.
     The SyncBoundary all-reduce is autograd-invisible, so this backward is
     collective-free on every rank (the record's gradient semantics).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from parallm.model.pt_model import PTWrappedModel
from parallm.model.seam import checkpointed_halves
from parallm.train.losses import block_mse


@dataclass
class DistillConfig:
    sync_layer_indices: tuple[int, ...] = field(default_factory=tuple)
    lambda_block: float = 1.0
    lambda_kl: float = 1.0
    lambda_ce: float = 0.5
    lambda_logit_mse: float = 0.0
    kl_temperature: float = 1.0
    normalize_block_mse: bool = True
    block_mse_clamp: float | None = 10.0
    intra_window_mse: bool = False
    kl_ce_chunk_size: int = 256


def student_forcing_schedule(step: int, prob: float, warmup: int) -> float:
    """Per-step student-forcing probability: linear ``prob * min(1, step/warmup)``
    then hold (``warmup=0`` ⇒ constant ``prob``, the record recipe)."""
    return prob * min(1.0, step / warmup) if warmup > 0 else prob


def freeze_slice_teacher(model: PTWrappedModel) -> PTWrappedModel:
    """Configure the ORIGINAL track slices as the frozen exact-schedule teacher.

    ``model`` is a PTWrappedModel loaded from the raw convert output; in eval
    mode with ``requires_grad=False`` and ``sync_phase="exact"`` its forward is
    the dense teacher's up to fp summation order (the engine's dense rail), at
    one track slice of memory per rank.
    """
    model.eval()
    model.set_sync_phase("exact")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def teacher_forward(
    teacher: PTWrappedModel,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
    post_attn_layers: set[int],
    post_mlp_layers: set[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """The post-norm final hidden (identical on every rank) and ``{layer: target}``
    captures: post-attn residual at ``post_attn_layers``, post-MLP hidden at
    ``post_mlp_layers`` — the lever-B target contract."""
    return teacher(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_sync_hiddens=True,
        return_hidden_pre_lm_head=True,
        capture_post_attn=post_attn_layers,
        capture_post_mlp=post_mlp_layers,
    )


def capture_sets(
    sync_layer_indices, num_layers: int, intra_window_mse: bool = False
) -> tuple[set[int], set[int]]:
    """``(post_attn_layers, post_mlp_layers)`` the teacher captures: post-attn
    residual at every boundary, post-MLP at the final layer (plus mid-window
    taps under ``intra_window_mse``)."""
    last = num_layers - 1
    post_attn = set(sync_layer_indices) - {last}
    post_mlp = {last} | (
        set(range(num_layers)) - post_attn - {last} if intra_window_mse else set()
    )
    return post_attn, post_mlp


def kl_ce_chunked(
    hidden: torch.Tensor,             # (B, T, H), grad-connected to student params
    teacher_hidden: torch.Tensor,     # (B, T, H) detached (teacher post-norm)
    lm_head: nn.Module,               # frozen head, identical on every rank
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    lambda_kl: float,
    lambda_ce: float,
    lambda_logit_mse: float = 0.0,
    kl_temperature: float = 1.0,
    chunk_size: int = 256,
    loss_scale: float = 1.0,
    compute_grads: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
    """λ_kl·KL + λ_ce·CE (+ λ_lm·centered logit-MSE) in seq-chunks.

    Never materializes a (B, T, V) logits tensor for either side: both the
    student's and the teacher's logits are produced per chunk from their final
    hiddens through the shared frozen ``lm_head``. Each chunk's gradient is
    taken against a detached ``h_anchor`` and accumulated into a (B, T, H)
    buffer the caller backwards through the real graph; the chunk's fp32
    softmax tensors are freed before the next chunk (no retain_graph).

    Returns ``(kl, ce, logit_mse, grad_h_or_None)`` — detached scalars, UNSCALED
    (``loss_scale`` multiplies only the accumulated gradient). Identical on
    every rank (replicated loss).
    """
    B, T, H = hidden.shape
    device = hidden.device
    temp_sq = kl_temperature * kl_temperature

    if attention_mask is not None:
        kl_denom = attention_mask.sum().clamp(min=1).float()
    else:
        kl_denom = torch.tensor(float(B * T), device=device)
    shift_labels = labels[:, 1:]
    ce_denom = (shift_labels != -100).sum().clamp(min=1).float()

    if compute_grads:
        h_anchor = hidden.detach().requires_grad_(True)
        grad_h_accum = torch.zeros_like(h_anchor)
    else:
        h_anchor = hidden
        grad_h_accum = None

    kl_total = torch.zeros((), device=device)
    ce_total = torch.zeros((), device=device)
    lm_total = torch.zeros((), device=device)

    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        with torch.enable_grad() if compute_grads else torch.no_grad():
            s_logits = lm_head(h_anchor[:, t0:t1, :]).float()
            with torch.no_grad():
                t_logits = lm_head(teacher_hidden[:, t0:t1, :]).float()

            chunk_loss = torch.zeros((), device=device)

            # KL is always computed — it is also the headline logged metric.
            s_logp = F.log_softmax(s_logits / kl_temperature, dim=-1)
            t_logp = F.log_softmax(t_logits / kl_temperature, dim=-1)
            per_token = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)  # (B, chunk)
            if attention_mask is not None:
                per_token = per_token * attention_mask[:, t0:t1]
            kl_chunk = per_token.sum() / kl_denom * temp_sq
            kl_total += kl_chunk.detach()
            if lambda_kl != 0.0:
                chunk_loss = chunk_loss + lambda_kl * kl_chunk
            del s_logp, t_logp, per_token

            if lambda_logit_mse != 0.0:
                d = s_logits - t_logits
                d = d - d.mean(dim=-1, keepdim=True)
                per_token = d.pow(2).mean(dim=-1)
                if attention_mask is not None:
                    per_token = per_token * attention_mask[:, t0:t1]
                    lm_chunk = per_token.sum() / kl_denom
                else:
                    lm_chunk = per_token.sum() / (B * T)
                lm_total += lm_chunk.detach()
                chunk_loss = chunk_loss + lambda_logit_mse * lm_chunk
                del d, per_token

            # CE: logits[t] predicts labels[t+1]; the chunk's last position is
            # scored by the NEXT chunk's first label except at the seq end.
            lbl_hi = min(t1, T - 1)
            if lbl_hi > t0:
                ce_logits = s_logits[:, : lbl_hi - t0, :]
                ce_labels = labels[:, t0 + 1 : lbl_hi + 1]
                ce_chunk = F.cross_entropy(
                    ce_logits.reshape(-1, ce_logits.size(-1)),
                    ce_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ) / ce_denom
                ce_total += ce_chunk.detach()
                if lambda_ce != 0.0:
                    chunk_loss = chunk_loss + lambda_ce * ce_chunk

            if compute_grads and chunk_loss.requires_grad:
                (g,) = torch.autograd.grad(chunk_loss * loss_scale, [h_anchor])
                grad_h_accum += g
            del s_logits, t_logits

    return kl_total, ce_total, lm_total, grad_h_accum


def distill_step(
    student: PTWrappedModel,
    teacher: PTWrappedModel,
    lm_head: nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: DistillConfig,
    student_forcing_prob: float = 0.0,
    forcing_seed: object = 0,
    loss_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """One heal step (backward done internally; caller syncs grads + steps).

    The TF block loop backwards per boundary segment (graph freed each flush,
    peak memory ≈ one segment); the output objective backwards once through
    the checkpointed full forward. All RNG driven by ``forcing_seed`` so every
    rank draws identical sf decisions (the SyncBoundary collectives require
    lock-step choices).
    """
    # str(): random.Random rejects a tuple seed, and str is stable across ranks
    # and processes (unlike hash()).
    forcing_rng = random.Random(str(forcing_seed))
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]
    device = input_ids.device

    tm0 = student.text_models[0]
    L = len(tm0.layers)
    last = L - 1
    need_klce = cfg.lambda_kl != 0.0 or cfg.lambda_ce != 0.0 or cfg.lambda_logit_mse != 0.0

    # Teacher targets: post-attn at boundaries, post-MLP at the final layer and
    # (with intra-window supervision) the non-boundary layers.
    sync_attn_set, post_mlp_set = capture_sets(cfg.sync_layer_indices, L, cfg.intra_window_mse)
    t_hidden, t_caps = teacher_forward(
        teacher, input_ids, attention_mask,
        post_attn_layers=sync_attn_set, post_mlp_layers=post_mlp_set,
    )

    # ----- Scaffolding (embed broadcast + masks + rotary, shared by the loop) -----
    inputs_embeds = student.embed(input_ids)
    position_ids, text_position_ids = tm0._resolve_position_ids(inputs_embeds, None)
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    causal_mask = create_causal_mask(
        config=tm0.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = (
        None if (attention_mask is not None and torch.all(attention_mask == 1)) else attention_mask
    )
    position_embeddings = tm0.rotary_emb(inputs_embeds, position_ids)

    def tap_loss(h_synced: torch.Tensor, t_target: torch.Tensor) -> torch.Tensor:
        r = block_mse(
            h_synced, t_target, attention_mask=attention_mask,
            normalize=cfg.normalize_block_mse,
            clamp_max=cfg.block_mse_clamp if cfg.normalize_block_mse else None,
        )
        return r

    # ----- Teacher-forced post-attn block loop (per-segment backward) -----
    block_loss_val = torch.zeros((), device=device)
    layer_relmse: dict[int, float] = {}
    block_input = [inputs_embeds.detach() for _ in student.text_models]
    block_start = inputs_embeds.detach()
    seg_loss = torch.zeros((), device=device)
    seg_count = 0

    def _flush(sl, sc):
        if sc > 0 and cfg.lambda_block != 0.0 and sl.requires_grad:
            (cfg.lambda_block * (sl / sc) * loss_scale).backward()

    # Per-sublayer activation checkpointing bounds the peak graph to ONE
    # recomputed layer regardless of own-carry segment length. But at d1b (every
    # non-last layer a boundary) each segment is length-1 and flushed
    # immediately, so checkpointing the TF loop saves ZERO memory and is pure
    # recompute on the launch-bound hot path — skip it there. The FR forward
    # below keeps the model's own checkpointing regardless (it holds all L layers).
    # ponytail: gapless <=> D=1; a stray missing boundary => one length-2 segment,
    # still memory-safe uncheckpointed.
    gapless = len(sync_attn_set) >= L - 1
    run_mixer, run_mlp = checkpointed_halves(
        student.use_checkpoint and torch.is_grad_enabled() and not gapless,
        position_embeddings,
        text_position_ids,
    )

    for i in range(L):
        layer_mask = (
            linear_attn_mask
            if tm0.config.layer_types[i] == "linear_attention"
            else causal_mask
        )
        h_attn = [run_mixer(tm.layers[i], block_input[k], layer_mask)
                  for k, tm in enumerate(student.text_models)]
        is_boundary = i in sync_attn_set
        supervise = is_boundary or i == last or cfg.intra_window_mse
        if is_boundary:
            h_synced = student.sync_module(h_attn, block_start)  # post-attn, carried
        else:
            new_h = [run_mlp(student.text_models[k].layers[i], h_attn[k])
                     for k in range(len(student.text_models))]
            # Final layer: post-MLP carried sync. Non-boundary: loss-only tap
            # (synced reconstruction; the forward keeps carrying the partial).
            h_synced = student.sync_module(new_h, block_start) if supervise else None
        if supervise:
            t_l = t_caps.pop(i).detach()
            r = tap_loss(h_synced, t_l)
            seg_loss = seg_loss + r
            seg_count += 1
            block_loss_val = block_loss_val + r.detach()
            layer_relmse[i] = float(r.detach())
        if is_boundary:
            _flush(seg_loss, seg_count)
            seg_loss = torch.zeros((), device=device)
            seg_count = 0
            use_student = forcing_rng.random() < student_forcing_prob
            chosen = h_synced.detach() if use_student else t_l
            block_start = chosen
            block_input = [run_mlp(tm.layers[i], chosen) for tm in student.text_models]
        elif i == last:
            _flush(seg_loss, seg_count)
        else:
            block_input = new_h  # carry the partial

    n_taps = max(1, len(layer_relmse))
    block_loss_val = block_loss_val / n_taps

    # ----- Output objective: full free-running forward + chunked KL/CE -----
    # The free-running output needs a direct KL objective: teacher-forced
    # block-MSE alone left it untrained (val_kl plateaued ~1.9) — reconstructing
    # own-carry from teacher inputs didn't transfer to free-running generation.
    kl_val = torch.zeros((), device=device)
    ce_val = torch.zeros((), device=device)
    lm_val = torch.zeros((), device=device)
    if need_klce:
        hidden, _ = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=False,
            return_hidden_pre_lm_head=True,
        )
        kl_val, ce_val, lm_val, grad_h = kl_ce_chunked(
            hidden, t_hidden, lm_head, labels, attention_mask,
            lambda_kl=cfg.lambda_kl, lambda_ce=cfg.lambda_ce,
            lambda_logit_mse=cfg.lambda_logit_mse,
            kl_temperature=cfg.kl_temperature,
            chunk_size=cfg.kl_ce_chunk_size,
            loss_scale=loss_scale,
        )
        if grad_h is not None:
            torch.autograd.backward([hidden], [grad_h])

    total_val = (
        cfg.lambda_block * block_loss_val
        + cfg.lambda_kl * kl_val
        + cfg.lambda_ce * ce_val
        + cfg.lambda_logit_mse * lm_val
    )
    return {
        "total": total_val,
        "block_mse": block_loss_val,
        "kl": kl_val,
        "ce": ce_val,
        "logit_mse": lm_val,
        "layer_relmse": layer_relmse,  # detached floats, per supervised tap
    }


@torch.no_grad()
def validate_step(
    student: PTWrappedModel,
    teacher: PTWrappedModel,
    lm_head: nn.Module,
    batch: dict[str, torch.Tensor],
    kl_temperature: float = 1.0,
    chunk_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Forward-only KL(teacher‖student) + LM CE on a held-out batch. Scalars are
    identical on every rank — log any single one.

    `chunk_size` MUST match the training call: the logit chunk is the largest
    transient at V=248320, so the 256 default OOMs a slice that trained fine at
    32 — and only at the first eval, after the steps are spent.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    t_hidden, _ = teacher_forward(teacher, input_ids, attention_mask, set(), set())
    hidden, _ = student(
        input_ids=input_ids, attention_mask=attention_mask,
        return_sync_hiddens=False, return_hidden_pre_lm_head=True,
    )
    kl, ce, _lm, _ = kl_ce_chunked(
        hidden, t_hidden, lm_head, batch["labels"], attention_mask,
        lambda_kl=1.0, lambda_ce=1.0, kl_temperature=kl_temperature,
        chunk_size=chunk_size, compute_grads=False,
    )
    return {"kl": kl, "ce": ce}
