"""SPD-style distillation training step.

Implements the loop described in the plan:

1. Teacher forward (no_grad), capture {layer_idx -> hidden_state} at the
   sync boundaries plus final logits.
2. For each PT block (group of D student layers between syncs):
   - Feed the *teacher's* pre-block hidden state into the student's block.
   - Each of the rank's K local tracks runs D layers locally; one
     SyncBoundary call combines (local-sum across K) + (NCCL all-reduce
     across the world) to produce the synced post-block hidden.
   - block_MSE(student_post_block, teacher_post_block).
   Backprop each block independently (memory-bounded, mirrors SPD's
   block-to-block formulation).
3. Final-logit KL + LM CE on a full forward of the student (no teacher
   forcing) so the student also learns the end-to-end objective.

The block-wise teacher-forced loop and the full-forward loop are run on the
same minibatch; gradients accumulate before `optimizer.step()`.
"""
from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.losses import block_mse, logit_kl, lm_cross_entropy
from pt_converter.train.teacher import HookedTeacher


@contextmanager
def _phase(
    name: str,
    timings: dict[str, float] | None,
    mem: dict[str, dict[str, float]] | None = None,
):
    """Time and/or memory-profile a distill_step phase, CUDA-synced.

    When both ``timings`` and ``mem`` are None (the default, non-profiling path)
    this is a pure ``yield`` — no ``cuda.synchronize()``, no ``record_function``,
    zero overhead. Otherwise the body is wrapped in a
    ``torch.profiler.record_function`` range (so phase names appear in any trace)
    and bracketed by device syncs so the recorded wall time / memory reflects GPU
    completion, not just kernel launch.

    ``timings`` accumulates ``timings[name]`` wall-clock seconds. ``mem`` records
    ``mem[name] = {"peak_gb", "resident_gb"}`` — the transient peak *within* this
    phase (via ``reset_peak_memory_stats`` at entry) and the still-resident
    allocation at exit. ``mem`` resets peak stats, so the caller should only pass
    it on a single designated step (not every step).
    """
    if timings is None and mem is None:
        yield
        return
    torch.cuda.synchronize()
    if mem is not None:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.profiler.record_function(name):
        try:
            yield
        finally:
            torch.cuda.synchronize()
            if timings is not None:
                timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)
            if mem is not None:
                mem[name] = {
                    "peak_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                    "resident_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                }


def _kl_ce_chunked(
    hidden: torch.Tensor,                # (B, T, D) bf16, grad-connected to student params
    lm_head: nn.Module,                  # student's lm_head (owner rank only)
    teacher_logits: torch.Tensor,        # (B, T, V) detached
    labels: torch.Tensor,                # (B, T)
    attention_mask: torch.Tensor | None, # (B, T) or None
    *,
    lambda_kl: float,
    lambda_ce: float,
    kl_temperature: float,
    chunk_size: int,
    loss_scale: float = 1.0,
    compute_grads: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute lambda_kl * KL + lambda_ce * CE in seq-chunks; return the
    accumulated gradient w.r.t. ``hidden`` for the caller to backward.

    ``loss_scale`` (gradient-accumulation: 1/grad_accum_steps) multiplies the
    accumulated gradients only — the returned KL/CE scalars stay UNSCALED so
    logging reports the true loss.

    Two memory pressures motivate this design:

    1. The per-chunk fp32 saved tensors (log_softmax, softmax materialized by
       CE backward) are vocab-wide — at V=248320, chunk=128 each is ~127 MB.
       We get rid of them by computing each chunk's gradient *into the hidden
       state* (autograd.grad against ``h_anchor``) and discarding the chunk's
       forward graph immediately — no ``retain_graph`` across chunks.

    2. Materialising (B, T, V) bf16 logits in one shot (~2 GB at seq=4096)
       to hand to the caller is itself the dominant non-activation tensor on
       the lm_head-owning rank. We chunk the lm_head application here, so the
       caller never sees a full-T logits tensor.

    Flow:
      - Detach ``hidden`` into ``h_anchor`` (requires_grad=True): a leaf-like
        grad target whose grads we accumulate into ``grad_h_accum``
        (B, T, D) bf16 — orders of magnitude smaller than (B, T, V).
      - Per chunk: ``logits = lm_head(h_anchor[:, t0:t1, :])`` → KL + CE →
        ``autograd.grad(loss, h_anchor)`` returns a (B, T, D) tensor that is
        non-zero only on [t0:t1]. Add into the accumulator; the chunk's fp32
        tensors and lm_head's per-chunk graph are freed at function return.
      - The caller drives ``hidden.backward(grad_h)`` itself (combinable with
        other losses rooted in the same forward graph in ONE traversal). No
        ``retain_graph`` is needed anywhere, and no full-T logits tensor is
        ever materialized.

    Returns ``(kl_detached, ce_detached, grad_h_or_None)``. lm_head's own
    ``.grad`` is accumulated in place here. With ``compute_grads=False`` the
    chunk loop runs under ``no_grad`` (metrics only — kl/ce logging without
    building or backwarding any graph) and ``grad_h`` is ``None``.
    """
    B, T, D = hidden.shape
    V = teacher_logits.shape[-1]
    device = hidden.device
    temp_sq = kl_temperature * kl_temperature

    if attention_mask is not None:
        kl_denom = attention_mask.sum().clamp(min=1).float()
    else:
        kl_denom = torch.tensor(B * T, dtype=torch.float32, device=device)

    # CE uses next-token shifting: logits[t] predicts labels[t+1].
    shift_labels = labels[:, 1:]
    ce_denom = (shift_labels != -100).sum().clamp(min=1).float()

    if compute_grads:
        h_anchor = hidden.detach().requires_grad_(True)
        grad_h_accum = torch.zeros_like(h_anchor)
        # lm_head's own parameter gradients must be accumulated alongside
        # grad_h_accum — autograd.grad with only h_anchor in `inputs` would
        # silently drop them. Initialize `.grad` so we can add into it in place.
        lm_head_params = [p for p in lm_head.parameters() if p.requires_grad]
        for p in lm_head_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
    else:
        h_anchor = hidden
        grad_h_accum = None
        lm_head_params = []

    kl_acc = hidden.new_zeros((), dtype=torch.float32)
    ce_acc = hidden.new_zeros((), dtype=torch.float32)

    with torch.set_grad_enabled(compute_grads):
        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            h_chunk = h_anchor[:, t0:t1, :]                 # view, has grad to h_anchor
            s_chunk = lm_head(h_chunk)                      # (B, chunk, V) bf16
            t_chunk = teacher_logits[:, t0:t1, :]           # detached

            # KL contribution for this chunk. Reuses t_logp.exp() in place of a
            # second log_softmax — saves one fp32 vocab-wide intermediate.
            s_logp = F.log_softmax(s_chunk.float() / kl_temperature, dim=-1)
            t_logp = F.log_softmax(t_chunk.float() / kl_temperature, dim=-1)
            per_token_kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
            if attention_mask is not None:
                per_token_kl = per_token_kl * attention_mask[:, t0:t1]
            kl_chunk = per_token_kl.sum() / kl_denom * temp_sq

            # CE contribution: shifted alignment. Positions [t0, ce_t1) of logits
            # pair with labels [t0+1, ce_t1+1). The very last logit position
            # (T-1) has no label and is skipped.
            ce_t1 = min(t1, T - 1)
            if t0 < ce_t1:
                n_ce = ce_t1 - t0
                ce_logits = s_chunk[:, :n_ce, :].float().reshape(-1, V)
                ce_lbl = shift_labels[:, t0:ce_t1].reshape(-1)
                ce_sum = F.cross_entropy(
                    ce_logits, ce_lbl, ignore_index=-100, reduction="sum"
                )
            else:
                ce_sum = hidden.new_zeros((), dtype=torch.float32)
            ce_chunk = ce_sum / ce_denom

            if compute_grads:
                chunk_loss = lambda_kl * kl_chunk + lambda_ce * ce_chunk
                grads = torch.autograd.grad(
                    chunk_loss, [h_anchor, *lm_head_params], retain_graph=False
                )
                grad_h_accum.add_(grads[0] * loss_scale)
                for p, g in zip(lm_head_params, grads[1:]):
                    p.grad.add_(g * loss_scale)
            kl_acc = kl_acc + kl_chunk.detach()
            ce_acc = ce_acc + ce_chunk.detach()

    # The caller backwards grad_h_accum into the (bf16) student forward graph
    # (one traversal, combinable with other graph-rooted losses). lm_head's
    # grads are already populated above and are NOT touched there (hidden's
    # graph ends at the post-norm hidden state, before lm_head).
    return kl_acc, ce_acc, grad_h_accum


def _vp_all_reduce(t: torch.Tensor, op, group, world_size: int) -> None:
    """In-place SUM/MAX across the vocab-parallel group; no-op when single-shard."""
    if world_size > 1 and group is not None and dist.is_initialized():
        dist.all_reduce(t, op=op, group=group)


class _VocabParallelKLCE(torch.autograd.Function):
    """Vocab-parallel KL(teacher‖student) + LM CE for one seq-chunk.

    Each rank holds only its vocab shard of the student logits (``s_local``)
    and the teacher logits (``t_local``), both ``(B, c, Vs)``. The global
    (full-vocab) softmax normalizers are formed with three small all-reduces —
    a MAX over per-shard maxima and two SUMs (Σexp and the vocab-summed cross
    terms). Collectives live ONLY in ``forward``; ``backward`` returns the grad
    w.r.t. ``s_local`` with no collective, so autograd flows it through the
    local ``h @ Wᵀ`` matmul to the hidden state and the lm_head shard exactly
    as a dense softmax would (Megatron VocabParallelCrossEntropy pattern,
    extended with the KL term).

    With ``world_size == 1`` (single shard = full vocab, no group) this reduces
    bit-for-bit to a dense forward-KL + CE.
    """

    @staticmethod
    def forward(ctx, s_local, t_local, ce_target, ce_valid, kl_mask, cfg):
        T, lam_kl, lam_ce, kl_denom, ce_denom, group, ws = cfg
        s = s_local.float()
        t = t_local.float()
        s_kl = s / T
        t_kl = t / T

        # 1) global maxima (one MAX all-reduce over [student/T, student-raw, teacher/T]).
        maxes = torch.stack([s_kl.amax(-1), s.amax(-1), t_kl.amax(-1)], dim=0)
        _vp_all_reduce(maxes, dist.ReduceOp.MAX, group, ws)
        gmax_s_kl, gmax_s_ce, gmax_t_kl = maxes[0], maxes[1], maxes[2]

        # 2) global Σexp (one SUM all-reduce over the three exp-sums).
        exp_s_kl = (s_kl - gmax_s_kl.unsqueeze(-1)).exp()
        exp_s_ce = (s - gmax_s_ce.unsqueeze(-1)).exp()
        exp_t_kl = (t_kl - gmax_t_kl.unsqueeze(-1)).exp()
        sums = torch.stack([exp_s_kl.sum(-1), exp_s_ce.sum(-1), exp_t_kl.sum(-1)], dim=0)
        _vp_all_reduce(sums, dist.ReduceOp.SUM, group, ws)
        gsum_s_kl, gsum_s_ce, gsum_t_kl = sums[0], sums[1], sums[2]
        glse_s_kl = gmax_s_kl + gsum_s_kl.log()
        glse_s_ce = gmax_s_ce + gsum_s_ce.log()
        glse_t_kl = gmax_t_kl + gsum_t_kl.log()

        # shard-local probabilities and shard-local cross-term partials.
        p_s_kl = exp_s_kl / gsum_s_kl.unsqueeze(-1)      # student probs (T-scaled)
        p_s_ce = exp_s_ce / gsum_s_ce.unsqueeze(-1)      # student probs (raw)
        p_t_kl = exp_t_kl / gsum_t_kl.unsqueeze(-1)      # teacher probs (T-scaled)
        logps_kl = s_kl - glse_s_kl.unsqueeze(-1)        # log p_s (shard)
        logpt_kl = t_kl - glse_t_kl.unsqueeze(-1)        # log p_t (shard)
        B_term = (p_t_kl * logps_kl).sum(-1)             # Σ_shard p_t·log p_s
        A_term = (p_t_kl * logpt_kl).sum(-1)             # Σ_shard p_t·log p_t

        valid_in_shard = ce_valid & (ce_target >= 0)
        sel_idx = ce_target.clamp(min=0).unsqueeze(-1)
        ce_sel = s.gather(-1, sel_idx).squeeze(-1) * valid_in_shard.to(s.dtype)

        # 3) vocab-summed cross terms (one SUM all-reduce over [B, A, ce_sel]).
        crosses = torch.stack([B_term, A_term, ce_sel], dim=0)
        _vp_all_reduce(crosses, dist.ReduceOp.SUM, group, ws)
        B_full, A_full, ce_sel_full = crosses[0], crosses[1], crosses[2]

        # KL(t‖s) per token = Σ p_t log p_t − Σ p_t log p_s; ×T² (KD convention).
        per_tok_kl = (A_full - B_full) * kl_mask
        kl_val = per_tok_kl.sum() / kl_denom * (T * T)
        # CE per predicting position = global_logsumexp − selected-label logit.
        per_pos_ce = (glse_s_ce - ce_sel_full) * ce_valid.to(glse_s_ce.dtype)
        ce_val = per_pos_ce.sum() / ce_denom

        loss = lam_kl * kl_val + lam_ce * ce_val
        ctx.save_for_backward(p_s_kl, p_t_kl, p_s_ce, ce_target, ce_valid, kl_mask)
        ctx.consts = (T, lam_kl, lam_ce, kl_denom, ce_denom)
        return loss, kl_val.detach(), ce_val.detach()

    @staticmethod
    def backward(ctx, grad_loss, grad_kl, grad_ce):
        p_s_kl, p_t_kl, p_s_ce, ce_target, ce_valid, kl_mask = ctx.saved_tensors
        T, lam_kl, lam_ce, kl_denom, ce_denom = ctx.consts
        # KL grad: ∂kl/∂s = (T/kl_denom)·mask·(p_s − p_t)  (after the ×T² factor).
        g_kl = lam_kl * (T / kl_denom) * kl_mask.unsqueeze(-1) * (p_s_kl - p_t_kl)
        # CE grad: ∂ce/∂s = (1/ce_denom)·valid·(p_s − onehot(label)).
        onehot = torch.zeros_like(p_s_ce)
        valid_in_shard = ce_valid & (ce_target >= 0)
        onehot.scatter_(-1, ce_target.clamp(min=0).unsqueeze(-1),
                        valid_in_shard.unsqueeze(-1).to(p_s_ce.dtype))
        g_ce = lam_ce / ce_denom * ce_valid.unsqueeze(-1).to(p_s_ce.dtype) * (p_s_ce - onehot)
        grad_s = (g_kl + g_ce) * grad_loss
        return grad_s, None, None, None, None, None


def _kl_ce_vocab_parallel(
    hidden: torch.Tensor,        # (B, T, D) bf16, grad-connected to student params
    lm_head: nn.Module,          # this rank's [Vs, H] lm_head shard
    v_lo: int,
    v_hi: int,
    teacher_logits: torch.Tensor,  # (B, T, Vs) detached — this rank's TEACHER vocab shard
    labels: torch.Tensor,          # (B, T)
    attention_mask: torch.Tensor | None,
    *,
    lambda_kl: float,
    lambda_ce: float,
    kl_temperature: float,
    chunk_size: int,
    group,
    world_size: int,
    compute_grads: bool = True,
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Vocab-parallel KL+CE: runs on EVERY rank, each owning vocab rows [v_lo, v_hi).

    Mirrors ``_kl_ce_chunked`` (seq-chunked lm_head, grad-w.r.t.-hidden returned
    for the caller to backward, lm_head grads accumulated in place) but the
    per-chunk softmax is vocab-parallel via ``_VocabParallelKLCE``.
    ``teacher_logits`` is already this rank's ``(B, T, Vs)`` teacher vocab shard
    (the teacher's lm_head is itself vocab-row-sharded), so the teacher's full
    ``(B,c,V)`` fp32 expansion is avoided entirely. Returns
    ``(kl_detached, ce_detached, grad_h_or_None)`` — the scalars identical on
    every rank.

    With ``compute_grads=False`` (validation / metrics-only logging) the chunk
    loop runs under ``no_grad`` — no graph is built and ``grad_h`` is ``None``.
    """
    B, T, D = hidden.shape
    device = hidden.device

    if attention_mask is not None:
        kl_denom = attention_mask.sum().clamp(min=1).float()
    else:
        kl_denom = torch.tensor(B * T, dtype=torch.float32, device=device)
    ce_denom = (labels[:, 1:] != -100).sum().clamp(min=1).float()

    if compute_grads:
        h_src = hidden.detach().requires_grad_(True)
        grad_h_accum = torch.zeros_like(h_src)
        W = lm_head.weight
        if W.grad is None:
            W.grad = torch.zeros_like(W)
    else:
        h_src = hidden
        grad_h_accum = None

    kl_acc = hidden.new_zeros((), dtype=torch.float32)
    ce_acc = hidden.new_zeros((), dtype=torch.float32)
    Vs = v_hi - v_lo

    with torch.set_grad_enabled(compute_grads):
        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            c = t1 - t0
            h_chunk = h_src[:, t0:t1, :]
            s_local = lm_head(h_chunk)                          # (B, c, Vs)
            t_local = teacher_logits[:, t0:t1, :]               # (B, c, Vs) teacher shard, detached

            # CE next-token shift: position p predicts labels[p+1]; the final
            # position (T-1) has no label and is skipped. ce_target is the label's
            # index within THIS shard, or -1 when the label lives on another shard.
            pos = torch.arange(t0, t1, device=device)
            valid_pos = (pos < T - 1).unsqueeze(0).expand(B, c)
            lab = labels[:, (pos + 1).clamp(max=T - 1)]          # (B, c)
            ce_valid = valid_pos & (lab != -100)
            local_lab = lab - v_lo
            ce_target = torch.where(
                ce_valid & (local_lab >= 0) & (local_lab < Vs),
                local_lab, torch.full_like(local_lab, -1),
            )
            kl_mask = (
                attention_mask[:, t0:t1].to(torch.float32)
                if attention_mask is not None
                else torch.ones(B, c, device=device)
            )

            cfg = (kl_temperature, lambda_kl, lambda_ce, kl_denom, ce_denom, group, world_size)
            loss, kl_d, ce_d = _VocabParallelKLCE.apply(
                s_local.float(), t_local.float(), ce_target, ce_valid, kl_mask, cfg
            )
            if compute_grads:
                grads = torch.autograd.grad(loss, [h_src, W], retain_graph=False)
                grad_h_accum.add_(grads[0] * loss_scale)
                W.grad.add_(grads[1] * loss_scale)
            kl_acc = kl_acc + kl_d
            ce_acc = ce_acc + ce_d

    # The caller backwards grad_h_accum into the student forward graph (one
    # traversal, combinable with other graph-rooted losses).
    return kl_acc, ce_acc, grad_h_accum


@dataclass
class DistillConfig:
    sync_layer_indices: tuple[int, ...]
    lambda_block: float = 1.0
    lambda_kl: float = 1.0
    lambda_ce: float = 0.5
    kl_temperature: float = 1.0
    # Relative (scale-free) block MSE: Σ(s−t)²/Σt² per block instead of the raw
    # masked mean. Rescales the block term to O(1)/block so deep, large-norm
    # layers no longer dominate the gradient. See losses.block_mse.
    normalize_block_mse: bool = False
    # Cap the (normalized) per-block relative MSE so a single student-forced batch
    # whose ratio blows up can't spike the gradient and trip the grad-norm clip.
    # None = no clamp. Only applies when normalize_block_mse is True.
    block_mse_clamp: float | None = None
    # Supervise EVERY layer inside each sync window, not just the boundary.
    # At each within-window layer the synced reconstruction is MSE'd against the
    # teacher's hidden at that depth (the forward still feeds each track its
    # PARTIAL residual — the loss taps are sync-for-loss-only), pinning the
    # within-window layers (which run on partial residuals at D≥2) to the teacher
    # trajectory. The per-window loss is AVERAGED over its layers so the loss
    # scale (and `lambda_block`) stays comparable to the boundary-only path; at
    # D=1 (one layer per window) it is bit-identical to the legacy boundary MSE.
    # Requires the teacher to be hooked at every layer (see scripts/train_*.py).
    intra_window_mse: bool = False
    # Free-running feature matching: relative-MSE the END-TO-END student forward's
    # synced hiddens (the same full forward the KL/CE pass uses — the student runs
    # on its OWN hiddens throughout) against the teacher hiddens at the sync
    # boundaries, with gradients flowing through the WHOLE forward. Unlike the
    # block loop (detached at every boundary) this trains multi-window error
    # compounding directly — the deep free-running relMSE plateau the block loop
    # cannot see. Reuses the already-paid student_fwd pass, so the marginal cost
    # is ~zero (it shares the single full-graph backward with KL/CE).
    free_running_mse: bool = False
    # Weight on the free-running feature-matching term (multiplied by the
    # per-step schedule scale passed to distill_step). The term is always the
    # relative MSE (mean over taps, clamped by block_mse_clamp), so 1.0 is
    # comparable to a normalized lambda_block.
    lambda_free_running: float = 1.0
    # Which sync boundaries the free-running MSE supervises: "all", or
    # "deep-half" (the deeper half — where the free-running error concentrates;
    # halves the retained-teacher-hidden memory).
    free_running_taps: str = "all"
    # Per-boundary gradient damping of the free-running unroll: during the FULL
    # student forward the hidden continuing past each sync boundary is replaced
    # by ``h.detach() + alpha*(h - h.detach())`` — forward value exactly h, but
    # a tap j's gradient into window w is scaled alpha^(j-w). This geometrically
    # truncates the through-depth Jacobian products whose amplification (and the
    # gain-raising feedback they reward) makes the raw full-unroll term diverge.
    # 1.0 = legacy full unroll; 0.0 = each tap trains only its own window on the
    # true free-running input. The taps themselves are stored undamped. Applies
    # only to the full forward (the block loop already detaches per boundary);
    # the KL/CE backward shares the damped graph, so alpha<1 requires
    # lambda_kl == lambda_ce == 0.
    fr_grad_alpha: float = 1.0
    # Seq-chunk size for the KL+CE pass. The (B, T, V) fp32 softmax expansions
    # would OOM at training seq_len; chunking caps the per-chunk transient at
    # (chunk_size/T)x. Under vocab-parallel the expansion is per-rank only
    # (B, chunk, V/world_size). Each chunk costs 3 small all-reduces + one
    # autograd.grad, and profiling shows the klce phase is collective/dispatch-
    # bound, so #chunks = ceil(T/chunk) is the cost driver — 512 (8 chunks at
    # T=4096 vs 32 at 128) cuts klce collectives ~4x at negligible extra memory
    # (per-rank fp32 transient (B, chunk, V/world) ≈ 63 MB/tensor at B=1). The
    # transient grows ×B, so at large batch keep this moderate rather than maxing.
    kl_ce_chunk_size: int = 512


def _combine_seed(forcing_seed: object) -> object:
    """Fold a tuple/list seed (e.g. ``(seed, step)``) into a single int.

    Pure integer arithmetic, so the result is deterministic and
    process-independent — every rank derives the same value (unlike ``hash`` of
    str/bytes, which is randomized under PYTHONHASHSEED). Non-tuple seeds
    (int/str/bytes/None) pass straight through to ``random.Random``.
    """
    if isinstance(forcing_seed, (tuple, list)):
        combined = 0
        for v in forcing_seed:
            combined = combined * 1_000_003 + int(v)
        return combined
    return forcing_seed


def _effective_block_weights(
    adaptive_weights: "dict[int, float] | None",
    tap_indices: list[int],
) -> dict[int, float]:
    """Per-tap block-MSE weights from the optional adaptive (running-relMSE) map.

    ``adaptive_weights is None`` ⇒ all-ones (uniform; bit-identical to the
    unweighted path). When given, each tap weight is ``adaptive_weights.get(l,
    1.0)`` renormalized to **mean 1 over the supervised taps**, so the total
    block-loss magnitude (and the meaning of ``lambda_block``) is preserved while
    gradient budget shifts toward the taps with the largest running relative error.
    """
    if adaptive_weights is None:
        return {l: 1.0 for l in tap_indices}
    raw = {l: adaptive_weights.get(l, 1.0) for l in tap_indices}
    mean = sum(raw.values()) / max(1, len(raw))
    if mean <= 0:
        return {l: 1.0 for l in tap_indices}
    return {l: w / mean for l, w in raw.items()}


def adaptive_weights_from_relmse(
    relmse: "dict[int, float]", power: float = 1.0
) -> dict[int, float]:
    """Mean-1 per-tap weights from a running relative-MSE map.

    ``weight(l) ∝ relmse[l] ** power``, normalized so the mean over taps is 1 (the
    form ``_effective_block_weights`` consumes). ``power`` sharpens (>1) or softens
    (<1) the tilt toward the worst-fitting layers. Empty input ⇒ empty output; an
    all-zero map ⇒ uniform 1.0 (no information to tilt on yet).
    """
    if not relmse:
        return {}
    raw = {l: max(0.0, v) ** power for l, v in relmse.items()}
    mean = sum(raw.values()) / len(raw)
    if mean <= 0:
        return {l: 1.0 for l in relmse}
    return {l: w / mean for l, w in raw.items()}


def student_forcing_schedule(
    step: int,
    prob: float,
    warmup: int,
    max_steps: int,
    shape: str = "hold",
    power: float = 1.0,
) -> float:
    """Per-step student-forcing probability under one of two schedule shapes.

    ``shape="hold"`` (default): the legacy ramp — linear ``prob * min(1, step/warmup)``
    that reaches ``prob`` at ``warmup`` and then HOLDS there for the rest of the run
    (``warmup == 0`` ⇒ constant ``prob`` from step 0). Bit-identical to the old inline
    computation. ``power`` is ignored.

    ``shape="cosine-full"``: a free-running CURRICULUM that ramps 0 → ``prob`` across
    the WHOLE run with a cosine ease, so the high-forcing regime (where deep blocks
    compound free-running drift) is approached gently and only reached near the end.
    This closes the train(teacher-forced) / eval(free-running) gap without the unstable
    long tail of holding at a high ``prob``. ``warmup`` is ignored in this shape (the
    whole run is the ramp).

    ``power`` (>0) sets the steepness: the per-step *gap to target*
    ``gap = 0.5*(1 + cos(pi*frac))`` decays 1 → 0 over the run, and ``sf_p =
    prob*(1 - gap**power)``. ``power=1`` recovers the plain cosine ``prob*0.5*(1 -
    cos(pi*frac))`` (bit-identical); ``power>1`` closes the gap faster, reaching the
    high-forcing regime EARLIER (steeper curriculum — useful on long runs where the
    plain cosine only reaches high forcing near the end); ``power<1`` reaches it later.
    """
    if shape == "cosine-full":
        if max_steps <= 0:
            return prob
        frac = min(1.0, max(0.0, step / max_steps))
        if power == 1.0:  # exact legacy expression (bit-identical default)
            return prob * 0.5 * (1.0 - math.cos(math.pi * frac))
        gap = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 → 0 over the run
        return prob * (1.0 - gap ** power)
    # "hold": legacy linear-ramp-then-hold.
    if warmup > 0:
        return prob * min(1.0, step / warmup)
    return prob


def _block_ranges(num_layers: int, sync_indices: tuple[int, ...]) -> list[tuple[int, int]]:
    """Convert sync layer indices into (start_layer, end_layer_inclusive) ranges."""
    ranges = []
    prev_end = -1
    for idx in sync_indices:
        ranges.append((prev_end + 1, idx))
        prev_end = idx
    if prev_end != num_layers - 1:
        ranges.append((prev_end + 1, num_layers - 1))
    return ranges


def _run_student_block(
    student: PTWrappedModel,
    h_in: torch.Tensor,
    start: int,
    end_inclusive: int,
    position_embeddings,
    text_position_ids,
    causal_mask,
    linear_attn_mask,
) -> list[torch.Tensor]:
    """Run student layers [start..end_inclusive] on each of the K local tracks.

    Every track starts from the same `h_in` (the synced pre-block hidden).
    Returns the K post-block tensors (one per local track), without sync.
    The caller calls `student.sync_module(h_post_list, h_in)` once to produce
    the single synced post-block tensor.
    """
    per_track_h = [h_in for _ in student.text_models]
    for layer_idx in range(start, end_inclusive + 1):
        new_h: list[torch.Tensor] = []
        for k, tm in enumerate(student.text_models):
            layer = tm.layers[layer_idx]
            layer_mask = (
                linear_attn_mask
                if tm.config.layer_types[layer_idx] == "linear_attention"
                else causal_mask
            )
            new_h.append(
                layer(
                    per_track_h[k],
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
            )
        per_track_h = new_h
    return per_track_h


def distill_step(
    student: PTWrappedModel,
    teacher: HookedTeacher,
    batch: dict[str, torch.Tensor],
    cfg: DistillConfig,
    timings: dict[str, float] | None = None,
    mem: dict[str, dict[str, float]] | None = None,
    student_forcing_prob: float = 0.0,
    forcing_seed: object = 0,
    loss_scale: float = 1.0,
    adaptive_weights: "dict[int, float] | None" = None,
    track_layer_relmse: bool = False,
    free_running_scale: float = 1.0,
    compute_klce_metrics: bool = True,
) -> dict[str, torch.Tensor]:
    """Run one distillation step. Backward is done internally, per block.

    Each block's autograd graph is freed before the next block's forward, so
    peak memory holds at most a single block's activations rather than every
    block plus the final forward simultaneously. Mathematically equivalent to
    a single `backward()` on the summed loss because the block forwards are
    independent — each block reads its input ``.detach()``ed.

    Scheduled sampling (``student_forcing_prob`` > 0): per block, with probability
    ``student_forcing_prob`` the *next* block is fed the student's own synced
    hidden (``h_synced.detach()``) instead of the teacher's hidden, while the MSE
    target stays the teacher hidden. This trains each block to correct from a
    drifted (student) input toward the teacher output — closing the exposure-bias
    gap between teacher-forced training and free-running inference. The decision is
    drawn from ``random.Random(forcing_seed)`` so it is IDENTICAL on every rank:
    all ranks run the same block on the same input and all-reduce in
    ``sync_module``; a per-rank-divergent choice would corrupt that sum.
    ``forcing_prob == 0`` reproduces the legacy fully-teacher-forced path bit-for-bit.

    ``loss_scale`` (gradient accumulation: 1/grad_accum_steps) multiplies every
    backward'd loss so the grads accumulated across microbatches average rather than
    sum; the returned scalars stay UNSCALED for logging. ``adaptive_weights`` (per
    supervised-tap layer index → weight) sets the per-layer block-MSE weight
    (renormalized to mean 1; ``None`` ⇒ uniform, bit-identical to the unweighted
    path). ``track_layer_relmse`` additionally returns a detached
    per-tap relative MSE (``Σ(s−t)²/Σt²``) under ``losses["layer_relmse"]`` — the signal
    the caller folds into the adaptive-weight EMA. The relMSE is computed from the
    *synced* hidden (identical on every rank), so the EMA stays in lock-step.

    ``free_running_scale`` multiplies ``cfg.lambda_free_running`` for this step
    (the caller's ramp schedule, mirroring ``student_forcing_prob``); it depends
    only on the step, so it is identical on every rank. When the effective
    free-running weight is 0 (ramp start) the term is still computed for logging
    but under ``no_grad``.

    ``compute_klce_metrics=False``: when the logit losses are OFF
    (``lambda_kl == lambda_ce == 0``) the whole KL/CE pass — the teacher
    lm_head matmul feeding it included — is metrics-only logging. Passing
    False skips it entirely (zero ``kl``/``ce`` returned); the caller enables
    it only on the steps whose losses are actually printed. With either
    lambda non-zero the pass always runs (the flag only gates metrics-only
    work). Must be identical on every rank (it gates collectives).

    Returns a dict of *detached* scalar tensors for logging. The caller is
    responsible for ``sync_replicated_grads(plan)``, clip, and ``optim.step()``.
    """
    # A tuple seed (e.g. ``(seed, step)``) is combined into a single int so every
    # rank derives the same generator — see _combine_seed.
    forcing_rng = random.Random(_combine_seed(forcing_seed))
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]

    # KL/CE work is needed when the logit losses backward (a lambda is non-zero)
    # or when the caller wants the metrics logged this step. Both are step-level
    # constants, identical on every rank, so the collectives they gate (teacher
    # logit gathers, vocab-parallel softmax reduces) stay matched.
    need_klce_grads = cfg.lambda_kl != 0.0 or cfg.lambda_ce != 0.0
    need_klce = need_klce_grads or compute_klce_metrics

    # ----- Teacher forward (frozen) -----
    # The teacher logits feed only the KL/CE pass — skip the lm_head matmul
    # (and its gather, in batch-sharded mode) when that pass won't run.
    with _phase("teacher_fwd", timings, mem):
        teacher_logits, teacher_hiddens = teacher.forward(
            input_ids, attention_mask=attention_mask, need_logits=need_klce,
        )

    # ----- Embedding broadcast -----
    # Vocab-parallel: each rank embeds its vocab shard, summed across ranks.
    # Legacy: owner embeds the full vocab, peers contribute zeros. Either way
    # one all-reduce (inside student.embed) delivers the embedding to every track.
    with _phase("setup", timings, mem):
        inputs_embeds = student.embed(input_ids)

        tm0 = student.text_models[0]
        position_ids, text_position_ids = tm0._resolve_position_ids(inputs_embeds, None)
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # local import

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

    # ----- Per-block teacher-forced backward -----
    # Each iteration: run the student block, sync, compute block_mse, backward
    # immediately, drop the graph. Gradients accumulate on student params
    # across iterations; the next block's forward starts fresh from a
    # detached teacher hidden state.
    num_layers = len(tm0.layers)
    ranges = _block_ranges(num_layers, cfg.sync_layer_indices)
    # Supervised taps: every layer (intra-window) or just the sync boundaries.
    # Per-tap block-MSE weight = optional adaptive (running-relMSE) weight, mean-1
    # over the taps (adaptive None ⇒ uniform, bit-identical to the unweighted path).
    if cfg.intra_window_mse:
        tap_indices = [l for (s, e) in ranges for l in range(s, e + 1)]
    else:
        tap_indices = [e for (_, e) in ranges]
    eff_w = _effective_block_weights(adaptive_weights, tap_indices)
    # Free-running feature matching: effective weight for THIS step, whether the
    # full-forward graph must carry gradients, and which boundary teacher hiddens
    # to retain past the block loop (normally they're released as consumed). All
    # of this depends only on cfg + step-level scalars — identical on every rank.
    eff_lambda_fr = cfg.lambda_free_running * free_running_scale if cfg.free_running_mse else 0.0
    do_fr_backward = eff_lambda_fr != 0.0
    fwd_needs_grad = need_klce_grads or do_fr_backward
    # The full student forward feeds the KL/CE pass and the free-running MSE.
    # When neither runs this step (block-only recipe, metrics not being
    # logged), the forward itself is pure waste — skip it. Step-level
    # constant, identical on every rank (the skipped sync all-reduces match).
    run_full_forward = need_klce or cfg.free_running_mse
    boundaries = [e for (_, e) in ranges]
    if cfg.free_running_mse:
        fr_tap_set = set(
            boundaries if cfg.free_running_taps == "all" else boundaries[len(boundaries) // 2:]
        )
    else:
        fr_tap_set: set[int] = set()
    fr_targets: dict[int, torch.Tensor] = {}
    # Detached per-tap relative MSE signal for the adaptive-weight EMA (only when
    # requested; the loss values are unaffected either way — the signal is detached).
    layer_relmse: dict[int, torch.Tensor] = {}

    def tap_loss_and_relmse(
        h_synced: torch.Tensor, t_target: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Per-tap (loss, detached relmse-or-None) in ONE block_mse eval where possible.

        On the normalized path the unclamped relative MSE doubles as the loss
        (clamped after — same op as clamping inside block_mse, so the loss is
        bit-identical to the former two-call form) and, detached, as the
        adaptive-EMA signal. Only the raw-loss + tracking combination still
        needs a second (normalized) eval.
        """
        if cfg.normalize_block_mse:
            r = block_mse(h_synced, t_target, attention_mask=attention_mask, normalize=True)
            rel = r.detach() if track_layer_relmse else None
            if cfg.block_mse_clamp is not None:
                r = r.clamp(max=cfg.block_mse_clamp)
            return r, rel
        loss = block_mse(h_synced, t_target, attention_mask=attention_mask, normalize=False)
        rel = (
            block_mse(h_synced, t_target, attention_mask=attention_mask, normalize=True).detach()
            if track_layer_relmse
            else None
        )
        return loss, rel
    block_loss_val = torch.zeros((), device=input_ids.device)
    # Block 0 reads the synced student embedding (detached); subsequent blocks
    # read the teacher's post-block hidden state at the previous sync index.
    prev_h = inputs_embeds.detach()
    with _phase("block_loop", timings, mem):
        for start, end in ranges:
            if cfg.intra_window_mse:
                # Per-layer supervision: run the window track-locally and at EACH
                # layer compute the synced reconstruction and MSE it against the
                # teacher hidden at that depth. The synced taps are loss-only — the
                # forward keeps feeding each track its PARTIAL hidden to the next
                # layer (the D≥2 semantics). h_synced after the last layer is the
                # real boundary reconstruction (what carries to the next window).
                per_track_h = [prev_h for _ in student.text_models]
                win_loss = torch.zeros((), device=input_ids.device)
                t_end = None
                for layer_idx in range(start, end + 1):
                    new_h: list[torch.Tensor] = []
                    for k, tm in enumerate(student.text_models):
                        layer = tm.layers[layer_idx]
                        layer_mask = (
                            linear_attn_mask
                            if tm.config.layer_types[layer_idx] == "linear_attention"
                            else causal_mask
                        )
                        new_h.append(
                            layer(
                                per_track_h[k],
                                position_embeddings=position_embeddings,
                                attention_mask=layer_mask,
                                position_ids=text_position_ids,
                                past_key_values=None,
                                use_cache=False,
                            )
                        )
                    h_synced = student.sync_module(new_h, prev_h)
                    t_l = teacher_hiddens.pop(layer_idx).detach()
                    if layer_idx == end:
                        t_end = t_l  # boundary target, reused for teacher forcing
                        if end in fr_tap_set:
                            # Retain (by reference) for the free-running MSE after
                            # the block loop — extends this hidden's lifetime only.
                            fr_targets[end] = t_end
                    tap_loss, tap_rel = tap_loss_and_relmse(h_synced, t_l)
                    win_loss = win_loss + eff_w[layer_idx] * tap_loss
                    if tap_rel is not None:
                        layer_relmse[layer_idx] = tap_rel
                    per_track_h = new_h
                # Average over the window so the scale (and lambda_block) matches
                # the boundary-only path; identical to it when the window is 1 layer.
                block_loss_b = win_loss / (end - start + 1)
            else:
                h_post_list = _run_student_block(
                    student,
                    prev_h,
                    start,
                    end,
                    position_embeddings,
                    text_position_ids,
                    causal_mask,
                    linear_attn_mask,
                )
                h_synced = student.sync_module(h_post_list, prev_h)
                # pop (not index): release this hidden from the captures dict once
                # consumed so only ~1 teacher hidden stays resident, not all 8.
                # The teacher hidden is always the MSE target (detached: no grad
                # flows back into the teacher).
                t_end = teacher_hiddens.pop(end).detach()
                if end in fr_tap_set:
                    # Retain (by reference) for the free-running MSE after the
                    # block loop — extends this hidden's lifetime only.
                    fr_targets[end] = t_end
                tap_loss, tap_rel = tap_loss_and_relmse(h_synced, t_end)
                block_loss_b = eff_w[end] * tap_loss
                if tap_rel is not None:
                    layer_relmse[end] = tap_rel
            if cfg.lambda_block != 0.0:
                (cfg.lambda_block * block_loss_b * loss_scale).backward()
            block_loss_val = block_loss_val + block_loss_b.detach()
            # Next block's input: teacher hidden (teacher forcing) or the
            # student's own synced output (student forcing). Either way detach()
            # keeps the per-block backward memory-bounded. The draw is seeded by
            # forcing_seed so it is identical on every rank, keeping the cross-rank
            # SyncBoundary choices in lock-step.
            use_student = forcing_rng.random() < student_forcing_prob
            prev_h = h_synced.detach() if use_student else t_end

    # ----- Final-logit KL + LM CE (full student forward, chunked lm_head) -----
    # All ranks call the full forward so cross-track SyncBoundary all-reduces
    # line up. We request the pre-lm_head hidden state so the KL+CE can chunk
    # the lm_head application itself — avoids materializing a (B, T, V) bf16
    # logits tensor. When NOTHING will backward through this forward (all of
    # lambda_kl / lambda_ce / the free-running term are 0) it runs under
    # no_grad: kl/ce/fr_mse are still computed for logging, but the graph —
    # and the full (previously zero-gradient) backward — are skipped entirely.
    hidden, sync_hiddens = None, None
    if run_full_forward:
        with _phase("student_fwd", timings, mem):
            with torch.set_grad_enabled(fwd_needs_grad):
                hidden, sync_hiddens = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_sync_hiddens=cfg.free_running_mse,
                    return_hidden_pre_lm_head=True,
                    boundary_grad_alpha=cfg.fr_grad_alpha,
                )

    # ----- Free-running feature matching -----
    # Relative-MSE the free-running forward's synced hiddens against the teacher
    # hiddens at the tapped boundaries. Gradients (when the effective weight is
    # non-zero) flow through the WHOLE forward — the only loss that trains
    # multi-window error compounding. Each tap is checkpointed so the fp32
    # (s−t)² intermediate (~(B,T,H) per tap) is recomputed in backward instead
    # of held across all taps; the saved inputs are bf16 refs already resident.
    fr_val = torch.zeros((), device=input_ids.device)
    fr_layer_relmse: dict[int, torch.Tensor] = {}
    fr_root = None
    if cfg.free_running_mse:
        with _phase("fr_mse", timings, mem):
            terms = []
            for l in sorted(fr_targets):
                if do_fr_backward:
                    r = checkpoint(
                        block_mse, sync_hiddens[l], fr_targets[l], attention_mask,
                        True, 1e-6, cfg.block_mse_clamp,   # normalize=True, eps, clamp
                        use_reentrant=False, preserve_rng_state=False,
                    )
                else:
                    with torch.no_grad():
                        r = block_mse(
                            sync_hiddens[l], fr_targets[l], attention_mask=attention_mask,
                            normalize=True, clamp_max=cfg.block_mse_clamp,
                        )
                fr_layer_relmse[l] = r.detach()
                terms.append(r)
            # Mean over taps: the term stays O(1) regardless of tap count, so
            # lambda_free_running keeps its meaning across D / taps choices.
            fr_loss = torch.stack(terms).mean()
            fr_val = fr_loss.detach()
            if do_fr_backward:
                fr_root = fr_loss * (eff_lambda_fr * loss_scale)

    # Metrics-only KL/CE (logit lambdas 0, computed just for logging) builds no
    # graph, so smaller chunks cost almost nothing — cap at 256 to halve the
    # fp32 (B, chunk, V/world) transients that otherwise set the klce peak.
    # Gradient-carrying passes keep the configured chunk untouched.
    klce_chunk = (
        cfg.kl_ce_chunk_size if need_klce_grads else min(cfg.kl_ce_chunk_size, 256)
    )
    grad_h = None
    if not need_klce:
        # Logit losses off AND metrics not wanted this step: the KL/CE pass —
        # student lm_head + softmax reduces — is skipped wholesale. Zeros keep
        # the returned dict's shape/semantics for the caller's accumulation.
        kl_val = torch.zeros((), device=input_ids.device)
        ce_val = torch.zeros((), device=input_ids.device)
    elif student.vocab_parallel:
        # Vocab-parallel: EVERY rank computes its vocab shard's KL+CE and the
        # softmax normalizers are all-reduced — no rank-0 serial tail, and the
        # full-vocab fp32 expansion is sharded 1/world_size.
        with _phase("klce", timings, mem):
            kl_val, ce_val, grad_h = _kl_ce_vocab_parallel(
                hidden, student.lm_head, student.v_lo, student.v_hi,
                teacher_logits.detach(), labels, attention_mask,
                lambda_kl=cfg.lambda_kl, lambda_ce=cfg.lambda_ce,
                kl_temperature=cfg.kl_temperature, chunk_size=klce_chunk,
                group=student.vp_group, world_size=student.vp_world_size,
                loss_scale=loss_scale, compute_grads=need_klce_grads,
            )
    elif student.lm_head is not None:
        # Legacy: only the track-0 owner has lm_head; peers run the forward for
        # collective ordering, contribute zero KL/CE, and never backward through it.
        with _phase("klce", timings, mem):
            kl_val, ce_val, grad_h = _kl_ce_chunked(
                hidden, student.lm_head, teacher_logits.detach(), labels, attention_mask,
                lambda_kl=cfg.lambda_kl, lambda_ce=cfg.lambda_ce,
                kl_temperature=cfg.kl_temperature, chunk_size=klce_chunk,
                loss_scale=loss_scale, compute_grads=need_klce_grads,
            )
    else:
        kl_val = torch.zeros((), device=input_ids.device)
        ce_val = torch.zeros((), device=input_ids.device)

    # ----- One backward through the full free-running forward graph -----
    # KL/CE inject their accumulated grad at the pre-lm_head hidden; the
    # free-running MSE is rooted at the tapped sync hiddens. Backwarding both
    # together traverses the (shared) graph ONCE — no retain_graph. The
    # SyncBoundary all-reduce is autograd-invisible (in-place on the partial
    # sum), so this backward is collective-free on every rank.
    with _phase("bwd_full", timings, mem):
        roots: list[torch.Tensor] = []
        grad_tensors: list[torch.Tensor | None] = []
        if grad_h is not None:
            roots.append(hidden)
            grad_tensors.append(grad_h)
        if fr_root is not None:
            roots.append(fr_root)
            grad_tensors.append(None)  # scalar root
        if roots:
            torch.autograd.backward(roots, grad_tensors)

    total_val = (
        cfg.lambda_block * block_loss_val
        + cfg.lambda_kl * kl_val
        + cfg.lambda_ce * ce_val
        + eff_lambda_fr * fr_val
    )
    return {
        "total": total_val,
        "block_mse": block_loss_val,
        "kl": kl_val,
        "ce": ce_val,
        # Free-running feature-matching relMSE (mean over taps; zeros when off).
        "fr_mse": fr_val,
        # Per-tap relative MSE (detached; empty unless track_layer_relmse). NOT a
        # scalar loss — the caller reads it to update the adaptive-weight EMA.
        "layer_relmse": layer_relmse,
        # Per-tap free-running relMSE (detached; empty unless free_running_mse).
        # Observability only — not folded into the adaptive EMA.
        "fr_layer_relmse": fr_layer_relmse,
    }


@torch.no_grad()
def validate_step(
    student: PTWrappedModel,
    batch: dict[str, torch.Tensor],
    teacher: HookedTeacher,
    kl_temperature: float = 1.0,
    chunk_size: int = 512,
) -> dict[str, torch.Tensor]:
    """Forward-only KL(teacher || student) and LM CE on a held-out batch.

    All ranks call the full student forward so the cross-track SyncBoundary
    all-reduces match.

    Vocab-parallel: every rank computes its vocab shard and the all-reduced
    KL/CE are GLOBAL (identical on every rank) — the caller must NOT sum them
    across ranks (that would over-count by world_size); divide by world_size or
    read any single rank.

    Legacy: only the track-0 owner has lm_head and computes the metrics; peers
    return zero placeholders, so the caller aggregates with all_reduce SUM.

    Always metrics-only (no graph), so the KL/CE chunk is capped at 256
    regardless of ``chunk_size`` — halves the fp32 ``(B, chunk, V/world)``
    transients (validation has OOM'd on them historically) at negligible cost.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]

    if student.vocab_parallel:
        hidden, _ = student(
            input_ids=input_ids, attention_mask=attention_mask,
            return_sync_hiddens=False, return_hidden_pre_lm_head=True,
        )
        teacher_logits, _ = teacher.forward(input_ids, attention_mask=attention_mask)
        kl, ce, _ = _kl_ce_vocab_parallel(
            hidden, student.lm_head, student.v_lo, student.v_hi,
            teacher_logits.detach(), labels, attention_mask,
            lambda_kl=1.0, lambda_ce=1.0, kl_temperature=kl_temperature,
            chunk_size=min(chunk_size, 256), group=student.vp_group,
            world_size=student.vp_world_size, compute_grads=False,
        )
        return {"ce": ce, "kl": kl}

    student_logits, _ = student(
        input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=False
    )
    if student_logits is not None:
        ce = lm_cross_entropy(student_logits, labels)
        teacher_logits, _ = teacher.forward(input_ids, attention_mask=attention_mask)
        kl = logit_kl(
            student_logits, teacher_logits, attention_mask, temperature=kl_temperature
        )
    else:
        ce = torch.zeros((), device=input_ids.device)
        kl = torch.zeros((), device=input_ids.device)
    return {"ce": ce, "kl": kl}
