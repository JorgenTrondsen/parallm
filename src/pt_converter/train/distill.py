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

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute lambda_kl * KL + lambda_ce * CE in seq-chunks; one backward
    into the student forward graph at the end.

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
      - Once all chunks are done, ``hidden.backward(grad_h_accum)`` drives
        a single backward through the (bf16) student forward graph. No
        ``retain_graph`` is needed anywhere, and no full-T logits tensor is
        ever materialized.

    Returns ``(kl_detached, ce_detached)``. Backward is already done.
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

    h_anchor = hidden.detach().requires_grad_(True)
    grad_h_accum = torch.zeros_like(h_anchor)
    # lm_head's own parameter gradients must be accumulated alongside
    # grad_h_accum — autograd.grad with only h_anchor in `inputs` would
    # silently drop them. Initialize `.grad` so we can add into it in place.
    lm_head_params = [p for p in lm_head.parameters() if p.requires_grad]
    for p in lm_head_params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    kl_acc = hidden.new_zeros((), dtype=torch.float32)
    ce_acc = hidden.new_zeros((), dtype=torch.float32)

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

        chunk_loss = lambda_kl * kl_chunk + lambda_ce * ce_chunk
        grads = torch.autograd.grad(
            chunk_loss, [h_anchor, *lm_head_params], retain_graph=False
        )
        grad_h_accum.add_(grads[0])
        for p, g in zip(lm_head_params, grads[1:]):
            p.grad.add_(g)
        kl_acc = kl_acc + kl_chunk.detach()
        ce_acc = ce_acc + ce_chunk.detach()

    # Single backward into the (bf16) student forward graph. lm_head's grads
    # are already populated above and are NOT touched here (hidden's graph
    # ends at the post-norm hidden state, before lm_head).
    hidden.backward(grad_h_accum)
    return kl_acc, ce_acc


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vocab-parallel KL+CE: runs on EVERY rank, each owning vocab rows [v_lo, v_hi).

    Mirrors ``_kl_ce_chunked`` (seq-chunked lm_head, one ``hidden.backward`` at
    the end, lm_head grads accumulated in place) but the per-chunk softmax is
    vocab-parallel via ``_VocabParallelKLCE``. ``teacher_logits`` is already this
    rank's ``(B, T, Vs)`` teacher vocab shard (the teacher's lm_head is itself
    vocab-row-sharded), so the teacher's full ``(B,c,V)`` fp32 expansion is
    avoided entirely. Returns the (unweighted) detached KL and CE sums, identical
    on every rank.

    With ``compute_grads=False`` (validation) no graph / backward is built — it
    just accumulates the global KL and CE scalars.
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

    kl_acc = hidden.new_zeros((), dtype=torch.float32)
    ce_acc = hidden.new_zeros((), dtype=torch.float32)
    Vs = v_hi - v_lo

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
            grad_h_accum.add_(grads[0])
            W.grad.add_(grads[1])
        kl_acc = kl_acc + kl_d
        ce_acc = ce_acc + ce_d

    if compute_grads:
        hidden.backward(grad_h_accum)
    return kl_acc, ce_acc


@dataclass
class DistillConfig:
    sync_layer_indices: tuple[int, ...]
    lambda_block: float = 1.0
    lambda_kl: float = 1.0
    lambda_ce: float = 0.5
    kl_temperature: float = 1.0
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
) -> dict[str, torch.Tensor]:
    """Run one distillation step. Backward is done internally, per block.

    Each block's autograd graph is freed before the next block's forward, so
    peak memory holds at most a single block's activations rather than every
    block plus the final forward simultaneously. Mathematically equivalent to
    a single `backward()` on the summed loss because the block forwards are
    independent (teacher-forced — each block reads ``prev_teacher_h.detach()``).

    Returns a dict of *detached* scalar tensors for logging. The caller is
    responsible for ``sync_replicated_grads(plan)``, clip, and ``optim.step()``.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]

    # ----- Teacher forward (frozen) -----
    with _phase("teacher_fwd", timings, mem):
        teacher_logits, teacher_hiddens = teacher.forward(input_ids, attention_mask=attention_mask)

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
    ranges = _block_ranges(len(tm0.layers), cfg.sync_layer_indices)
    block_loss_val = torch.zeros((), device=input_ids.device)
    # Block 0 reads the synced student embedding (detached); subsequent blocks
    # read the teacher's post-block hidden state at the previous sync index.
    prev_h = inputs_embeds.detach()
    with _phase("block_loop", timings, mem):
        for start, end in ranges:
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
            # Reused as both the MSE target and the next block's input (detach:
            # teacher-forced, no grad flows back into the teacher).
            t_end = teacher_hiddens.pop(end).detach()
            block_loss_b = block_mse(h_synced, t_end, attention_mask=attention_mask)
            (cfg.lambda_block * block_loss_b).backward()
            block_loss_val = block_loss_val + block_loss_b.detach()
            prev_h = t_end

    # ----- Final-logit KL + LM CE (full student forward, chunked lm_head) -----
    # All ranks call the full forward so cross-track SyncBoundary all-reduces
    # line up. We request the pre-lm_head hidden state so the KL+CE can chunk
    # the lm_head application itself — avoids materializing a (B, T, V) bf16
    # logits tensor.
    with _phase("student_fwd", timings, mem):
        hidden, _ = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=False,
            return_hidden_pre_lm_head=True,
        )
    if student.vocab_parallel:
        # Vocab-parallel: EVERY rank computes its vocab shard's KL+CE and the
        # softmax normalizers are all-reduced — no rank-0 serial tail, and the
        # full-vocab fp32 expansion is sharded 1/world_size.
        with _phase("klce", timings, mem):
            kl_val, ce_val = _kl_ce_vocab_parallel(
                hidden, student.lm_head, student.v_lo, student.v_hi,
                teacher_logits.detach(), labels, attention_mask,
                lambda_kl=cfg.lambda_kl, lambda_ce=cfg.lambda_ce,
                kl_temperature=cfg.kl_temperature, chunk_size=cfg.kl_ce_chunk_size,
                group=student.vp_group, world_size=student.vp_world_size,
            )
    elif student.lm_head is not None:
        # Legacy: only the track-0 owner has lm_head; peers run the forward for
        # collective ordering, contribute zero KL/CE, and never backward through it.
        with _phase("klce", timings, mem):
            kl_val, ce_val = _kl_ce_chunked(
                hidden, student.lm_head, teacher_logits.detach(), labels, attention_mask,
                lambda_kl=cfg.lambda_kl, lambda_ce=cfg.lambda_ce,
                kl_temperature=cfg.kl_temperature, chunk_size=cfg.kl_ce_chunk_size,
            )
    else:
        kl_val = torch.zeros((), device=input_ids.device)
        ce_val = torch.zeros((), device=input_ids.device)

    total_val = cfg.lambda_block * block_loss_val + cfg.lambda_kl * kl_val + cfg.lambda_ce * ce_val
    return {
        "total": total_val,
        "block_mse": block_loss_val,
        "kl": kl_val,
        "ce": ce_val,
    }


@torch.no_grad()
def validate_step(
    student: PTWrappedModel,
    batch: dict[str, torch.Tensor],
    teacher: HookedTeacher,
    kl_temperature: float = 1.0,
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
        kl, ce = _kl_ce_vocab_parallel(
            hidden, student.lm_head, student.v_lo, student.v_hi,
            teacher_logits.detach(), labels, attention_mask,
            lambda_kl=1.0, lambda_ce=1.0, kl_temperature=kl_temperature,
            chunk_size=hidden.shape[1], group=student.vp_group,
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
