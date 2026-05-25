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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.losses import block_mse, logit_kl, lm_cross_entropy
from pt_converter.train.teacher import HookedTeacher


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


@dataclass
class DistillConfig:
    sync_layer_indices: tuple[int, ...]
    lambda_block: float = 1.0
    lambda_kl: float = 1.0
    lambda_ce: float = 0.5
    kl_temperature: float = 1.0
    # Seq-chunk size for the KL+CE pass. The full (B, T, V) fp32 expansions
    # (V=151936 for Qwen3.5) OOM the embed+lm_head-owning rank at training
    # seq_len; chunking caps the per-chunk transient at (chunk_size/T)x.
    # Larger values run fewer per-chunk backwards (faster) at the cost of a
    # bigger per-chunk transient. 128 fits comfortably on the asymmetric
    # n16_d4 layout with --rank0-tracks 1; the old smoke runs used 32 only
    # to survive the uniform-K layout where rank 0 was memory-tight.
    kl_ce_chunk_size: int = 128


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
    teacher_logits, teacher_hiddens = teacher.forward(input_ids, attention_mask=attention_mask)

    # ----- Embedding broadcast -----
    # Owner runs embed_tokens; peers contribute zeros. sync_module's local sum
    # + all-reduce delivers the owner's embedding to every track.
    embeds_per_track: list[torch.Tensor] = []
    for tm in student.text_models:
        if tm.embed_tokens is not None:
            embeds_per_track.append(tm.embed_tokens(input_ids))
        else:
            B, S = input_ids.shape
            embeds_per_track.append(
                torch.zeros(
                    B,
                    S,
                    tm.config.hidden_size,
                    device=input_ids.device,
                    dtype=tm.norm.weight.dtype,
                )
            )
    inputs_embeds = student.sync_module(embeds_per_track, torch.zeros_like(embeds_per_track[0]))

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
        block_loss_b = block_mse(
            h_synced, teacher_hiddens[end].detach(), attention_mask=attention_mask
        )
        (cfg.lambda_block * block_loss_b).backward()
        block_loss_val = block_loss_val + block_loss_b.detach()
        prev_h = teacher_hiddens[end].detach()

    # ----- Final-logit KL + LM CE (full student forward, chunked lm_head) -----
    # All ranks call the full forward so cross-track SyncBoundary all-reduces
    # line up. We request the pre-lm_head hidden state so `_kl_ce_chunked`
    # can chunk the lm_head application itself — avoids materializing a
    # (B, T, V) bf16 logits tensor on the owner rank. Only the rank that
    # owns track 0 has lm_head; peers run the forward only for collective
    # ordering, contribute zero KL/CE, and never backward through it.
    hidden, _ = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_sync_hiddens=False,
        return_hidden_pre_lm_head=True,
    )
    if student.lm_head is not None:
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
) -> dict[str, torch.Tensor]:
    """Forward-only LM CE on a held-out batch.

    All ranks call the full student forward so the cross-track SyncBoundary
    all-reduces match; only the rank that owns track 0 has lm_head and
    computes CE — peer ranks return a zero placeholder. The caller is
    responsible for aggregating across ranks (typically all_reduce SUM,
    since peers are 0).
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]
    student_logits, _ = student(
        input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=False
    )
    if student_logits is not None:
        ce = lm_cross_entropy(student_logits, labels)
    else:
        ce = torch.zeros((), device=input_ids.device)
    return {"ce": ce}
