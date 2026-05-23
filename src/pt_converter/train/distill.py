"""SPD-style distillation training step.

Implements the loop described in the plan:

1. Teacher forward (no_grad), capture {layer_idx -> hidden_state} at the
   sync boundaries plus final logits.
2. For each PT block (group of D student layers between syncs):
   - Feed the *teacher's* pre-block hidden state into the student's block.
   - Student runs D layers locally and the SyncBoundary all-reduces deltas.
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
from torch import nn

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.losses import block_mse, logit_kl, lm_cross_entropy
from pt_converter.train.teacher import HookedTeacher


@dataclass
class DistillConfig:
    sync_layer_indices: tuple[int, ...]
    lambda_block: float = 1.0
    lambda_kl: float = 1.0
    lambda_ce: float = 0.5
    kl_temperature: float = 1.0


def _block_ranges(num_layers: int, sync_indices: tuple[int, ...]) -> list[tuple[int, int]]:
    """Convert sync layer indices into (start_layer, end_layer_inclusive) ranges."""
    ranges = []
    prev_end = -1
    for idx in sync_indices:
        ranges.append((prev_end + 1, idx))
        prev_end = idx
    # If the last sync isn't at num_layers-1, we'd have a trailing block — for
    # our D=4/32-layer case the last sync IS at layer 31, so this is a no-op.
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
):
    """Run student layers [start..end_inclusive] *without* the sync at the end.

    Returns (h_out_pre_sync, h_in) so the caller can call sync_module(h_out, h_in).
    """
    tm = student.text_model
    hidden_states = h_in
    for layer_idx in range(start, end_inclusive + 1):
        layer = tm.layers[layer_idx]
        layer_mask = (
            linear_attn_mask
            if tm.config.layer_types[layer_idx] == "linear_attention"
            else causal_mask
        )
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
    return hidden_states


def distill_step(
    student: PTWrappedModel,
    teacher: HookedTeacher,
    batch: dict[str, torch.Tensor],
    cfg: DistillConfig,
) -> dict[str, torch.Tensor]:
    """Run one distillation step. Returns a dict of scalar tensors for logging.

    The caller is responsible for `loss.backward()` and `optimizer.step()`.
    The `total` loss is `loss_block + loss_kl + loss_ce` already weighted.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch["labels"]

    # ----- Teacher forward (frozen) -----
    teacher_logits, teacher_hiddens = teacher.forward(input_ids, attention_mask=attention_mask)

    # ----- Block-wise teacher-forced MSE -----
    tm = student.text_model
    # We need to recompute student-side scaffolding (rotary, masks) once.
    # `embed_tokens` lives only on the owner track; peers allocate a zero
    # placeholder and the SyncBoundary all-reduce broadcasts the owner's
    # embedding to every track (mirrors PTTrackTextModel.forward).
    if tm.embed_tokens is not None:
        inputs_embeds = tm.embed_tokens(input_ids)
    else:
        B, S = input_ids.shape
        inputs_embeds = torch.zeros(
            B,
            S,
            tm.config.hidden_size,
            device=input_ids.device,
            dtype=tm.norm.weight.dtype,
        )
    inputs_embeds = tm.sync_module(inputs_embeds, torch.zeros_like(inputs_embeds))
    position_ids, text_position_ids = tm._resolve_position_ids(inputs_embeds, None)
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # local import

    causal_mask = create_causal_mask(
        config=tm.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = (
        None if (attention_mask is not None and torch.all(attention_mask == 1)) else attention_mask
    )
    position_embeddings = tm.rotary_emb(inputs_embeds, position_ids)

    block_loss = torch.zeros((), device=input_ids.device)
    ranges = _block_ranges(len(tm.layers), cfg.sync_layer_indices)
    # The first block reads `inputs_embeds` (the embedding output, same on every
    # rank because embed_tokens is replicated). Subsequent blocks read the
    # *teacher's* post-block hidden state at the previous sync index.
    prev_teacher_h = inputs_embeds
    for start, end in ranges:
        h_post = _run_student_block(
            student,
            prev_teacher_h.detach(),  # teacher forcing: feed teacher's pre-block hidden
            start,
            end,
            position_embeddings,
            text_position_ids,
            causal_mask,
            linear_attn_mask,
        )
        h_synced = tm.sync_module(h_post, prev_teacher_h.detach())
        block_loss = block_loss + block_mse(
            h_synced, teacher_hiddens[end].detach(), attention_mask=attention_mask
        )
        prev_teacher_h = teacher_hiddens[end]

    # ----- Final-logit KL + LM CE (full student forward, no teacher forcing) -----
    # All ranks call the full forward so the cross-track SyncBoundary
    # all-reduces line up. Only the lm_head owner has logits to score; peer
    # tracks contribute zero KL/CE for this step.
    student_logits, _ = student(
        input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=False
    )
    if student_logits is not None:
        kl_loss = logit_kl(
            student_logits, teacher_logits.detach(), attention_mask=attention_mask, temperature=cfg.kl_temperature
        )
        ce_loss = lm_cross_entropy(student_logits, labels)
    else:
        kl_loss = torch.zeros((), device=input_ids.device)
        ce_loss = torch.zeros((), device=input_ids.device)

    total = cfg.lambda_block * block_loss + cfg.lambda_kl * kl_loss + cfg.lambda_ce * ce_loss
    return {
        "total": total,
        "block_mse": block_loss.detach(),
        "kl": kl_loss.detach(),
        "ce": ce_loss.detach(),
    }


@torch.no_grad()
def validate_step(
    student: PTWrappedModel,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Forward-only LM CE on a held-out batch.

    All ranks call the full student forward so the cross-track SyncBoundary
    all-reduces match; only the lm_head owner has logits and computes CE,
    peer tracks return a zero placeholder. The caller is responsible for
    aggregating across ranks (typically all_reduce SUM, since peers are 0).
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
