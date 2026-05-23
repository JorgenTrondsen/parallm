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
    # Embedding broadcast: owner runs embed_tokens; peers contribute zeros.
    # The sync_module local-sum-then-all-reduce delivers the owner's embedding
    # to every track.
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

    block_loss = torch.zeros((), device=input_ids.device)
    ranges = _block_ranges(len(tm0.layers), cfg.sync_layer_indices)
    # The first block reads `inputs_embeds` (the embedding output, identical
    # on every rank after the broadcast). Subsequent blocks read the
    # *teacher's* post-block hidden state at the previous sync index.
    prev_teacher_h = inputs_embeds
    for start, end in ranges:
        h_post_list = _run_student_block(
            student,
            prev_teacher_h.detach(),  # teacher forcing
            start,
            end,
            position_embeddings,
            text_position_ids,
            causal_mask,
            linear_attn_mask,
        )
        h_synced = student.sync_module(h_post_list, prev_teacher_h.detach())
        block_loss = block_loss + block_mse(
            h_synced, teacher_hiddens[end].detach(), attention_mask=attention_mask
        )
        prev_teacher_h = teacher_hiddens[end]

    # ----- Final-logit KL + LM CE (full student forward, no teacher forcing) -----
    # All ranks call the full forward so the cross-track SyncBoundary
    # all-reduces line up. Only the rank that owns track 0 has lm_head; peer
    # ranks contribute zero KL/CE for this step.
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
