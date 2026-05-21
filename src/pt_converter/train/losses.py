"""Distillation losses for SPD-style PT conversion.

Three components:
- `block_mse`: at each sync boundary, MSE between the student's all-reduced
  hidden state and the teacher's hidden state at the same depth. We mask
  padded positions out so the loss is a true per-token mean.
- `logit_kl`: temperature-softmaxed KL between teacher and student final logits.
- `lm_cross_entropy`: standard next-token prediction loss on the labels.

All three return scalar tensors that the caller can weight and sum.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of `x` over positions where `mask` is non-zero.

    `x` is (B, T, ...) and `mask` is (B, T) of {0,1}. The hidden-dim trailing
    axes are always averaged in full; only the (B, T) positions are masked.
    """
    if mask is None:
        return x.mean()
    mask_b = mask
    while mask_b.ndim < x.ndim:
        mask_b = mask_b.unsqueeze(-1)
    trailing_elems = 1
    for d in x.shape[2:]:
        trailing_elems *= d
    return (x * mask_b).sum() / (mask.sum() * trailing_elems).clamp(min=1)


def block_mse(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between (B, T, H) student and teacher hidden states. Mean over non-pad positions."""
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"block_mse shape mismatch: student {student_hidden.shape} vs teacher {teacher_hidden.shape}"
        )
    diff = (student_hidden.float() - teacher_hidden.float()).pow(2)
    return _masked_mean(diff, attention_mask)


def logit_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL: KL(teacher_dist || student_dist), temperature-softmaxed.

    Convention used in distillation: student matches teacher's distribution.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logit_kl shape mismatch: student {student_logits.shape} vs teacher {teacher_logits.shape}"
        )
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_p = F.softmax(teacher_logits.float() / temperature, dim=-1)
    # KL(t || s) = sum t * (log t - log s) — we compute per-token, then mean.
    t_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = (t_p * (t_logp - s_logp)).sum(dim=-1)  # (B, T)
    if attention_mask is None:
        return per_token.mean() * (temperature * temperature)
    masked = per_token * attention_mask
    return masked.sum() / attention_mask.sum().clamp(min=1) * (temperature * temperature)


def lm_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard shifted next-token CE. Logits and labels are (B, T, V) and (B, T)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
