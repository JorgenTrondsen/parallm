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


def _masked_sum(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Sum of `x` over positions where `mask` is non-zero (no averaging)."""
    if mask is None:
        return x.sum()
    mask_b = mask
    while mask_b.ndim < x.ndim:
        mask_b = mask_b.unsqueeze(-1)
    return (x * mask_b).sum()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of `x` over positions where `mask` is non-zero.

    `x` is (B, T, ...) and `mask` is (B, T) of {0,1}. The hidden-dim trailing
    axes are always averaged in full; only the (B, T) positions are masked.
    """
    if mask is None:
        return x.mean()
    trailing_elems = 1
    for d in x.shape[2:]:
        trailing_elems *= d
    return _masked_sum(x, mask) / (mask.sum() * trailing_elems).clamp(min=1)


def block_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    normalize: bool = False,
    eps: float = 1e-6,
    clamp_max: float | None = None,
) -> torch.Tensor:
    """MSE between (B, T, H) student and teacher hidden states, over non-pad positions.

    ``normalize=False`` (default): masked mean of the squared error — the raw,
    scale-dependent loss. The residual-stream norm grows with depth, so under this
    form deep blocks dominate the gradient.

    ``normalize=True``: relative (scale-free) MSE ``Σ_mask (s−t)² / Σ_mask t²``, so
    each block's loss is O(1) regardless of its activation magnitude and every depth
    contributes comparably. The trailing hidden-dim elements cancel in the ratio.

    ``clamp_max`` (normalized path only): cap the returned relative-MSE at this value.
    Under student forcing a block can be fed the student's own drifted hidden, which
    occasionally makes the ratio blow up (e.g. 100+) on a single batch — a spike that
    inflates the gradient and trips the global grad-norm clip, throttling every param.
    Clamping saturates the gradient above the cap (outlier rejection): normal per-block
    ratios (~0.5–1.5) are untouched, only the spikes are capped. Ignored when
    ``normalize=False``.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"block_mse shape mismatch: student {student_hidden.shape} vs teacher {teacher_hidden.shape}"
        )
    diff = (student_hidden.float() - teacher_hidden.float()).pow(2)
    if not normalize:
        return _masked_mean(diff, attention_mask)
    denom = _masked_sum(teacher_hidden.float().pow(2), attention_mask).clamp(min=eps)
    ratio = _masked_sum(diff, attention_mask) / denom
    if clamp_max is not None:
        ratio = ratio.clamp(max=clamp_max)
    return ratio


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


def logit_mse(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    center: bool = True,
) -> torch.Tensor:
    """Mean-squared error between student and teacher final logits, over non-pad positions.

    Output-aware alternative to ``logit_kl``: it supervises the residual stream in the
    directions the ``lm_head`` actually reads, with a gradient ``∝ (d − d̄)`` that does
    NOT saturate when the student is close-but-mis-directed (KL's gradient does). The
    per-token error is the mean over the vocab of the squared logit difference.

    ``center=True`` (default) removes the per-token mean over the vocab from the
    difference before squaring. The softmax is invariant to a per-token additive shift
    of the logits, so that mean component is a free direction the student should not
    spend capacity matching. Using ``d = s − t`` and ``d̄ = mean_v(d)`` the centered
    form ``mean_v((d − d̄)²) = mean_v(d²) − d̄²`` (the identity the vocab-parallel path
    relies on). ``center=False`` is the plain (uncentered) logit MSE.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logit_mse shape mismatch: student {student_logits.shape} vs teacher {teacher_logits.shape}"
        )
    d = student_logits.float() - teacher_logits.float()
    if center:
        d = d - d.mean(dim=-1, keepdim=True)
    per_token = d.pow(2).mean(dim=-1)  # (B, T)
    return _masked_mean(per_token, attention_mask)


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
