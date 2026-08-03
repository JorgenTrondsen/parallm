"""Hidden-state fidelity loss for PT conversion.

`block_mse`: at each sync boundary, MSE between the student's all-reduced hidden
state and the teacher's hidden state at the same depth (padded positions masked
out so it's a true per-token mean). The heal step's block-MSE tap
(`train/distill.py`) and the fidelity score in `eval/fidelity.py`. The output
objective (CE) lives in `train/distill.ce_chunked`, which
computes them per seq-chunk so full-vocab logits never materialize.
Returns a scalar tensor.
"""
from __future__ import annotations

import torch


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
) -> torch.Tensor:
    """MSE between (B, T, H) student and teacher hidden states, over non-pad positions.

    ``normalize=False`` (default): masked mean of the squared error — the raw,
    scale-dependent loss. The residual-stream norm grows with depth, so under this
    form deep blocks dominate the gradient.

    ``normalize=True``: relative (scale-free) MSE ``Σ_mask (s−t)² / Σ_mask t²``, so
    each block's loss is O(1) regardless of its activation magnitude and every depth
    contributes comparably. The trailing hidden-dim elements cancel in the ratio.

    There is deliberately NO clamp here. A ``clamp_max`` existed to cap spikes caused
    by student forcing feeding a block its own drifted hidden; sf was removed from the
    trainer, and the cap then did active harm. At D=2 the first boundary's ratio starts
    at 15.9 against a cap of 10, so it sat pinned at exactly the cap while being ~93%
    of the block loss — and ``clamp`` passes ZERO gradient above its max, leaving the
    dominant term untrainable until unrelated CE drift happened to carry it under 10.
    That was the multi-hundred-step "ledge". Removing the cap took macro 0.592 -> 0.638
    at 500 steps and made the first boundary train from step 0 (15.9 -> 0.23).
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"block_mse shape mismatch: student {student_hidden.shape} vs teacher {teacher_hidden.shape}"
        )
    diff = (student_hidden.float() - teacher_hidden.float()).pow(2)
    if not normalize:
        return _masked_mean(diff, attention_mask)
    denom = _masked_sum(teacher_hidden.float().pow(2), attention_mask).clamp(min=eps)
    return _masked_sum(diff, attention_mask) / denom
