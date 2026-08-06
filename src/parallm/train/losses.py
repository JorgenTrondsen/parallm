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


def block_split(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The boundary tap split into ``(direction, magnitude)``, both differentiable.

    ``direction`` = masked mean of ``2(1−cos)``, which is exactly ``‖ŝ−t̂‖²`` for
    the RMS/L2-normalized states — i.e. relMSE with the per-token scale divided
    out. ``magnitude`` = masked mean of ``log(‖s‖/‖t‖)²``, symmetric in over- and
    under-shoot and well behaved at the r≈5-7 seen at step 0 (``(r−1)²`` would put
    ~40x more weight on the first boundary than the last).

    Why this exists (measured 2026-08-04): ``block_mse(normalize=True)`` is
    ``1 − 2·cos·r + r²``, and at D=2 the first boundary sits at cos 0.935 with
    r 5.4 — **99% of that tap is magnitude with the direction already right**, yet
    it is ~93% of the whole block loss. Meanwhile the residual that survives
    training is direction (cos 0.93 → 0.80 with r ≈ 1.0). Splitting the two lets
    the objective be weighted by which failure it is.

    ⚠ Deliberately NO learned norm weight here. `norm.weight` is trainable, so
    applying it to the teacher target would let the model shrink the loss by
    shrinking the norm rather than by matching the teacher.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"block_split shape mismatch: student {student_hidden.shape} "
            f"vs teacher {teacher_hidden.shape}"
        )
    s = student_hidden.float()
    t = teacher_hidden.float()
    s_norm = s.pow(2).sum(-1).sqrt().clamp(min=eps)
    t_norm = t.pow(2).sum(-1).sqrt().clamp(min=eps)
    cos = (s * t).sum(-1) / (s_norm * t_norm)
    direction = _masked_mean(2.0 * (1.0 - cos), attention_mask)
    magnitude = _masked_mean((s_norm / t_norm).log().pow(2), attention_mask)
    return direction, magnitude


def block_direction_magnitude(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a boundary's error into ``(cosine, norm_ratio)``, per-token then masked-mean.

    ``block_mse(normalize=True)`` conflates the two: relMSE ``≈ 1 − 2ρ·r + r²`` for
    per-token cosine ρ and norm ratio ``r = ‖s‖/‖t‖``, so a state that points the
    right way but is 20% short and one that has the right length but is 20° off can
    score identically. They are not the same failure and they do not have the same
    remedy — a magnitude deficit is a gain the network can absorb, a direction error
    is missing content.

    Reported alongside relMSE rather than instead of it: relMSE is what the
    objective optimizes, these say what it is made of.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"block_direction_magnitude shape mismatch: student {student_hidden.shape} "
            f"vs teacher {teacher_hidden.shape}"
        )
    s = student_hidden.float()
    t = teacher_hidden.float()
    s_norm = s.pow(2).sum(-1).sqrt()
    t_norm = t.pow(2).sum(-1).sqrt()
    cos = (s * t).sum(-1) / (s_norm * t_norm).clamp(min=eps)
    ratio = s_norm / t_norm.clamp(min=eps)
    return _masked_mean(cos, attention_mask), _masked_mean(ratio, attention_mask)
