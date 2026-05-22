"""FSDP wrapping helpers.

Under the KV-replicated rule (world_size == n_tracks), the per-track student
is small (~562M params for Qwen3.5-9B at N=16) and we skip intra-track FSDP
entirely. `wrap_student_with_fsdp` is therefore a no-op that just moves the
student to the current CUDA device.

The teacher is still FSDP-sharded across the global group so each rank only
materializes a fraction of the 9B teacher in memory.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
except ImportError as e:  # pragma: no cover - older torch
    raise ImportError(
        "FSDP2 (fully_shard) requires torch>=2.4. The installed torch version "
        "is too old."
    ) from e

from pt_converter.dist.groups import ProcessGroupLayout
from pt_converter.model.pt_model import PTWrappedModel


def wrap_student_with_fsdp(
    student: PTWrappedModel,
    layout: ProcessGroupLayout,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
) -> PTWrappedModel:
    """No-op under the KV-replicated layout (intra_track_group is None).

    Kept as a function so the training script's surface stays the same — when
    a future model is large enough to require intra-track sharding we wire
    that in here without touching the caller.
    """
    _ = (param_dtype, reduce_dtype)  # silence unused-arg lints; kept for API compatibility
    if layout.intra_track_group is not None:
        # Reserved path for future large-model layouts. Not used today.
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        for layer in student.text_model.layers:
            fully_shard(layer, mesh=layout.intra_track_group, mp_policy=mp_policy)
        fully_shard(student, mesh=layout.intra_track_group, mp_policy=mp_policy)
    return student


def wrap_teacher_with_fsdp(
    teacher: nn.Module,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Shard the frozen teacher across the global group. Caller has already set
    `requires_grad=False` and `eval()`."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=param_dtype)
    fully_shard(teacher, mp_policy=mp_policy)
    return teacher
