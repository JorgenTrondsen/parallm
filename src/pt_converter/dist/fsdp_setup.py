"""Apply FSDP2 to a per-track student. Teacher is sharded across the global group.

We use the new fully_shard API from torch.distributed.fsdp (FSDP2). Each
`Qwen3_5DecoderLayer` is wrapped individually so activation memory is
bounded.

Teacher sharding strategy: a single FSDP2 wrap over the whole teacher on the
*global* process group, frozen (`requires_grad=False`), put in eval mode.
This minimizes per-rank teacher VRAM (~2.25GB per rank for a 9B teacher
in bf16 sharded across 8 ranks) without any custom plumbing.
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
    """FSDP2-wrap each decoder layer of `student` on its intra_track_group."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    mesh = layout.intra_track_group  # FSDP2 accepts a ProcessGroup directly via mesh-like arg
    for layer in student.text_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    # Wrap the outer model so embed/norm/lm_head are also sharded.
    fully_shard(student, mesh=mesh, mp_policy=mp_policy)
    return student


def wrap_teacher_with_fsdp(
    teacher: nn.Module,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Frozen-teacher FSDP wrap across the default (global) group.

    Caller should already have set requires_grad=False on every teacher param
    and put it in eval mode. We do not reduce grads on the teacher.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=param_dtype)
    fully_shard(teacher, mp_policy=mp_policy)
    return teacher
