"""FSDP wrapping helpers.

With one rank per GPU and K = n_tracks / world_size tracks hosted per rank,
the per-rank student is still small (≈ K × 562M params for Qwen3.5-9B at
N=16) and we skip intra-track FSDP entirely. `wrap_student_with_fsdp` is
therefore a no-op that just leaves the student on the current CUDA device.

The teacher is FSDP-sharded across the world group (one rank per GPU, so
the communicator is duplicate-free and NCCL-clean), and each rank only
materializes a 1/world_size fraction of the dense teacher.
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
    """No-op today (intra_track_group is None).

    Kept as a function so the training script's surface stays the same. The
    reserved branch below shards each local track's layers across an
    intra-track group if a future layout introduces one.
    """
    _ = (param_dtype, reduce_dtype)  # silence unused-arg lints; kept for API compatibility
    if layout.intra_track_group is not None:
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        for tm in student.text_models:
            for layer in tm.layers:
                fully_shard(layer, mesh=layout.intra_track_group, mp_policy=mp_policy)
        fully_shard(student, mesh=layout.intra_track_group, mp_policy=mp_policy)
    return student


def wrap_teacher_with_fsdp(
    text_model: nn.Module,
    lm_head: nn.Module,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Shard the frozen teacher across the world group (one rank per GPU).

    HookedTeacher invokes `text_model(...)` and `lm_head(...)` directly, so the
    FSDP boundaries (which install the DTensor input-conversion hooks) must be
    on those exact modules — not on a parent wrapper. We also shard each
    decoder layer individually so only one layer's parameters are
    all-gathered at a time during the forward pass.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=param_dtype)
    for layer in text_model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(text_model, mp_policy=mp_policy)
    fully_shard(lm_head, mp_policy=mp_policy)
