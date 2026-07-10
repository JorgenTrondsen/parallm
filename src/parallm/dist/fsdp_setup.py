"""Teacher FSDP wrapping.

The teacher is FSDP-sharded across the world group (one rank per GPU, so the
communicator is duplicate-free and NCCL-clean), and each rank only materializes
a 1/world_size fraction of the dense teacher. The per-rank student is small
enough (K tracks, each ≈ 562M params for Qwen3.5-9B at N=16) that it needs no
intra-track sharding — it just stays on the current CUDA device.
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


def wrap_teacher_with_fsdp(
    text_model: nn.Module,
    lm_head: nn.Module,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Shard the frozen teacher across the world group (one rank per GPU).

    HookedTeacher invokes `text_model(...)` and `lm_head(...)` directly, so the
    FSDP boundaries (which install the DTensor input-conversion hooks) must be on
    those exact modules — not on a parent wrapper. We also shard each decoder
    layer individually so only one layer's parameters are all-gathered at a time
    during the forward pass.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=param_dtype)
    for layer in text_model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(text_model, mp_policy=mp_policy)
    fully_shard(lm_head, mp_policy=mp_policy)
