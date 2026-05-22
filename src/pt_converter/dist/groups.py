"""Process-group layout for PT training (KV-replicated, N=num_attention_heads).

Layout under the new rule:
  - world_size = n_tracks (e.g., 16 for Qwen3.5-9B)
  - track_id = rank (one rank per track)
  - track_group = dist.group.WORLD (every cross-track sync is a global all-reduce)
  - intra_track_group = None (per-track student is small enough to skip FSDP)

For 8-GPU nodes running 16 ranks, 2 ranks share a GPU via
`torch.cuda.set_device(local_rank % torch.cuda.device_count())`. NCCL handles
this fine on modern PyTorch; the launcher script is responsible for the
device-set call.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass
class ProcessGroupLayout:
    world_size: int
    rank: int
    n_tracks: int
    intra_track_size: int  # always 1 under the new layout
    track_id: int
    intra_track_rank: int  # always 0 under the new layout
    intra_track_group: "dist.ProcessGroup | None"  # always None under the new layout
    track_group: "dist.ProcessGroup | None"  # = WORLD under the new layout


def build_groups(n_tracks: int) -> ProcessGroupLayout:
    """Construct the (degenerate) intra/track group layout for this process.

    Under the KV-replicated rule, world_size == n_tracks, intra_track_size == 1,
    and the track_group is the world group. We keep the function signature for
    forward compatibility with future layouts that may re-introduce intra-track
    sharding for very large models.
    """
    if not dist.is_initialized():
        raise RuntimeError("dist.init_process_group must be called before build_groups()")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != n_tracks:
        raise ValueError(
            f"world_size {world_size} != n_tracks {n_tracks}. Under the KV-replicated rule "
            f"every rank is one track; launch with --nproc-per-node={n_tracks} "
            f"(potentially > GPU count; ranks share GPUs via set_device(local_rank % gpus))."
        )

    return ProcessGroupLayout(
        world_size=world_size,
        rank=rank,
        n_tracks=n_tracks,
        intra_track_size=1,
        track_id=rank,
        intra_track_rank=0,
        intra_track_group=None,
        track_group=dist.group.WORLD,
    )
