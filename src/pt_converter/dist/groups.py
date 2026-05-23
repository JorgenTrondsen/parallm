"""Process-group layout for PT training.

Layout:
  - one rank per visible GPU (no oversubscription — NCCL 2.19+ rejects
    communicators that contain two ranks pinned to the same physical device)
  - each rank hosts ``tracks_per_rank = n_tracks // world_size`` tracks
  - rank ``r`` owns the contiguous block ``[r*K, r*K+1, ..., r*K+K-1]``
    (keeps track 0 — the embed/lm_head owner — on rank 0)
  - track_group = dist.group.WORLD; the cross-rank all-reduce in SyncBoundary
    spans world_size ranks, after a local sum across the K local tracks
  - intra_track_group = None (per-track student is small enough to skip FSDP)

Multi-node: ``world_size = nnodes * nproc_per_node`` = total GPUs across the
job. Standard torchrun, nothing custom.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass
class ProcessGroupLayout:
    world_size: int
    rank: int
    n_tracks: int
    tracks_per_rank: int
    local_track_ids: tuple[int, ...]
    track_group: "dist.ProcessGroup | None"  # = WORLD
    intra_track_size: int  # always 1 today
    intra_track_rank: int  # always 0 today
    intra_track_group: "dist.ProcessGroup | None"  # always None today


def build_groups(n_tracks: int) -> ProcessGroupLayout:
    """Construct the per-rank track layout.

    Requires ``n_tracks % world_size == 0``. Each rank owns a contiguous block
    of ``K = n_tracks // world_size`` track ids; rank 0 always owns track 0.
    """
    if not dist.is_initialized():
        raise RuntimeError("dist.init_process_group must be called before build_groups()")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if n_tracks % world_size != 0:
        raise ValueError(
            f"n_tracks {n_tracks} not divisible by world_size {world_size}. "
            f"Launch with --nproc-per-node*--nnodes evenly dividing {n_tracks}."
        )
    tracks_per_rank = n_tracks // world_size
    local_track_ids = tuple(range(rank * tracks_per_rank, (rank + 1) * tracks_per_rank))

    return ProcessGroupLayout(
        world_size=world_size,
        rank=rank,
        n_tracks=n_tracks,
        tracks_per_rank=tracks_per_rank,
        local_track_ids=local_track_ids,
        track_group=dist.group.WORLD,
        intra_track_size=1,
        intra_track_rank=0,
        intra_track_group=None,
    )
