"""Process-group layout for PT training.

Layout:
  - one rank per visible GPU (no oversubscription — NCCL 2.19+ rejects
    communicators that contain two ranks pinned to the same physical device)
  - by default, each rank hosts ``tracks_per_rank = n_tracks // world_size``
    tracks; rank ``r`` owns the contiguous block ``[r*K, …, r*K+K-1]``
  - optionally, ``build_groups`` accepts ``tracks_per_rank_list`` — a per-rank
    K vector summing to ``n_tracks`` — for non-uniform distributions (e.g.
    rank 0 hosts fewer tracks because it also owns embed_tokens + lm_head).
    Track ids are still assigned contiguously: rank 0 gets the first ``K_0``
    tracks, rank 1 the next ``K_1``, etc., so track 0 (the embed/lm_head
    owner) stays on rank 0.
  - track_group = dist.group.WORLD; the cross-rank all-reduce in SyncBoundary
    spans world_size ranks, after a local sum across the K local tracks
  - fuse_group = a contiguous block of ``fuse_ranks`` ranks, or None when
    ``fuse_ranks == 1``. This is the CROSS-RANK fusion tier: between global
    syncs the block's tracks pool their partials with a subgroup all-reduce, so
    they compute as one track. It is what lets ``--fuse-tracks F`` exceed a
    rank's own track count — without it F is capped at n_tracks/world_size, so
    adding GPUs would LOWER the reachable F.
  - intra_track_group = None (per-track student is small enough to skip FSDP)

Multi-node: ``world_size = nnodes * nproc_per_node`` = total GPUs across the
job. Standard torchrun, nothing custom.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.distributed as dist


def compute_contiguous_assignment(
    tracks_per_rank_list: Sequence[int],
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pure helper: derive (local_track_ids, track_to_rank) from a per-rank K list.

    Tracks are assigned in contiguous blocks: rank 0 gets ``[0, K_0)``,
    rank 1 gets ``[K_0, K_0+K_1)``, and so on. ``track_to_rank[t]`` is the
    rank that owns track ``t``; ``local_track_ids`` is the slice belonging
    to ``rank``.
    """
    track_to_rank: list[int] = []
    rank_track_starts: list[int] = []
    cursor = 0
    for r, k in enumerate(tracks_per_rank_list):
        rank_track_starts.append(cursor)
        track_to_rank.extend([r] * k)
        cursor += k
    start = rank_track_starts[rank]
    end = start + tracks_per_rank_list[rank]
    return tuple(range(start, end)), tuple(track_to_rank)


@dataclass
class ProcessGroupLayout:
    world_size: int
    rank: int
    n_tracks: int
    tracks_per_rank: int  # this rank's K (may differ from peers under non-uniform layouts)
    local_track_ids: tuple[int, ...]
    track_to_rank: tuple[int, ...]  # length n_tracks: track_to_rank[t] = owning rank
    track_group: "dist.ProcessGroup | None"  # = WORLD
    intra_track_size: int  # always 1 today
    intra_track_rank: int  # always 0 today
    intra_track_group: "dist.ProcessGroup | None"  # always None today
    # Cross-rank fusion tier. `fuse_ranks` == 1 means off and `fuse_group` is None.
    fuse_ranks: int = 1
    fuse_rank: int = 0  # this rank's index WITHIN its fusion block; 0 = the leader
    fuse_group: "dist.ProcessGroup | None" = None


def build_groups(
    n_tracks: int,
    *,
    tracks_per_rank_list: Sequence[int] | None = None,
    fuse_ranks: int = 1,
) -> ProcessGroupLayout:
    """Construct the per-rank track layout.

    Default (``tracks_per_rank_list=None``): uniform K = n_tracks // world_size;
    requires ``n_tracks % world_size == 0`` and rank ``r`` owns
    ``[r*K, r*K+K-1]``.

    Custom (``tracks_per_rank_list`` given): must have length ``world_size``,
    contain positive ints, and sum to ``n_tracks``. Each rank gets a
    contiguous block sized by its entry. Used for asymmetric layouts where
    rank 0 carries embed_tokens + lm_head on top of its tracks and needs
    fewer student tracks than its peers.

    ``fuse_ranks`` > 1 additionally partitions the ranks into contiguous blocks of
    that size and builds one subgroup per block (see `SyncBoundary.fuse`). Requires
    a uniform layout — a fusion block pools whole ranks, so its members must hold
    equal track counts for the grouping to mean the same thing on each.
    """
    if not dist.is_initialized():
        raise RuntimeError("dist.init_process_group must be called before build_groups()")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if tracks_per_rank_list is None:
        if n_tracks % world_size != 0:
            raise ValueError(
                f"n_tracks {n_tracks} not divisible by world_size {world_size}. "
                f"Launch with --nproc-per-node*--nnodes evenly dividing {n_tracks}, "
                f"or pass tracks_per_rank_list for a non-uniform layout."
            )
        k = n_tracks // world_size
        k_list = [k] * world_size
    else:
        k_list = list(tracks_per_rank_list)
        if len(k_list) != world_size:
            raise ValueError(
                f"tracks_per_rank_list length {len(k_list)} != world_size {world_size}"
            )
        if any(k <= 0 for k in k_list):
            raise ValueError(f"tracks_per_rank_list must be all positive; got {k_list}")
        if sum(k_list) != n_tracks:
            raise ValueError(
                f"tracks_per_rank_list sums to {sum(k_list)} but n_tracks={n_tracks}"
            )

    local_track_ids, track_to_rank = compute_contiguous_assignment(k_list, rank)

    fuse_group = None
    if fuse_ranks > 1:
        if world_size % fuse_ranks:
            raise ValueError(
                f"fuse_ranks {fuse_ranks} must divide world_size {world_size}"
            )
        if len(set(k_list)) != 1:
            raise ValueError(
                f"fuse_ranks > 1 needs a uniform layout (a fusion block pools whole "
                f"ranks); got tracks_per_rank_list {k_list}"
            )
        # `dist.new_group` is itself collective: EVERY rank must construct EVERY
        # subgroup, in the same order, or the job deadlocks. Same discipline as
        # `train.sync_grads.build_replication_plan`.
        for start in range(0, world_size, fuse_ranks):
            block = list(range(start, start + fuse_ranks))
            pg = dist.new_group(ranks=block)
            if rank in block:
                fuse_group = pg

    return ProcessGroupLayout(
        world_size=world_size,
        rank=rank,
        n_tracks=n_tracks,
        tracks_per_rank=k_list[rank],
        local_track_ids=local_track_ids,
        track_to_rank=tuple(track_to_rank),
        track_group=dist.group.WORLD,
        intra_track_size=1,
        intra_track_rank=0,
        intra_track_group=None,
        fuse_ranks=fuse_ranks,
        fuse_rank=rank % fuse_ranks,
        fuse_group=fuse_group,
    )
