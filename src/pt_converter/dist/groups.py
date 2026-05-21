"""Process-group layout for PT training on a single 8-GPU node.

We carve the global group into two orthogonal sets:

- `intra_track_group`: ranks that hold the same *track* of the model. These
  shard a single track's parameters via FSDP. With n_tracks=4 and world=8
  this is a pair of ranks per track.
- `track_group`: ranks that act as the "track-id channel" — one rank per
  track. These all-reduce at sync boundaries.

For world=8, n_tracks=4, intra_track=2 the rank layout is:

    track 0:  ranks 0, 1     (intra_track_group_0)
    track 1:  ranks 2, 3     (intra_track_group_1)
    track 2:  ranks 4, 5     (intra_track_group_2)
    track 3:  ranks 6, 7     (intra_track_group_3)

    track_group_for_intra_rank_0: {0, 2, 4, 6}
    track_group_for_intra_rank_1: {1, 3, 5, 7}

Each rank participates in exactly one intra_track_group and one track_group.
The track_group for intra-rank j is {track_id * intra_size + j for track_id in
range(n_tracks)} — this preserves a 1:1 hidden-state correspondence between
each FSDP shard and the matching shard on every other track.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ProcessGroupLayout:
    world_size: int
    rank: int
    n_tracks: int
    intra_track_size: int
    track_id: int
    intra_track_rank: int
    intra_track_group: "dist.ProcessGroup | None"
    track_group: "dist.ProcessGroup | None"


def build_groups(n_tracks: int) -> ProcessGroupLayout:
    """Construct intra_track and track_group for the current process.

    Assumes torch.distributed is already initialized (e.g., via torchrun and
    `dist.init_process_group("nccl")` from the entrypoint script).
    """
    if not dist.is_initialized():
        raise RuntimeError("dist.init_process_group must be called before build_groups()")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % n_tracks != 0:
        raise ValueError(f"world_size {world_size} not divisible by n_tracks {n_tracks}")
    intra_track_size = world_size // n_tracks
    track_id = rank // intra_track_size
    intra_track_rank = rank % intra_track_size

    intra_track_group = None
    track_group = None

    # Build intra_track groups: one group per track, containing ranks
    # [track_id * intra_size, (track_id + 1) * intra_size). NCCL requires
    # *every* rank to call new_group in the same order — so we iterate over
    # all tracks and intra-ranks even though we only keep our own.
    for tid in range(n_tracks):
        ranks = list(range(tid * intra_track_size, (tid + 1) * intra_track_size))
        g = dist.new_group(ranks=ranks)
        if tid == track_id:
            intra_track_group = g

    for itr in range(intra_track_size):
        ranks = [tid * intra_track_size + itr for tid in range(n_tracks)]
        g = dist.new_group(ranks=ranks)
        if itr == intra_track_rank:
            track_group = g

    return ProcessGroupLayout(
        world_size=world_size,
        rank=rank,
        n_tracks=n_tracks,
        intra_track_size=intra_track_size,
        track_id=track_id,
        intra_track_rank=intra_track_rank,
        intra_track_group=intra_track_group,
        track_group=track_group,
    )
