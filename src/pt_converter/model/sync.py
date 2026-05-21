"""SyncBoundary: the only cross-track collective in the PT forward.

The PT/SPD math we implement: between syncs, each track adds a *partial*
residual update to its track-local copy of the hidden state. At a sync
boundary we want to recombine the partials into the full update:

    h_synced = h_pre_block + sum_t (h_t - h_pre_block)
             = h_pre_block + sum_t(delta_t)

This is mathematically equivalent to `sum_t h_t - (N-1) * h_pre_block`,
implemented as a single all-reduce on `delta_t = h_t - h_pre_block`
followed by an add of `h_pre_block`. After the sync, every track holds the
same `h_synced`, and the next block starts with `h_pre_block = h_synced`.

For N=1 (or when no process group is provided) the module is a pure no-op,
which is the key correctness gate: `PTWrappedModel(N=1, D=*).forward` must
be numerically equivalent to the dense forward.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class SyncBoundary(nn.Module):
    """All-reduce SUM on (h_t - h_pre_block) across the track_group, then add back.

    Args:
        track_group: torch.distributed ProcessGroup spanning one rank per track.
                     Pass None for single-track / single-process testing.
        n_tracks: stored only for sanity assertions; the group's size is what matters.
    """

    def __init__(self, track_group: "dist.ProcessGroup | None" = None, n_tracks: int = 1):
        super().__init__()
        self.track_group = track_group
        self.n_tracks = n_tracks

    def forward(self, h_t: torch.Tensor, h_pre_block: torch.Tensor) -> torch.Tensor:
        if self.n_tracks <= 1 or self.track_group is None:
            return h_t  # no-op: degenerates to dense forward
        delta = h_t - h_pre_block
        dist.all_reduce(delta, op=dist.ReduceOp.SUM, group=self.track_group)
        return h_pre_block + delta
