"""SyncBoundary: the only cross-track collective in the PT forward.

The PT/SPD math we implement: between syncs, each track adds a *partial*
residual update to its track-local copy of the hidden state. At a sync
boundary we want to recombine the partials into the full update:

    h_synced = h_pre_block + Σ_{t=0..n_tracks-1} (h_t - h_pre_block)
             = h_pre_block + Σ_t (delta_t)

Each rank may host K = tracks_per_rank local tracks. We split the global
sum into a per-rank local part and a cross-rank all-reduce:

    partial_r = Σ_{k=0..K-1} (h_{r,k} - h_pre_block)        # local sum
    global    = all_reduce_SUM(partial_r) over track_group  # one rank per GPU
    h_synced  = h_pre_block + global

`h_pre_block` is identical across all K local tracks (they all start each
block from the previous synced state), so factoring it out is exact. For
K=1 this degenerates to the original single-track-per-rank behaviour; for
single-process testing (no track_group) it degenerates to a pure local sum.

The N=1 single-track case is a full no-op, which is the dense-parity
correctness gate: PTWrappedModel(N=1) must equal the dense forward.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class SyncBoundary(nn.Module):
    """Local-sum then NCCL all-reduce on delta_t across track_group, then add back.

    Args:
        track_group: torch.distributed ProcessGroup spanning one rank per GPU.
                     Pass None for single-process (in-process multi-track) testing.
        n_tracks: total number of tracks in the model. The N=1 case short-circuits
                  to a no-op for dense parity.
    """

    def __init__(self, track_group: "dist.ProcessGroup | None" = None, n_tracks: int = 1):
        super().__init__()
        self.track_group = track_group
        self.n_tracks = n_tracks

    def forward(
        self,
        h_list: "list[torch.Tensor] | torch.Tensor",
        h_pre_block: torch.Tensor,
    ) -> torch.Tensor:
        # Single-tensor shim: a leftover K=1 caller can pass a bare tensor.
        if isinstance(h_list, torch.Tensor):
            h_list = [h_list]

        if self.n_tracks <= 1:
            # Dense-parity short-circuit. With N=1 there is exactly one track.
            return h_list[0]

        # Local reduce across the K tracks this rank hosts.
        partial = h_list[0] - h_pre_block
        for h in h_list[1:]:
            partial = partial + (h - h_pre_block)

        if self.track_group is not None:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.track_group)

        return h_pre_block + partial
