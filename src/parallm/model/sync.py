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

`fuse_size` adds a second, collective-free tier BETWEEN syncs: F rank-local
tracks pool their partials at every sublayer, so a group computes as one
F-wide track and the model behaves as N/F tracks between global syncs while
the checkpoint stays N shards. Off (F=1) by default.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class _AllReduceSum(torch.autograd.Function):
    """All-reduce with the trainer's gradient semantics made explicit.

    Backward is IDENTITY: the loss is replicated on every rank, so each rank's
    local gradient of the boundary output is already the full dL/dy, and the
    other ranks' partials are constants from this rank's graph perspective —
    the record trainer's collective-free backward, without relying on torch's
    deprecated not-implemented fallback for c10d ops. The clone keeps the
    graph tensor un-mutated (in-place all_reduce would bump its version).
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor, group) -> torch.Tensor:
        out = t.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class SyncBoundary(nn.Module):
    """Local-sum then NCCL all-reduce on delta_t across track_group, then add back.

    Args:
        track_group: torch.distributed ProcessGroup spanning one rank per GPU.
                     Pass None for single-process (in-process multi-track) testing.
        n_tracks: total number of tracks in the model. The N=1 case short-circuits
                  to a no-op for dense parity.
        fuse_size: F rank-local tracks that pool their partials between global
                   syncs (see `fuse`). 1 = off, today's behaviour.
    """

    def __init__(
        self,
        track_group: "dist.ProcessGroup | None" = None,
        n_tracks: int = 1,
        fuse_size: int = 1,
    ):
        super().__init__()
        self.track_group = track_group
        self.n_tracks = n_tracks
        self.fuse_size = fuse_size

    def fuse(
        self,
        h_list: "list[torch.Tensor]",
        h_pre_list: "list[torch.Tensor]",
    ) -> "list[torch.Tensor]":
        """Rank-local group sum: each run of F consecutive tracks adopts its
        COMBINED delta, so between global syncs the group computes as one F-wide
        track (an N/F-track model on the same slices).

        ``h_pre_list[i]`` is track i's input to the sublayer just run; within a
        group those are identical (the previous fuse handed every member the same
        tensor), so ``h_pre_list[g]`` is the group's common pre-state. Members
        come back as the SAME tensor object, which keeps every per-track loop
        downstream unchanged.

        Purely local — no collective — so groups must not straddle ranks;
        `PTWrappedModel` validates that.
        """
        if self.fuse_size <= 1:
            return h_list
        out: list[torch.Tensor] = []
        for g in range(0, len(h_list), self.fuse_size):
            grp = h_list[g : g + self.fuse_size]
            p = h_pre_list[g]
            fused = p + sum((h - p for h in grp[1:]), grp[0] - p)
            out.extend([fused] * len(grp))
        return out

    def leaders(self, h_list: "list[torch.Tensor]") -> "list[torch.Tensor]":
        """One representative per fused group, for feeding a global sync.

        Post-`fuse` the F members of a group are the same tensor, each already
        carrying the group's whole delta — summing all F would count every delta
        F times. Fusing BEFORE the boundary is what makes this exact: the members'
        shared pre-state is folded into one leader delta, not F copies of it.
        """
        return h_list if self.fuse_size <= 1 else h_list[:: self.fuse_size]

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
            if partial.requires_grad:
                partial = _AllReduceSum.apply(partial, self.track_group)
            else:
                dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.track_group)

        return h_pre_block + partial
