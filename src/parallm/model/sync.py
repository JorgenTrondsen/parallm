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

`fuse_group` extends that tier ACROSS ranks: a contiguous block of R ranks
pools its partials with a subgroup all-reduce at every sublayer no global sync
covers, so the block computes as one track of R*K shards. This is the only way
to reach a fusion width larger than a rank's own track count — the rank-local
tier caps F at n_tracks/world_size, which means adding GPUs LOWERS the
reachable F. It costs one extra collective per non-boundary sublayer (forward
only; the backward is identity) and, because the R members then hold the same
state, only the block LEADER may feed the global boundary — see `_LeaderOnly`.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

# Distinguishes "caller did not name a group, use track_group" from "caller passed
# None", which legitimately means no collective (single-process multi-track testing,
# and the rank-local degenerate case of a fusion block).
_UNSET = object()


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


class _LeaderOnly(torch.autograd.Function):
    """Contribute this tensor to the global sum only on a fusion block's leader.

    After a cross-rank fuse all R ranks of a block hold the SAME state, each already
    carrying the block's whole delta — so all R feeding the boundary all-reduce would
    count that delta R times. The leader contributes it; peers contribute zero.

    Backward is IDENTITY, and that is the whole point of this being a Function rather
    than a `torch.zeros_like` in the caller: zeroing the forward would also zero the
    gradient, cutting every non-leader rank's parameters out of the graph from this
    boundary onward and silently training only 1/R of the model. The peer's state is
    numerically identical to the leader's, so its local dL/dstate is the right
    gradient — the same argument `_AllReduceSum` already makes for the boundary sum.
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor, is_leader: bool) -> torch.Tensor:
        return t if is_leader else torch.zeros_like(t)

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
        fuse_group: process group over the R ranks of this rank's CROSS-RANK fusion
                    block, or None for the rank-local-only tier.
        fuse_ranks: R, the block size. 1 = off.
        fuse_rank: this rank's index within its block; 0 is the block leader, the
                   only member that feeds a global boundary (see `_LeaderOnly`).
    """

    def __init__(
        self,
        track_group: "dist.ProcessGroup | None" = None,
        n_tracks: int = 1,
        fuse_size: int = 1,
        fuse_group: "dist.ProcessGroup | None" = None,
        fuse_ranks: int = 1,
        fuse_rank: int = 0,
    ):
        super().__init__()
        self.track_group = track_group
        self.n_tracks = n_tracks
        self.fuse_size = fuse_size
        self.fuse_group = fuse_group
        self.fuse_ranks = fuse_ranks
        self.fuse_rank = fuse_rank
        # Turned off under `sync_phase="exact"`, where every sublayer syncs globally
        # so any grouping sums the same deltas — the cross-rank collective would be
        # pure cost. `PTWrappedModel.set_sync_phase` owns this.
        self.cross_rank_enabled = True

    @property
    def cross_rank_active(self) -> bool:
        return self.fuse_ranks > 1 and self.cross_rank_enabled

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

        Local-only when `cross_rank_active` is False — so a rank-local group must
        not straddle ranks; `PTWrappedModel` validates that. When cross-rank fusion
        IS active the block spans whole ranks, so ALL local tracks are one group and
        their combined delta goes through a subgroup all-reduce; every member of the
        block comes back holding the same state.
        """
        cross = self.cross_rank_active
        if self.fuse_size <= 1 and not cross:
            return h_list
        step = self._group_step(len(h_list))
        out: list[torch.Tensor] = []
        for g in range(0, len(h_list), step):
            grp = h_list[g : g + step]
            p = h_pre_list[g]
            delta = sum((h - p for h in grp[1:]), grp[0] - p)
            if cross:
                delta = self._all_reduce(delta, self.fuse_group)
            out.extend([p + delta] * len(grp))
        return out

    def _group_step(self, n_local: int) -> int:
        """Local tracks per fusion group. A cross-rank block pools WHOLE ranks, so
        every local track is in the same group."""
        return n_local if self.cross_rank_active else self.fuse_size

    def leaders(self, h_list: "list[torch.Tensor]") -> "list[torch.Tensor]":
        """One representative per fused group, for feeding a global sync.

        Post-`fuse` the F members of a group are the same tensor, each already
        carrying the group's whole delta — summing all F would count every delta
        F times. Fusing BEFORE the boundary is what makes this exact: the members'
        shared pre-state is folded into one leader delta, not F copies of it.

        This dedupes WITHIN a rank. Across ranks the same double-count is handled in
        `forward` by `_LeaderOnly`, because every rank of a block holds a full copy.
        """
        if self.fuse_size <= 1 and not self.cross_rank_active:
            return h_list
        return h_list[:: self._group_step(len(h_list))]

    def forward(
        self,
        h_list: "list[torch.Tensor] | torch.Tensor",
        h_pre_block: torch.Tensor,
        stacked: bool = False,
        block_shared: bool = True,
    ) -> torch.Tensor:
        """``stacked`` says ``h_list`` is ONE ``[G, ...]`` tensor of G members'
        states (the batched merged path), not a per-track list — so the local
        reduce is a single ``sum(0)`` instead of G-1 full-residual adds. Explicit
        rather than sniffed from the shape: a bare tensor otherwise still means
        the legacy K=1 caller, and the two must not be confused.

        ``block_shared`` says the incoming states are post-`fuse`, i.e. every rank
        of a cross-rank fusion block already holds that block's whole delta, so only
        the leader may contribute. False for the embedding broadcast, which runs
        before any sublayer and therefore before any block state exists."""
        if stacked:
            if self.n_tracks <= 1 and h_list.shape[0] == 1:
                return h_list[0]
            partial = (h_list - h_pre_block).sum(0)
            if block_shared:
                partial = self._block_share(partial)
            return h_pre_block + self._all_reduce(partial)

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

        if block_shared:
            partial = self._block_share(partial)
        return h_pre_block + self._all_reduce(partial)

    def _block_share(self, partial: torch.Tensor) -> torch.Tensor:
        """This rank's contribution to the global sum under cross-rank fusion.

        Every rank of a fusion block holds the block's whole delta after `fuse`, so
        only the leader may contribute it. No-op when cross-rank fusion is off.
        """
        if not self.cross_rank_active:
            return partial
        return _LeaderOnly.apply(partial, self.fuse_rank == 0)

    def _all_reduce(
        self, partial: torch.Tensor, group: "dist.ProcessGroup | None" = _UNSET
    ) -> torch.Tensor:
        if group is _UNSET:
            group = self.track_group
        if group is None:
            return partial
        if partial.requires_grad:
            return _AllReduceSum.apply(partial, group)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        return partial
