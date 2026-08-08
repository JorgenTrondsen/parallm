"""Merge F consecutive per-track shards into one F-wide track (and back).

With ``--fuse-tracks F`` set to a rank's whole track count, the F tracks share
their input at every sublayer and the rank contributes exactly one delta to each
global sync. So one wide track holding the concatenated slabs computes the same
function as F narrow tracks summing their deltas — while issuing 1/F the kernels
and carrying ONE full-width residual stream instead of F. On the 27B at N=24 that
is the difference between 6.4-7.7 s/step and N=8's ~3.0 s/step.

This is a training-time representation only: ``split_track_state`` puts the
checkpoint back into N shards on save, so eval, serve and the deploy artifact are
untouched.

Every parameter's concatenation rule already lives on its ``SlicerSpec``
(``merge``/``split`` beside ``slice``/``reassemble``); this module is just the
canonical-name dispatch over ``resolve_param_specs``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from parallm.slicer.base import OwnerOnly
from parallm.slicer.convert import resolve_param_specs


@dataclass(frozen=True)
class TrackLayoutPlan:
    """How a requested fusion width F is realised on ``world_size`` ranks.

    Four numbers, three of which are implementations of the SAME grouping — F
    consecutive shards computing as one track between global syncs:

    - ``merge_group`` m: shards CONCATENATED into one wide module. The only one that
      changes step time (fewer, wider GEMMs; one residual stream instead of m).
    - ``exec_groups`` G: that merged module RUN as G independent streams
      (`parallm.model.batched`), which UNDOES m/G members' worth of fusion. Buys the
      merged step time back at a smaller F. Dense families only.
    - ``fuse_size``: rank-local SUMMING of logical tracks' partials
      (`SyncBoundary.fuse`). The original, speed-neutral implementation.
    - ``fuse_ranks`` R: logical tracks pooled ACROSS ranks by a subgroup all-reduce.
      The only way to reach F > tracks-per-rank, and the only one that costs a
      collective.

    Their product is the effective fusion width: ``m / G * fuse_size * R == F``.
    """

    merge_group: int = 1
    exec_groups: int = 1
    fuse_size: int = 1
    fuse_ranks: int = 1
    n_logical_tracks: int = 1

    @property
    def effective_fuse(self) -> int:
        return (self.merge_group // self.exec_groups) * self.fuse_size * self.fuse_ranks


def _valid_fuse_widths(n_tracks: int, world_size: int) -> list[int]:
    """Every F this layout can express: the divisors of K (rank-local groups) plus
    the multiples of K up to n_tracks (whole-rank blocks)."""
    k = n_tracks // world_size
    below = [d for d in range(1, k + 1) if k % d == 0]
    above = [k * r for r in range(2, world_size + 1) if world_size % r == 0]
    return below + above


def plan_track_layout(
    n_tracks: int,
    world_size: int,
    fuse_tracks: int,
    *,
    supports_merged_tracks: bool = True,
    supports_batched_exec: bool = True,
    allow_merge: bool = True,
) -> TrackLayoutPlan:
    """Choose (merge_group, exec_groups, fuse_size, fuse_ranks) for F=``fuse_tracks``.

    The policy, given K = n_tracks/world_size shards per rank:

    ==========================  ===========================================
    F <= K, batched-capable     m=K, G=K/F  — merged storage at any F
    F <= K, merge-only (MoE)    m=F, G=1    — K/F merged tracks/rank, looped
    F >  K                      m=K, G=1, R=F/K — cross-rank fusion groups
    merging refused             m=1, fuse_size=min(F,K), R=F/K
    ==========================  ===========================================

    Why ``m=F`` rather than ``m=K, G=K/F`` for an MoE: `exec_groups` needs a batched
    fold, and an MoE has none — under G > 1 each stream carries its own residual, so
    the shared router picks a different top-k per stream and no single ``grouped_mm``
    exists. Merging only F shards still collapses F kernels into one F-times-wider
    one; the rank just loops over K/F of them instead of 1.
    """
    if n_tracks % world_size:
        raise ValueError(
            f"n_tracks {n_tracks} not divisible by world_size {world_size}"
        )
    k = n_tracks // world_size
    f = int(fuse_tracks)
    if f < 1 or n_tracks % f:
        raise ValueError(
            f"--fuse-tracks {f} must be >= 1 and divide n_tracks {n_tracks}"
        )
    if f <= k and k % f:
        raise ValueError(
            f"--fuse-tracks {f} must divide tracks-per-rank {k} "
            f"(n_tracks={n_tracks}, world_size={world_size}); valid: "
            f"{_valid_fuse_widths(n_tracks, world_size)}"
        )
    if f > k and f % k:
        # A cross-rank block is R WHOLE ranks, so F must be a multiple of K. This
        # does NOT follow from F dividing n_tracks: at n=24/world=8, K=3 and F=4
        # divides 24 but straddles ranks 1.33 deep, which has no meaning.
        raise ValueError(
            f"--fuse-tracks {f} exceeds tracks-per-rank {k}, so it must be a "
            f"MULTIPLE of {k} (a cross-rank fusion block is whole ranks); valid: "
            f"{_valid_fuse_widths(n_tracks, world_size)}"
        )

    can_merge = supports_merged_tracks and allow_merge
    ranks = f // k if f > k else 1

    if not can_merge:
        return TrackLayoutPlan(
            merge_group=1,
            exec_groups=1,
            fuse_size=min(f, k),
            fuse_ranks=ranks,
            n_logical_tracks=n_tracks,
        )
    if f > k:
        m = k
    elif supports_batched_exec:
        m = k
    else:
        m = f
    return TrackLayoutPlan(
        merge_group=m,
        exec_groups=m // f if f <= m else 1,
        fuse_size=1,
        fuse_ranks=ranks,
        n_logical_tracks=n_tracks // m,
    )


def _spec_for(spec_map: dict, key: str):
    spec = spec_map.get(key)
    if spec is None:
        raise KeyError(f"no slicer spec for per-track key {key!r} — cannot merge it")
    if not hasattr(spec, "merge"):
        raise TypeError(
            f"{type(spec).__name__} has no merge()/split(); add them to the spec "
            f"rather than guessing a concat dim for {key!r}"
        )
    return spec


def merge_track_states(
    adapter,
    text_cfg,
    n_tracks: int,
    states: dict[int, dict[str, torch.Tensor]],
    fuse: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Fold per-track shards into merged tracks, keyed by *logical* track id.

    ``states`` maps global track id → the shard's state_dict; ids must form whole
    ``fuse``-aligned runs. Logical id ``t`` covers global ids
    ``[t*fuse, (t+1)*fuse)`` — the same grouping ``SyncBoundary.fuse`` uses, which
    is what makes the merge equivalent to fusing.
    """
    if fuse < 1:
        raise ValueError(f"fuse must be >= 1, got {fuse}")
    spec_map = resolve_param_specs(adapter, text_cfg)
    tids = sorted(states)
    if len(tids) % fuse or tids[0] % fuse:
        raise ValueError(
            f"merge_track_states needs whole {fuse}-aligned runs of track ids; got {tids}"
        )

    out: dict[int, dict[str, torch.Tensor]] = {}
    for g in range(0, len(tids), fuse):
        group = [states[t] for t in tids[g : g + fuse]]
        keys = list(dict.fromkeys(k for sd in group for k in sd))  # stable union
        merged: dict[str, torch.Tensor] = {}
        for key in keys:
            present = [sd[key] for sd in group if key in sd]
            spec = _spec_for(spec_map, key)
            # Only OwnerOnly legitimately appears on a subset of the group
            # (embed_tokens / lm_head live on track 0 alone); anything else missing
            # means mismatched shards, which would merge to the wrong width.
            if len(present) != fuse and not isinstance(spec, OwnerOnly):
                raise ValueError(
                    f"{key!r} present on {len(present)}/{fuse} shards of group "
                    f"{tids[g]}..{tids[g + fuse - 1]}"
                )
            merged[key] = spec.merge(present, n_tracks)
        out[tids[g] // fuse] = merged
    return out


def split_track_state(
    adapter,
    text_cfg,
    n_tracks: int,
    merged_sd: dict[str, torch.Tensor],
    fuse: int,
    first_tid: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Inverse of `merge_track_states` for one merged track.

    Returns ``{global_track_id: shard_state_dict}`` for the ``fuse`` global ids
    starting at ``first_tid``, so a merged run saves as an ordinary N-shard
    checkpoint. Keys a spec drops for a member (``OwnerOnly`` on peers) are
    omitted from that shard, exactly as the slicer emits them.
    """
    spec_map = resolve_param_specs(adapter, text_cfg)
    out: dict[int, dict[str, torch.Tensor]] = {
        first_tid + i: {} for i in range(fuse)
    }
    for key, val in merged_sd.items():
        parts = _spec_for(spec_map, key).split(val, fuse, n_tracks)
        for i, part in enumerate(parts):
            if part is not None:
                out[first_tid + i][key] = part
    return out
