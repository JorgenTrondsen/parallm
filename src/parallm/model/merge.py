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

import torch

from parallm.slicer.base import OwnerOnly
from parallm.slicer.convert import resolve_param_specs


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
