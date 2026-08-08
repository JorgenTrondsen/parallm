"""`plan_track_layout`: how a requested fusion width F becomes a real layout.

F is the quality knob (F shards computing as one track between syncs). It has four
possible implementations, three of which are the same grouping expressed differently,
and picking the wrong one either loses the step-time win or asks for a fold that does
not exist. The policy this pins:

  - dense family, F <= K:  merge the rank's whole set, undo with `exec_groups`
  - MoE family,   F <= K:  merge exactly F, loop over K/F merged tracks
  - any family,   F >  K:  merge the whole set AND pool across F/K ranks

The invariant every case must satisfy is ``effective_fuse == F``.
"""
from __future__ import annotations

import pytest

from parallm.model.merge import _valid_fuse_widths, plan_track_layout


def _plan(n, w, f, **kw):
    return plan_track_layout(n, w, f, **kw)


# --------------------------------------------------------------------------- #
# The invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,w", [(64, 8), (24, 8), (16, 8), (8, 8), (64, 4)])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("merge", [True, False])
def test_effective_fuse_always_equals_requested_f(n, w, batched, merge):
    widths = _valid_fuse_widths(n, w)
    assert widths, (n, w)
    for f in widths:
        plan = _plan(n, w, f, supports_batched_exec=batched, supports_merged_tracks=merge)
        assert plan.effective_fuse == f, (n, w, f, batched, merge, plan)
        # And the logical track count is always the shard count over the merge width.
        assert plan.n_logical_tracks == n // plan.merge_group


@pytest.mark.parametrize("n,w", [(64, 8), (24, 8), (16, 8), (64, 4)])
def test_valid_widths_are_exactly_the_accepted_ones(n, w):
    """The error messages advertise `_valid_fuse_widths`; it must not lie."""
    ok = set(_valid_fuse_widths(n, w))
    for f in range(1, n + 1):
        try:
            _plan(n, w, f)
        except ValueError:
            assert f not in ok, f"{f} rejected but advertised as valid for {n}/{w}"
        else:
            assert f in ok, f"{f} accepted but not advertised for {n}/{w}"


# --------------------------------------------------------------------------- #
# Per-case policy
# --------------------------------------------------------------------------- #


def test_dense_family_keeps_todays_behaviour():
    """Unchanged for qwen: merge the whole rank, express F through exec_groups."""
    plan = _plan(24, 8, 1)  # K=3, the default --fuse-tracks 1
    assert (plan.merge_group, plan.exec_groups, plan.fuse_size, plan.fuse_ranks) == (3, 3, 1, 1)
    plan = _plan(24, 8, 3)  # F == K: one wide fused track
    assert (plan.merge_group, plan.exec_groups, plan.fuse_size, plan.fuse_ranks) == (3, 1, 1, 1)


def test_moe_family_expresses_f_as_the_merge_width():
    """No batched fold, so F becomes the merge width and the rank loops K/F tracks.

    At N=64 on 8 GPUs this is the difference between 8 narrow tracks and 2 wide ones
    at F=4 — most of the step-time win with no new kernels.
    """
    plan = _plan(64, 8, 4, supports_batched_exec=False)  # K=8
    assert (plan.merge_group, plan.exec_groups, plan.fuse_size, plan.fuse_ranks) == (4, 1, 1, 1)
    assert plan.n_logical_tracks == 16  # 2 merged tracks on each of 8 ranks

    plan = _plan(64, 8, 8, supports_batched_exec=False)  # F == K: one wide track
    assert (plan.merge_group, plan.exec_groups) == (8, 1)
    assert plan.n_logical_tracks == 8

    # F=1 is the honest cost of having no batched fold: nothing to merge.
    plan = _plan(64, 8, 1, supports_batched_exec=False)
    assert plan.merge_group == 1 and plan.n_logical_tracks == 64


def test_f_above_tracks_per_rank_goes_cross_rank():
    """The capability this whole change exists for: F is no longer capped by K."""
    for f, expect_ranks in ((16, 2), (32, 4), (64, 8)):
        plan = _plan(64, 8, f, supports_batched_exec=False)
        assert plan.fuse_ranks == expect_ranks, (f, plan)
        assert plan.merge_group == 8 and plan.exec_groups == 1
        assert plan.effective_fuse == f


def test_cross_rank_never_combines_with_the_batched_fold():
    """`exec_groups` and `fuse_ranks` both need the rank to hold ONE logical track,
    and the batched walk does not call `fuse` at all — so they must not co-occur."""
    for batched in (True, False):
        plan = _plan(64, 8, 16, supports_batched_exec=batched)
        assert plan.fuse_ranks > 1
        assert plan.exec_groups == 1


def test_no_merge_falls_back_to_summed_fusion():
    """`--no-merge-fused` (and any family that cannot merge) keeps the looped path,
    but still gets the cross-rank tier — it is what eval uses to score a run
    trained at F > K."""
    plan = _plan(64, 8, 4, allow_merge=False)
    assert (plan.merge_group, plan.exec_groups, plan.fuse_size, plan.fuse_ranks) == (1, 1, 4, 1)
    assert plan.n_logical_tracks == 64

    plan = _plan(64, 8, 16, allow_merge=False)
    assert (plan.merge_group, plan.fuse_size, plan.fuse_ranks) == (1, 8, 2)
    assert plan.effective_fuse == 16


# --------------------------------------------------------------------------- #
# Refusals — each should name the valid options rather than fail downstream
# --------------------------------------------------------------------------- #


def test_rejects_indivisible_layouts():
    with pytest.raises(ValueError, match="not divisible by world_size"):
        _plan(24, 5, 1)
    with pytest.raises(ValueError, match="must be >= 1 and divide n_tracks"):
        _plan(24, 8, 5)
    # F below K must still divide K, or the rank cannot form whole groups.
    with pytest.raises(ValueError, match="must divide tracks-per-rank"):
        _plan(24, 4, 4)  # K=6, and 4 does not divide 6
    # F above K must be a MULTIPLE of K — this does NOT follow from F | n_tracks:
    # at n=24/world=8, K=3 and F=4 divides 24 but straddles ranks 1.33 deep.
    with pytest.raises(ValueError, match="must be a MULTIPLE"):
        _plan(24, 8, 4)
