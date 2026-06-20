"""Unit tests for dynamic sync-boundary placement (`place_sync_boundaries`).

Pure-python, CPU-only. The schedule is chosen at train time from a per-layer
sensitivity score: the last layer is always a boundary (a correctness
requirement — the post-stack norm reads the last synced hidden), and the
remaining budget goes to the highest-sensitivity layers.
"""
from __future__ import annotations

import pytest

from pt_converter.eval.sensitivity import debias_partial_error
from pt_converter.slicer.convert import place_sync_boundaries


def test_final_layer_always_included():
    num_layers = 8
    sens = {L: 0.0 for L in range(num_layers)}
    # Budget of 1 spends the whole budget on the forced final-layer sync.
    assert place_sync_boundaries(num_layers, 1, sens) == [num_layers - 1]


def test_picks_top_sensitivity():
    num_layers = 8
    sens = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.3, 5: 0.7, 6: 0.05, 7: 0.0}
    # Budget 4 → forced final (7) + the 3 highest-sensitivity among 0..6.
    assert place_sync_boundaries(num_layers, 4, sens) == [1, 3, 5, 7]


def test_returns_sorted_unique_with_final():
    num_layers = 6
    sens = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    chosen = place_sync_boundaries(num_layers, 3, sens)
    assert chosen == sorted(set(chosen))
    assert chosen[-1] == num_layers - 1
    assert len(chosen) == 3


def test_ties_break_by_index_deterministic():
    num_layers = 6
    sens = {L: 1.0 for L in range(num_layers)}  # all tied
    chosen = place_sync_boundaries(num_layers, 3, sens)
    # Final layer 5 forced; the remaining two go to the LOWEST indices on a tie.
    assert chosen == [0, 1, 5]
    # Same inputs → same schedule (every rank must agree).
    assert place_sync_boundaries(num_layers, 3, sens) == chosen


def test_full_budget_includes_all_layers():
    num_layers = 4
    sens = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    assert place_sync_boundaries(num_layers, num_layers, sens) == [0, 1, 2, 3]


def test_list_and_dict_sensitivity_equivalent():
    num_layers = 6
    as_list = [0.1, 0.5, 0.2, 0.9, 0.3, 0.0]
    as_dict = {L: v for L, v in enumerate(as_list)}
    assert place_sync_boundaries(num_layers, 3, as_list) == place_sync_boundaries(
        num_layers, 3, as_dict
    )


def test_invalid_budget_raises():
    sens = {L: 0.0 for L in range(8)}
    with pytest.raises(ValueError):
        place_sync_boundaries(8, 0, sens)
    with pytest.raises(ValueError):
        place_sync_boundaries(8, 9, sens)  # > num_layers


# ----- de-biased sensitivity scoring (debias_partial_error) -----

# Synthetic 6-layer profile: teacher norm grows with depth (the real residual-stream
# behaviour), so the plain relative score (abs/norm) ranks the small-norm early layers
# highest even though their absolute error is smallest.
_ABS = {0: 0.12, 1: 0.21, 2: 0.30, 3: 0.45, 4: 0.60, 5: 0.90}
_TNORM = {0: 0.10, 1: 0.15, 2: 0.40, 3: 0.80, 4: 1.50, 5: 3.00}


def _rank(score):
    return sorted(score, key=lambda L: -score[L])


def test_debias_q0_is_pure_relative_early_biased():
    score = debias_partial_error(_ABS, _TNORM, quantile=0.0)
    # floor = min norm → no clamp → score == abs/norm (relative); early layers on top.
    for L in _ABS:
        assert score[L] == pytest.approx(_ABS[L] / _TNORM[L])
    assert _rank(score)[:2] == [1, 0]  # the small-norm early layers rank highest


def test_debias_q1_is_raw_deep_biased():
    score = debias_partial_error(_ABS, _TNORM, quantile=1.0)
    # floor = max norm → every layer divided by the same global scale → raw ranking.
    assert _rank(score) == [5, 4, 3, 2, 1, 0]  # deepest (largest abs err) on top


def test_debias_median_demotes_small_norm_early_layers():
    score = debias_partial_error(_ABS, _TNORM, quantile=0.5)
    rank = _rank(score)
    # The early spike (L1 was #1 under relative) is demoted below the mid/deep layers.
    assert rank[0] in (2, 3)            # a mid/deep layer now leads
    assert rank.index(1) > rank.index(2)  # L1 below L2
    assert rank.index(0) > rank.index(3)  # L0 below L3
    # Layers at/above the median norm keep their natural (relative) order: L3 > L2.
    assert rank.index(3) < rank.index(2)
