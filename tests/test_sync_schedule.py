"""Verify the sync-block-depth → layer-indices mapping."""
from __future__ import annotations

from pt_converter.slicer.convert import _resolve_sync_schedule


def test_d4_on_32_layers():
    # D=4 → sync after layers 3, 7, 11, ..., 31 (eight syncs).
    assert _resolve_sync_schedule(32, 4) == [3, 7, 11, 15, 19, 23, 27, 31]


def test_d2_on_32_layers():
    assert _resolve_sync_schedule(32, 2) == list(range(1, 32, 2))


def test_d8_on_32_layers():
    assert _resolve_sync_schedule(32, 8) == [7, 15, 23, 31]


def test_non_dividing_d_raises():
    try:
        _resolve_sync_schedule(32, 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-dividing D")
