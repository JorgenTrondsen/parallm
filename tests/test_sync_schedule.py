"""Verify the sync-block-depth → layer-indices mapping."""
from __future__ import annotations

from pt_converter.slicer.convert import _resolve_sync_schedule

# Qwen3.5 layer pattern: full-attention every 4th layer (indices 3, 7, ..., 31).
QWEN_LAYER_TYPES = [
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)
]


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


# ----- Full-attention-aligned schedule -----
# A sync fires immediately BEFORE each full-attention layer (after f-1) plus a
# final sync before the norm; gaps longer than D are subdivided.


def test_aligned_d4():
    # D == full-attention interval: every full-attn layer (3,7,...,31) reads a
    # synced residual; 9 syncs (one extra vs uniform — the final norm sync).
    assert _resolve_sync_schedule(
        32, 4, layer_types=QWEN_LAYER_TYPES, align_full_attention=True
    ) == [2, 6, 10, 14, 18, 22, 26, 30, 31]


def test_aligned_d2():
    # D < interval: full-attn layers still aligned, intermediate syncs keep
    # windows <= 2. Matches the literal shift-by-1 of the uniform D=2 schedule.
    assert _resolve_sync_schedule(
        32, 2, layer_types=QWEN_LAYER_TYPES, align_full_attention=True
    ) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31]


def test_aligned_d8_collapses_to_interval():
    # D >= interval: aligning every full-attn layer forces one sync per interval,
    # so D=8 aligned == D=4 aligned.
    assert _resolve_sync_schedule(
        32, 8, layer_types=QWEN_LAYER_TYPES, align_full_attention=True
    ) == [2, 6, 10, 14, 18, 22, 26, 30, 31]


def test_aligned_does_not_require_dividing_d():
    # The subdivide loop handles any D — no divisibility constraint in aligned mode.
    sched = _resolve_sync_schedule(
        32, 3, layer_types=QWEN_LAYER_TYPES, align_full_attention=True
    )
    full_attn = {i for i, t in enumerate(QWEN_LAYER_TYPES) if t == "full_attention"}
    # Every full-attention layer reads a synced residual: f-1 is a sync point.
    assert all((f - 1) in sched for f in full_attn)
    # Final layer is synced for the post-stack norm.
    assert sched[-1] == 31
    # No window exceeds D=3 layers (gap between consecutive syncs, and the prefix).
    prev = -1
    for idx in sched:
        assert idx - prev <= 3
        prev = idx


def test_aligned_requires_layer_types():
    try:
        _resolve_sync_schedule(32, 4, align_full_attention=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError when layer_types missing in aligned mode")
