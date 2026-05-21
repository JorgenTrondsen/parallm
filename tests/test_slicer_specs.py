"""Unit-test each SlicerSpec on synthetic tensors."""
from __future__ import annotations

import torch

from pt_converter.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    PerHead,
    Replicated,
    Rowwise,
)


def _assert_roundtrip(spec, full_shape, n_tracks):
    full = torch.arange(int(torch.tensor(full_shape).prod()), dtype=torch.float32).reshape(full_shape)
    slices = [spec.slice(full, t, n_tracks) for t in range(n_tracks)]
    # per_track_shape matches
    for s in slices:
        assert tuple(s.shape) == spec.per_track_shape(full_shape, n_tracks)
    # reassemble round-trips for non-replicated specs
    if not isinstance(spec, Replicated):
        reassembled = spec.reassemble(slices)
        assert torch.equal(reassembled, full)


def test_colwise_roundtrip():
    _assert_roundtrip(Colwise(dim=0), (16, 4), n_tracks=4)


def test_rowwise_roundtrip():
    _assert_roundtrip(Rowwise(dim=1), (4, 16), n_tracks=4)


def test_perhead_roundtrip():
    _assert_roundtrip(PerHead(num_heads=8), (8,), n_tracks=4)


def test_replicated_returns_full_copy_on_every_track():
    spec = Replicated()
    full = torch.randn(4, 4)
    for t in range(3):
        s = spec.slice(full, t, n_tracks=3)
        assert torch.equal(s, full)
        assert s.data_ptr() != full.data_ptr()  # cloned


def test_fused_segment_colwise_roundtrip():
    # Mimic in_proj_qkv where segments = (key_dim, key_dim, value_dim).
    # Use tiny values: key_dim=8, value_dim=16, n_tracks=4.
    spec = FusedSegmentColwise(segments=(8, 8, 16), dim=0)
    full_shape = (8 + 8 + 16, 4)
    full = torch.arange(32 * 4, dtype=torch.float32).reshape(full_shape)
    slices = [spec.slice(full, t, 4) for t in range(4)]
    # Per-track shape: (2 + 2 + 4, 4) = (8, 4)
    for s in slices:
        assert s.shape == (8, 4)
    reassembled = spec.reassemble(slices)
    assert torch.equal(reassembled, full)


def test_gated_q_colwise_round_trip():
    # 4 heads, head_dim=2, so q_proj has out_features = 4 * 2 * 2 = 16.
    # Heads laid out as [h0_q, h0_g, h1_q, h1_g, h2_q, h2_g, h3_q, h3_g],
    # each segment has length head_dim=2.
    spec = GatedQColwise(num_heads=4, head_dim=2, dim=0)
    full = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    slices = [spec.slice(full, t, 2) for t in range(2)]
    # Track 0 holds heads [0,1] -> rows 0..7. Track 1 holds heads [2,3] -> rows 8..15.
    assert slices[0].shape == (8, 3)
    assert slices[1].shape == (8, 3)
    assert torch.equal(slices[0], full[:8])
    assert torch.equal(slices[1], full[8:])
    assert torch.equal(spec.reassemble(slices), full)


def test_colwise_rejects_indivisible_dim():
    spec = Colwise(dim=0)
    full = torch.zeros(10, 4)
    try:
        spec.slice(full, 0, n_tracks=3)
    except ValueError as e:
        assert "not divisible" in str(e)
        return
    raise AssertionError("expected ValueError")
