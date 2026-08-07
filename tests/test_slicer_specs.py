"""Unit-test each SlicerSpec on synthetic tensors."""
from __future__ import annotations

import pytest
import torch

from parallm.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    KVReplicatedColwise,
    OwnerOnly,
    PerHead,
    Replicated,
    Rowwise,
    SummedBias,
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


def test_owner_only_returns_full_tensor_on_owner_and_none_elsewhere():
    spec = OwnerOnly(owner_track=0)
    full = torch.randn(4, 4)
    on_owner = spec.slice(full, 0, n_tracks=4)
    assert torch.equal(on_owner, full)
    assert on_owner.data_ptr() != full.data_ptr()
    for t in range(1, 4):
        assert spec.slice(full, t, n_tracks=4) is None


def test_owner_only_reassemble_picks_unique_non_none_slice():
    spec = OwnerOnly(owner_track=2)
    full = torch.randn(3, 5)
    slices = [None, None, full, None]
    assert torch.equal(spec.reassemble(slices), full)


def test_owner_only_per_track_shape_is_full_shape():
    spec = OwnerOnly(owner_track=0)
    assert spec.per_track_shape((128, 64), n_tracks=8) == (128, 64)


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


def test_kv_replicated_groups_match_within_group():
    """Tracks in the same kv-group must see identical slices; different groups differ."""
    # 4 kv-heads, head_dim=8 → out_features=32. N=8 → tracks_per_kv_group=2.
    spec = KVReplicatedColwise(num_kv_heads=4)
    full = torch.arange(32 * 5, dtype=torch.float32).reshape(32, 5)
    n_tracks = 8
    slices = [spec.slice(full, t, n_tracks) for t in range(n_tracks)]
    # tracks 0,1 → kv-group 0; tracks 2,3 → group 1; etc.
    for g in range(4):
        a, b = slices[g * 2], slices[g * 2 + 1]
        assert torch.equal(a, b), f"kv-group {g} replicas differ"
    # Different groups must hold different rows.
    assert not torch.equal(slices[0], slices[2])
    # reassemble with one slice per kv-group (the unique slices) reproduces the dense.
    uniques = [slices[g * 2] for g in range(4)]
    assert torch.equal(spec.reassemble(uniques), full)


def test_kv_replicated_rejects_n_not_multiple_of_kv():
    spec = KVReplicatedColwise(num_kv_heads=4)
    full = torch.zeros(32, 5)
    with pytest.raises(ValueError):
        spec.slice(full, 0, n_tracks=6)


def test_kv_replicated_per_track_shape():
    spec = KVReplicatedColwise(num_kv_heads=4)
    # full_shape (32, 5) -> per-track (8, 5) regardless of n_tracks.
    assert spec.per_track_shape((32, 5), n_tracks=4) == (8, 5)
    assert spec.per_track_shape((32, 5), n_tracks=8) == (8, 5)
    assert spec.per_track_shape((32, 5), n_tracks=16) == (8, 5)


def test_colwise_rejects_indivisible_dim():
    spec = Colwise(dim=0)
    full = torch.zeros(10, 4)
    try:
        spec.slice(full, 0, n_tracks=3)
    except ValueError as e:
        assert "not divisible" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_summed_bias_owner_keeps_it_peers_get_zeros():
    spec = SummedBias()
    full = torch.randn(8)
    slices = [spec.slice(full, t, 4) for t in range(4)]
    assert torch.equal(slices[0], full)
    for s in slices[1:]:
        assert torch.count_nonzero(s) == 0
    # every track keeps the full shape (it is added to a full-width output)
    for s in slices:
        assert tuple(s.shape) == spec.per_track_shape(tuple(full.shape), 4)


def test_summed_bias_reassembles_by_summing():
    spec = SummedBias()
    full = torch.randn(3, 5)
    slices = [spec.slice(full, t, 3) for t in range(3)]
    assert torch.equal(spec.reassemble(slices), full)
    # A DIVERGED set (what training produces) reassembles to its sum, which is the
    # effective bias — not to any single copy.
    diverged = [torch.randn(3, 5) for _ in range(3)]
    assert torch.allclose(spec.reassemble(diverged), sum(diverged))


def test_summed_bias_groups_are_singletons():
    # Averaging these copies would not preserve their sum, so they never share a grad.
    assert SummedBias().replication_groups(4) == [[0], [1], [2], [3]]
    assert SummedBias().replication_groups(4, force_sync=True) == [[0], [1], [2], [3]]


def test_summed_bias_merge_split_preserve_the_sum():
    spec = SummedBias()
    members = [torch.randn(6) for _ in range(3)]
    merged = spec.merge(members, n_tracks=6)
    assert torch.allclose(merged, sum(members))
    back = spec.split(merged, fuse=3, n_tracks=6)
    assert len(back) == 3
    assert torch.allclose(sum(back), merged)
