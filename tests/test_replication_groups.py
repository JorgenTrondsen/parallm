"""Unit tests for SlicerSpec.replication_groups.

`replication_groups(n_tracks)` partitions tracks into sets that hold
*identical* slices. The trainer uses this partition to keep replicated
copies bit-identical under gradient updates. See
[src/parallm/train/sync_grads.py](src/parallm/train/sync_grads.py).
"""
from __future__ import annotations

import pytest

from parallm.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    KVReplicatedColwise,
    OwnerOnly,
    PerHead,
    Replicated,
    Rowwise,
)
from parallm.train.sync_grads import get_replication_groups


def test_replicated_one_group_with_all_tracks():
    assert Replicated().replication_groups(4) == [[0, 1, 2, 3]]
    assert Replicated().replication_groups(1) == [[0]]


def test_replicated_sync_false_diverges_into_singletons():
    # sync=False ⇒ each copy is its own group (free to diverge).
    assert Replicated(sync=False).replication_groups(4) == [[0], [1], [2], [3]]
    # force_sync overrides sync=False back to one shared group.
    assert Replicated(sync=False).replication_groups(4, force_sync=True) == [[0, 1, 2, 3]]
    # force_sync is a no-op when already syncing.
    assert Replicated().replication_groups(4, force_sync=True) == [[0, 1, 2, 3]]


def test_owner_only_returns_singleton_owner_group():
    assert OwnerOnly(owner_track=0).replication_groups(8) == [[0]]
    assert OwnerOnly(owner_track=3).replication_groups(8) == [[3]]


def test_kv_replicated_partitions_into_kv_groups():
    # n_tracks=8, num_kv_heads=2 -> tracks_per_group=4 -> two groups of four.
    spec = KVReplicatedColwise(num_kv_heads=2)
    assert spec.replication_groups(8) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_kv_replicated_at_singleton_when_n_equals_num_kv_heads():
    # n_tracks=4, num_kv_heads=4 -> each track is its own kv-group (no replication).
    spec = KVReplicatedColwise(num_kv_heads=4)
    assert spec.replication_groups(4) == [[0], [1], [2], [3]]


def test_kv_replicated_sync_false_diverges_into_singletons():
    # sync=False ⇒ every track's KV head trains independently (no kv-group sync).
    spec = KVReplicatedColwise(num_kv_heads=2, sync=False)
    assert spec.replication_groups(8) == [[t] for t in range(8)]
    # force_sync overrides back to the bit-identical kv-group partition.
    assert spec.replication_groups(8, force_sync=True) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    # Validation still fires when n_tracks is not a multiple of num_kv_heads.
    with pytest.raises(ValueError):
        KVReplicatedColwise(num_kv_heads=3, sync=False).replication_groups(8)


def test_kv_replicated_rejects_n_not_multiple_of_kv():
    spec = KVReplicatedColwise(num_kv_heads=3)
    with pytest.raises(ValueError):
        spec.replication_groups(8)


@pytest.mark.parametrize(
    "spec",
    [
        Colwise(),
        Rowwise(),
        PerHead(num_heads=8),
        FusedSegmentColwise(segments=(8, 8, 16)),
        GatedQColwise(num_heads=4, head_dim=2),
    ],
)
def test_unique_specs_default_to_singleton_groups(spec):
    # These specs partition the dense tensor uniquely per track; no replication.
    groups = get_replication_groups(spec, n_tracks=4)
    assert groups == [[0], [1], [2], [3]]


def test_get_replication_groups_dispatches_to_method():
    # Replicated implements the method; result matches direct call.
    spec = Replicated()
    assert get_replication_groups(spec, 6) == spec.replication_groups(6)


def test_get_replication_groups_threads_force_sync():
    # A diverging spec returns singletons by default but re-syncs under force_sync.
    spec = Replicated(sync=False)
    assert get_replication_groups(spec, 4) == [[0], [1], [2], [3]]
    assert get_replication_groups(spec, 4, force_sync=True) == [[0, 1, 2, 3]]


def test_get_replication_groups_force_sync_with_legacy_signature():
    # A spec whose replication_groups predates the force_sync kwarg is still
    # callable (the dispatch falls back to the (n_tracks)-only call).
    class _Legacy:
        def replication_groups(self, n_tracks):
            return [list(range(n_tracks))]

    assert get_replication_groups(_Legacy(), 3, force_sync=True) == [[0, 1, 2]]


def test_get_replication_groups_falls_back_for_specs_without_method():
    # A bare object without `replication_groups` falls back to singletons.
    class _Bare:
        pass

    assert get_replication_groups(_Bare(), n_tracks=3) == [[0], [1], [2]]
