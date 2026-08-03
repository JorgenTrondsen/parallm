"""End-to-end KV-replication invariants on a synthetic Qwen3.5 text model.

Uses a tiny config with num_attention_heads=8, num_key_value_heads=2 to
exercise N=8 (each kv-group spans 4 tracks). The invariants:

- Tracks in the same kv-group hold pairwise-identical k_proj / v_proj slices.
- Tracks in different kv-groups hold *different* slices.
- The two unique k_proj slices (one per kv-group) concatenated reconstruct
  the dense k_proj bit-equal.
"""
from __future__ import annotations

import pytest
import torch

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.slicer.base import KVReplicatedColwise
from parallm.slicer.convert import slice_model_to_tracks


def _kv_replicated_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,  # 2 kv-groups, each spanning N//2 tracks
        head_dim=8,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
        vocab_size=64,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def kv_model():
    cfg = _kv_replicated_config()
    m = Qwen3_5TextModel(cfg)
    m.eval()
    return m


def test_k_proj_replicated_within_each_kv_group_at_n8(kv_model):
    n_tracks = 8
    num_kv = 2
    tracks_per_group = n_tracks // num_kv  # = 4

    tracks, _ = slice_model_to_tracks(
        kv_model, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    # The only full-attention layer is at idx 3.
    layer_key = "layers.3.self_attn.k_proj.weight"
    slices = [t[layer_key] for t in tracks]
    # Same group: tracks {0,1,2,3} share group 0; {4,5,6,7} share group 1.
    for g in range(num_kv):
        base = slices[g * tracks_per_group]
        for offset in range(1, tracks_per_group):
            other = slices[g * tracks_per_group + offset]
            assert torch.equal(base, other), f"kv-group {g} replicas differ at track offset {offset}"
    # Different groups must hold different rows.
    assert not torch.equal(slices[0], slices[tracks_per_group]), "different kv-groups produced identical k_proj"


def test_v_proj_replicated_within_each_kv_group_at_n8(kv_model):
    n_tracks = 8
    num_kv = 2
    tracks_per_group = n_tracks // num_kv
    tracks, _ = slice_model_to_tracks(
        kv_model, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    layer_key = "layers.3.self_attn.v_proj.weight"
    slices = [t[layer_key] for t in tracks]
    for g in range(num_kv):
        base = slices[g * tracks_per_group]
        for offset in range(1, tracks_per_group):
            assert torch.equal(slices[g * tracks_per_group + offset], base)


def test_unique_kv_group_slices_reconstruct_dense(kv_model):
    """One representative slice per kv-group, concatenated → dense k_proj."""
    n_tracks = 8
    num_kv = 2
    tracks_per_group = n_tracks // num_kv
    tracks, _ = slice_model_to_tracks(
        kv_model, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    state = dict(kv_model.state_dict())
    layer_key = "layers.3.self_attn.k_proj.weight"
    track_slices = [t[layer_key] for t in tracks]
    uniques = [track_slices[g * tracks_per_group] for g in range(num_kv)]
    reassembled = torch.cat(uniques, dim=0)
    assert torch.equal(reassembled, state[layer_key])


def test_per_track_full_attention_shapes_at_n8(kv_model):
    """Sanity-check per-track shapes for the full-attention layer at N=8."""
    tracks, _ = slice_model_to_tracks(
        kv_model, n_tracks=8, sync_block_depth=4, text_config_attr="config"
    )
    # q_proj dense: (8 heads * 8 head_dim * 2, 64) = (128, 64). N=8 → 1 head/track → (16, 64).
    assert tracks[0]["layers.3.self_attn.q_proj.weight"].shape == (16, 64)
    # k_proj dense: (2 kv * 8, 64) = (16, 64). KV-replicated chunk = 16/2 = 8 → per-track (8, 64).
    assert tracks[0]["layers.3.self_attn.k_proj.weight"].shape == (8, 64)
    # Tracks in the same kv-group all have the same shape (and the same values, tested above).
    assert tracks[3]["layers.3.self_attn.k_proj.weight"].shape == (8, 64)
    # o_proj dense: (64, 8*8) = (64, 64). Rowwise N=8: (64, 8).
    assert tracks[0]["layers.3.self_attn.o_proj.weight"].shape == (64, 8)


def test_diverged_kv_replication_groups_do_not_require_divisibility():
    """`sync=False` must give singletons for ANY track count.

    Under merged tracks the replication plan is built over the LOGICAL track count
    (shards / F per rank), which need not be a multiple of num_kv_heads: 27B at
    N=24 with F=2 on 6 ranks gives 6 logical tracks against 4 kv heads. The
    divisibility guard used to run before the sync branch and killed that run at
    startup — but the factor it protects is only ever formed when the copies are
    kept bit-identical. Diverged copies (the DEFAULT) share nothing, so the answer
    is singletons and no factor exists.
    """
    spec = KVReplicatedColwise(num_kv_heads=4, sync=False)
    assert spec.replication_groups(6) == [[t] for t in range(6)]
    assert spec.replication_groups(8) == [[t] for t in range(8)]

    # force_sync still partitions BY kv-group, so there it must still divide.
    assert spec.replication_groups(8, force_sync=True) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        spec.replication_groups(6, force_sync=True)
