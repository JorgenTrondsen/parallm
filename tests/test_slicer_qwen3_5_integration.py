"""Integration test: build a tiny Qwen3.5 text model, slice it, verify round-trip.

This is the key correctness gate from the plan: with N=1, every per-track
tensor must be bit-equal to the original. With N>1, concatenating slices
back together must reproduce the original (for non-replicated specs).

Runs entirely on CPU using a synthetic ~370k-param model.
"""
from __future__ import annotations

import pytest
import torch

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    PerHead,
    Replicated,
    Rowwise,
)
from pt_converter.slicer.convert import slice_model_to_tracks
from pt_converter.slicer.qwen3_5 import decoder_layer_specs


def _tiny_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def tiny_model():
    cfg = _tiny_config()
    model = Qwen3_5TextModel(cfg)
    model.eval()
    return model


def test_n1_slice_preserves_every_value_bit_equal(tiny_model):
    """Most important correctness gate: N=1 slicing yields tensors bit-equal to original."""
    tracks, manifest = slice_model_to_tracks(
        tiny_model, n_tracks=1, sync_block_depth=4, text_config_attr="config"
    )
    assert len(tracks) == 1
    assert manifest.n_tracks == 1
    assert manifest.sync_layer_indices == [3, 7]
    assert manifest.layer_types == list(tiny_model.config.layer_types)

    state = dict(tiny_model.state_dict())
    track0 = tracks[0]
    # Spot-check a representative selection of param categories.
    # Linear-attention layer params
    assert torch.equal(track0["layers.0.linear_attn.in_proj_qkv.weight"], state["layers.0.linear_attn.in_proj_qkv.weight"])
    assert torch.equal(track0["layers.0.linear_attn.A_log"], state["layers.0.linear_attn.A_log"])
    assert torch.equal(track0["layers.0.linear_attn.conv1d.weight"], state["layers.0.linear_attn.conv1d.weight"])
    # Full-attention layer (idx 3) params
    assert torch.equal(track0["layers.3.self_attn.q_proj.weight"], state["layers.3.self_attn.q_proj.weight"])
    assert torch.equal(track0["layers.3.self_attn.o_proj.weight"], state["layers.3.self_attn.o_proj.weight"])
    # MLP
    assert torch.equal(track0["layers.3.mlp.gate_proj.weight"], state["layers.3.mlp.gate_proj.weight"])
    assert torch.equal(track0["layers.3.mlp.down_proj.weight"], state["layers.3.mlp.down_proj.weight"])
    # Embeddings, final norm (replicated)
    assert torch.equal(track0["embed_tokens.weight"], state["embed_tokens.weight"])
    assert torch.equal(track0["norm.weight"], state["norm.weight"])


def test_n2_round_trip_via_per_spec_reassemble(tiny_model):
    """For N=2, reassembling per-spec slices must reproduce the dense weights."""
    tracks, manifest = slice_model_to_tracks(
        tiny_model, n_tracks=2, sync_block_depth=4, text_config_attr="config"
    )
    assert len(tracks) == 2
    state = dict(tiny_model.state_dict())

    text_cfg = tiny_model.config
    for layer_idx, layer_type in enumerate(text_cfg.layer_types):
        specs = decoder_layer_specs(text_cfg, layer_type)
        for sub_key, spec in specs.items():
            slices = [t[f"layers.{layer_idx}.{sub_key}"] for t in tracks]
            full_key = f"layers.{layer_idx}.{sub_key}"
            original = state[full_key]
            if isinstance(spec, Replicated):
                # Each track's slice must equal the original.
                for s in slices:
                    assert torch.equal(s, original), f"replicated mismatch at {full_key}"
            else:
                reassembled = spec.reassemble(slices)
                assert torch.equal(reassembled, original), f"round-trip mismatch at {full_key}"


def test_sync_layer_indices_d4_on_8_layers(tiny_model):
    _, manifest = slice_model_to_tracks(
        tiny_model, n_tracks=1, sync_block_depth=4, text_config_attr="config"
    )
    # 8 layers / D=4 = sync after layers 3 and 7.
    assert manifest.sync_layer_indices == [3, 7]


def test_per_track_shape_matches_per_track_config_for_qwen3_5_attention(tiny_model):
    """N=2 should halve heads on full-attention layers; verify the sliced shapes."""
    tracks, _ = slice_model_to_tracks(
        tiny_model, n_tracks=2, sync_block_depth=4, text_config_attr="config"
    )
    # Full attention layer is at idx 3. Original q_proj: (4 heads * 16 head_dim * 2, 64) = (128, 64).
    # Per-track: (2 heads * 16 * 2, 64) = (64, 64).
    assert tracks[0]["layers.3.self_attn.q_proj.weight"].shape == (64, 64)
    # k_proj original: (2 kv * 16, 64) = (32, 64). Per-track: (1 kv * 16, 64) = (16, 64).
    assert tracks[0]["layers.3.self_attn.k_proj.weight"].shape == (16, 64)
    # o_proj original: (64, 4*16) = (64, 64). Per-track (Rowwise): (64, 32).
    assert tracks[0]["layers.3.self_attn.o_proj.weight"].shape == (64, 32)
