"""N=1 PT forward must match dense Qwen3_5 forward (the core correctness gate).

When n_tracks=1, the SyncBoundary becomes a no-op, so the PT model is
mathematically identical to the dense model with the same weights. Any drift
indicates a bug in:
- slicer reassembly (key remapping in load_track_state_dict),
- per-track decoder-layer construction (build_per_track_text_config),
- the order in which we apply ops in PTTrackTextModel.forward.

Run on CPU with a tiny config; tolerance is ~1e-5 in float32.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks


def _tiny_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
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


def test_n1_matches_dense_forward():
    cfg = _tiny_config()
    torch.manual_seed(42)

    # Dense reference: a TextModel + a fresh lm_head.
    dense = Qwen3_5TextModel(cfg)
    dense.eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    dense.eval()

    # Slice with N=1, sync_block_depth=4 (sync indices [3, 7]).
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=1, sync_block_depth=4, text_config_attr="config"
    )

    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=1,
        track_id=0,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    pt.eval()
    pt.load_track_state_dict(tracks[0], strict=False)

    # Same input.
    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    with torch.no_grad():
        dense_out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        dense_logits = dense.lm_head(dense_out.last_hidden_state)
        pt_logits, _ = pt(input_ids=input_ids, attention_mask=attention_mask)

    max_abs_diff = (dense_logits - pt_logits).abs().max().item()
    assert max_abs_diff < 1e-4, f"N=1 forward drift {max_abs_diff} exceeds tolerance"


def test_n1_returns_sync_hiddens_at_correct_depths():
    cfg = _tiny_config()
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=1, sync_block_depth=4, text_config_attr="config"
    )

    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=1,
        track_id=0,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    pt.eval()
    pt.load_track_state_dict(tracks[0], strict=False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        _, sync_hiddens = pt(input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=True)

    # Two sync points (after layer 3 and layer 7) -> two captures.
    assert len(sync_hiddens) == 2
    for h in sync_hiddens:
        assert h.shape == (1, 16, cfg.hidden_size)
