"""K=2 per rank, single-process (no NCCL): exercises the new lockstep sync path.

With n_tracks=2 and local_track_ids=(0,1) hosted in a single PTWrappedModel,
the SyncBoundary's local-sum-then-all-reduce degenerates to a pure local
sum (track_group=None skips the NCCL collective). This validates the
K>1 forward path end-to-end without needing a distributed launcher.
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


def test_k2_local_only_forward_is_finite_and_matches_manual_sync():
    cfg = _tiny_config()
    n_tracks = 2
    sync_block_depth = 4

    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=sync_block_depth, text_config_attr="config"
    )
    assert manifest.sync_layer_indices == [3, 7]

    # Single PTWrappedModel hosting both tracks (K=2, world_size=1).
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=(0, 1),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    ).eval()
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    with torch.no_grad():
        pt_logits, sync_hiddens = pt(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=True
        )

    assert pt_logits is not None  # rank hosts track 0 (the owner)
    assert pt_logits.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(pt_logits).all()
    assert set(sync_hiddens.keys()) == {3, 7}
    for h in sync_hiddens.values():
        assert h.shape == (1, 16, cfg.hidden_size)
        assert torch.isfinite(h).all()


def test_k2_intra_window_taps_observe_without_perturbing():
    """Mid-window taps add loss-only reconstructions at every non-boundary layer
    and must leave the carried state (boundary hiddens, logits) bit-identical."""
    cfg = _tiny_config()
    n_tracks = 2

    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=(0, 1),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    ).eval()
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    with torch.no_grad():
        base_logits, base_hiddens = pt(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=True
        )
        tap_logits, tap_hiddens = pt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sync_hiddens=True,
            return_intra_window_hiddens=True,
        )

    # Every layer is reported: boundaries carry state, the rest are loss-only taps.
    assert set(tap_hiddens.keys()) == set(range(cfg.num_hidden_layers))
    for h in tap_hiddens.values():
        assert h.shape == (1, 16, cfg.hidden_size)
        assert torch.isfinite(h).all()
    # Observation must not perturb the forward.
    assert torch.equal(base_logits, tap_logits)
    for idx in manifest.sync_layer_indices:
        assert torch.equal(base_hiddens[idx], tap_hiddens[idx])


def test_k2_intra_window_taps_reject_window_checkpointing():
    """The 'window' checkpoint granule hides per-layer state — taps must refuse."""
    cfg = _tiny_config()
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=4,
        local_track_ids=(0, 1),
        sync_after_layers=[3, 7],
        track_group=None,
        activation_checkpoint=True,
        checkpoint_granularity="window",
    ).train()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    try:
        pt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_intra_window_hiddens=True,
        )
        assert False, "expected RuntimeError for taps under window checkpointing"
    except RuntimeError as e:
        assert "per-layer" in str(e)


def test_k2_peer_rank_returns_no_logits():
    """A rank that does NOT own track 0 should have lm_head=None and emit logits=None."""
    cfg = _tiny_config()
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=4,
        local_track_ids=(2, 3),  # peer rank, no owner
        sync_after_layers=[3, 7],
        track_group=None,
    ).eval()
    assert pt.lm_head is None
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    with torch.no_grad():
        logits, _ = pt(input_ids=input_ids, attention_mask=attention_mask)
    assert logits is None
