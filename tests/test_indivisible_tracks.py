"""N does not have to divide every dim: GDN key heads replicate, MLP width zero-pads.

Both mechanisms exist so Qwen3.5-27B can reach its N=24 ceiling (it has 24 q-heads)
despite 16 GDN key heads and intermediate_size=17408. They are only worth having if
they are EXACT, so the rail here is a whole forward at the `exact` sync schedule —
2 syncs/layer, equivalent to dense by construction — at an N that divides neither dim.

Scaled-down mirror of the 27B: 6 q-heads / 2 kv-heads, 4 GDN k-heads and 12 GDN
v-heads (ratio 3, as at 16/48), intermediate_size 17, all at N=6. So 4 % 6 != 0 and
17 % 6 != 0 — both mechanisms fire — while 6 % 6 == 0, 6 % 2 == 0 and 12 % 6 == 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.base import FusedSegmentColwise, GDNFusedQKV
from parallm.slicer.convert import slice_model_to_tracks
from parallm.utils.max_tracks import valid_track_counts

N_TRACKS = 6


def _indivisible_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=17,  # 17 % 6 != 0 -> zero-padded to 18 (3 per track)
        num_hidden_layers=8,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,  # 4 % 6 != 0 -> one k-head copy per v-head
        linear_num_value_heads=12,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
        vocab_size=64,
        rms_norm_eps=1e-6,
    )


def test_exact_schedule_matches_dense_when_n_divides_neither_dim():
    cfg = _indivisible_config()
    assert N_TRACKS in valid_track_counts(cfg), "N=6 must be an accepted track count"

    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config"
    )
    per_track = manifest.per_track_param_shapes
    # 2 v-heads/track, each with its own k-head copy: 2*8 q + 2*8 k + 2*8 v.
    assert per_track["layers.0.linear_attn.in_proj_qkv.weight"] == (48, 64)
    assert per_track["layers.0.linear_attn.conv1d.weight"] == (48, 1, 2)
    assert per_track["layers.0.mlp.gate_proj.weight"] == (3, 64)  # ceil(17/6)
    assert per_track["layers.0.mlp.down_proj.weight"] == (64, 3)

    # All tracks in one process: SyncBoundary degenerates to a local sum.
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=N_TRACKS,
        local_track_ids=tuple(range(N_TRACKS)),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    pt.set_sync_phase("exact")
    pt.eval()
    pt.load_track_state_dicts({t: tracks[t] for t in range(N_TRACKS)}, strict=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        dense_logits = dense.lm_head(dense_out.last_hidden_state)
        pt_logits, _ = pt(input_ids=input_ids, attention_mask=attention_mask)

    max_abs_diff = (dense_logits - pt_logits).abs().max().item()
    assert max_abs_diff < 1e-4, f"exact-schedule drift {max_abs_diff} exceeds tolerance"


def test_gdn_compact_mode_is_a_plain_segment_split():
    """When the k-heads DO divide N (every N=8 dense and every MoE convert today),
    GDNFusedQKV must reproduce the old FusedSegmentColwise output byte-for-byte."""
    n_tracks = 4
    spec = GDNFusedQKV(num_k_heads=8, num_v_heads=16, head_k_dim=4, head_v_dim=4)
    old = FusedSegmentColwise(segments=(32, 32, 64))
    weight = torch.randn(128, 5)
    for t in range(n_tracks):
        assert torch.equal(spec.slice(weight, t, n_tracks), old.slice(weight, t, n_tracks))
    assert spec.per_track_shape((128, 5), n_tracks) == old.per_track_shape((128, 5), n_tracks)
