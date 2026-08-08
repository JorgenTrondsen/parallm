"""gpt-oss slicing rails: round-trip reassembly, N=1 dense parity, and N=4 parity
at the exact sync schedule.

The N=4 test is the one that matters. It exercises, simultaneously:
  - `SummedBias` on `o_proj.bias` and `experts.down_proj_bias` (a `Replicated` bias
    there would be added N times by the sync's partial-sum semantics),
  - the per-head split of `self_attn.sinks`,
  - the interleaved gate/up expert slabs surviving a plain `Colwise(dim=2)`,
  - the replicated router picking the same top-k on every track,
  - the `{full_attention, sliding_attention}` mask map.

At `sync_phase="exact"` every sublayer syncs, so the PT walk IS the dense forward up
to floating-point summation order — any of the above being wrong shows up as a large
delta, not a small one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel

from parallm.adapters import get_adapter_for_config
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks
from parallm.slicer.gpt_oss import decoder_layer_specs, top_level_specs


def _tiny_config(num_kv: int = 2, inter: int = 32) -> GptOssConfig:
    cfg = GptOssConfig(
        hidden_size=64,
        intermediate_size=inter,  # per-expert width; divides 1/2/4
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=num_kv,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
        sliding_window=3,  # short enough that sliding != full on a 12-token input
        layer_types=["sliding_attention", "full_attention"] * 2,
        rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _dense(cfg: GptOssConfig, seed: int = 42):
    torch.manual_seed(seed)
    dense = GptOssModel(cfg).eval()
    # Non-trivial biases/sinks: the HF init leaves several of them at zero, which
    # would make the SummedBias rail vacuous.
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name.endswith(("bias", "sinks")):
                p.normal_(mean=0.0, std=0.05)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    return dense.eval()


def _build_pt(cfg, tracks, manifest, n_tracks: int, phase: str) -> PTWrappedModel:
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    pt.eval()
    pt.load_track_state_dicts({t: tracks[t] for t in range(n_tracks)}, strict=True)
    pt.set_sync_phase(phase)
    return pt


def _dense_logits(dense, cfg, input_ids):
    with torch.no_grad():
        out = dense(input_ids=input_ids, use_cache=False)
        return dense.lm_head(out.last_hidden_state)


# --------------------------------------------------------------------------- #
# Adapter wiring
# --------------------------------------------------------------------------- #


def test_adapter_registered_and_layer_types():
    cfg = _tiny_config()
    adapter = get_adapter_for_config(cfg)
    assert adapter.model_type == "gpt_oss"
    assert set(adapter.valid_layer_types) == {"full_attention", "sliding_attention"}
    # No `get_layer_types` override needed — the default reads the config.
    assert adapter.layer_types_for(cfg) == list(cfg.layer_types)
    # Merging (concatenated slabs) yes; the batched fold (exec_groups) no — the MoE
    # router picks a different top-k per stream, so there is no single grouped_mm.
    assert adapter.supports_merged_tracks is True
    assert adapter.supports_batched_exec is False


def test_build_masks_gives_one_mask_per_layer_type():
    cfg = _tiny_config()
    adapter = get_adapter_for_config(cfg)
    embeds = torch.zeros(1, 12, cfg.hidden_size)
    masks = adapter.build_masks(cfg, embeds, None, torch.arange(12).unsqueeze(0))
    assert set(masks) == {"full_attention", "sliding_attention"}
    # The sliding mask must actually be narrower than the full causal one.
    full, sliding = masks["full_attention"], masks["sliding_attention"]
    assert full is not None and sliding is not None
    assert (sliding == torch.finfo(sliding.dtype).min).sum() > (
        full == torch.finfo(full.dtype).min
    ).sum()


def test_valid_track_counts_for_the_shipped_shape():
    """The real 20b/120b shape: 64 q heads, 8 kv heads, 2880-wide experts."""
    from parallm.utils.max_tracks import valid_track_counts

    cfg = GptOssConfig(
        hidden_size=2880, intermediate_size=2880, num_hidden_layers=24,
        num_attention_heads=64, num_key_value_heads=8, head_dim=64,
        num_local_experts=32, num_experts_per_tok=4, vocab_size=201088,
    )
    assert valid_track_counts(cfg) == [64, 32, 16, 8]


# --------------------------------------------------------------------------- #
# Slicer
# --------------------------------------------------------------------------- #


def test_slice_reassemble_round_trip():
    cfg = _tiny_config()
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(dense, n_tracks=4, text_config_attr="config")

    assert manifest.model_type == "gpt_oss"
    assert manifest.layer_types == list(cfg.layer_types)
    assert len(tracks) == 4

    state = dense.state_dict()
    for li, layer_type in enumerate(cfg.layer_types):
        for sub_key, spec in decoder_layer_specs(cfg, layer_type).items():
            key = f"layers.{li}.{sub_key}"
            slices = [tracks[t][key] for t in range(4)]
            if type(spec).__name__ == "KVReplicatedColwise":
                # reassemble wants one unique slice per kv-group.
                tpg = 4 // spec.num_kv_heads
                slices = [slices[g * tpg] for g in range(spec.num_kv_heads)]
            got = spec.reassemble(slices)
            want = state[key]
            assert torch.allclose(got, want, atol=1e-6), key

    for canonical, spec in top_level_specs(cfg).items():
        got = spec.reassemble([tracks[t].get(canonical) for t in range(4)])
        want = state[canonical]
        assert torch.allclose(got, want, atol=1e-6), canonical


def test_summed_bias_lands_on_owner_only():
    """The row-parallel biases must be owner-only zeros — the whole point of
    `SummedBias`. A replicated copy would be added N times at the sync."""
    cfg = _tiny_config()
    tracks, _ = slice_model_to_tracks(_dense(cfg), n_tracks=4, text_config_attr="config")
    for key in ("layers.0.self_attn.o_proj.bias", "layers.0.mlp.experts.down_proj_bias"):
        assert tracks[0][key].abs().sum() > 0, key
        for t in (1, 2, 3):
            assert torch.count_nonzero(tracks[t][key]) == 0, f"{key} on track {t}"


def test_per_track_shapes_at_n4():
    cfg = _tiny_config()
    tracks, _ = slice_model_to_tracks(_dense(cfg), n_tracks=4, text_config_attr="config")
    t0 = tracks[0]
    assert t0["layers.0.self_attn.q_proj.weight"].shape == (16, 64)  # 1 head * 16
    assert t0["layers.0.self_attn.q_proj.bias"].shape == (16,)
    assert t0["layers.0.self_attn.k_proj.weight"].shape == (16, 64)  # 1 kv head
    assert t0["layers.0.self_attn.o_proj.weight"].shape == (64, 16)
    assert t0["layers.0.self_attn.sinks"].shape == (1,)
    assert t0["layers.0.mlp.router.weight"].shape == (4, 64)  # replicated
    assert t0["layers.0.mlp.experts.gate_up_proj"].shape == (4, 64, 16)  # 2*(32/4), already 8-aligned
    assert t0["layers.0.mlp.experts.gate_up_proj_bias"].shape == (4, 16)
    assert t0["layers.0.mlp.experts.down_proj"].shape == (4, 8, 64)  # 32/4
    assert t0["layers.0.mlp.experts.down_proj_bias"].shape == (4, 64)


def test_gate_up_slice_keeps_interleaved_pairs():
    """`gate_up_proj` is `[g0,u0,g1,u1,...]` along dim 2, so a track's slab must be
    the contiguous run of whole pairs — not a gate/up phase shift."""
    cfg = _tiny_config()
    dense = _dense(cfg)
    tracks, _ = slice_model_to_tracks(dense, n_tracks=4, text_config_attr="config")
    full = dense.state_dict()["layers.0.mlp.experts.gate_up_proj"]
    per = full.shape[2] // 4
    for t in range(4):
        got = tracks[t]["layers.0.mlp.experts.gate_up_proj"]
        assert torch.equal(got, full[:, :, t * per : (t + 1) * per])
        # gate lanes stay on even indices in the slice
        assert torch.equal(got[..., ::2], full[:, :, t * per : (t + 1) * per : 2])


# --------------------------------------------------------------------------- #
# Forward parity
# --------------------------------------------------------------------------- #


def test_n1_matches_dense_forward():
    """N=1: SyncBoundary is a no-op, so the PT model IS the dense model."""
    cfg = _tiny_config(num_kv=1)  # N=1 must be a multiple of num_kv
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=1, sync_block_depth=2, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 1, "post-mlp")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    assert torch.allclose(got, _dense_logits(dense, cfg, ids), atol=2e-5, rtol=1e-4)


def test_n4_exact_schedule_matches_dense_forward():
    """N=4 at the exact schedule (sync after every sublayer) must reproduce the dense
    forward: each track holds a disjoint head set / expert-width slab, and every
    partial is summed before the next sublayer reads it."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 4, "exact")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    want = _dense_logits(dense, cfg, ids)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, f"relL2 {rel:.3e}"


def test_n4_post_attn_schedule_runs_and_stays_close():
    """The training schedule (lever B, d1b): every layer a boundary. Not bit-exact —
    each layer's MLP reads the synced residual but carries its own delta — so this
    only guards that the walk runs and does not blow up."""
    cfg = _tiny_config(num_kv=2)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 4, "post-attn")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    # Post-attn captures are opt-in per layer (the trainer's target contract):
    # boundaries tap post-attention, the final layer taps post-MLP.
    last = cfg.num_hidden_layers - 1
    boundaries = set(manifest.sync_layer_indices) - {last}
    with torch.no_grad():
        got, sync_h = pt(
            input_ids=ids,
            return_sync_hiddens=True,
            capture_post_attn=boundaries,
            capture_post_mlp={last},
        )
    assert torch.isfinite(got).all()
    assert set(sync_h) == boundaries | {last}


# --------------------------------------------------------------------------- #
# Expert-width padding (the `torch._grouped_mm` 16-byte stride requirement)
# --------------------------------------------------------------------------- #


def test_per_track_expert_width_is_padded_to_a_multiple_of_8():
    """`torch._grouped_mm` rejects a contracted dim whose row stride is not a
    multiple of 16 bytes, so the bf16 per-track expert width must be a multiple of 8.
    The slicer pads, and the per-track config MUST agree or the shards will not
    load."""
    from parallm.slicer.gpt_oss import moe_mlp_specs

    cfg = _tiny_config(inter=24)  # 24/4 = 6 -> padded to 8
    adapter = get_adapter_for_config(cfg)
    specs = moe_mlp_specs(cfg)
    E, H, I = cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size

    for n in (2, 4):  # N must be a multiple of num_kv_heads (=2 here)
        w = adapter.build_per_track_text_config(cfg, n).intermediate_size
        assert w % 8 == 0, (n, w)
        assert specs["experts.down_proj"].per_track_shape((E, I, H), n)[1] == w
        # gate_up carries interleaved (gate, up) pairs, so exactly twice the width.
        assert specs["experts.gate_up_proj"].per_track_shape((E, H, 2 * I), n)[2] == 2 * w
        assert specs["experts.gate_up_proj_bias"].per_track_shape((E, 2 * I), n)[1] == 2 * w


def test_padded_expert_lanes_are_zero_in_weight_and_bias():
    """The padding is only exact because the padded lanes have gate = up = 0 in BOTH
    the weight and the bias: `(0 + 1) * glu(0) = 0`. A nonzero bias on a padded lane
    would leak a real contribution."""
    cfg = _tiny_config(inter=24)
    tracks, _ = slice_model_to_tracks(_dense(cfg), n_tracks=4, text_config_attr="config")
    # 24 lanes over 4 tracks of 8 -> track 3 is entirely padding.
    assert torch.count_nonzero(tracks[3]["layers.0.mlp.experts.gate_up_proj"]) == 0
    assert torch.count_nonzero(tracks[3]["layers.0.mlp.experts.gate_up_proj_bias"]) == 0
    assert torch.count_nonzero(tracks[3]["layers.0.mlp.experts.down_proj"]) == 0


def test_n4_exact_schedule_matches_dense_with_padded_expert_width():
    """The padding rail on the real forward: 24/4 = 6 pads to 8, so every track
    carries 2 dead lanes (and track 3 is all padding). Still bit-for-bit the dense
    model up to fp summation order."""
    cfg = _tiny_config(num_kv=2, inter=24)
    dense = _dense(cfg)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    pt = _build_pt(cfg, tracks, manifest, 4, "exact")
    pt.lm_head.weight = dense.lm_head.weight

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got, _ = pt(input_ids=ids)
    want = _dense_logits(dense, cfg, ids)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-5, f"relL2 {rel:.3e}"


def test_merged_config_is_f_aligned_slabs():
    """A merged track is F ALIGNED per-track slabs, not one aligned F-wide slab.

    That distinction is what makes the merged module loadable from the shards on
    disk: each shard was padded to `EXPERT_WIDTH_ALIGN` independently, so the merged
    width must be F times the padded width, never `align(F * raw)`.
    """
    from parallm.model.tracks.gpt_oss import build_per_track_text_config

    cfg = _tiny_config()
    one = build_per_track_text_config(cfg, 4, fuse_size=1)
    two = build_per_track_text_config(cfg, 4, fuse_size=2)
    assert two.intermediate_size == one.intermediate_size * 2
    assert two.num_attention_heads == one.num_attention_heads * 2
    assert two.num_key_value_heads == one.num_key_value_heads * 2
