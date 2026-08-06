"""Unit tests for the unified streaming loader (`parallm.slicer.loader`)."""
from __future__ import annotations

import torch
from safetensors.torch import save_file

from parallm.slicer.loader import slicer_key, stream_text_state_dict


def test_slicer_key_remap():
    cases = {
        # VLM and text-only layer layouts both collapse to layers.{i}.*
        "model.language_model.layers.3.self_attn.q_proj.weight": "layers.3.self_attn.q_proj.weight",
        "model.layers.3.mlp.gate_proj.weight": "layers.3.mlp.gate_proj.weight",
        "model.language_model.layers.7.mlp.experts.gate_up_proj": "layers.7.mlp.experts.gate_up_proj",
        # top-level, both namings
        "model.language_model.embed_tokens.weight": "embed_tokens.weight",
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        # dropped: vision, MTP, scale siblings
        "model.visual.blocks.0.mlp.linear_fc1.weight": None,
        "mtp.layers.0.mlp.down_proj.weight": None,
        "mtp.fc.weight": None,
        "model.language_model.layers.2.self_attn.k_proj.weight_scale": None,
    }
    for k, expected in cases.items():
        assert slicer_key(k) == expected, (k, slicer_key(k))


def test_stream_single_file_remaps_and_drops(tmp_path):
    """Single-file (no index) checkpoint: bf16 passthrough + remap + drop vision/MTP/lm_head."""
    src = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.language_model.embed_tokens.weight": torch.randn(6, 4, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(4, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(6, 4, dtype=torch.bfloat16),
        "model.visual.blocks.0.mlp.linear_fc1.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    }
    save_file(src, str(tmp_path / "model.safetensors"))

    sd = stream_text_state_dict(str(tmp_path))
    assert set(sd) == {
        "layers.0.mlp.gate_proj.weight", "embed_tokens.weight", "norm.weight", "lm_head.weight",
    }
    assert torch.equal(sd["layers.0.mlp.gate_proj.weight"], src["model.language_model.layers.0.mlp.gate_proj.weight"])
    assert all(v.dtype == torch.bfloat16 for v in sd.values())

    # drop_lm_head excludes the head (used by the replica calibration loader)
    sd2 = stream_text_state_dict(str(tmp_path), drop_lm_head=True)
    assert "lm_head.weight" not in sd2


def test_fuse_unfused_moe_experts():
    """Per-expert 2-D Linears fuse into the slicer's 3-D slabs (NVFP4-checkpoint layout)."""
    from parallm.slicer.loader import _fuse_moe_experts

    E, I, H = 3, 4, 6
    gate = [torch.randn(I, H) for _ in range(E)]
    up = [torch.randn(I, H) for _ in range(E)]
    down = [torch.randn(H, I) for _ in range(E)]
    sd = {}
    for e in range(E):
        sd[f"layers.0.mlp.experts.{e}.gate_proj.weight"] = gate[e]
        sd[f"layers.0.mlp.experts.{e}.up_proj.weight"] = up[e]
        sd[f"layers.0.mlp.experts.{e}.down_proj.weight"] = down[e]
    sd["layers.0.self_attn.q_proj.weight"] = torch.randn(8, H)  # untouched

    out = _fuse_moe_experts(sd)
    assert out["layers.0.mlp.experts.gate_up_proj"].shape == (E, 2 * I, H)
    assert out["layers.0.mlp.experts.down_proj"].shape == (E, H, I)
    # per expert the fused rows are [gate | up]
    for e in range(E):
        assert torch.equal(out["layers.0.mlp.experts.gate_up_proj"][e], torch.cat([gate[e], up[e]], dim=0))
        assert torch.equal(out["layers.0.mlp.experts.down_proj"][e], down[e])
    assert not any(".experts.0." in k for k in out)  # per-expert keys consumed
    assert "layers.0.self_attn.q_proj.weight" in out  # non-expert keys preserved

    # already-fused input is a no-op
    fused = {"layers.0.mlp.experts.gate_up_proj": torch.randn(E, 2 * I, H)}
    assert _fuse_moe_experts(dict(fused)).keys() == fused.keys()


def test_tied_embeddings_get_an_untied_lm_head(tmp_path):
    """`tie_word_embeddings` checkpoints ship NO `lm_head.weight`.

    Every small Qwen3.5 is tied (the 0.8B is), and the PT model has a real
    untied head per `OwnerOnly(owner_track=0)`, so without this the convert
    loads with `missing=['lm_head.weight']` and nothing downstream runs. Untie
    at stream time: the head is frozen everywhere, so a copy cannot drift.
    """
    src = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.language_model.embed_tokens.weight": torch.randn(6, 4, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(4, dtype=torch.bfloat16),
    }
    save_file(src, str(tmp_path / "model.safetensors"))

    sd = stream_text_state_dict(str(tmp_path))
    assert "lm_head.weight" in sd, "tied checkpoint produced no lm_head"
    assert torch.equal(sd["lm_head.weight"], sd["embed_tokens.weight"])
    # A copy, not an alias — the head is frozen, but aliasing would make any
    # future embedding update silently rewrite the head too.
    assert sd["lm_head.weight"].data_ptr() != sd["embed_tokens.weight"].data_ptr()

    # An UNTIED checkpoint must be left exactly as it is.
    src["lm_head.weight"] = torch.randn(6, 4, dtype=torch.bfloat16)
    save_file(src, str(tmp_path / "model.safetensors"))
    sd2 = stream_text_state_dict(str(tmp_path))
    assert torch.equal(sd2["lm_head.weight"], src["lm_head.weight"])
    assert not torch.equal(sd2["lm_head.weight"], sd2["embed_tokens.weight"])

    # ...and drop_lm_head must still drop it (replica calibration relies on that).
    assert "lm_head.weight" not in stream_text_state_dict(str(tmp_path), drop_lm_head=True)
