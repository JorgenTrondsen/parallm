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
