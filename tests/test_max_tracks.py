"""Verify max-tracks gcd detection."""
from __future__ import annotations

from types import SimpleNamespace

from pt_converter.utils.max_tracks import max_tracks_for_config


def _fake_qwen3_5_cfg(**overrides):
    base = dict(
        model_type="qwen3_5_text",
        num_attention_heads=16,
        num_key_value_heads=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_qwen3_5_9b_max_tracks_is_4():
    # The binding dimension is num_key_value_heads=4.
    cfg = _fake_qwen3_5_cfg()
    assert max_tracks_for_config(cfg) == 4


def test_kv_heads_bind_below_other_dims():
    cfg = _fake_qwen3_5_cfg(num_key_value_heads=2)
    assert max_tracks_for_config(cfg) == 2


def test_top_level_config_with_text_subconfig():
    text = _fake_qwen3_5_cfg()
    top = SimpleNamespace(model_type="qwen3_5", text_config=text)
    assert max_tracks_for_config(top) == 4


def test_unregistered_model_type_raises():
    cfg = SimpleNamespace(model_type="not_a_real_model")
    try:
        max_tracks_for_config(cfg)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for unregistered model_type")
