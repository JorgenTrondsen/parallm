"""Verify max-tracks detection under the new KV-replicated rule.

Rule (recap):
  max_tracks is the largest N satisfying:
    1. num_attention_heads % N == 0
    2. N % num_key_value_heads == 0
    3. linear_num_key_heads % N == 0 and linear_num_value_heads % N == 0
    4. intermediate_size % N == 0
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from parallm.utils.max_tracks import max_tracks_for_config, valid_track_counts


def _fake_qwen3_5_cfg(**overrides):
    base = dict(
        model_type="qwen3_5_text",
        num_attention_heads=16,
        num_key_value_heads=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        intermediate_size=12288,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_qwen3_5_9b_max_tracks_is_16():
    """The new rule lifts the kv-head bottleneck: max = num_attention_heads = 16."""
    cfg = _fake_qwen3_5_cfg()
    assert max_tracks_for_config(cfg) == 16


def test_valid_track_counts_for_qwen3_5_9b():
    """Valid Ns are intersection of: divides 16, multiple of 4, divides {16, 32, 12288}."""
    cfg = _fake_qwen3_5_cfg()
    # Multiples of 4 that divide 16 and 32 and 12288: {4, 8, 16}.
    assert valid_track_counts(cfg) == [16, 8, 4]


def test_no_gqa_model_caps_at_num_q():
    """When num_q == num_kv (no GQA), only N == num_q is valid."""
    cfg = _fake_qwen3_5_cfg(num_attention_heads=4, num_key_value_heads=4)
    # divides 4 ∩ multiple of 4 = {4}
    assert max_tracks_for_config(cfg) == 4


def test_num_q_12_num_kv_4_yields_12():
    """divides 12 ∩ multiple of 4 = {4, 12}; max = 12."""
    cfg = _fake_qwen3_5_cfg(
        num_attention_heads=12,
        num_key_value_heads=4,
        linear_num_key_heads=12,
        linear_num_value_heads=24,
        intermediate_size=12 * 256,  # divisible by 4 and 12
    )
    assert max_tracks_for_config(cfg) == 12


def test_kv_1_means_every_q_divisor_is_valid():
    """When num_kv == 1, the multiple-of-num_kv rule is vacuous."""
    cfg = _fake_qwen3_5_cfg(num_key_value_heads=1)
    # divisors of 16 ∩ divisors of 16/32/12288 = {1, 2, 4, 8, 16}
    assert max_tracks_for_config(cfg) == 16


def test_top_level_config_with_text_subconfig():
    text = _fake_qwen3_5_cfg()
    top = SimpleNamespace(model_type="qwen3_5", text_config=text)
    assert max_tracks_for_config(top) == 16


def test_unregistered_model_type_raises():
    cfg = SimpleNamespace(model_type="not_a_real_model")
    with pytest.raises(NotImplementedError):
        max_tracks_for_config(cfg)


def test_unsatisfiable_constraints_raise():
    """num_q=4 cannot be a multiple of num_kv=8, so no N is valid."""
    cfg = _fake_qwen3_5_cfg(
        num_attention_heads=4,
        num_key_value_heads=8,
    )
    with pytest.raises(ValueError):
        max_tracks_for_config(cfg)
