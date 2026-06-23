"""Unit tests for the cross-track intervention harness (``eval/intervention.py``).

Two layers of coverage:

1. **Channel math** — pure-tensor, single-process, CPU. Exercises each channel's
   reconstruction directly on small tensors.
2. **Anchor equivalence** — the calibration guarantee. On a tiny K=2 single-process
   model (``track_group=None`` ⇒ no NCCL), the ``zero`` channel forward must match the
   deployed D-window forward and the ``oracle`` channel forward must match the D=1
   (sync-every-layer) forward. If either anchor drifts, the harness is wrong — not the
   channel under test.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.eval.intervention import (
    CalibratedFixedLowRankChannel,
    CausalAvgChannel,
    FixedLowRankChannel,
    LowRankChannel,
    MaskedOracleChannel,
    OracleChannel,
    ZeroChannel,
    intervention_forward,
    parse_channel,
)
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks


# --------------------------------------------------------------------------- #
# Channel math
# --------------------------------------------------------------------------- #
def test_zero_channel_returns_zeros():
    others = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]
    out = ZeroChannel()(others)
    assert all(torch.equal(s, torch.zeros_like(o)) for s, o in zip(out, others))


def test_oracle_channel_returns_inputs():
    others = [torch.randn(1, 3, 4), torch.randn(2, 5, 8)]
    out = OracleChannel()(others)
    assert all(torch.equal(s, o) for s, o in zip(out, others))


def test_masked_oracle_applies_only_at_listed_layers():
    others = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]
    ch = MaskedOracleChannel({2}, invert=False)
    out2 = ch(others, layer_idx=2)  # listed ⇒ oracle (returns inputs)
    assert all(torch.equal(s, o) for s, o in zip(out2, others))
    out1 = ch(others, layer_idx=1)  # not listed ⇒ zeros
    assert all(torch.equal(s, torch.zeros_like(o)) for s, o in zip(out1, others))


def test_masked_oracle_invert_is_leave_one_out():
    o = torch.randn(1, 3, 4)
    ch = MaskedOracleChannel({2}, invert=True)
    assert torch.equal(ch([o], layer_idx=1)[0], o)  # everywhere EXCEPT 2 ⇒ oracle
    assert torch.equal(ch([o], layer_idx=2)[0], torch.zeros_like(o))  # the held-out layer


def test_masked_oracle_empty_set_is_zero():
    o = torch.randn(1, 3, 4)
    ch = MaskedOracleChannel(set())
    assert torch.equal(ch([o], layer_idx=0)[0], torch.zeros_like(o))
    assert ch.name == "oracle@"


def test_stale_channel_is_one_token_shift():
    # window=1 ⇒ S[:, t] = o[:, t-1]; position 0 → 0 (no history).
    o = torch.tensor([[[1.0], [2.0], [3.0]]])  # (B=1, T=3, H=1)
    out = CausalAvgChannel(1)([o])[0]
    assert torch.allclose(out, torch.tensor([[[0.0], [1.0], [2.0]]]))


def test_causal_avg_window2_hand_computed():
    o = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])  # (1, 4, 1)
    out = CausalAvgChannel(2)([o])[0]
    # t0: no history → 0 ; t1: avg(o0)=1 ; t2: avg(o0,o1)=2 ; t3: avg(o1,o2)=4
    assert torch.allclose(out, torch.tensor([[[0.0], [1.0], [2.0], [4.0]]]))


def test_causal_avg_is_strictly_causal():
    # Changing the LAST token must not change any earlier position's output.
    o = torch.randn(1, 6, 3)
    a = CausalAvgChannel(3)([o])[0]
    o2 = o.clone()
    o2[:, -1] += 100.0
    b = CausalAvgChannel(3)([o2])[0]
    assert torch.allclose(a[:, :-1], b[:, :-1])


def test_lowrank_high_rank_recovers():
    o = torch.randn(1, 8, 4)  # rank ≤ 4
    out = LowRankChannel(4)([o])[0]
    assert torch.allclose(out, o, atol=1e-4)


def test_lowrank_rank1_keeps_top_singular_energy():
    # Rank-2 matrix with singular values 3 and 2 ⇒ a rank-1 projection keeps the
    # dominant direction (ideal energy 9 of 13). svd_lowrank is randomized, so the
    # recovered energy is ~9 (seeded for determinism), always below the total 13.
    torch.manual_seed(0)
    o = torch.zeros(1, 3, 4)
    o[0, 0, 0] = 3.0
    o[0, 1, 1] = 2.0
    out = LowRankChannel(1)([o])[0]
    assert out.pow(2).sum() < o.pow(2).sum()  # dropped the smaller direction
    assert math.isclose(out.pow(2).sum().item(), 9.0, rel_tol=2e-2)


def test_fixed_lowrank_freezes_basis_per_layer():
    # The basis is fit once (first batch) per (layer, track) and frozen.
    torch.manual_seed(0)
    ch = FixedLowRankChannel(1)
    # First batch at layer 0: all energy along hidden dim 0 ⇒ fitted basis ≈ e0.
    a = torch.zeros(1, 4, 3)
    a[0, 0, 0] = 5.0
    a[0, 1, 0] = 3.0
    out_a = ch([a], layer_idx=0)[0]
    assert torch.allclose(out_a, a, atol=1e-4)  # rank-1 along e0 captures it
    # Second batch at the SAME layer, energy along dim 1 (orthogonal to frozen e0):
    # the frozen basis projects it to ~0 (proves the basis is reused, not refit).
    b = torch.zeros(1, 4, 3)
    b[0, 0, 1] = 7.0
    assert ch([b], layer_idx=0)[0].abs().max() < 1e-3
    # A DIFFERENT layer fits its own fresh basis from b ⇒ captures b.
    assert torch.allclose(ch([b], layer_idx=1)[0], b, atol=1e-4)


def test_calibrated_fixed_lowrank_pca_over_batches():
    # Observe accumulates per-(layer,track) covariance over MANY batches and returns
    # zeros (D=2 trajectory); finalize fits the top-r PCA basis from all of them.
    ch = CalibratedFixedLowRankChannel(1)
    ch.start_observing()
    b1 = torch.zeros(1, 4, 3)
    b1[0, 0, 0] = 5.0  # energy along hidden dim 0
    assert ch([b1], layer_idx=0)[0].abs().max() == 0.0  # zeros while observing
    b2 = torch.zeros(1, 4, 3)
    b2[0, 1, 0] = 3.0  # more energy along dim 0 (consistent)
    ch([b2], layer_idx=0)
    ch.finalize()
    # Frozen basis ≈ e0: a vector along e0 is preserved, one along e1 projects to ~0.
    v0 = torch.zeros(1, 2, 3)
    v0[0, 0, 0] = 9.0
    assert torch.allclose(ch([v0], layer_idx=0)[0], v0, atol=1e-4)
    v1 = torch.zeros(1, 2, 3)
    v1[0, 0, 1] = 9.0
    assert ch([v1], layer_idx=0)[0].abs().max() < 1e-3
    # A layer never observed has no basis ⇒ returns zeros.
    assert ch([v0], layer_idx=7)[0].abs().max() == 0.0


def test_parse_channel():
    assert isinstance(parse_channel("zero"), ZeroChannel)
    assert isinstance(parse_channel("oracle"), OracleChannel)
    assert parse_channel("stale").window == 1
    assert parse_channel("avg:8").window == 8
    assert parse_channel("lowrank:64").rank == 64
    assert parse_channel("fixed-lowrank:128").rank == 128
    assert parse_channel("fixed-lowrank:128").name == "fixed-lowrank:128"
    assert isinstance(parse_channel("calib-fixed-lowrank:256"), CalibratedFixedLowRankChannel)
    assert parse_channel("calib-fixed-lowrank:256").rank == 256
    assert parse_channel("calib-fixed-lowrank:256").name == "calib-fixed-lowrank:256"
    assert parse_channel("zero").name == "zero"
    assert parse_channel("avg:8").name == "avg:8"
    ch = parse_channel("oracle@1,3")
    assert isinstance(ch, MaskedOracleChannel)
    assert ch.layers == frozenset({1, 3}) and ch.invert is False and ch.name == "oracle@1,3"
    ch2 = parse_channel("oracle~2,5")
    assert ch2.layers == frozenset({2, 5}) and ch2.invert is True and ch2.name == "oracle~2,5"


# --------------------------------------------------------------------------- #
# Anchor equivalence (the calibration guarantee)
# --------------------------------------------------------------------------- #
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


def _build_pt(sync_after_layers):
    """A tiny K=2 single-process PT model with deterministic (seeded) weights."""
    cfg = _tiny_config()
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=4, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=sync_after_layers, track_group=None,
    ).eval()
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    return pt, cfg


def test_zero_channel_matches_deployed_d2_forward():
    pt, cfg = _build_pt([3, 7])  # the deployed D-window schedule
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        ref_h, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        iv_h = intervention_forward(pt, ids, mask, sync_indices=[3, 7], channel=ZeroChannel())
    # Equal up to fp summation order (incremental running sum vs the window sync).
    assert torch.allclose(ref_h, iv_h, atol=1e-4, rtol=1e-4)


def test_oracle_channel_matches_d1_forward():
    # Reference = a real model that syncs after EVERY layer (D=1). Same seed ⇒ same
    # weights as the model we run the intervention on (whose own schedule is
    # irrelevant — intervention_forward takes sync_indices explicitly).
    pt, cfg = _build_pt([3, 7])
    pt_d1, _ = _build_pt(list(range(8)))
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        d1_h, _ = pt_d1(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        iv_h = intervention_forward(pt, ids, mask, sync_indices=[3, 7], channel=OracleChannel())
    assert torch.allclose(d1_h, iv_h, atol=1e-5, rtol=1e-5)


def test_oracle_and_zero_diverge_when_window_has_partial_reads():
    # Sanity: with a real partial-read window the two anchors are NOT equal (there
    # is genuine cross-track headroom for a channel to recover).
    pt, cfg = _build_pt([3, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        zero_h = intervention_forward(pt, ids, mask, [3, 7], ZeroChannel())
        oracle_h = intervention_forward(pt, ids, mask, [3, 7], OracleChannel())
    assert not torch.allclose(zero_h, oracle_h, atol=1e-3)


def test_masked_oracle_full_and_empty_collapse_to_the_anchors():
    # The sweep's two limits must be exactly the anchors: oracle on EVERY mid-window
    # (partial-read) layer == oracle; oracle on NONE == zero. Mid-window layers for
    # sync_indices=[3,7] (num_layers=8) are {0,1,2,4,5,6}; 3 and 7 are boundaries.
    pt, cfg = _build_pt([3, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    mid = {0, 1, 2, 4, 5, 6}
    with torch.no_grad():
        oracle_h = intervention_forward(pt, ids, mask, [3, 7], OracleChannel())
        full_h = intervention_forward(pt, ids, mask, [3, 7], MaskedOracleChannel(mid))
        zero_h = intervention_forward(pt, ids, mask, [3, 7], ZeroChannel())
        empty_h = intervention_forward(pt, ids, mask, [3, 7], MaskedOracleChannel(set()))
    assert torch.allclose(oracle_h, full_h, atol=1e-6)
    assert torch.allclose(zero_h, empty_h, atol=1e-6)
