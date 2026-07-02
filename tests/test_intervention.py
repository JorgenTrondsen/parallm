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
    PhasedMode,
    ZeroChannel,
    _MixerWriteSwap,
    _ridge_oos_relmse,
    _ridge_relmse,
    _seam_token_mixer,
    intervention_forward,
    parse_channel,
    phased_intervention_forward,
    seam_intervention_forward,
    seam_predictability_analysis,
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


# --------------------------------------------------------------------------- #
# Seam (intra-block, post-attention) intervention — the D=1 cross-head anchors.
# --------------------------------------------------------------------------- #
def _build_pt_and_dense(sync_after_layers):
    """Tiny K=2 single-process PT model + the dense model it was sliced from.

    Both use sdpa attention so the oracle==dense comparison holds to fp noise.
    """
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
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
    return pt, dense, cfg


def test_seam_zero_matches_plain_d1_forward():
    # zero seam channel ⇒ MLP reads X + Y_self ⇒ exactly the current D=1 forward.
    pt, _dense, cfg = _build_pt_and_dense(list(range(8)))
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        ref_h, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        seam_h = seam_intervention_forward(pt, ids, mask, list(range(8)), ZeroChannel())
    assert torch.allclose(ref_h, seam_h, atol=1e-4, rtol=1e-4)


def test_seam_oracle_matches_dense_teacher():
    # THE premise check: oracle seam ⇒ every track's MLP reads X + ΣY ⇒ the seam
    # forward reconstructs the dense model EXACTLY (D=1's only gap is this seam).
    pt, dense, cfg = _build_pt_and_dense(list(range(8)))
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_h = dense(input_ids=ids, attention_mask=mask).last_hidden_state
        seam_h = seam_intervention_forward(pt, ids, mask, list(range(8)), OracleChannel())
    assert torch.allclose(dense_h, seam_h, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Predictability decomposition (ridge ceilings + the seam analysis forward).
# --------------------------------------------------------------------------- #
def test_ridge_relmse_recovers_linear_map():
    # O = F @ M exactly ⇒ the best linear F→O has ~zero residual ⇒ relMSE ≈ 0.
    torch.manual_seed(0)
    N, df, do = 400, 16, 8
    F = torch.randn(N, df)
    M = torch.randn(df, do)
    O = F @ M
    A, C, tot = F.T @ F, F.T @ O, O.pow(2).sum()
    assert _ridge_relmse(A, C, tot, lam=1e-6) < 1e-3


def test_ridge_relmse_orthogonal_target_is_one():
    # O independent of F ⇒ no linear map helps ⇒ relMSE ≈ 1 (predicting zero).
    torch.manual_seed(1)
    N, df, do = 4000, 16, 8
    F = torch.randn(N, df)
    O = torch.randn(N, do)  # uncorrelated with F
    A, C, tot = F.T @ F, F.T @ O, O.pow(2).sum()
    assert abs(_ridge_relmse(A, C, tot, lam=1e-3) - 1.0) < 0.1


def test_ridge_oos_recovers_linear_map_and_flags_overfit():
    # O = F@M generalizes ⇒ held-out relMSE ≈ 0. But a random target with d≈N
    # overfits in-sample (relMSE→0) yet held-out relMSE ≈ 1 — the inflation the OOS
    # split is designed to catch (the piqa-3-batch artifact in miniature).
    torch.manual_seed(3)
    d, do = 20, 6
    Ff, Fe = torch.randn(2000, d), torch.randn(2000, d)
    M = torch.randn(d, do)
    af = (Ff.T @ Ff, Ff.T @ (Ff @ M))
    ae = (Fe.T @ Fe, Fe.T @ (Fe @ M), (Fe @ M).pow(2).sum())
    assert _ridge_oos_relmse(af[0], af[1], ae[0], ae[1], ae[2], lam=1e-6) < 1e-2
    # Overfit case: few fit tokens (N≈d), random target ⇒ in-sample ~0, OOS ~1.
    torch.manual_seed(4)
    Ff2, Fe2 = torch.randn(24, d), torch.randn(2000, d)
    Of2, Oe2 = torch.randn(24, do), torch.randn(2000, do)
    in_sample = _ridge_relmse(Ff2.T @ Ff2, Ff2.T @ Of2, Of2.pow(2).sum(), lam=1e-4)
    oos = _ridge_oos_relmse(Ff2.T @ Ff2, Ff2.T @ Of2, Fe2.T @ Fe2, Fe2.T @ Oe2, Oe2.pow(2).sum(), lam=1e-4)
    assert in_sample < 0.5 and oos > 0.8   # in-sample lies, OOS tells the truth


def test_seam_predictability_analysis_runs_and_is_well_formed():
    # End-to-end on the tiny CPU model: every layer reports the expected keys, all
    # relMSE are finite and ~in range, drift_by_hln ∈ [0,1], SVD energy monotone.
    pt, _dense, cfg = _build_pt_and_dense(list(range(8)))
    # ≥3 batches so the stride-3 fit/eval split yields both fit and held-out batches.
    batches = []
    for s in range(3):
        torch.manual_seed(100 + s)
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        batches.append({"input_ids": ids, "attention_mask": torch.ones((1, 24), dtype=torch.long)})
    res = seam_predictability_analysis(
        pt, batches, tuple(range(8)),
        svd_r_grid=(2, 8, 32), ridge_lambda=1e-2, avg_windows=(2, 4), eval_stride=3,
    )
    assert set(res) == set(range(8))
    for L, r in res.items():
        # All are relMSE = SSE/norm ≥ 0; out-of-sample they can exceed 1 when the fit
        # underfits (tiny model, few tokens) — only require finite + non-negative.
        assert 0.0 <= r["stale"] < 1e4
        assert 0.0 <= r["hln"] < 1e4 and 0.0 <= r["hybrid"] < 1e4
        assert r["drift_by_hln"] <= 1.0 + 1e-4   # = 1 − (OOS drift residual, ≥ 0)
        # SVD cumulative energy is non-decreasing in r and ≤ 1.
        e = r["svd_sumy"]
        assert e[2] <= e[8] + 1e-5 <= e[32] + 1e-5 and e[32] <= 1.0 + 1e-5


def test_seam_predictability_oracle_target_self_predicts():
    # Sanity on the math path: a fully-predictable target (h_ln itself) is recovered
    # by the ridge ⇒ relMSE ~0 — guards the closed-form trace against sign/scale bugs.
    torch.manual_seed(2)
    N, d = 500, 12
    F = torch.randn(N, d)
    A, C, tot = F.T @ F, F.T @ F, F.pow(2).sum()  # predict F from F
    assert _ridge_relmse(A, C, tot, lam=1e-6) < 1e-3


def test_seam_oracle_and_zero_diverge():
    # Sanity: the D=1 post-attention gap is real — the two anchors are not equal.
    pt, _dense, cfg = _build_pt_and_dense(list(range(8)))
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        z = seam_intervention_forward(pt, ids, mask, list(range(8)), ZeroChannel())
        o = seam_intervention_forward(pt, ids, mask, list(range(8)), OracleChannel())
    assert not torch.allclose(z, o, atol=1e-3)


# --------------------------------------------------------------------------- #
# Phased (post-attn) intervention — the write-side memory-correction gate.
# --------------------------------------------------------------------------- #
def _build_pt_phased(n_tracks, sync_after_layers):
    """Tiny single-process PT model + the dense model it was sliced from, sdpa
    attention (needed for the N=1 dense-parity check to hold to fp noise)."""
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=sync_after_layers, track_group=None,
    ).eval()
    pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    return pt, dense, cfg


def _mixer_scaffolding(pt, ids, mask):
    """(embeds, position_embeddings, causal_mask, text_position_ids) — the shared
    per-forward scaffolding, for driving a single token mixer in isolation."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    with torch.no_grad():
        emb = pt.embed(ids)
        tm0 = pt.text_models[0]
        position_ids, text_position_ids = tm0._resolve_position_ids(emb, None)
        causal_mask = create_causal_mask(
            config=tm0.config, inputs_embeds=emb, attention_mask=mask,
            past_key_values=None, position_ids=text_position_ids,
        )
        pos_emb = tm0.rotary_emb(emb, position_ids)
    return emb, pos_emb, causal_mask, text_position_ids


def test_phased_mode_names_and_rejects_unknown():
    for name in ("zero", "oracle", "kv", "q"):
        assert PhasedMode(name).name == name
    try:
        PhasedMode("stale")
        raise AssertionError("PhasedMode should reject unknown modes")
    except ValueError:
        pass


def test_phased_zero_matches_deployed_postattn_forward():
    # The floor anchor: zero mode must reproduce the deployed phased forward
    # (set_sync_phase('post-attn'), sparse D=2-style schedule), fp order aside.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    pt.set_sync_phase("post-attn")
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        ref_h, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        iv_h = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode("zero"))
    assert torch.allclose(ref_h, iv_h, atol=1e-4, rtol=1e-4)


def test_phased_n1_all_modes_match_dense():
    # N=1 ⇒ the track IS the whole model: no deficiency anywhere, every swap is a
    # value-no-op and every sync a no-op ⇒ all four modes equal the dense forward.
    pt, dense, cfg = _build_pt_phased(1, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_h = dense(input_ids=ids, attention_mask=mask).last_hidden_state
        for name in ("zero", "oracle", "kv", "q"):
            iv_h = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode(name))
            assert torch.allclose(dense_h, iv_h, atol=1e-4, rtol=1e-4), name


def test_phased_oracle_on_fresh_slice_matches_dense():
    # THE oracle-correctness check: on a freshly sliced (untrained) model the
    # per-track sublayers sum EXACTLY to the dense ones, so perfect delivery at
    # every sublayer (oracle mode) must reconstruct the dense forward — at N=2
    # with a sparse schedule, not just the N=1 degenerate case. If this holds,
    # an at-scale oracle landing below the floor is a property of the TRAINED
    # weights (co-adaptation to the partial regime), not a harness bug.
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        dense_h = dense(input_ids=ids, attention_mask=mask).last_hidden_state
        iv_h = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode("oracle"))
    assert torch.allclose(dense_h, iv_h, atol=1e-4, rtol=1e-4)


def test_phased_modes_are_distinct_when_deficiency_is_real():
    # At N=2 with a sparse schedule the partial-input deficiency is real, so the
    # write-side (kv), read-side (q) and full (oracle) corrections must each move
    # the output off the floor — and kv/q are different interventions.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode(name))
            for name in ("zero", "oracle", "kv", "q")
        }
    for name in ("oracle", "kv", "q"):
        assert not torch.allclose(out["zero"], out[name], atol=1e-5), name
    assert not torch.allclose(out["kv"], out["q"], atol=1e-5)
    assert not torch.allclose(out["kv"], out["oracle"], atol=1e-5)


def test_mixer_write_swap_is_noop_when_alt_equals_input():
    # kv/q with alt == the layer's own layernormed input must be bit-equivalent to
    # the unhooked mixer (the no-deficiency invariant) — on BOTH layer types.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    emb, pos_emb, causal_mask, pos_ids = _mixer_scaffolding(pt, ids, mask)
    tm0 = pt.text_models[0]
    for idx, layer_mask in ((0, None), (3, causal_mask)):  # linear_attention, full_attention
        layer = tm0.layers[idx]
        alt = layer.input_layernorm(emb)
        with torch.no_grad():
            ref = _seam_token_mixer(layer, emb, pos_emb, layer_mask, pos_ids)
            for mode in ("kv", "q"):
                with _MixerWriteSwap(layer, mode, alt):
                    got = _seam_token_mixer(layer, emb, pos_emb, layer_mask, pos_ids)
                assert torch.allclose(ref[0], got[0], atol=1e-6), (idx, mode)
                assert torch.allclose(ref[1], got[1], atol=1e-6), (idx, mode)
            # And with a genuinely different alt the swap must be live.
            with _MixerWriteSwap(layer, "kv", alt + 0.5):
                moved = _seam_token_mixer(layer, emb, pos_emb, layer_mask, pos_ids)
            assert not torch.allclose(ref[1], moved[1], atol=1e-5), idx


def test_mixer_write_swap_gdn_slice_surgery():
    # The fused in_proj_qkv is split at its OUTPUT: kv mode keeps the q slice from
    # the partial input and takes k/v from alt (and reroutes b/a); q mode is the
    # complement (q slice from alt, z rerouted). Hooks must vanish on exit.
    import torch.nn.functional as F

    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    layer = pt.text_models[0].layers[0]  # linear_attention
    gdn = layer.linear_attn
    kd = gdn.key_dim
    torch.manual_seed(7)
    x = torch.randn(1, 5, cfg.hidden_size)
    alt = torch.randn(1, 5, cfg.hidden_size)
    with torch.no_grad():
        raw_x = F.linear(x, gdn.in_proj_qkv.weight)
        raw_alt = F.linear(alt, gdn.in_proj_qkv.weight)
        with _MixerWriteSwap(layer, "kv", alt):
            hooked = gdn.in_proj_qkv(x)
            assert torch.allclose(hooked[..., :kd], raw_x[..., :kd], atol=1e-6)   # q: partial
            assert torch.allclose(hooked[..., kd:], raw_alt[..., kd:], atol=1e-6)  # k,v: alt
            assert torch.allclose(gdn.in_proj_b(x), F.linear(alt, gdn.in_proj_b.weight), atol=1e-6)
            assert torch.allclose(gdn.in_proj_a(x), F.linear(alt, gdn.in_proj_a.weight), atol=1e-6)
            assert torch.allclose(gdn.in_proj_z(x), F.linear(x, gdn.in_proj_z.weight), atol=1e-6)
        with _MixerWriteSwap(layer, "q", alt):
            hooked = gdn.in_proj_qkv(x)
            assert torch.allclose(hooked[..., :kd], raw_alt[..., :kd], atol=1e-6)  # q: alt
            assert torch.allclose(hooked[..., kd:], raw_x[..., kd:], atol=1e-6)    # k,v: partial
            assert torch.allclose(gdn.in_proj_z(x), F.linear(alt, gdn.in_proj_z.weight), atol=1e-6)
            assert torch.allclose(gdn.in_proj_b(x), F.linear(x, gdn.in_proj_b.weight), atol=1e-6)
        # Context exited ⇒ hooks removed ⇒ plain projection again.
        assert torch.allclose(gdn.in_proj_qkv(x), raw_x, atol=1e-6)
        assert torch.allclose(gdn.in_proj_b(x), F.linear(x, gdn.in_proj_b.weight), atol=1e-6)
