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
    allocate_layer_fracs,
    block_wanda_prune_weight,
    collect_input_covs,
    collect_input_norms,
    compute_shared_lsparse_slices,
    fake_quant_weight,
    lsparse_decompose_weight,
    prune_weight,
    sparsegpt_prune_weight,
    wanda24_prune_weight,
    wanda_prune_weight,
    intervention_forward,
    parse_channel,
    svd_truncate_weight,
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


# --------------------------------------------------------------------------- #
# Replica modes (degraded local recomputation) — the gate's rails.
# --------------------------------------------------------------------------- #
def test_replica_spec_parsing():
    for name in ("replica:exact", "replica:int8", "replica:int4", "replica:svd:16",
                 "replica:prune:0.5", "replica:wanda:0.5", "replica:blockwanda:16:0.5",
                 "replica:profwanda:0.5"):
        m = PhasedMode(name)
        assert m.name == name and m.replica
    assert PhasedMode("replica:exact").degrade is None
    assert not PhasedMode("zero").replica
    for bad in ("replica", "replica:fp16", "replica:svd", "replica:svd:x",
                "replica:prune", "replica:prune:1.5", "replica:wanda:1.5",
                "replica:blockwanda:1:0.5", "replica:blockwanda:16:1.5",
                "replica:profwanda:0.0"):
        try:
            PhasedMode(bad)
            raise AssertionError(f"PhasedMode should reject {bad!r}")
        except ValueError:
            pass


def test_degradation_helpers():
    torch.manual_seed(3)
    w = torch.randn(32, 48)
    q8 = fake_quant_weight(w, 8)
    q4 = fake_quant_weight(w, 4)
    # int8 is a fine grid (tiny error); int4 is coarser but bounded by its step.
    assert (q8 - w).abs().max() < w.abs().max() / 100
    assert (q4 - w).abs().max() < w.abs().max() / 6
    assert (q4 - w).abs().sum() > (q8 - w).abs().sum()
    t = svd_truncate_weight(w, 8)
    assert torch.linalg.matrix_rank(t.float()).item() <= 8
    # rank >= full rank ⇒ ~exact reconstruction.
    assert torch.allclose(svd_truncate_weight(w, 64), w, atol=1e-5)


def test_wanda_prune_weight_helper():
    torch.manual_seed(3)
    w = torch.randn(16, 32)
    # Uniform norms ⇒ wanda reduces to plain magnitude pruning.
    assert torch.equal(wanda_prune_weight(w, 0.5, torch.ones(32)), prune_weight(w, 0.5))
    # A huge-norm input column survives in every row despite small |w|.
    w2 = torch.randn(16, 32)
    w2[:, 5] = 0.01 * torch.sign(torch.randn(16)) + 0.005
    norms = torch.ones(32)
    norms[5] = 1e4
    p = wanda_prune_weight(w2, 0.75, norms)
    assert (p[:, 5] != 0).all()
    assert torch.equal(p[p != 0], w2[p != 0])  # survivors exact
    try:
        wanda_prune_weight(w, 0.5, torch.ones(31))
        raise AssertionError("should reject mismatched norms")
    except ValueError:
        pass


def test_collect_input_norms_tiny_dense():
    _pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    # 8 layers × 4 distinct input spaces each (mixer in, mixer out-proj in, mlp in, mlp down in).
    assert len(norms) == 32
    for li, layer in enumerate(dense.layers):
        if hasattr(layer, "linear_attn"):
            assert norms[f"{li}.linear_attn.in_proj_qkv"].numel() == cfg.hidden_size
            assert norms[f"{li}.linear_attn.out_proj"].numel() == \
                cfg.linear_num_value_heads * cfg.linear_value_head_dim
        else:
            assert norms[f"{li}.self_attn.q_proj"].numel() == cfg.hidden_size
            assert norms[f"{li}.self_attn.o_proj"].numel() == \
                cfg.num_attention_heads * cfg.head_dim
        assert norms[f"{li}.mlp.gate_proj"].numel() == cfg.hidden_size
        assert norms[f"{li}.mlp.down_proj"].numel() == cfg.intermediate_size
        assert all(v.min() > 0 for k, v in norms.items())


def test_wanda_channel_end_to_end():
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")

    bare = PhasedMode("replica:wanda:0.5")
    try:
        bare.ensure_shadow(pt)
        raise AssertionError("ensure_shadow must demand calibration norms")
    except RuntimeError:
        pass

    ch = PhasedMode("replica:wanda:0.5")
    ch.set_input_norms(norms, 2)
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(
                pt, ids, mask, [1, 3, 5, 7],
                ch if name == "wanda" else PhasedMode(name),
            )
            for name in ("zero", "oracle", "wanda")
        }
    assert torch.isfinite(out["wanda"]).all()
    assert not torch.allclose(out["wanda"], out["zero"], atol=1e-5)
    assert not torch.allclose(out["wanda"], out["oracle"], atol=1e-6)


def test_sparsegpt_identity_H_is_blockwise_magnitude():
    torch.manual_seed(5)
    w = torch.randn(8, 64)
    out = sparsegpt_prune_weight(w, 0.5, torch.eye(64), block=32)
    # With H = I there is nothing to reconstruct: survivors keep exact values
    # and each row loses half its entries per 32-column block.
    surv = out != 0
    assert torch.allclose(out[surv], w[surv], atol=1e-5)
    for r in range(8):
        for s in (0, 32):
            assert int((out[r, s:s + 32] == 0).sum()) == 16


def test_sparsegpt_reconstruction_beats_wanda_on_correlated_inputs():
    torch.manual_seed(11)
    n, d = 4096, 48
    base = torch.randn(n, d // 2)
    X = torch.cat([base, base + 0.3 * torch.randn(n, d // 2)], dim=1)  # correlated cols
    w = torch.randn(16, d)
    H = X.T @ X
    norms = X.norm(dim=0)
    w_sgpt = sparsegpt_prune_weight(w, 0.5, H, block=16)
    w_wanda = wanda_prune_weight(w, 0.5, norms)
    err_sgpt = (X @ (w - w_sgpt).T).pow(2).mean()
    err_wanda = (X @ (w - w_wanda).T).pow(2).mean()
    assert err_sgpt < err_wanda  # the survivor update compensates pruned mass


def test_qwanda_compose():
    torch.manual_seed(3)
    w = torch.randn(16, 32)
    ch = PhasedMode("replica:qwanda:4:0.5")
    assert ch.wanda and ch.pre_quant_bits == 4
    expected = wanda_prune_weight(fake_quant_weight(w, 4), 0.5, torch.ones(32))
    assert (expected == 0).float().mean() >= 0.5 - 1e-6
    # survivors sit on the int4 grid, not the original values
    q = fake_quant_weight(w, 4)
    surv = expected != 0
    assert torch.equal(expected[surv], q[surv])


def test_collect_input_covs_tiny_dense():
    _pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.long)
    batch = {"input_ids": ids, "attention_mask": mask}
    covs = collect_input_covs(
        dense, [batch], "cpu",
        slice_keys={"self_attn.o_proj": 2, "linear_attn.out_proj": 2, "mlp.down_proj": 2},
        gpu_budget_bytes=2 ** 20,  # force multiple passes
    )
    assert len(covs) == 32
    norms = collect_input_norms(dense, [batch], "cpu")
    for li, layer in enumerate(dense.layers):
        key = (f"{li}.linear_attn.in_proj_qkv" if hasattr(layer, "linear_attn")
               else f"{li}.self_attn.q_proj")
        H = covs[key]
        assert H.shape == (cfg.hidden_size, cfg.hidden_size)
        assert torch.allclose(H, H.T, atol=1e-3)  # symmetric
        assert torch.allclose(H.diagonal().clamp(min=0).sqrt(), norms[key], atol=1e-2)
        Hd = covs[f"{li}.mlp.down_proj"]
        assert Hd.shape == (2, cfg.intermediate_size // 2, cfg.intermediate_size // 2)


def test_chanwanda_and_sparsegpt_channels_end_to_end():
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    batch = {"input_ids": ids, "attention_mask": mask}
    norms = collect_input_norms(dense, [batch], "cpu")
    covs = collect_input_covs(
        dense, [batch], "cpu",
        slice_keys={"self_attn.o_proj": 2, "linear_attn.out_proj": 2, "mlp.down_proj": 2},
    )

    chw = PhasedMode("replica:chanwanda:0.5")
    chw.set_input_norms(norms, 2)
    sg = PhasedMode("replica:sparsegpt:0.5")
    sg.set_input_covs(covs, 2)
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(
                pt, ids, mask, [1, 3, 5, 7],
                {"chanwanda": chw, "sparsegpt": sg}.get(name) or PhasedMode(name),
            )
            for name in ("zero", "oracle", "chanwanda", "sparsegpt")
        }
    for name in ("chanwanda", "sparsegpt"):
        assert torch.isfinite(out[name]).all(), name
        assert not torch.allclose(out[name], out["zero"], atol=1e-5), name
        assert not torch.allclose(out[name], out["oracle"], atol=1e-6), name

    # chanwanda: gate/up zero-ROWS == down zero-COLS (the joint channel mask).
    shadow_layer = chw._shadow[0][0]
    gate_zero = (shadow_layer.mlp.gate_proj.weight == 0).all(dim=1)
    up_zero = (shadow_layer.mlp.up_proj.weight == 0).all(dim=1)
    down_zero = (shadow_layer.mlp.down_proj.weight == 0).all(dim=0)
    assert torch.equal(gate_zero, down_zero) and torch.equal(up_zero, down_zero)
    assert int(down_zero.sum()) == cfg.intermediate_size // 2 // 2


def test_block_wanda_prune_weight():
    torch.manual_seed(4)
    w = torch.randn(8, 64)
    p = block_wanda_prune_weight(w, 0.5, torch.ones(64), block_size=8)
    zero = (p == 0).reshape(8, 8, 8)
    # Zeros arrive as whole aligned blocks: each 8-wide block is all-zero or all-kept.
    per_block = zero.all(dim=-1) | (~zero.any(dim=-1))
    assert per_block.all()
    assert int(zero.all(dim=-1).sum(dim=-1).float().mean()) == 4  # half the blocks per row
    surv = p != 0
    assert torch.equal(p[surv], w[surv])  # survivors exact
    # A high-norm column rescues its whole block.
    norms = torch.ones(64)
    norms[3] = 1e5
    p2 = block_wanda_prune_weight(w, 0.5, norms, block_size=8)
    assert (p2[:, 0:8] != 0).all()
    try:
        block_wanda_prune_weight(w, 0.5, torch.ones(64), block_size=7)
        raise AssertionError("should reject non-divisible block size")
    except ValueError:
        pass


def test_profwanda_layer_fracs_and_allocator():
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")

    ch = PhasedMode("replica:profwanda:0.5")
    ch.set_input_norms(norms, 2)
    try:
        ch.ensure_shadow(pt)
        raise AssertionError("profwanda must demand a layer profile")
    except RuntimeError:
        pass

    fracs = allocate_layer_fracs(pt, ch, [1, 3, 5, 7], 0.5)
    assert len(fracs) == 8
    assert all(0.35 <= f <= 0.85 for f in fracs)
    # Parameter-weighted average ≈ the uniform budget.
    params = torch.tensor([
        sum(m.weight.numel() for tm in pt.text_models
            for m in tm.layers[i].modules() if isinstance(m, nn.Linear))
        for i in range(8)
    ], dtype=torch.float64)
    avg = float((params * torch.tensor(fracs, dtype=torch.float64)).sum() / params.sum())
    assert abs(avg - 0.5) < 0.02
    # Window starts (deeper remaining depth) get DENSER copies than window ends.
    assert fracs[2] <= fracs[3] + 1e-9  # layer 2 opens the [2,3] window

    ch.set_layer_fracs(fracs)
    ch.ensure_shadow(pt)
    # Per-layer shadow sparsity follows the profile.
    for li in (0, 5):
        mod = ch._shadow[0][li].mlp.gate_proj
        sparsity = float((mod.weight == 0).float().mean())
        assert abs(sparsity - fracs[li]) < 0.05


def test_wanda24_prune_weight():
    torch.manual_seed(6)
    w = torch.randn(8, 32)
    p = wanda24_prune_weight(w, torch.ones(32))
    kept = (p != 0).reshape(8, 8, 4).sum(-1)
    assert (kept == 2).all()  # exactly 2 of every 4
    surv = p != 0
    assert torch.equal(p[surv], w[surv])
    # Norms steer the within-group selection.
    norms = torch.ones(32)
    norms[0] = 1e5
    p2 = wanda24_prune_weight(w, norms)
    assert (p2[:, 0] != 0).all()
    assert PhasedMode("replica:wanda24").wanda24


def test_prune_weight_helper():
    torch.manual_seed(3)
    w = torch.randn(32, 48)
    p = prune_weight(w, 0.5)
    # Per row: ~half the entries zeroed, and every survivor outranks every
    # zeroed entry in magnitude (pure magnitude criterion).
    for r in range(w.shape[0]):
        zeroed = p[r] == 0
        assert 20 <= int(zeroed.sum()) <= 28  # ~24, ties aside
        assert w[r][~zeroed].abs().min() >= w[r][zeroed].abs().max()
    surv = p != 0
    assert torch.equal(p[surv], w[surv])  # survivors are EXACT (no requantization)
    assert (prune_weight(w, 0.75) != 0).float().mean() < 0.30
    for bad in (0.0, 1.0, -0.5):
        try:
            prune_weight(w, bad)
            raise AssertionError("should reject frac outside (0,1)")
        except ValueError:
            pass


def test_replica_exact_matches_oracle():
    # THE rail: between-sync trajectories are deterministic functions of the
    # synced residual + weights, so an undegraded shadow replay must reproduce
    # the oracle forward (fp order aside). Everything the degraded arms measure
    # is then pure degradation effect.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        oracle_h = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode("oracle"))
        replica_h = phased_intervention_forward(
            pt, ids, mask, [1, 3, 5, 7], PhasedMode("replica:exact")
        )
    assert torch.allclose(oracle_h, replica_h, atol=1e-4, rtol=1e-4)


def test_replica_degraded_moves_between_anchors():
    # Degraded replicas must leave the floor (they deliver real content) without
    # being exactly the ceiling (the degradation is real): int8 lands near oracle,
    # a hard svd truncation lands strictly between and off both anchors.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode(name))
            for name in ("zero", "oracle", "replica:int8", "replica:svd:8")
        }
    for name in ("replica:int8", "replica:svd:8"):
        assert not torch.allclose(out["zero"], out[name], atol=1e-5), name
        assert not torch.allclose(out["oracle"], out[name], atol=1e-6), name
    err_int8 = (out["replica:int8"] - out["oracle"]).abs().mean()
    err_svd8 = (out["replica:svd:8"] - out["oracle"]).abs().mean()
    assert err_int8 < err_svd8  # finer copies ⇒ closer to perfect delivery


def test_mlp_subspec_parsing():
    m = PhasedMode("replica:wanda:0.5:mlp:none")
    assert m.replica and m.wanda and m.mlp_none and m.mlp_mode is None
    m = PhasedMode("replica:exact:mlp:none")
    assert m.mlp_none and m.degrade is None and not m.wanda
    m = PhasedMode("replica:qwanda:4:0.5:mlp:none")
    assert m.mlp_none and m.wanda and m.pre_quant_bits == 4
    m = PhasedMode("replica:wanda:0.5:mlp:wanda:0.9")
    assert not m.mlp_none and m.mlp_mode is not None
    assert m.mlp_mode.wanda and m.mlp_mode.wanda_frac == 0.9
    for bad in (
        "replica:wanda:0.5:mlp:wanda:1.5",       # sub-spec validated too
        "replica:wanda:0.5:mlp:chanwanda:0.5",   # structured subs unsupported
        "replica:wanda:0.5:mlp:sparsegpt:0.5",
        "replica:chanwanda:0.5:mlp:none",        # chanwanda already owns the MLP
        "replica:wanda:0.5:mlp:wanda:0.8:mlp:none",  # one sub-spec only
        "replica:wanda:0.5:mlp:",
    ):
        try:
            PhasedMode(bad)
            raise AssertionError(f"PhasedMode should reject {bad!r}")
        except ValueError:
            pass


def test_mlp_subspec_split_is_lossless():
    # Grammar rail: splitting one spec into identical base + mlp sub-specs must
    # reproduce the unsuffixed channel exactly (same weights, same forward).
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    plain = PhasedMode("replica:wanda:0.5")
    split = PhasedMode("replica:wanda:0.5:mlp:wanda:0.5")
    for ch in (plain, split):
        ch.set_input_norms(norms, 2)
    with torch.no_grad():
        h_plain = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], plain)
        h_split = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], split)
    assert torch.allclose(h_plain, h_split, atol=1e-6)


def test_mlp_none_shadow_holds_no_mlp_params():
    # The memory claim itself: attn-only shadow clones contain zero mlp weights.
    pt, _dense, _cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ch = PhasedMode("replica:exact:mlp:none")
    shadow = ch.ensure_shadow(pt)
    for layers in shadow:
        for clone in layers:
            assert clone.mlp is None
            assert not any(n.startswith("mlp.") for n, _ in clone.named_parameters())


def test_mlp_subspec_routing_targets_only_mlp():
    # replica:exact:mlp:wanda:0.5 — mixer weights bit-identical to the originals,
    # mlp weights pruned. Also: the sub-spec alone must demand calibration norms.
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    bare = PhasedMode("replica:exact:mlp:wanda:0.5")
    try:
        bare.ensure_shadow(pt)
        raise AssertionError("ensure_shadow must demand the sub-spec's norms")
    except RuntimeError:
        pass
    ch = PhasedMode("replica:exact:mlp:wanda:0.5")
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    ch.set_input_norms(norms, 2)
    shadow = ch.ensure_shadow(pt)
    for tm, layers in zip(pt.text_models, shadow):
        for layer, clone in zip(tm.layers, layers):
            for (rel, mod), (_, cmod) in zip(layer.named_modules(), clone.named_modules()):
                if not isinstance(mod, nn.Linear):
                    continue
                if rel.startswith("mlp."):
                    assert int((cmod.weight == 0).sum()) >= cmod.weight.numel() // 2, rel
                else:
                    assert torch.equal(mod.weight, cmod.weight), rel


def test_attn_only_exact_matches_oracle_when_mlp_is_dead():
    # Semantics rail: with every track's down_proj zeroed, ALL MLP deltas vanish,
    # so dropping the MLP copies loses nothing — replica:exact:mlp:none must
    # reproduce replica:exact (≡ oracle per the existing rail) to fp order.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    with torch.no_grad():
        for tm in pt.text_models:
            for layer in tm.layers:
                layer.mlp.down_proj.weight.zero_()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        h_exact = phased_intervention_forward(
            pt, ids, mask, [1, 3, 5, 7], PhasedMode("replica:exact")
        )
        h_attn_only = phased_intervention_forward(
            pt, ids, mask, [1, 3, 5, 7], PhasedMode("replica:exact:mlp:none")
        )
    assert torch.allclose(h_exact, h_attn_only, atol=1e-5, rtol=1e-5)


def test_lsparse_spec_parsing():
    m = PhasedMode("replica:lsparse:64:0.7")
    assert m.replica and m.lsparse and m.lsparse_rank == 64 and m.wanda_frac == 0.7
    assert not m.wanda and m.pre_quant_bits is None
    m = PhasedMode("replica:qlsparse:4:64:0.7")
    assert m.lsparse and m.lsparse_rank == 64 and m.pre_quant_bits == 4
    for bad in ("replica:lsparse:64", "replica:lsparse:0:0.7", "replica:lsparse:64:1.5",
                "replica:qlsparse:4:64", "replica:qlsparse:4:64:0.0"):
        try:
            PhasedMode(bad)
            raise AssertionError(f"PhasedMode should reject {bad!r}")
        except ValueError:
            pass


def test_lsparse_decompose_identity_at_full_rank():
    # Rail: at rank >= min(m, n) the first SVD is exact, the residual is zero,
    # and the output must reproduce w regardless of frac.
    torch.manual_seed(7)
    w = torch.randn(12, 24)
    norms = torch.rand(24) + 0.5
    out = lsparse_decompose_weight(w, rank=12, frac=0.7, in_norms=norms)
    assert torch.allclose(out, w, atol=1e-4)


def test_lsparse_recovers_planted_lowrank_plus_sparse():
    # On a matrix that IS low-rank + sparse spikes, the decomposition recovers
    # it ~exactly while plain wanda at the same frac cannot (it must either
    # drop bulk mass or spike mass).
    torch.manual_seed(11)
    m, n, r = 32, 64, 4
    low = torch.randn(m, r) @ torch.randn(r, n) * 0.5
    spikes = torch.zeros(m, n)
    idx = torch.rand(m, n) < 0.1  # 10% dense spikes
    spikes[idx] = torch.randn(int(idx.sum())) * 5.0
    w = low + spikes
    norms = torch.ones(n)
    ls = lsparse_decompose_weight(w, rank=8, frac=0.7, in_norms=norms)
    wd = wanda_prune_weight(w, 0.7, norms)
    err_ls = (ls - w).norm() / w.norm()
    err_wd = (wd - w).norm() / w.norm()
    assert err_ls < 0.15
    assert err_ls < err_wd / 2


def test_lsparse_qlsparse_channels_end_to_end():
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    bare = PhasedMode("replica:lsparse:8:0.7")
    try:
        bare.ensure_shadow(pt)
        raise AssertionError("lsparse must demand calibration norms")
    except RuntimeError:
        pass
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    out = {}
    for name in ("zero", "oracle", "replica:lsparse:8:0.7", "replica:qlsparse:4:8:0.7"):
        ch = PhasedMode(name)
        if ch.replica:
            ch.set_input_norms(norms, 2)
        with torch.no_grad():
            out[name] = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], ch)
    for name in ("replica:lsparse:8:0.7", "replica:qlsparse:4:8:0.7"):
        assert torch.isfinite(out[name]).all(), name
        assert not torch.allclose(out[name], out["zero"], atol=1e-5), name
        assert not torch.allclose(out[name], out["oracle"], atol=1e-6), name
    # The quantized variant is a strictly coarser copy than the plain one.
    assert not torch.allclose(out["replica:lsparse:8:0.7"],
                              out["replica:qlsparse:4:8:0.7"], atol=1e-6)


def test_slsparse_spec_parsing():
    m = PhasedMode("replica:slsparse:256:0.7")
    assert m.replica and m.shared_lsparse and m.lsparse_rank == 256
    assert m.wanda_frac == 0.7 and not m.lsparse and m.pre_quant_bits is None
    m = PhasedMode("replica:qslsparse:4:256:0.7")
    assert m.shared_lsparse and m.pre_quant_bits == 4
    for bad in ("replica:slsparse:256", "replica:slsparse:0:0.7",
                "replica:slsparse:256:1.5", "replica:qslsparse:4:256:0.0"):
        try:
            PhasedMode(bad)
            raise AssertionError(f"PhasedMode should reject {bad!r}")
        except ValueError:
            pass


def test_shared_lsparse_identity_matches_exact():
    # THE mapping rail: at rank >= every dense min-dim the decomposition is the
    # identity, so the shared slices must be bit-equal re-slices of the dense
    # weights — the channel must reproduce replica:exact (and the internal
    # dense→slice correspondence assertion runs on every slab type:
    # GatedQ/KVReplicated/FusedSegment/Colwise/Rowwise).
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    slices = compute_shared_lsparse_slices(dense, pt, rank=100000, frac=0.5, input_norms=norms)
    ch = PhasedMode("replica:slsparse:100000:0.5")
    ch.set_shared_slices(slices)
    with torch.no_grad():
        h_exact = phased_intervention_forward(
            pt, ids, mask, [1, 3, 5, 7], PhasedMode("replica:exact")
        )
        h_shared = phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], ch)
    assert torch.allclose(h_exact, h_shared, atol=1e-4, rtol=1e-4)


def test_shared_lsparse_small_rank_lands_between_anchors():
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    bare = PhasedMode("replica:slsparse:4:0.6")
    try:
        bare.ensure_shadow(pt)
        raise AssertionError("slsparse must demand precomputed shared slices")
    except RuntimeError:
        pass
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    ch = PhasedMode("replica:slsparse:4:0.6")
    ch.set_shared_slices(compute_shared_lsparse_slices(
        dense, pt, rank=4, frac=0.6, input_norms=norms
    ))
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(
                pt, ids, mask, [1, 3, 5, 7],
                ch if name == "shared" else PhasedMode(name),
            )
            for name in ("zero", "oracle", "shared")
        }
    assert torch.isfinite(out["shared"]).all()
    assert not torch.allclose(out["shared"], out["zero"], atol=1e-5)
    assert not torch.allclose(out["shared"], out["oracle"], atol=1e-6)


def test_attn_none_subspec_parsing():
    m = PhasedMode("replica:none:mlp:wanda:0.5")
    assert m.replica and m.attn_none and not m.wanda
    assert m.mlp_mode is not None and m.mlp_mode.wanda
    m = PhasedMode("replica:none:mlp:qwanda:2:0.5")
    assert m.attn_none and m.mlp_mode.pre_quant_bits == 2
    for bad in ("replica:none",              # no MLP copies either = zero channel
                "replica:none:mlp:none"):
        try:
            PhasedMode(bad)
            raise AssertionError(f"PhasedMode should reject {bad!r}")
        except ValueError:
            pass


def test_mlp_only_shadow_holds_no_mixer_params():
    # The memory claim: MLP-only shadow clones contain zero mixer weights
    # (and would need no shadow KV cache / GDN state in deployment).
    pt, dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    ch = PhasedMode("replica:none:mlp:wanda:0.5")
    ch.set_input_norms(norms, 2)
    shadow = ch.ensure_shadow(pt)
    for layers in shadow:
        for clone in layers:
            assert getattr(clone, "self_attn", None) is None
            assert getattr(clone, "linear_attn", None) is None
            for n, _ in clone.named_parameters():
                assert not n.startswith(("self_attn.", "linear_attn.")), n


def test_mlp_only_exact_matches_oracle_when_attention_is_dead():
    # Semantics rail (mirror of the dead-MLP rail): with every track's mixer
    # OUTPUT projection zeroed, ALL attention deltas vanish, so dropping the
    # attention copies loses nothing — replica:none:mlp:exact must reproduce
    # replica:exact (≡ oracle per the existing rail) to fp order.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    with torch.no_grad():
        for tm in pt.text_models:
            for layer in tm.layers:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn.out_proj.weight.zero_()
                else:
                    layer.self_attn.o_proj.weight.zero_()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        h_exact = phased_intervention_forward(
            pt, ids, mask, [1, 3, 5, 7], PhasedMode("replica:exact")
        )
        h_mlp_only = phased_intervention_forward(
            pt, ids, mask, [1, 3, 5, 7], PhasedMode("replica:none:mlp:exact")
        )
    assert torch.allclose(h_exact, h_mlp_only, atol=1e-5, rtol=1e-5)


def test_mlp_only_lands_between_anchors_when_attention_is_live():
    # With real attention the MLP-only estimate delivers MLP content but misses
    # the other track's attention deltas: distinct from zero, oracle, AND the
    # full exact replica.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode(name))
            for name in ("zero", "oracle", "replica:exact", "replica:none:mlp:exact")
        }
    mo = out["replica:none:mlp:exact"]
    assert torch.isfinite(mo).all()
    assert not torch.allclose(mo, out["zero"], atol=1e-5)
    assert not torch.allclose(mo, out["oracle"], atol=1e-6)
    assert not torch.allclose(mo, out["replica:exact"], atol=1e-6)


def test_attn_only_lands_between_anchors_when_mlp_is_live():
    # With real MLPs the attn-only estimate delivers true attention content but
    # misses the other track's MLP deltas: distinct from zero, oracle, AND from
    # the full exact replica.
    pt, _dense, cfg = _build_pt_phased(2, [1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        out = {
            name: phased_intervention_forward(pt, ids, mask, [1, 3, 5, 7], PhasedMode(name))
            for name in ("zero", "oracle", "replica:exact", "replica:exact:mlp:none")
        }
    ao = out["replica:exact:mlp:none"]
    assert torch.isfinite(ao).all()
    assert not torch.allclose(ao, out["zero"], atol=1e-5)
    assert not torch.allclose(ao, out["oracle"], atol=1e-6)
    assert not torch.allclose(ao, out["replica:exact"], atol=1e-6)
