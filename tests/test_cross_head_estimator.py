"""Unit tests for the cross-head attention-output estimator (``model/cross_head_estimator.py``).

Covers the module math in isolation (no distributed, CPU): the zero-init no-op
(the warm-start guarantee), the fresh/stale substitute forms, and the seam split
helpers. The end-to-end "seam forward with the estimator == plain D=1 at init"
equivalence is exercised against a built model in the integration tests.
"""
from __future__ import annotations

import pytest
import torch

torch.set_default_dtype(torch.float32)

from pt_converter.model.cross_head_estimator import (
    CrossHeadEstimator,
    _roll1,
    seam_mlp,
    seam_token_mixer,
)


def _y_list(n_tracks, B=1, T=4, H=8):
    return [torch.randn(B, T, H) for _ in range(n_tracks)]


def test_zero_init_gate_makes_substitutes_zero_fresh():
    est = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="fresh", rank=4)
    ys = _y_list(2)
    sum_y = ys[0] + ys[1]
    h_ln = torch.randn(1, 4, 8)
    subs, ghat = est.subs(1, ys, sum_y, h_ln)
    # No-op warm start: every substitute is exactly zero at the zero gate.
    assert all(torch.equal(s, torch.zeros_like(s)) for s in subs)
    # The predictor still produces a (non-gated) estimate for the predict loss.
    assert ghat is not None and ghat.shape == (1, 4, 8)


def test_zero_init_gate_makes_substitutes_zero_stale():
    est = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="stale")
    ys = _y_list(2)
    sum_y = ys[0] + ys[1]
    subs, ghat = est.subs(0, ys, sum_y, h_ln=None)
    assert all(torch.equal(s, torch.zeros_like(s)) for s in subs)
    assert ghat is None  # stale has no predictor


def test_fresh_substitute_form_when_gate_open():
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="fresh", rank=4)
    with torch.no_grad():
        est.gate[1].fill_(1.0)  # fully open at layer 1
    ys = _y_list(2)
    sum_y = ys[0] + ys[1]
    h_ln = torch.randn(1, 4, 8)
    subs, ghat = est.subs(1, ys, sum_y, h_ln)
    # S_k = gate * (ghat - y_k) ⇒ mlp_in_k = X + y_k + S_k = X + ghat (gate=1).
    for s, y in zip(subs, ys):
        assert torch.allclose(s, ghat - y)


def test_stale_substitute_is_one_token_stale_cross_head():
    est = CrossHeadEstimator(num_layers=2, hidden_size=4, backend="stale")
    with torch.no_grad():
        est.gate[0].fill_(1.0)
    ys = _y_list(3, T=5, H=4)
    sum_y = ys[0] + ys[1] + ys[2]
    subs, _ = est.subs(0, ys, sum_y, h_ln=None)
    for s, y in zip(subs, ys):
        assert torch.allclose(s, _roll1(sum_y - y))


def test_fresh_yself_conditions_on_y_self_and_stacks_ghat():
    # fresh_yself reads [h_ln, y_self] (down input = 2H) and returns a PER-TRACK ghat.
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="fresh_yself", rank=4)
    assert est.down[0].in_features == 16  # 2 * hidden_size
    ys = _y_list(3, T=5, H=8)
    sum_y = ys[0] + ys[1] + ys[2]
    h_ln = torch.randn(1, 5, 8)
    # zero-init gate ⇒ exact no-op.
    subs0, ghat0 = est.subs(0, ys, sum_y, h_ln)
    assert all(torch.count_nonzero(s) == 0 for s in subs0)
    # ghat is stacked per-track (K, B, T, H) so the caller averages relMSE over tracks.
    assert ghat0.shape == (3, 1, 5, 8)
    # gate open ⇒ S_k = ghat_k - y_k ⇒ mlp_in_k = X + y_k + S_k = X + ghat_k.
    with torch.no_grad():
        est.gate[1].fill_(1.0)
    subs1, ghat1 = est.subs(1, ys, sum_y, h_ln)
    for s, gh, y in zip(subs1, ghat1, ys):
        assert torch.allclose(s, gh - y)
    # each track's prediction differs (conditioned on its own y_self).
    assert not torch.allclose(ghat1[0], ghat1[1])


def test_gate_scale_warmup_holds_gate_shut():
    # _gate_scale (set by the train loop's gate-warmup) multiplies the effective gate:
    # at 0 the seam is a no-op even with a fully-open gate param.
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="fresh_yself", rank=4)
    with torch.no_grad():
        est.gate.fill_(1.0)
    ys = _y_list(2, H=8)
    sum_y = ys[0] + ys[1]
    h_ln = torch.randn(1, 4, 8)
    est._gate_scale = 0.0
    subs, _ = est.subs(0, ys, sum_y, h_ln)
    assert all(torch.count_nonzero(s) == 0 for s in subs)
    est._gate_scale = 1.0
    subs1, _ = est.subs(0, ys, sum_y, h_ln)
    assert any(torch.count_nonzero(s) > 0 for s in subs1)


def test_fresh_yself_config_roundtrips():
    est = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="fresh_yself", rank=4, depth=2)
    rebuilt = CrossHeadEstimator(**est.config_dict())
    assert rebuilt.backend == "fresh_yself"
    rebuilt.load_state_dict(est.state_dict())
    for a, b in zip(est.parameters(), rebuilt.parameters()):
        assert torch.equal(a, b)


def test_temporal_backend_reads_cached_history_shared_ghat():
    # temporal predicts current ΣY from window past frames of ΣY; in_dim = window*H,
    # ghat is SHARED across tracks (coherent, like fresh — not per-track).
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="temporal", rank=4, window=3)
    assert est.down[0].in_features == 24  # window * hidden_size
    ys = _y_list(2, T=6, H=8)
    sum_y = ys[0] + ys[1]
    subs0, ghat0 = est.subs(0, ys, sum_y, h_ln=None)  # temporal uses sum_y history, not h_ln
    assert all(torch.count_nonzero(s) == 0 for s in subs0)  # zero-gate no-op
    assert ghat0.shape == sum_y.shape  # shared (not stacked per-track)
    with torch.no_grad():
        est.gate[1].fill_(1.0)
    subs1, ghat1 = est.subs(1, ys, sum_y, h_ln=None)
    for s, y in zip(subs1, ys):
        assert torch.allclose(s, ghat1 - y)  # S_k = g*(ghat - y_k)
    rebuilt = CrossHeadEstimator(**est.config_dict())
    assert rebuilt.window == 3 and rebuilt.backend == "temporal"
    rebuilt.load_state_dict(est.state_dict())


def test_oracle_lowrank_compresses_real_sumy_shared_ghat():
    # oracle_lowrank reconstructs the REAL ΣY through a rank-`rank` linear bottleneck
    # (the down projection = the rank-r second all-reduce); shared ghat across tracks.
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="oracle_lowrank", rank=2)
    assert est.down[0].in_features == 8 and est.down[0].out_features == 2  # ΣY → rank-2 bottleneck
    ys = _y_list(3, T=4, H=8)
    sum_y = ys[0] + ys[1] + ys[2]
    subs0, ghat0 = est.subs(0, ys, sum_y, h_ln=None)  # reads sum_y, not h_ln
    assert all(torch.count_nonzero(s) == 0 for s in subs0)  # zero-gate no-op
    assert ghat0.shape == sum_y.shape  # shared (coherent)
    with torch.no_grad():
        est.gate[1].fill_(1.0)
    subs1, ghat1 = est.subs(1, ys, sum_y, h_ln=None)
    for s, y in zip(subs1, ys):
        assert torch.allclose(s, ghat1 - y)


def test_stale_corr_is_stale_skip_plus_correction():
    # stale_corr (HYBRID): reads [h_ln, roll(ΣY,1)] (in_dim 2H), predicts the drift,
    # ghat = roll(ΣY,1) + correction; shared ghat. S_k = g*(ghat - y_k).
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="stale_corr", rank=4)
    assert est.down[0].in_features == 16  # [h_ln, stale] = 2 * hidden_size
    ys = _y_list(3, T=6, H=8)
    sum_y = ys[0] + ys[1] + ys[2]
    h_ln = torch.randn(1, 6, 8)
    # zero-init gate ⇒ exact no-op (warm-start safe).
    subs0, ghat0 = est.subs(0, ys, sum_y, h_ln)
    assert all(torch.count_nonzero(s) == 0 for s in subs0)
    assert ghat0.shape == sum_y.shape  # shared (coherent), ndim matches sum_y
    # gate open ⇒ S_k = ghat - y_k ⇒ mlp_in_k = X + y_k + S_k = X + ghat.
    with torch.no_grad():
        est.gate[1].fill_(1.0)
    subs1, ghat1 = est.subs(1, ys, sum_y, h_ln)
    for s, y in zip(subs1, ys):
        assert torch.allclose(s, ghat1 - y)
    # ghat carries the real stale frame as a skip: ghat - prediction == roll(ΣY,1).
    pred = est.predict(1, h_ln, _roll1(sum_y))
    assert torch.allclose(ghat1 - pred, _roll1(sum_y), atol=1e-6)


def test_stale_corr_degrades_to_stale_skip_when_correction_zero():
    # The graceful-degradation guarantee: if the predictor outputs ~0 (drift
    # unpredictable), ghat → roll(ΣY,1) so the MLP reads X + stale ΣY (stale-quality).
    est = CrossHeadEstimator(num_layers=2, hidden_size=8, backend="stale_corr", rank=4)
    with torch.no_grad():
        for lin in est.up:  # zero the output projection ⇒ correction == 0
            lin.weight.zero_()
        est.gate[0].fill_(1.0)
    ys = _y_list(3, T=6, H=8)
    sum_y = ys[0] + ys[1] + ys[2]
    h_ln = torch.randn(1, 6, 8)
    subs, ghat = est.subs(0, ys, sum_y, h_ln)
    assert torch.allclose(ghat, _roll1(sum_y), atol=1e-6)        # pure stale skip
    for s, y in zip(subs, ys):
        assert torch.allclose(s, _roll1(sum_y) - y, atol=1e-6)   # mlp_in_k = X + stale ΣY


def test_stale_corr_config_roundtrips():
    est = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="stale_corr", rank=4, depth=2)
    rebuilt = CrossHeadEstimator(**est.config_dict())
    assert rebuilt.backend == "stale_corr"
    rebuilt.load_state_dict(est.state_dict())
    for a, b in zip(est.parameters(), rebuilt.parameters()):
        assert torch.equal(a, b)


def test_oracle_backend_injects_exact_missing_content():
    # oracle: S_k = gate * (ΣY − y_k) = gate * Σ_others Y_k (exact current token, no roll).
    est = CrossHeadEstimator(num_layers=2, hidden_size=4, backend="oracle")
    assert est.backend == "oracle"
    ys = _y_list(3, T=5, H=4)
    sum_y = ys[0] + ys[1] + ys[2]
    # zero-init gate ⇒ exact no-op (warm-start safe), no predictor.
    subs0, ghat0 = est.subs(0, ys, sum_y, h_ln=None)
    assert ghat0 is None
    assert all(torch.count_nonzero(s) == 0 for s in subs0)
    # gate open ⇒ S_k = Σ_others Y_k ⇒ mlp_in_k = X + y_k + S_k = X + ΣY.
    with torch.no_grad():
        est.gate[1].fill_(1.0)
    subs1, _ = est.subs(1, ys, sum_y, h_ln=None)
    for s, y in zip(subs1, ys):
        assert torch.allclose(s, sum_y - y)


def test_oracle_backend_has_no_predictor():
    est = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="oracle")
    assert not hasattr(est, "down") or len(est.down) == 0 if hasattr(est, "down") else True
    # config round-trips the backend so the sidecar rebuilds it.
    rebuilt = CrossHeadEstimator(**est.config_dict())
    assert rebuilt.backend == "oracle"
    rebuilt.load_state_dict(est.state_dict())


def test_roll1_is_one_token_causal_shift():
    o = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
    assert torch.allclose(_roll1(o), torch.tensor([[[0.0], [1.0], [2.0]]]))


def test_config_dict_roundtrips_shape():
    est = CrossHeadEstimator(num_layers=5, hidden_size=16, backend="fresh", rank=8)
    cfg = est.config_dict()
    rebuilt = CrossHeadEstimator(**cfg)
    rebuilt.load_state_dict(est.state_dict())
    # Identical params after a state_dict round-trip through the sidecar config.
    for a, b in zip(est.parameters(), rebuilt.parameters()):
        assert torch.equal(a, b)


def test_depth_adds_rank_space_blocks_and_roundtrips():
    # depth=1 has no mid blocks (plain low-rank map); depth>1 adds (depth-1) per layer.
    shallow = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="fresh", rank=4, depth=1)
    assert all(len(m) == 0 for m in shallow.mid)
    deep = CrossHeadEstimator(num_layers=3, hidden_size=8, backend="fresh", rank=4, depth=3)
    assert all(len(m) == 2 for m in deep.mid)
    assert deep.config_dict()["depth"] == 3
    rebuilt = CrossHeadEstimator(**deep.config_dict())
    rebuilt.load_state_dict(deep.state_dict())  # shapes match through the sidecar config
    # Still an exact no-op at the zero-init gate regardless of depth.
    ys = _y_list(2)
    subs, ghat = deep.subs(1, ys, ys[0] + ys[1], torch.randn(1, 4, 8))
    assert ghat.shape == ys[0].shape
    for s in subs:
        assert torch.count_nonzero(s) == 0


def test_depth_invalid_raises():
    with pytest.raises(ValueError):
        CrossHeadEstimator(num_layers=2, hidden_size=8, backend="fresh", rank=4, depth=0)


def _build_pt_d1(n_tracks=2):
    import torch.nn as nn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from pt_converter.model.pt_model import PTWrappedModel
    from pt_converter.slicer.convert import slice_model_to_tracks

    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4, vocab_size=128, rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(dense, n_tracks=n_tracks, sync_block_depth=1, text_config_attr="config")

    def build():
        pt = PTWrappedModel(
            text_config=cfg, n_tracks=n_tracks, local_track_ids=tuple(range(n_tracks)),
            sync_after_layers=list(range(8)), track_group=None,
        ).eval()
        pt.load_track_state_dicts({i: tracks[i] for i in range(n_tracks)}, strict=False)
        return pt

    return build, cfg


def test_attached_zero_init_estimator_is_noop_d1():
    # THE warm-start guarantee: attaching a zero-init estimator must leave the D=1
    # forward bit-identical (the split seam at a zero substitute == the stock layer).
    build, cfg = _build_pt_d1()
    pt_base = build()
    pt_seam = build()
    pt_seam.cross_head = CrossHeadEstimator(
        num_layers=8, hidden_size=cfg.hidden_size, backend="fresh", rank=8
    ).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        h0, _ = pt_base(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        h1, _ = pt_seam(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    assert torch.allclose(h0, h1, atol=1e-5)


def test_attached_open_gate_estimator_changes_output_d1():
    # Sanity: once the gate opens the seam actually alters the forward (it isn't a
    # silent no-op for some unrelated reason).
    build, cfg = _build_pt_d1()
    pt_seam = build()
    est = CrossHeadEstimator(num_layers=8, hidden_size=cfg.hidden_size, backend="fresh", rank=8).eval()
    with torch.no_grad():
        for lin in est.up:
            torch.nn.init.normal_(lin.weight, std=0.1)
        est.gate.fill_(1.0)
    pt_seam.cross_head = est
    pt_base = build()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        h0, _ = pt_base(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        h1, _ = pt_seam(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    assert not torch.allclose(h0, h1, atol=1e-3)


def test_distill_step_trains_estimator_fresh():
    # End-to-end (CPU, K=2 single-process): a distill step with a fresh estimator
    # attached must run and deliver gradients to BOTH the gate (via the end-to-end
    # block-MSE) and the predictor (via the direct predict loss).
    import torch.nn as nn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from pt_converter.model.pt_model import PTWrappedModel
    from pt_converter.slicer.convert import slice_model_to_tracks
    from pt_converter.train.distill import DistillConfig, distill_step
    from pt_converter.train.teacher import HookedTeacher

    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4, vocab_size=128, rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(dense, n_tracks=2, sync_block_depth=1, text_config_attr="config")
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=list(range(8)), track_group=None,
    )
    student.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    student.cross_head = CrossHeadEstimator(
        num_layers=8, hidden_size=cfg.hidden_size, backend="fresh", rank=8
    )
    student.train()
    teacher = HookedTeacher(text_model=dense, lm_head=lm_head, sync_layer_indices=list(range(8)))

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones((1, 16), dtype=torch.long),
        "labels": ids.clone(),
    }
    dcfg = DistillConfig(
        sync_layer_indices=tuple(range(8)), lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0, intra_window_mse=True,
        lambda_cross_head_predict=1.0,
    )
    out = distill_step(student, teacher, batch, dcfg, compute_klce_metrics=False)
    assert out["cross_head_predict"].item() > 0.0
    # The gate gets gradient even at its zero init (∂/∂g of g*(ghat-y) ∝ ghat-y ≠ 0).
    assert student.cross_head.gate.grad is not None
    assert student.cross_head.gate.grad.abs().sum().item() > 0.0
    # The predictor (down/up) gets gradient from the predict loss.
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for p in student.cross_head.down.parameters()
    )


def test_distill_step_trains_estimator_stale_corr():
    # stale_corr through the full distill step: the predict loss (relMSE(ghat, ΣY))
    # trains the drift predictor; the gate trains end-to-end. Mirrors the fresh test.
    import torch.nn as nn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from pt_converter.model.pt_model import PTWrappedModel
    from pt_converter.slicer.convert import slice_model_to_tracks
    from pt_converter.train.distill import DistillConfig, distill_step
    from pt_converter.train.teacher import HookedTeacher

    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4, vocab_size=128, rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(dense, n_tracks=2, sync_block_depth=1, text_config_attr="config")
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=list(range(8)), track_group=None,
    )
    student.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    student.cross_head = CrossHeadEstimator(
        num_layers=8, hidden_size=cfg.hidden_size, backend="stale_corr", rank=8
    )
    student.train()
    teacher = HookedTeacher(text_model=dense, lm_head=lm_head, sync_layer_indices=list(range(8)))

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones((1, 16), dtype=torch.long),
        "labels": ids.clone(),
    }
    dcfg = DistillConfig(
        sync_layer_indices=tuple(range(8)), lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0, intra_window_mse=True,
        lambda_cross_head_predict=1.0,
    )
    out = distill_step(student, teacher, batch, dcfg, compute_klce_metrics=False)
    assert out["cross_head_predict"].item() > 0.0
    assert student.cross_head.gate.grad is not None
    assert student.cross_head.gate.grad.abs().sum().item() > 0.0
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for p in student.cross_head.down.parameters()
    )


_SEAM_METRICS = ["seam_total", "oracle_seam", "substitute_err", "residual_drift", "stale_ratio"]


def test_accumulate_seam_exact_substitute_zero_headroom():
    # With the ORACLE substitute (S_k = ΣY − y_k ⇒ mlp_in_k = X + ΣY) the injected
    # input equals the perfect-substitute input, so substitute_err == 0 and
    # seam_total == oracle_seam (headroom 0) regardless of the teacher — the core
    # identity the headroom read relies on.
    build, cfg = _build_pt_d1()
    pt = build()
    H = cfg.hidden_size
    B, T = 1, 5
    X = torch.randn(B, T, H)
    y0, y1 = torch.randn(B, T, H), torch.randn(B, T, H)
    sum_y = y0 + y1
    per_track_h, y_list = [X, X], [y0, y1]
    h_attn = [X + y0, X + y1]
    subs = [sum_y - y0, sum_y - y1]                       # oracle substitute
    pt._seam_teacher = {0: (torch.randn(B, T, H), torch.randn(B, T, H))}  # arbitrary teacher
    pt._seam_sums = {n: torch.zeros(8) for n in _SEAM_METRICS}
    pt._accumulate_seam(0, per_track_h, h_attn, y_list, sum_y, subs)
    s = pt._seam_sums
    assert s["substitute_err"][0].abs() < 1e-5
    assert torch.allclose(s["seam_total"][0], s["oracle_seam"][0], atol=1e-5)


def test_accumulate_seam_perfect_inputs_all_zero():
    # Teacher matches exactly (X_t = X, Y_t = ΣY), oracle substitute, and
    # time-constant per-track y (zero cross-track staleness) ⇒ ALL five metrics 0.
    build, cfg = _build_pt_d1()
    pt = build()
    H = cfg.hidden_size
    B, T = 1, 5
    X = torch.randn(B, T, H)
    y0 = torch.randn(B, 1, H).expand(B, T, H).contiguous()
    y1 = torch.randn(B, 1, H).expand(B, T, H).contiguous()
    sum_y = y0 + y1
    per_track_h, y_list = [X, X], [y0, y1]
    h_attn = [X + y0, X + y1]
    subs = [sum_y - y0, sum_y - y1]
    pt._seam_teacher = {0: (X.clone(), sum_y.clone())}
    pt._seam_sums = {n: torch.zeros(8) for n in _SEAM_METRICS}
    pt._accumulate_seam(0, per_track_h, h_attn, y_list, sum_y, subs)
    for n in _SEAM_METRICS:
        assert pt._seam_sums[n][0].abs() < 1e-4, n


def test_seam_mlp_none_is_stock_layer():
    # seam_mlp(s=None) must equal the stock post-attn + MLP residual path.
    import torch.nn as nn
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    class _TinyLayer(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.post_attention_layernorm = Qwen3_5RMSNorm(H, eps=1e-6)
            self.mlp = nn.Linear(H, H, bias=False)

    H = 8
    layer = _TinyLayer(H)
    h_attn = torch.randn(1, 4, H)
    ref = h_attn + layer.mlp(layer.post_attention_layernorm(h_attn))
    assert torch.allclose(seam_mlp(layer, h_attn, None), ref, atol=1e-6)
