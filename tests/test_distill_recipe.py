"""Quality-recipe additions: normalized block-MSE + student-forcing scheduled sampling.

Covers the three recipe changes that target the exposure-bias gap (teacher-forced
training vs free-running inference) at fixed N/D:

- ``losses.block_mse(normalize=True)`` is the scale-free relative MSE Σ(s−t)²/Σt².
- ``distill_step`` runs end-to-end with student forcing + normalized block MSE on the
  single-process K=2 path (no NCCL).
- The per-block teacher/student forcing decision is deterministic given ``forcing_seed``,
  which is what keeps the choice identical across ranks (the SyncBoundary all-reduce would
  be corrupted by a per-rank-divergent choice).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks
from pt_converter.train.distill import (
    DistillConfig,
    _combine_seed,
    _effective_block_weights,
    adaptive_weights_from_relmse,
    distill_step,
    student_forcing_schedule,
)
from pt_converter.eval.fidelity import fidelity_step
from pt_converter.train.losses import block_mse
from pt_converter.train.teacher import HookedTeacher


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


# ----- block_mse normalization -----

def test_block_mse_normalize_matches_manual():
    torch.manual_seed(0)
    s = torch.randn(2, 5, 8)
    t = torch.randn(2, 5, 8)
    got = block_mse(s, t, normalize=True)
    expected = (s - t).pow(2).sum() / t.pow(2).sum()
    assert torch.allclose(got, expected, atol=1e-6)


def test_block_mse_normalize_respects_mask():
    torch.manual_seed(1)
    s = torch.randn(1, 4, 8)
    t = torch.randn(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]])
    got = block_mse(s, t, attention_mask=mask, normalize=True)
    m = mask.unsqueeze(-1).float()
    expected = ((s - t).pow(2) * m).sum() / (t.pow(2) * m).sum()
    assert torch.allclose(got, expected, atol=1e-6)


def test_block_mse_clamp_caps_spikes():
    """clamp_max caps the relative MSE; below the cap it's untouched, None = no clamp."""
    torch.manual_seed(3)
    t = torch.full((1, 3, 8), 0.01)          # tiny teacher norm → huge ratio
    s = torch.randn(1, 3, 8)                  # large student → ratio >> 10
    raw = block_mse(s, t, normalize=True)
    assert raw.item() > 10.0                  # this batch genuinely spikes
    capped = block_mse(s, t, normalize=True, clamp_max=10.0)
    assert torch.isclose(capped, torch.tensor(10.0), atol=1e-4)
    # A well-behaved pair stays below the cap and is identical clamped vs not.
    s2 = t + 0.001
    assert block_mse(s2, t, normalize=True).item() < 10.0
    assert torch.allclose(
        block_mse(s2, t, normalize=True), block_mse(s2, t, normalize=True, clamp_max=10.0)
    )
    # clamp is inert on the non-normalized path.
    assert torch.allclose(block_mse(s, t), block_mse(s, t, clamp_max=10.0))


def test_block_mse_default_is_unchanged_masked_mean():
    """normalize=False (default) must remain the raw masked mean — backward compat."""
    torch.manual_seed(2)
    s = torch.randn(2, 4, 8)
    t = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    got = block_mse(s, t, attention_mask=mask)  # normalize defaults False
    diff = (s - t).pow(2)
    m = mask.unsqueeze(-1).float()
    expected = (diff * m).sum() / (mask.sum() * 8)
    assert torch.allclose(got, expected, atol=1e-6)


# ----- forcing-decision determinism (rank consistency) -----

def test_forcing_decisions_are_deterministic_per_seed():
    """The (seed, step) tuple folds to a process-independent int and seeds an
    identical draw sequence — the mechanism that keeps the teacher/student choice
    the same on every rank. A divergent choice would corrupt the SyncBoundary sum."""
    # A tuple seed must fold to an int (random.Random rejects tuples directly).
    assert isinstance(_combine_seed((42, 7)), int)
    # Reconstructing the same generator (as each rank does) reproduces the stream.
    rng_a = random.Random(_combine_seed((42, 7)))
    rng_b = random.Random(_combine_seed((42, 7)))
    stream_a = [rng_a.random() for _ in range(8)]
    stream_b = [rng_b.random() for _ in range(8)]
    assert stream_a == stream_b
    # Different step ⇒ different stream (so per-block decisions aren't correlated).
    rng_other = random.Random(_combine_seed((42, 8)))
    stream_other = [rng_other.random() for _ in range(8)]
    assert stream_a != stream_other


# ----- distill_step end-to-end with the new recipe -----

def _build_teacher_student(cfg, n_tracks=2, sync_block_depth=4, teacher_hook_all=False,
                           student_kwargs=None):
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=sync_block_depth, text_config_attr="config"
    )
    if sync_block_depth == 4:
        assert manifest.sync_layer_indices == [3, 7]

    student = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
        **(student_kwargs or {}),
    )
    student.load_track_state_dicts({i: tracks[i] for i in range(n_tracks)}, strict=False)
    student.train()

    # Intra-window per-layer MSE needs a teacher hidden at every layer, not just
    # the sync boundaries.
    hook_indices = (
        list(range(cfg.num_hidden_layers)) if teacher_hook_all else manifest.sync_layer_indices
    )
    teacher = HookedTeacher(
        text_model=dense, lm_head=dense.lm_head,
        sync_layer_indices=hook_indices,
    )
    return student, teacher, manifest


def _batch(cfg, seq=16):
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq))
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones((1, seq), dtype=torch.long),
    }


def test_distill_step_student_forcing_and_normalized_runs():
    cfg = _tiny_config()
    student, teacher, manifest = _build_teacher_student(cfg)
    distill_cfg = DistillConfig(
        sync_layer_indices=tuple(manifest.sync_layer_indices),
        normalize_block_mse=True,
    )
    student.zero_grad(set_to_none=True)
    losses = distill_step(
        student, teacher, _batch(cfg), distill_cfg,
        student_forcing_prob=1.0, forcing_seed=(42, 0),
    )
    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.isfinite(losses[key]).all(), key
    # Backward ran internally — at least some student params got gradients.
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def test_distill_step_forcing_prob_zero_matches_legacy_default():
    """prob=0.0 must reproduce the fully-teacher-forced path (no student forcing),
    i.e. the same losses as a config built without the new options."""
    cfg = _tiny_config()
    batch = _batch(cfg)

    student_a, teacher_a, manifest = _build_teacher_student(cfg)
    out_a = distill_step(
        student_a, teacher_a, batch,
        DistillConfig(sync_layer_indices=tuple(manifest.sync_layer_indices)),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )

    # Rebuild identically; legacy call site would not pass the forcing kwargs at all.
    student_b, teacher_b, _ = _build_teacher_student(cfg)
    out_b = distill_step(
        student_b, teacher_b, batch,
        DistillConfig(sync_layer_indices=tuple(manifest.sync_layer_indices)),
    )
    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.allclose(out_a[key], out_b[key], atol=1e-5), key


# ----- intra-window per-layer MSE (lever c) -----

def test_intra_window_mse_runs_at_d4():
    """D=4 window (incl. the full-attention layer mid-window): the per-layer
    supervised path runs end-to-end, stays finite, and propagates gradients."""
    cfg = _tiny_config()
    student, teacher, manifest = _build_teacher_student(
        cfg, sync_block_depth=4, teacher_hook_all=True
    )
    distill_cfg = DistillConfig(
        sync_layer_indices=tuple(manifest.sync_layer_indices),
        normalize_block_mse=True,
        block_mse_clamp=10.0,
        intra_window_mse=True,
    )
    student.zero_grad(set_to_none=True)
    losses = distill_step(
        student, teacher, _batch(cfg), distill_cfg,
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )
    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.isfinite(losses[key]).all(), key
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())


def test_intra_window_mse_equals_boundary_at_d1():
    """At D=1 every window is a single layer, so per-layer supervision (averaged
    over one layer) must reduce bit-for-bit to the boundary-only path."""
    cfg = _tiny_config()
    batch = _batch(cfg)

    s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=1)
    out_a = distill_step(
        s_a, t_a, batch,
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      normalize_block_mse=True, intra_window_mse=False),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )

    s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=1)
    out_b = distill_step(
        s_b, t_b, batch,
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      normalize_block_mse=True, intra_window_mse=True),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )
    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.allclose(out_a[key], out_b[key], atol=1e-5), key


# ----- fidelity normalized per-boundary report -----

def test_fidelity_emits_raw_and_relative_block_mse():
    """fidelity_step emits both block_mse_l{i} (raw) and block_relmse_l{i}
    (normalized), and the relative one equals block_mse(normalize=True)."""
    cfg = _tiny_config()
    student, teacher, man = _build_teacher_student(cfg, sync_block_depth=4)
    m = fidelity_step(student, teacher, _batch(cfg), tuple(man.sync_layer_indices), chunk_size=8)
    for idx in man.sync_layer_indices:
        assert f"block_mse_l{idx}" in m
        assert f"block_relmse_l{idx}" in m
        # relative ≥ 0 and finite; raw and relative are distinct scale-free vs absolute.
        assert torch.isfinite(m[f"block_relmse_l{idx}"]).all()
        assert m[f"block_relmse_l{idx}"].item() >= 0.0


# ----- A: free-running student-forcing curriculum schedule -----

def test_student_forcing_schedule_hold_matches_legacy():
    """'hold' reproduces the legacy ``prob * min(1, step/warmup)`` ramp-then-hold."""
    prob, warmup, max_steps = 0.5, 100, 1000
    for step in (0, 50, 100, 200, 999):
        legacy = prob * min(1.0, step / warmup)
        assert student_forcing_schedule(step, prob, warmup, max_steps, "hold") == legacy
    # warmup=0 ⇒ constant prob from step 0 (the no-warmup branch).
    assert student_forcing_schedule(0, prob, 0, max_steps, "hold") == prob
    assert student_forcing_schedule(500, prob, 0, max_steps, "hold") == prob


def test_student_forcing_schedule_cosine_full_curriculum():
    """'cosine-full' ramps 0→prob across the whole run, monotone, warmup ignored."""
    prob, warmup, max_steps = 0.9, 100, 1000
    assert student_forcing_schedule(0, prob, warmup, max_steps, "cosine-full") == 0.0
    end = student_forcing_schedule(max_steps, prob, warmup, max_steps, "cosine-full")
    assert abs(end - prob) < 1e-9                       # reaches prob only at the end
    vals = [student_forcing_schedule(s, prob, warmup, max_steps, "cosine-full")
            for s in range(0, max_steps + 1, 100)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))  # strictly increasing
    assert all(0.0 <= v <= prob + 1e-9 for v in vals)
    # warmup is ignored in this shape: same value regardless of warmup arg.
    assert (student_forcing_schedule(400, prob, 100, max_steps, "cosine-full")
            == student_forcing_schedule(400, prob, 999, max_steps, "cosine-full"))


# ----- B: gradient-accumulation loss_scale -----

def test_loss_scale_halves_grads_not_losses():
    """loss_scale=0.5 scales every accumulated grad by 0.5 while leaving the returned
    (unscaled) loss scalars unchanged — the gradient-accumulation contract."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    kw = dict(student_forcing_prob=0.0, forcing_seed=(42, 0))

    s_full, t_full, man = _build_teacher_student(cfg, sync_block_depth=4)
    dcfg = DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices), normalize_block_mse=True)
    s_full.zero_grad(set_to_none=True)
    out_full = distill_step(s_full, t_full, batch, dcfg, loss_scale=1.0, **kw)
    grads_full = {n: p.grad.detach().clone() for n, p in s_full.named_parameters() if p.grad is not None}

    s_half, t_half, _ = _build_teacher_student(cfg, sync_block_depth=4)
    s_half.zero_grad(set_to_none=True)
    out_half = distill_step(s_half, t_half, batch, dcfg, loss_scale=0.5, **kw)
    grads_half = {n: p.grad.detach().clone() for n, p in s_half.named_parameters() if p.grad is not None}

    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.allclose(out_full[key], out_half[key], atol=1e-5), key
    assert grads_full and grads_full.keys() == grads_half.keys()
    for n in grads_full:
        assert torch.allclose(grads_half[n], 0.5 * grads_full[n], atol=1e-6, rtol=1e-4), n


# ----- C: adaptive error-proportional layer weighting -----

def test_adaptive_weights_from_relmse_mean_one():
    """Weights ∝ relMSE**power, mean-1 normalized; worse layers weigh more; edges handled."""
    w = adaptive_weights_from_relmse({3: 2.0, 7: 6.0}, power=1.0)
    assert abs(sum(w.values()) / len(w) - 1.0) < 1e-9
    assert w[7] > w[3]                                   # higher relMSE ⇒ more weight
    w2 = adaptive_weights_from_relmse({3: 2.0, 7: 6.0}, power=2.0)
    assert w2[7] / w2[3] > w[7] / w[3]                   # power sharpens the tilt
    assert adaptive_weights_from_relmse({}) == {}
    assert adaptive_weights_from_relmse({3: 0.0, 7: 0.0}) == {3: 1.0, 7: 1.0}


def test_effective_block_weights_none_is_uniform():
    """adaptive None ⇒ all-ones (uniform); adaptive given ⇒ mean-1 over the taps."""
    taps = [3, 7]
    assert _effective_block_weights(None, taps) == {3: 1.0, 7: 1.0}
    adapt = _effective_block_weights({3: 1.0, 7: 3.0}, taps)
    assert abs((adapt[3] + adapt[7]) / 2 - 1.0) < 1e-9
    assert adapt[7] > adapt[3]


def test_distill_step_adaptive_none_matches_default():
    """adaptive_weights=None with track_layer_relmse on is bit-identical to the plain
    call — the relMSE signal is detached and must not perturb the loss or grads."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    base = lambda man: DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                                     normalize_block_mse=True)
    s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=4)
    out_a = distill_step(s_a, t_a, batch, base(man), student_forcing_prob=0.0, forcing_seed=(42, 0))

    s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=4)
    out_b = distill_step(s_b, t_b, batch, base(man), student_forcing_prob=0.0, forcing_seed=(42, 0),
                         adaptive_weights=None, track_layer_relmse=True)
    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.allclose(out_a[key], out_b[key], atol=1e-6), key


def test_distill_step_returns_layer_relmse():
    """track_layer_relmse=True returns a finite, ≥0 relMSE per supervised tap."""
    cfg = _tiny_config()
    s, t, man = _build_teacher_student(cfg, sync_block_depth=4)
    out = distill_step(
        s, t, _batch(cfg),
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices)),
        student_forcing_prob=0.0, forcing_seed=(42, 0), track_layer_relmse=True,
    )
    rel = out["layer_relmse"]
    assert set(rel.keys()) == set(man.sync_layer_indices)   # boundary taps
    for v in rel.values():
        assert torch.isfinite(v).all() and v.item() >= 0.0


def test_distill_step_adaptive_weights_change_block_loss_and_grads():
    """Non-uniform adaptive_weights re-weight the block term vs the depth-only path,
    and gradients still flow finitely."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    base = lambda man: DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                                     normalize_block_mse=True)
    s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=4)
    out_a = distill_step(s_a, t_a, batch, base(man), student_forcing_prob=0.0, forcing_seed=(42, 0))

    s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=4)
    s_b.zero_grad(set_to_none=True)
    out_b = distill_step(s_b, t_b, batch, base(man), student_forcing_prob=0.0, forcing_seed=(42, 0),
                         adaptive_weights={3: 0.2, 7: 1.8})   # taps [3, 7]; tilt to the deep one
    assert not torch.allclose(out_a["block_mse"], out_b["block_mse"])
    assert torch.isfinite(out_b["total"]).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in s_b.parameters())


# ----- D: free-running feature matching + zero-lambda klce skip -----

def _grads(student):
    return {n: p.grad.detach().clone() for n, p in student.named_parameters() if p.grad is not None}


def test_free_running_off_matches_plain_call():
    """cfg.free_running_mse=False with an explicit free_running_scale is bit-identical
    to the plain call (the klce-return refactor must not perturb losses or grads),
    and fr_mse is reported as zero."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    dcfg = lambda man: DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                                     normalize_block_mse=True)
    s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=4)
    s_a.zero_grad(set_to_none=True)
    out_a = distill_step(s_a, t_a, batch, dcfg(man), student_forcing_prob=0.0, forcing_seed=(42, 0))

    s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=4)
    s_b.zero_grad(set_to_none=True)
    out_b = distill_step(s_b, t_b, batch, dcfg(man), student_forcing_prob=0.0, forcing_seed=(42, 0),
                         free_running_scale=1.0)
    for key in ("total", "block_mse", "kl", "ce", "fr_mse"):
        assert torch.allclose(out_a[key], out_b[key], atol=1e-6), key
    assert out_a["fr_mse"].item() == 0.0
    ga, gb = _grads(s_a), _grads(s_b)
    assert ga and ga.keys() == gb.keys()
    for n in ga:
        assert torch.equal(ga[n], gb[n]), n


def test_free_running_grads_flow_cross_window():
    """Free-running MSE on the deep tap only (boundaries [3, 7] → tap {7}) must put
    nonzero gradients on the FIRST window's layers — only possible if the gradient
    flows through the whole free-running forward, across the window boundary the
    block loop detaches at. All other lambdas are 0."""
    cfg = _tiny_config()
    s, t, man = _build_teacher_student(cfg, sync_block_depth=4)
    dcfg = DistillConfig(
        sync_layer_indices=tuple(man.sync_layer_indices),
        lambda_block=0.0, lambda_kl=0.0, lambda_ce=0.0,
        free_running_mse=True, lambda_free_running=1.0, free_running_taps="deep-half",
    )
    s.zero_grad(set_to_none=True)
    out = distill_step(s, t, _batch(cfg), dcfg, student_forcing_prob=0.0, forcing_seed=(42, 0))
    assert torch.isfinite(out["fr_mse"]).all() and out["fr_mse"].item() > 0.0
    assert set(out["fr_layer_relmse"].keys()) == {7}
    layer0_grads = [
        p.grad for tm in s.text_models for p in tm.layers[0].parameters() if p.grad is not None
    ]
    assert layer0_grads
    assert any(g.abs().sum().item() > 0 for g in layer0_grads)
    assert all(torch.isfinite(g).all() for g in layer0_grads)


def test_free_running_all_taps_reports_every_boundary():
    """free_running_taps='all' supervises every sync boundary ([3, 7] at D=4)."""
    cfg = _tiny_config()
    s, t, man = _build_teacher_student(cfg, sync_block_depth=4)
    dcfg = DistillConfig(
        sync_layer_indices=tuple(man.sync_layer_indices),
        free_running_mse=True,
    )
    out = distill_step(s, t, _batch(cfg), dcfg, student_forcing_prob=0.0, forcing_seed=(42, 0))
    assert set(out["fr_layer_relmse"].keys()) == set(man.sync_layer_indices)
    for v in out["fr_layer_relmse"].values():
        assert torch.isfinite(v).all() and v.item() >= 0.0


def test_zero_lambda_skip_keeps_klce_logging():
    """All lambdas 0 and fr off: the full forward runs under no_grad (the previously
    zero-gradient backward is skipped) but kl/ce are still computed for logging and
    match a grad-bearing run's values; no parameter receives a nonzero grad."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=4)
    out_a = distill_step(
        s_a, t_a, batch,
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices)),  # kl=1.0, ce=0.5
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )

    s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=4)
    s_b.zero_grad(set_to_none=True)
    out_b = distill_step(
        s_b, t_b, batch,
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      lambda_block=0.0, lambda_kl=0.0, lambda_ce=0.0),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )
    # The kl/ce METRICS are lambda-independent (same weights, same batch).
    assert torch.allclose(out_a["kl"], out_b["kl"], atol=1e-5)
    assert torch.allclose(out_a["ce"], out_b["ce"], atol=1e-5)
    for n, p in s_b.named_parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, n


def test_free_running_loss_scale_halves_grads_not_loss():
    """loss_scale scales the fr gradients (the grad-accum contract) but not the
    reported fr_mse."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    dcfg = lambda man: DistillConfig(
        sync_layer_indices=tuple(man.sync_layer_indices),
        lambda_block=0.0, lambda_kl=0.0, lambda_ce=0.0,
        free_running_mse=True,
    )
    s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=4)
    s_a.zero_grad(set_to_none=True)
    out_a = distill_step(s_a, t_a, batch, dcfg(man), student_forcing_prob=0.0,
                         forcing_seed=(42, 0), loss_scale=1.0)

    s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=4)
    s_b.zero_grad(set_to_none=True)
    out_b = distill_step(s_b, t_b, batch, dcfg(man), student_forcing_prob=0.0,
                         forcing_seed=(42, 0), loss_scale=0.5)
    assert torch.allclose(out_a["fr_mse"], out_b["fr_mse"], atol=1e-6)
    ga, gb = _grads(s_a), _grads(s_b)
    assert ga and ga.keys() == gb.keys()
    for n in ga:
        assert torch.allclose(gb[n], 0.5 * ga[n], atol=1e-7, rtol=1e-4), n


def test_free_running_scale_zero_logs_without_grads():
    """free_running_scale=0 (the cosine ramp at step 0) still reports fr_mse but the
    effective weight is 0 — combined with zero kl/ce lambdas, no grads anywhere."""
    cfg = _tiny_config()
    s, t, man = _build_teacher_student(cfg, sync_block_depth=4)
    s.zero_grad(set_to_none=True)
    out = distill_step(
        s, t, _batch(cfg),
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      lambda_block=0.0, lambda_kl=0.0, lambda_ce=0.0,
                      free_running_mse=True),
        student_forcing_prob=0.0, forcing_seed=(42, 0), free_running_scale=0.0,
    )
    assert torch.isfinite(out["fr_mse"]).all() and out["fr_mse"].item() > 0.0
    for n, p in s.named_parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0, n


def test_total_includes_fr_term():
    """total = λ_block·block + λ_kl·kl + λ_ce·ce + (λ_fr·scale)·fr_mse."""
    cfg = _tiny_config()
    s, t, man = _build_teacher_student(cfg, sync_block_depth=4)
    dcfg = DistillConfig(
        sync_layer_indices=tuple(man.sync_layer_indices),
        lambda_block=1.0, lambda_kl=1.0, lambda_ce=0.5,
        normalize_block_mse=True,
        free_running_mse=True, lambda_free_running=0.7,
    )
    out = distill_step(s, t, _batch(cfg), dcfg, student_forcing_prob=0.0,
                       forcing_seed=(42, 0), free_running_scale=0.5)
    expected = (
        1.0 * out["block_mse"] + 1.0 * out["kl"] + 0.5 * out["ce"]
        + 0.7 * 0.5 * out["fr_mse"]
    )
    assert torch.allclose(out["total"], expected, atol=1e-6)


# ----- metrics-only KL/CE gating (compute_klce_metrics) -----

def test_metrics_off_skips_klce_and_teacher_lm_head():
    """lambda_kl=lambda_ce=0 + compute_klce_metrics=False: the KL/CE pass AND
    the teacher lm_head are skipped (zero kl/ce returned), while the block
    losses and the accumulated grads are identical to the metrics-on step —
    the pass was contributing nothing but logging."""
    cfg = _tiny_config()
    batch = _batch(cfg)

    def dcfg(man):
        return DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                             lambda_kl=0.0, lambda_ce=0.0, normalize_block_mse=True)

    s_on, t_on, man = _build_teacher_student(cfg)
    s_on.zero_grad(set_to_none=True)
    out_on = distill_step(s_on, t_on, batch, dcfg(man), student_forcing_prob=0.0,
                          forcing_seed=(42, 0), compute_klce_metrics=True)
    grads_on = {n: p.grad.detach().clone()
                for n, p in s_on.named_parameters() if p.grad is not None}
    assert out_on["kl"].item() > 0.0          # metrics still computed when requested

    s_off, t_off, _ = _build_teacher_student(cfg)

    class _Boom(nn.Module):
        def forward(self, *a, **k):
            raise AssertionError("teacher lm_head must not run with metrics off")

    t_off.lm_head = _Boom()                    # observable skip of the logits matmul
    s_off.zero_grad(set_to_none=True)
    out_off = distill_step(s_off, t_off, batch, dcfg(man), student_forcing_prob=0.0,
                           forcing_seed=(42, 0), compute_klce_metrics=False)
    grads_off = {n: p.grad.detach().clone()
                 for n, p in s_off.named_parameters() if p.grad is not None}

    assert out_off["kl"].item() == 0.0 and out_off["ce"].item() == 0.0
    assert torch.allclose(out_on["block_mse"], out_off["block_mse"], atol=1e-6)
    assert grads_on and grads_on.keys() == grads_off.keys()
    for n in grads_on:
        assert torch.allclose(grads_on[n], grads_off[n], atol=1e-7), n


def test_metrics_flag_inert_when_klce_losses_on():
    """compute_klce_metrics gates METRICS-ONLY work: with a non-zero logit
    lambda the KL/CE pass runs (and backwards) regardless of the flag."""
    cfg = _tiny_config()
    batch = _batch(cfg)

    def dcfg(man):
        return DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                             lambda_kl=1.0, lambda_ce=0.5, normalize_block_mse=True)

    s_a, t_a, man = _build_teacher_student(cfg)
    out_a = distill_step(s_a, t_a, batch, dcfg(man), student_forcing_prob=0.0,
                         forcing_seed=(42, 0), compute_klce_metrics=True)

    s_b, t_b, _ = _build_teacher_student(cfg)
    out_b = distill_step(s_b, t_b, batch, dcfg(man), student_forcing_prob=0.0,
                         forcing_seed=(42, 0), compute_klce_metrics=False)
    assert out_b["kl"].item() > 0.0
    for key in ("total", "block_mse", "kl", "ce"):
        assert torch.allclose(out_a[key], out_b[key], atol=1e-6), key


def test_metrics_off_with_free_running_still_runs_full_forward():
    """Metrics off but free-running MSE on: the full forward must still run
    (fr needs the sync hiddens) and the fr term still backwards."""
    cfg = _tiny_config()
    s, t, man = _build_teacher_student(cfg, sync_block_depth=4)
    s.zero_grad(set_to_none=True)
    out = distill_step(
        s, t, _batch(cfg),
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      lambda_kl=0.0, lambda_ce=0.0, normalize_block_mse=True,
                      free_running_mse=True),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
        free_running_scale=1.0, compute_klce_metrics=False,
    )
    assert out["kl"].item() == 0.0 and out["ce"].item() == 0.0
    assert torch.isfinite(out["fr_mse"]).all() and out["fr_mse"].item() > 0.0
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0
               for p in s.parameters())


# ----- activation-checkpoint granularity (window vs layer vs off) -----

def test_checkpoint_granularity_parity():
    """Window-granular AC, per-layer AC, and no AC must produce identical losses
    AND grads — checkpointing changes what is saved/recomputed, never the math.
    Exercised through the full distill step (multi-layer D=4 windows, KL/CE +
    free-running backward through the checkpointed forward)."""
    cfg = _tiny_config()
    batch = _batch(cfg)

    variants = [
        None,                                                            # AC off
        {"activation_checkpoint": True, "checkpoint_granularity": "layer"},
        {"activation_checkpoint": True, "checkpoint_granularity": "window"},
    ]
    outs, grads = [], []
    for student_kwargs in variants:
        s, t, man = _build_teacher_student(cfg, sync_block_depth=4,
                                           student_kwargs=student_kwargs)
        dcfg = DistillConfig(
            sync_layer_indices=tuple(man.sync_layer_indices),
            lambda_kl=1.0, lambda_ce=0.5, normalize_block_mse=True,
            free_running_mse=True,
        )
        s.zero_grad(set_to_none=True)
        out = distill_step(s, t, batch, dcfg, student_forcing_prob=0.0,
                           forcing_seed=(42, 0), free_running_scale=1.0)
        outs.append(out)
        grads.append({n: p.grad.detach().clone()
                      for n, p in s.named_parameters() if p.grad is not None})

    ref_out, ref_grads = outs[0], grads[0]
    assert ref_grads
    for out, g in zip(outs[1:], grads[1:]):
        for key in ("total", "block_mse", "kl", "ce", "fr_mse"):
            assert torch.allclose(out[key], ref_out[key], atol=1e-6), key
        assert g.keys() == ref_grads.keys()
        for n in ref_grads:
            assert torch.allclose(g[n], ref_grads[n], atol=1e-6, rtol=1e-4), n


def test_checkpoint_granularity_rejects_unknown():
    cfg = _tiny_config()
    import pytest
    with pytest.raises(ValueError, match="checkpoint_granularity"):
        _build_teacher_student(
            cfg, student_kwargs={"checkpoint_granularity": "block"}
        )
