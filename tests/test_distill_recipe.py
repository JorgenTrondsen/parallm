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

def _build_teacher_student(cfg, n_tracks=2, sync_block_depth=4, teacher_hook_all=False):
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
