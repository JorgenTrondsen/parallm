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
    _depth_weights,
    distill_step,
)
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


# ----- depth-weighted block-MSE -----

def test_depth_weights_mean_one_and_monotone():
    """γ>0 gives a strictly increasing, mean-1 weight ramp; γ=0 (and L=1) is all-ones."""
    L = 8
    w = _depth_weights(L, gamma=3.0)
    assert len(w) == L
    assert abs(sum(w) / L - 1.0) < 1e-9              # mean exactly 1 (preserves magnitude)
    assert all(w[i] < w[i + 1] for i in range(L - 1))  # deeper layers weigh more
    assert w[0] < 1.0 < w[-1]                         # shallow down-weighted, deep up-weighted
    # γ=0 ⇒ uniform; single-layer ⇒ uniform regardless of γ.
    assert _depth_weights(L, gamma=0.0) == [1.0] * L
    assert _depth_weights(1, gamma=5.0) == [1.0]


def test_block_depth_weight_zero_matches_unweighted():
    """block_depth_weight=0.0 must reproduce the unweighted path bit-for-bit, on
    both the boundary-only and intra-window supervision paths."""
    cfg = _tiny_config()
    batch = _batch(cfg)
    for intra in (False, True):
        s_a, t_a, man = _build_teacher_student(cfg, sync_block_depth=4, teacher_hook_all=intra)
        base = DistillConfig(
            sync_layer_indices=tuple(man.sync_layer_indices),
            normalize_block_mse=True, intra_window_mse=intra,
        )
        out_a = distill_step(s_a, t_a, batch, base, student_forcing_prob=0.0, forcing_seed=(42, 0))

        s_b, t_b, _ = _build_teacher_student(cfg, sync_block_depth=4, teacher_hook_all=intra)
        weighted_zero = DistillConfig(
            sync_layer_indices=tuple(man.sync_layer_indices),
            normalize_block_mse=True, intra_window_mse=intra, block_depth_weight=0.0,
        )
        out_b = distill_step(s_b, t_b, batch, weighted_zero, student_forcing_prob=0.0, forcing_seed=(42, 0))
        for key in ("total", "block_mse", "kl", "ce"):
            assert torch.allclose(out_a[key], out_b[key], atol=1e-6), (intra, key)


def test_block_depth_weight_changes_block_loss_and_grads():
    """γ>0 actually re-weights: the block_mse term differs from the unweighted run,
    and gradients still flow (the run stays finite)."""
    cfg = _tiny_config()
    batch = _batch(cfg)

    s0, t0, man = _build_teacher_student(cfg, sync_block_depth=4, teacher_hook_all=True)
    out0 = distill_step(
        s0, t0, batch,
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      normalize_block_mse=True, intra_window_mse=True, block_depth_weight=0.0),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )

    sg, tg, _ = _build_teacher_student(cfg, sync_block_depth=4, teacher_hook_all=True)
    sg.zero_grad(set_to_none=True)
    outg = distill_step(
        sg, tg, batch,
        DistillConfig(sync_layer_indices=tuple(man.sync_layer_indices),
                      normalize_block_mse=True, intra_window_mse=True, block_depth_weight=4.0),
        student_forcing_prob=0.0, forcing_seed=(42, 0),
    )
    assert torch.isfinite(outg["block_mse"]).all()
    assert not torch.allclose(out0["block_mse"], outg["block_mse"])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in sg.parameters())
