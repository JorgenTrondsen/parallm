"""Rails for the rebuilt d1b-heal trainer (post-attn walk + distill step).

Rail 1 — exact schedule ≡ dense: ``sync_phase="exact"`` (2 syncs/layer) must
reproduce the dense forward at N=4 — the frozen-slice TEACHER's correctness.

Rail 2 — N=1 parity: every phase walk must reduce to the dense forward when
the SyncBoundary is a no-op.

Rail 3 — zero-loss identity: distill_step(student == teacher slices, N=1)
must read block_mse ≈ 0 (any target-phase misalignment between the teacher
captures and the student taps breaks this instantly).

Rail 4 — chunked CE ≡ direct full-logits computation (value and the gradient
w.r.t. the hidden state).

NOTE: the old rail 5 (sf=1.0 chain ≡ deployed forward) went with student
forcing when it was removed 2026-07-30 — the block loop is now always
teacher-forced, so there is no free-running variant of it left to compare.
That check has no replacement here; `git log` has it.

Rail 6 — gradients flow to every track's layer params through the step, at both
the d1b and a fixed-D=2 (gapped) schedule.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.engine import _submodule
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks
from parallm.train.distill import (
    DistillConfig,
    distill_step,
    freeze_slice_teacher,
    ce_chunked,
    teacher_forward,
)
from parallm.train.losses import block_mse


# Unchunked reference objective for rail 4 — the whole point of ce_chunked is
# that it never materializes these (B, T, V) logits, so they live only here.
def lm_cross_entropy(logits, labels, ignore_index=-100):
    shift = logits[:, :-1, :]
    return F.cross_entropy(shift.reshape(-1, shift.size(-1)).float(),
                           labels[:, 1:].reshape(-1), ignore_index=ignore_index)


def _tiny_config(n_layers: int = 8):
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * (n_layers // 4),
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def _build(n_tracks: int, sync_after=None, n_layers: int = 8, fuse_size: int = 1,
           merge_group: int = 1, exec_groups: int = 1):
    cfg = _tiny_config(n_layers)
    torch.manual_seed(42)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    dense.eval()

    n_shards = n_tracks * merge_group
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_shards, sync_block_depth=4, text_config_attr="config"
    )
    states = dict(enumerate(tracks))
    if merge_group > 1:
        from parallm.adapters import get_adapter_for_config
        from parallm.model.merge import merge_track_states

        states = merge_track_states(
            get_adapter_for_config(cfg), cfg, n_shards, states, merge_group
        )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=list(sync_after) if sync_after is not None else manifest.sync_layer_indices,
        track_group=None,
        fuse_size=fuse_size,
        merge_group=merge_group,
        exec_groups=exec_groups,
    )
    pt.eval()
    pt.load_track_state_dicts(states, strict=False)
    return cfg, dense, tracks, pt


def _dense_logits(dense, input_ids, attention_mask=None):
    with torch.no_grad():
        out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return dense.lm_head(out.last_hidden_state)


def _batch(cfg, B=1, T=16, seed=123):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones((B, T), dtype=torch.long),
    }


def test_exact_phase_matches_dense_n4():
    cfg, dense, _tracks, pt = _build(n_tracks=4)
    pt.set_sync_phase("exact")
    batch = _batch(cfg)
    with torch.no_grad():
        pt_logits, _ = pt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ref = _dense_logits(dense, batch["input_ids"], batch["attention_mask"])
    max_abs = (ref - pt_logits).abs().max().item()
    assert max_abs < 5e-4, f"exact schedule drifts from dense by {max_abs}"


def test_post_attn_n1_matches_dense():
    cfg, dense, _tracks, pt = _build(n_tracks=1, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    batch = _batch(cfg)
    with torch.no_grad():
        pt_logits, _ = pt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ref = _dense_logits(dense, batch["input_ids"], batch["attention_mask"])
    max_abs = (ref - pt_logits).abs().max().item()
    assert max_abs < 1e-4, f"post-attn N=1 drifts from dense by {max_abs}"


def test_distill_step_zero_loss_identity_n1():
    cfg, dense, tracks, pt = _build(n_tracks=1, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    pt.train()

    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=1, local_track_ids=(0,),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts({0: tracks[0]}, strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    dcfg = DistillConfig(sync_layer_indices=tuple(range(8)))
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg)
    assert losses["block_mse"].item() < 1e-8, f"block_mse {losses['block_mse'].item()} not ~0 at N=1"
    assert torch.isfinite(losses["ce"]), "ce not finite"


def test_ce_chunked_matches_direct():
    torch.manual_seed(0)
    B, T, H, V = 2, 13, 8, 31
    lm_head = nn.Linear(H, V, bias=False)
    lm_head.weight.requires_grad_(False)
    hidden = torch.randn(B, T, H, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    lam_ce = 0.3

    ce, grad_h = ce_chunked(hidden, lm_head, labels, lambda_ce=lam_ce, chunk_size=5)

    s_logits = lm_head(hidden)
    ce_ref = lm_cross_entropy(s_logits, labels)
    assert torch.allclose(ce, ce_ref, atol=1e-5), (ce.item(), ce_ref.item())

    (lam_ce * ce_ref).backward()
    assert torch.allclose(grad_h, hidden.grad, atol=1e-5)


def test_distill_step_grads_flow_n4():
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    dcfg = DistillConfig(sync_layer_indices=tuple(range(8)))
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg)
    assert torch.isfinite(losses["total"])
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no layer param received gradient"
    # Teacher stayed frozen and untouched.
    assert all(not p_.requires_grad for p_ in teacher.parameters())


def test_distill_step_batched_merged_matches_looped_unfused():
    """The TF block loop must mirror `pt_model._run_post_attn_stack` sublayer for
    sublayer, or the block targets are measured against a different model than the
    one the free-running forward trains. The two walks share `mix`/`mlp`/`sync`
    adapters precisely so they cannot drift; this pins that they haven't.

    Checkpointing is ON so the batched path is exercised through recompute, which
    is where the two representations could disagree on saved-tensor order.
    """
    sched = [1, 3, 5, 7]
    cfg, _dense, tracks, looped = _build(n_tracks=4, sync_after=sched)
    _cfg2, _d2, _t2, merged = _build(n_tracks=1, sync_after=sched,
                                     merge_group=4, exec_groups=4)
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)  # exact schedule: layout-independent

    dcfg = DistillConfig(sync_layer_indices=tuple(sched))
    batch = _batch(cfg)
    out = []
    for pt in (looped, merged):
        pt.set_sync_phase("post-attn")
        pt.use_checkpoint = True
        pt.train()
        out.append(distill_step(pt, teacher, pt.lm_head, batch, dcfg))
    lo, me = out

    assert abs(lo["block_mse"].item() - me["block_mse"].item()) < 1e-5, (
        f"block_mse {lo['block_mse'].item()} vs {me['block_mse'].item()}")
    assert abs(lo["ce"].item() - me["ce"].item()) < 1e-4, (
        f"ce {lo['ce'].item()} vs {me['ce'].item()}")
    assert lo["layer_relmse"].keys() == me["layer_relmse"].keys()
    for i, r in lo["layer_relmse"].items():
        assert abs(r - me["layer_relmse"][i]) < 1e-5, f"tap {i}: {r} vs {me['layer_relmse'][i]}"
    # Non-vacuous: the taps carry real error, not ~0 from a degenerate walk.
    assert max(lo["layer_relmse"].values()) > 1e-6

    got = sum(1 for p_ in merged.text_models[0].layers.parameters()
              if p_.grad is not None and p_.grad.abs().sum() > 0)
    assert got > 0, "no gradient reached the merged weights"


def test_merged_shadow_tracks_live_parameters():
    """`MergedShadow.stacked` must read through to the parameter every call.

    Caching it looks free — `DenseShadow` does exactly that — but this provider's
    weights are being TRAINED, and `_regroup_qkv` returns a `cat`, i.e. a copy.
    A cached copy would pin the GDN qkv and conv weights at their step-0 values
    while the optimizer moved the real ones, and nothing in the loss would say so.
    Checked on the GDN params specifically, because they are the only ones whose
    unmerge is not a view.
    """
    _cfg, _dense, _tracks, merged = _build(n_tracks=1, sync_after=[1, 3, 5, 7],
                                           merge_group=4, exec_groups=4)
    for path in ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight",
                 "mlp.down_proj.weight", "self_attn.q_norm.weight"):
        li = 0 if "linear_attn" in path else 3  # layer 3 is the full-attention one
        before = merged.shadow.stacked(li, path).clone()
        with torch.no_grad():
            _submodule(merged.text_models[0].layers[li], path).add_(1.0)
        after = merged.shadow.stacked(li, path)
        assert torch.allclose(after, before + 1.0), f"{path} is stale — stacked() cached"


def test_distill_step_fixed_d2_schedule_n4():
    """The fixed-D=2 re-heal: student BUILT at the gapped schedule (own-carry at
    layers 0,2,4,6), checkpointing on. Grads must reach every track through both
    the own-carry TF loop and the free-running CE forward."""
    sched = [1, 3, 5, 7]
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=sched)
    pt.set_sync_phase("post-attn")
    pt.use_checkpoint = True  # the deployed FR-at-D>1 path is checkpointed
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    dcfg = DistillConfig(sync_layer_indices=tuple(sched))
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg)
    assert torch.isfinite(losses["total"])
    assert losses["block_mse"].item() > 0 and losses["ce"].item() > 0
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters()
                  if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no grad at the fixed-D=2 schedule"
