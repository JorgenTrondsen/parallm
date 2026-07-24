"""Rails for the rebuilt d1b-heal trainer (post-attn walk + distill step).

Rail 1 — exact schedule ≡ dense: ``sync_phase="exact"`` (2 syncs/layer) must
reproduce the dense forward at N=4 — the frozen-slice TEACHER's correctness.

Rail 2 — N=1 parity: every phase walk must reduce to the dense forward when
the SyncBoundary is a no-op.

Rail 3 — zero-loss identity: distill_step(student == teacher slices, N=1)
must read block_mse ≈ 0 and KL ≈ 0 (any target-phase misalignment between the
teacher captures and the student taps breaks this instantly).

Rail 4 — chunked KL/CE/logit-MSE ≡ direct full-logits computation (values and
the gradient w.r.t. the hidden state).

Rail 5 — sf=1.0 chain ≡ deployed forward: at student_forcing_prob=1.0 the TF
block loop's per-tap relMSEs must equal the free-running post-attn forward's
(the tag-era "the sf-1.0 chain IS the deployed forward" property).

Rail 6 — gradients flow to every track's layer params through the step, at both
the d1b and a fixed-D=2 (gapped) schedule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks
from parallm.train.distill import (
    DistillConfig,
    distill_step,
    freeze_slice_teacher,
    kl_ce_chunked,
    teacher_forward,
    validate_step,
)
from parallm.train.losses import block_mse


# Unchunked reference objectives for rail 4 — the whole point of kl_ce_chunked is
# that it never materializes these (B, T, V) logits, so they live only here.
def logit_kl(student_logits, teacher_logits, attention_mask, temperature=1.0):
    s_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1) * attention_mask
    return per_token.sum() / attention_mask.sum().clamp(min=1) * temperature**2


def logit_mse(student_logits, teacher_logits, attention_mask):
    d = student_logits.float() - teacher_logits.float()
    d = d - d.mean(dim=-1, keepdim=True)  # centered: drop the softmax-free direction
    per_token = d.pow(2).mean(dim=-1) * attention_mask
    return per_token.sum() / attention_mask.sum().clamp(min=1)


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


def _build(n_tracks: int, sync_after=None, n_layers: int = 8):
    cfg = _tiny_config(n_layers)
    torch.manual_seed(42)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    dense.eval()

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=list(sync_after) if sync_after is not None else manifest.sync_layer_indices,
        track_group=None,
    )
    pt.eval()
    pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
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

    dcfg = DistillConfig(sync_layer_indices=tuple(range(8)), lambda_logit_mse=0.05)
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg, student_forcing_prob=0.0)
    assert losses["block_mse"].item() < 1e-8, f"block_mse {losses['block_mse'].item()} not ~0 at N=1"
    assert losses["kl"].item() < 1e-6, f"kl {losses['kl'].item()} not ~0 at N=1"
    assert torch.isfinite(losses["ce"]), "ce not finite"


def test_kl_ce_chunked_matches_direct():
    torch.manual_seed(0)
    B, T, H, V = 2, 13, 8, 31
    lm_head = nn.Linear(H, V, bias=False)
    lm_head.weight.requires_grad_(False)
    hidden = torch.randn(B, T, H, requires_grad=True)
    t_hidden = torch.randn(B, T, H)
    labels = torch.randint(0, V, (B, T))
    mask = torch.ones((B, T), dtype=torch.long)
    lam_kl, lam_ce, lam_lm, temp = 0.7, 0.3, 0.11, 2.0

    kl, ce, lm, grad_h = kl_ce_chunked(
        hidden, t_hidden, lm_head, labels, mask,
        lambda_kl=lam_kl, lambda_ce=lam_ce, lambda_logit_mse=lam_lm,
        kl_temperature=temp, chunk_size=5,
    )

    s_logits = lm_head(hidden)
    t_logits = lm_head(t_hidden)
    kl_ref = logit_kl(s_logits, t_logits, mask, temperature=temp)
    ce_ref = lm_cross_entropy(s_logits, labels)
    lm_ref = logit_mse(s_logits, t_logits, mask)
    assert torch.allclose(kl, kl_ref, atol=1e-5), (kl.item(), kl_ref.item())
    assert torch.allclose(ce, ce_ref, atol=1e-5), (ce.item(), ce_ref.item())
    assert torch.allclose(lm, lm_ref, atol=1e-5), (lm.item(), lm_ref.item())

    (lam_kl * kl_ref + lam_ce * ce_ref + lam_lm * lm_ref).backward()
    assert torch.allclose(grad_h, hidden.grad, atol=1e-6), \
        f"chunked grad drifts by {(grad_h - hidden.grad).abs().max().item()}"


def test_sf1_chain_matches_free_running_n4():
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(range(8)))
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher_pt = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(8)), track_group=None,
    )
    teacher_pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    teacher = freeze_slice_teacher(teacher_pt)

    # No clamp so the per-tap values are the raw relMSEs on both sides.
    dcfg = DistillConfig(
        sync_layer_indices=tuple(range(8)), lambda_kl=0.0, lambda_ce=0.0,
        block_mse_clamp=None,
    )
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg, student_forcing_prob=1.0)

    # Reference: the deployed (free-running) post-attn forward's synced states
    # vs the same teacher targets (same capture contract on both sides).
    post_attn, post_mlp = set(range(7)), {7}
    pt.eval()
    with torch.no_grad():
        _, fr_taps = pt(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            return_sync_hiddens=True,
            capture_post_attn=post_attn, capture_post_mlp=post_mlp,
        )
        _, t_caps = teacher_forward(
            teacher, batch["input_ids"], batch["attention_mask"],
            post_attn_layers=post_attn, post_mlp_layers=post_mlp,
        )
    assert len(fr_taps) == 8, f"expected a tap per layer, got {sorted(fr_taps)}"
    for i in sorted(fr_taps):
        ref = block_mse(fr_taps[i], t_caps[i], attention_mask=batch["attention_mask"],
                        normalize=True).item()
        got = losses["layer_relmse"][i]
        assert abs(got - ref) < 1e-5, f"layer {i}: sf=1.0 chain {got} != free-running {ref}"


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

    dcfg = DistillConfig(sync_layer_indices=tuple(range(8)), lambda_logit_mse=0.05)
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg, student_forcing_prob=0.25,
                          forcing_seed=(1, 2))
    assert torch.isfinite(losses["total"])
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no layer param received gradient"
    # Teacher stayed frozen and untouched.
    assert all(not p_.requires_grad for p_ in teacher.parameters())

    val = validate_step(pt, teacher, pt.lm_head, batch)
    assert torch.isfinite(val["kl"]) and torch.isfinite(val["ce"])


def test_distill_step_fixed_d2_schedule_n4():
    """The fixed-D=2 re-heal: student BUILT at the gapped schedule (own-carry at
    layers 0,2,4,6), checkpointing on. Grads must reach every track through both
    the own-carry TF loop and the free-running KL forward."""
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

    dcfg = DistillConfig(sync_layer_indices=tuple(sched), lambda_logit_mse=0.05,
                         block_mse_clamp=None)
    batch = _batch(cfg)
    losses = distill_step(pt, teacher, pt.lm_head, batch, dcfg,
                          student_forcing_prob=0.25, forcing_seed=(1, 2))
    assert torch.isfinite(losses["total"])
    assert losses["block_mse"].item() > 0 and losses["kl"].item() > 0
    for k, tm in enumerate(pt.text_models):
        got = sum(1 for p_ in tm.layers.parameters()
                  if p_.grad is not None and p_.grad.abs().sum() > 0)
        assert got > 0, f"track {k}: no grad at the fixed-D=2 schedule"
