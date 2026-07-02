"""Lever B: phase-shifted single per-block sync (``sync_phase='post-attn'``).

K>1, single-process (no NCCL, track_group=None so SyncBoundary is a pure local
sum). Validates: (1) the default is bit-identical to the legacy boundary forward;
(2) N=1 dense parity holds in both phases; (3) with all MLPs zeroed the post-attn
forward is bit-identical to the boundary forward — an independent check that the
post-attention all-reduce sums each per-track delta exactly once (no double-count,
correct ``block_start``); (4) for a normal K>1 model the phase actually changes the
forward and the output is well-formed; (5) the flag/validation guards.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks


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


def _build(n_tracks, local_track_ids, sync_indices=None, seed=13):
    cfg = _tiny_config()
    torch.manual_seed(seed)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=1, text_config_attr="config"
    )
    sched = sync_indices if sync_indices is not None else list(range(cfg.num_hidden_layers))
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=local_track_ids,
        sync_after_layers=sched,
        track_group=None,
    ).eval()
    pt.load_track_state_dicts({t: tracks[t] for t in local_track_ids}, strict=False)
    return cfg, pt


def _zero_all_mlps(pt):
    """Force every per-track MLP to output zero (down_proj=0)."""
    with torch.no_grad():
        for tm in pt.text_models:
            for layer in tm.layers:
                layer.mlp.down_proj.weight.zero_()


def test_default_phase_is_boundary():
    _cfg, pt = _build(2, (0, 1))
    assert pt.sync_phase == "boundary"


def test_n1_dense_parity_both_phases():
    """N=1: the sync is a no-op, so post-attn reduces to the normal layer sequence
    and must match the boundary forward bit-for-bit (= the dense forward)."""
    cfg, pt = _build(1, (0,))
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        b_logits, _ = pt(input_ids=ids, attention_mask=mask)
        pt.set_sync_phase("post-attn")
        p_logits, _ = pt(input_ids=ids, attention_mask=mask)
    assert torch.equal(b_logits, p_logits)


def test_zero_mlp_post_attn_equals_boundary():
    """With MLPs zeroed, every layer is attention-only and fully synced each layer
    in BOTH phases, so the outputs must be bit-identical — an independent check of
    the post-attention summation bookkeeping (no double-count, correct block_start)."""
    cfg, pt = _build(4, (0, 1, 2, 3))
    _zero_all_mlps(pt)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        b_logits, _ = pt(input_ids=ids, attention_mask=mask)
        pt.set_sync_phase("post-attn")
        p_logits, p_h = pt(input_ids=ids, attention_mask=mask, return_sync_hiddens=True)
    assert torch.allclose(b_logits, p_logits, atol=1e-6, rtol=0)
    # Post-attn reports one synced hidden per layer.
    assert set(p_h.keys()) == set(range(cfg.num_hidden_layers))


def test_post_attn_changes_forward_and_is_well_formed():
    """For a normal K>1 model the phase shift must actually change the forward
    (MLPs are non-trivial) and the output must be finite and correctly shaped."""
    cfg, pt = _build(4, (0, 1, 2, 3))
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        b_logits, _ = pt(input_ids=ids, attention_mask=mask)
        pt.set_sync_phase("post-attn")
        p_logits, p_h = pt(input_ids=ids, attention_mask=mask, return_sync_hiddens=True)
    assert p_logits is not None and p_logits.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(p_logits).all()
    assert not torch.allclose(b_logits, p_logits)  # the flag does something
    assert set(p_h.keys()) == set(range(cfg.num_hidden_layers))
    for h in p_h.values():
        assert h.shape == (1, 16, cfg.hidden_size)
        assert torch.isfinite(h).all()


def test_pre_lm_head_hidden_is_synced_and_finite():
    cfg, pt = _build(4, (0, 1, 2, 3))
    pt.set_sync_phase("post-attn")
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    mask = torch.ones((1, 12), dtype=torch.long)
    with torch.no_grad():
        h, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    assert h.shape == (1, 12, cfg.hidden_size)
    assert torch.isfinite(h).all()


def test_distill_step_post_attn_trains_every_sublayer():
    """A post-attn distill step must run, produce a finite block loss, and deliver
    gradients to EVERY sublayer: the first layer's attention (block 0), a middle
    layer's MLP and attention, and the LAST layer's MLP (the final block runs it)."""
    from pt_converter.train.distill import DistillConfig, distill_step
    from pt_converter.train.teacher import HookedTeacher

    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=1, text_config_attr="config"
    )
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=list(range(cfg.num_hidden_layers)), track_group=None,
    )
    student.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    student.set_sync_phase("post-attn")
    student.train()
    # Teacher with post-attention targets (X+Y for 0..L-2, post-MLP for the last).
    teacher = HookedTeacher(
        text_model=dense, lm_head=lm_head,
        sync_layer_indices=list(range(cfg.num_hidden_layers)), capture_post_attn=True,
    )

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones((1, 16), dtype=torch.long),
             "labels": ids.clone()}
    dcfg = DistillConfig(
        sync_layer_indices=tuple(range(cfg.num_hidden_layers)),
        lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0, intra_window_mse=True,
        sync_phase="post-attn",
    )
    out = distill_step(student, teacher, batch, dcfg, compute_klce_metrics=False,
                       student_forcing_prob=0.0)
    assert torch.isfinite(out["block_mse"]).all() and out["block_mse"].item() > 0.0

    L = cfg.num_hidden_layers
    layers0 = student.text_models[0].layers

    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0.0
                   for p in module.parameters())

    # Last layer's MLP is run + supervised in the final block.
    assert _has_grad(layers0[L - 1].mlp), "last-layer MLP got no gradient"
    # A middle layer's MLP (trained by the NEXT block) and attention (its own block).
    assert _has_grad(layers0[4].mlp), "middle MLP got no gradient"
    assert _has_grad(layers0[3].self_attn), "middle attention got no gradient"
    # First layer's attention is supervised by block 0.
    first_mixer = layers0[0].linear_attn if layers0[0].layer_type == "linear_attention" else layers0[0].self_attn
    assert _has_grad(first_mixer), "first-layer attention got no gradient"


def test_d2_n1_dense_parity():
    """D=2 (sync after odd layers): N=1 ⇒ post-attn reduces to the dense layer
    sequence and matches the boundary forward bit-for-bit."""
    cfg, pt = _build(1, (0,), sync_indices=[1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        b_logits, _ = pt(input_ids=ids, attention_mask=mask)
        pt.set_sync_phase("post-attn")
        p_logits, _ = pt(input_ids=ids, attention_mask=mask)
    assert torch.equal(b_logits, p_logits)


def test_d2_zero_mlp_post_attn_equals_boundary():
    """At D=2 with MLPs zeroed, each window is attention-only and fully synced at the
    window boundary in BOTH phases, so the outputs match — independent check that the
    sparse-boundary post-attn summation bookkeeping is correct."""
    cfg, pt = _build(4, (0, 1, 2, 3), sync_indices=[1, 3, 5, 7])
    _zero_all_mlps(pt)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        b_logits, _ = pt(input_ids=ids, attention_mask=mask)
        pt.set_sync_phase("post-attn")
        p_logits, p_h = pt(input_ids=ids, attention_mask=mask, return_sync_hiddens=True)
    assert torch.allclose(b_logits, p_logits, atol=1e-6, rtol=0)
    # Only the boundary layers report a synced hidden (the deployed sync count = D=2's).
    assert set(p_h.keys()) == {1, 3, 5, 7}


def test_d2_post_attn_changes_forward():
    cfg, pt = _build(4, (0, 1, 2, 3), sync_indices=[1, 3, 5, 7])
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        b_logits, _ = pt(input_ids=ids, attention_mask=mask)
        pt.set_sync_phase("post-attn")
        p_logits, _ = pt(input_ids=ids, attention_mask=mask)
    assert torch.isfinite(p_logits).all()
    assert not torch.allclose(b_logits, p_logits)


def test_d2_distill_post_attn_trains_every_sublayer():
    """D=2 phased distill step (intra-window): runs, finite loss, and gradients reach
    boundary AND non-boundary sublayers (including the boundary layers' MLPs, which are
    trained by the NEXT segment, and the final post-MLP layer)."""
    from pt_converter.train.distill import DistillConfig, distill_step
    from pt_converter.train.teacher import HookedTeacher

    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=2, text_config_attr="config"
    )
    sched = [1, 3, 5, 7]  # D=2
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=sched, track_group=None,
    )
    student.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    student.set_sync_phase("post-attn")
    student.train()
    # Teacher hooks every layer (intra-window taps); post-attn targets at the phased
    # boundaries {1,3,5}, post-MLP elsewhere ({0,2,4,6} taps + final 7).
    teacher = HookedTeacher(
        text_model=dense, lm_head=lm_head,
        sync_layer_indices=list(range(cfg.num_hidden_layers)), capture_post_attn=True,
        post_attn_layers={1, 3, 5},
    )

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones((1, 16), dtype=torch.long),
             "labels": ids.clone()}
    dcfg = DistillConfig(
        sync_layer_indices=tuple(sched), lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0, intra_window_mse=True,
        sync_phase="post-attn",
    )
    out = distill_step(student, teacher, batch, dcfg, compute_klce_metrics=False,
                       student_forcing_prob=0.0)
    assert torch.isfinite(out["block_mse"]).all() and out["block_mse"].item() > 0.0

    layers0 = student.text_models[0].layers

    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0.0
                   for p in module.parameters())

    assert _has_grad(layers0[0].mlp), "layer-0 MLP got no gradient"
    assert _has_grad(layers0[1].linear_attn), "boundary layer-1 attention got no gradient"
    assert _has_grad(layers0[1].mlp), "boundary layer-1 MLP got no gradient (next-segment train)"
    assert _has_grad(layers0[7].mlp), "final layer-7 MLP got no gradient"
    assert _has_grad(layers0[3].self_attn), "layer-3 attention got no gradient"


def test_window_stale_floor_equals_phased():
    """window_stale at the zero-init gate is an EXACT no-op: the D=2 post-attn forward
    with the depth-stale fill attached must be bit-identical to plain phased — the floor
    guarantee (it caches ΣY at boundaries but injects gate·(…)=0, so it can only help)."""
    from pt_converter.model.cross_head_estimator import CrossHeadEstimator
    ids = torch.randint(0, 128, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    cfg, plain = _build(4, (0, 1, 2, 3), sync_indices=[1, 3, 5, 7])
    plain.set_sync_phase("post-attn")
    _cfg, filled = _build(4, (0, 1, 2, 3), sync_indices=[1, 3, 5, 7])
    filled.cross_head = CrossHeadEstimator(
        num_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size, backend="window_stale"
    )
    filled.set_sync_phase("post-attn")  # window_stale is allowed with post-attn
    with torch.no_grad():
        a, _ = plain(input_ids=ids, attention_mask=mask)
        b, _ = filled(input_ids=ids, attention_mask=mask)
    assert torch.equal(a, b)


def test_window_stale_changes_forward_when_gate_open():
    """Opening the gate makes the non-boundary layers read the cached boundary ΣY, so
    the forward changes (and stays finite) — the fill is actually wired into the MLP."""
    from pt_converter.model.cross_head_estimator import CrossHeadEstimator
    ids = torch.randint(0, 128, (1, 16))
    mask = torch.ones((1, 16), dtype=torch.long)
    cfg, pt = _build(4, (0, 1, 2, 3), sync_indices=[1, 3, 5, 7])
    ch = CrossHeadEstimator(
        num_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size, backend="window_stale"
    )
    pt.cross_head = ch
    pt.set_sync_phase("post-attn")
    with torch.no_grad():
        base, _ = pt(input_ids=ids, attention_mask=mask)  # gate=0 ⇒ == plain phased
        ch.gate.fill_(0.5)
        opened, _ = pt(input_ids=ids, attention_mask=mask)
    assert torch.isfinite(opened).all()
    assert not torch.allclose(base, opened)


def test_d2_distill_window_stale_trains_gate():
    """D=2 phased + window_stale distill step: the non-boundary tap trains the gate.
    Gradient is nonzero even at the zero-init gate (it's the derivative, not the value).
    Non-boundary layers WITH a preceding boundary (2,4,6) get a fill ⇒ gate grad; layer
    0 has no preceding boundary (no fill) ⇒ exactly zero grad there."""
    from pt_converter.train.distill import DistillConfig, distill_step
    from pt_converter.train.teacher import HookedTeacher
    from pt_converter.model.cross_head_estimator import CrossHeadEstimator

    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=2, text_config_attr="config"
    )
    sched = [1, 3, 5, 7]  # D=2
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=sched, track_group=None,
    )
    student.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    ch = CrossHeadEstimator(
        num_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size, backend="window_stale"
    )
    student.cross_head = ch
    student.set_sync_phase("post-attn")
    student.train()
    teacher = HookedTeacher(
        text_model=dense, lm_head=lm_head,
        sync_layer_indices=list(range(cfg.num_hidden_layers)), capture_post_attn=True,
        post_attn_layers={1, 3, 5},
    )

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones((1, 16), dtype=torch.long),
             "labels": ids.clone()}
    dcfg = DistillConfig(
        sync_layer_indices=tuple(sched), lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0, intra_window_mse=True,
        sync_phase="post-attn",
    )
    out = distill_step(student, teacher, batch, dcfg, compute_klce_metrics=False,
                       student_forcing_prob=0.0)
    assert torch.isfinite(out["block_mse"]).all() and out["block_mse"].item() > 0.0
    assert ch.gate.grad is not None, "window_stale gate got no gradient"
    assert ch.gate.grad[2].abs().sum().item() > 0.0, "non-boundary layer-2 gate untrained"
    assert ch.gate.grad[0].abs().sum().item() == 0.0, "layer-0 has no fill ⇒ must be zero grad"


def test_set_sync_phase_validation():
    _cfg, pt = _build(2, (0, 1))
    with pytest.raises(ValueError):
        pt.set_sync_phase("nonsense")
    # A predictor backend is incompatible (B/post-attn delivers the real content instead).
    from pt_converter.model.cross_head_estimator import CrossHeadEstimator
    pt.cross_head = CrossHeadEstimator(num_layers=8, hidden_size=64, backend="stale")
    with pytest.raises(ValueError):
        pt.set_sync_phase("post-attn")
    # ...but window_stale IS allowed (it FILLS the partial non-boundary layers with the
    # cached boundary ΣY — it does not replace the sync).
    pt.cross_head = CrossHeadEstimator(num_layers=8, hidden_size=64, backend="window_stale")
    pt.set_sync_phase("post-attn")
    assert pt.sync_phase == "post-attn"


# --------------------------------------------------------------------------- #
# A2a enabling changes: fr damping on the phased forward + its checkpointing
# + the on-policy output objective end-to-end.
# --------------------------------------------------------------------------- #
def test_post_attn_grad_alpha_value_invariant_and_truncates():
    """boundary_grad_alpha on the PHASED forward: (a) forward value is EXACTLY
    unchanged (h.detach()+alpha*(h-h.detach()) == h); (b) alpha=0 hard-truncates —
    a loss on the final output reaches only sublayers downstream of the last
    post-attn sync boundary, so pre-boundary attention gets zero gradient."""
    cfg, pt = _build(2, (0, 1), sync_indices=[1, 3, 5, 7])
    pt.set_sync_phase("post-attn")
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    mask = torch.ones((1, 12), dtype=torch.long)

    def _run(alpha):
        pt.zero_grad(set_to_none=True)
        h, _ = pt(input_ids=ids, attention_mask=mask,
                  return_hidden_pre_lm_head=True, boundary_grad_alpha=alpha)
        h.float().pow(2).mean().backward()
        g_early = pt.text_models[0].layers[0].linear_attn.in_proj_qkv.weight.grad
        g_late = pt.text_models[0].layers[7].mlp.down_proj.weight.grad
        early = 0.0 if g_early is None else g_early.abs().sum().item()
        return h.detach().clone(), early, g_late.abs().sum().item()

    h1, early1, late1 = _run(1.0)
    h0, early0, late0 = _run(0.0)
    assert torch.equal(h1, h0)                # damping never changes the forward value
    assert late1 > 0.0 and late0 > 0.0        # the final window always trains
    assert early1 > 0.0 and early0 == 0.0     # alpha=0 cuts gradient at every boundary


def test_post_attn_checkpointing_matches_plain():
    """Per-sublayer activation checkpointing on the phased forward is recompute-only:
    identical output and identical gradients on every parameter (with damping active,
    so the ckpt/damping interaction is covered too)."""
    cfg, pt = _build(2, (0, 1), sync_indices=[1, 3, 5, 7])
    pt.set_sync_phase("post-attn")
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    mask = torch.ones((1, 12), dtype=torch.long)

    def _run(use_ckpt):
        pt._use_checkpoint = use_ckpt
        pt.zero_grad(set_to_none=True)
        h, _ = pt(input_ids=ids, attention_mask=mask,
                  return_hidden_pre_lm_head=True, boundary_grad_alpha=0.5)
        h.float().pow(2).mean().backward()
        return h.detach().clone(), {
            n: p.grad.clone() for n, p in pt.named_parameters() if p.grad is not None
        }

    h_plain, g_plain = _run(False)
    h_ckpt, g_ckpt = _run(True)
    pt._use_checkpoint = False
    assert torch.allclose(h_plain, h_ckpt, atol=1e-6)
    assert set(g_plain) == set(g_ckpt)
    for n in g_plain:
        assert torch.allclose(g_plain[n], g_ckpt[n], atol=1e-6), n


def test_d2_distill_post_attn_with_onpolicy_output_objective():
    """The A2a configuration end-to-end: D=2 phased distill step with a mode-seeking
    output divergence (reverse_kl) + fr damping (+ activation checkpointing). Must run
    finite, and the OUTPUT objective's gradient must reach the deep layers through the
    damped free-running phased forward (block loss alone can't be told apart here, so
    run with lambda_block=0 to isolate the KL path)."""
    from pt_converter.train.distill import DistillConfig, distill_step
    from pt_converter.train.teacher import HookedTeacher

    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=2, text_config_attr="config"
    )
    sched = [1, 3, 5, 7]
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=sched, track_group=None, activation_checkpoint=True,
    )
    student.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    student.set_sync_phase("post-attn")
    student.train()
    teacher = HookedTeacher(
        text_model=dense, lm_head=lm_head,
        sync_layer_indices=list(range(cfg.num_hidden_layers)), capture_post_attn=True,
        post_attn_layers={1, 3, 5},
    )

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones((1, 16), dtype=torch.long),
             "labels": ids.clone()}
    dcfg = DistillConfig(
        sync_layer_indices=tuple(sched), lambda_block=0.0, lambda_kl=0.5, lambda_ce=0.0,
        divergence="reverse_kl", fr_grad_alpha=0.5,
        normalize_block_mse=True, block_mse_clamp=10.0, intra_window_mse=True,
        sync_phase="post-attn",
    )
    out = distill_step(student, teacher, batch, dcfg, compute_klce_metrics=True,
                       student_forcing_prob=0.0)
    assert torch.isfinite(out["kl"]).all() and out["kl"].item() > 0.0

    layers0 = student.text_models[0].layers

    def _grad_sum(module):
        return sum(p.grad.abs().sum().item() for p in module.parameters()
                   if p.grad is not None)

    # The damped (alpha=0.5) reverse-KL flows through every boundary: deep layers get
    # full-strength gradient, shallow ones geometrically attenuated but NON-zero.
    assert _grad_sum(layers0[7].mlp) > 0.0, "final MLP got no KL gradient"
    assert _grad_sum(layers0[3].self_attn) > 0.0, "mid attention got no KL gradient"
    assert _grad_sum(layers0[0].linear_attn) > 0.0, "shallow attention got no (damped) KL gradient"
