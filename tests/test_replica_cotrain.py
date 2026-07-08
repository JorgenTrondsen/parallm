"""Rails for the sparse-copy CO-TRAINING branch (``distill_step`` with
``sync_phase='post-attn'`` + ``replica_channel``, the mixed replica design).

* Chain rail: under ``student_forcing_prob=1.0`` the TF chain's final tap must
  equal the DEPLOYED replica forward (``phased_intervention_forward`` with the
  same channel/shadow) — the same sf-1.0 rail the window-parallel branch has.
* Grad rail: exact track sublayers get gradients; the frozen shadow copies get
  none.
* Refresh rail: clearing ``channel._shadow`` re-derives the copies from the
  LIVE weights (the lagged target-network refresh the trainer drives).
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.eval.intervention import (
    PhasedMode,
    collect_input_norms,
    phased_intervention_forward,
)
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks
from pt_converter.train.distill import DistillConfig, distill_step
from pt_converter.train.teacher import HookedTeacher

D2_SCHEDULE = (1, 3, 5, 7)


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


def _build(sync_after_layers=D2_SCHEDULE):
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(11)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=2, text_config_attr="config"
    )
    student = PTWrappedModel(
        text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
        sync_after_layers=list(sync_after_layers), track_group=None,
    )
    student.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    student.set_sync_phase("post-attn")
    teacher = HookedTeacher(
        text_model=dense, lm_head=lm_head,
        sync_layer_indices=list(range(cfg.num_hidden_layers)),
        capture_post_attn=True, post_attn_layers=set(sync_after_layers) - {7},
    )
    return cfg, dense, student, teacher


def _channel(dense, cfg):
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    mask = torch.ones((2, 16), dtype=torch.long)
    norms = collect_input_norms(dense, [{"input_ids": ids, "attention_mask": mask}], "cpu")
    ch = PhasedMode("replica:wanda:0.5")
    ch.set_input_norms(norms, 2)
    return ch


def _batch(cfg, seq=16):
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    return {
        "input_ids": ids,
        "attention_mask": torch.ones((1, seq), dtype=torch.long),
        "labels": ids.clone(),
    }


def _dcfg(ch, **kw):
    base = dict(
        sync_layer_indices=D2_SCHEDULE,
        lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0,
        intra_window_mse=True, sync_phase="post-attn",
        replica_channel=ch,
    )
    base.update(kw)
    return DistillConfig(**base)


def test_sf1_chain_reproduces_deployed_replica_forward():
    # THE rail: the sf-1.0 TF chain's final post-MLP reconstruction must equal
    # the deployed replica forward with the identical shadow.
    cfg, dense, student, teacher = _build()
    ch = _channel(dense, cfg)
    student.train()
    batch = _batch(cfg)
    taps: dict = {}
    out = distill_step(
        student, teacher, batch, _dcfg(ch, replica_taps=taps),
        compute_klce_metrics=False, student_forcing_prob=1.0,
    )
    assert torch.isfinite(out["block_mse"]).all() and out["block_mse"].item() > 0.0
    assert set(taps) == set(range(8))  # every layer supervised (intra-window)

    student.eval()
    with torch.no_grad():
        deployed = phased_intervention_forward(
            student, batch["input_ids"], batch["attention_mask"], list(D2_SCHEDULE), ch
        )
        tf_final = student.text_models[0].norm(taps[7])
    assert torch.allclose(tf_final, deployed, atol=1e-5, rtol=1e-4)


def test_grads_reach_exact_sublayers_not_shadow():
    cfg, dense, student, teacher = _build()
    ch = _channel(dense, cfg)
    student.train()
    distill_step(
        student, teacher, _batch(cfg), _dcfg(ch),
        compute_klce_metrics=False, student_forcing_prob=0.0,
    )
    layers0 = student.text_models[0].layers

    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0.0
                   for p in module.parameters())

    assert _has_grad(layers0[7].mlp), "last-layer MLP got no gradient"
    assert _has_grad(layers0[4].mlp), "mid-window MLP got no gradient"
    assert _has_grad(layers0[3].self_attn), "boundary attention got no gradient"
    first_mixer = (layers0[0].linear_attn if layers0[0].layer_type == "linear_attention"
                   else layers0[0].self_attn)
    assert _has_grad(first_mixer), "first-layer attention got no gradient"
    for track_layers in ch._shadow:
        for clone in track_layers:
            assert all(p.grad is None and not p.requires_grad for p in clone.parameters())


def test_shadow_refresh_tracks_live_weights():
    cfg, dense, student, _teacher = _build()
    ch = _channel(dense, cfg)
    ch.ensure_shadow(student)
    old = ch._shadow[0][0].mlp.gate_proj.weight.clone()
    with torch.no_grad():
        student.text_models[0].layers[0].mlp.gate_proj.weight.mul_(2.0)
    ch._shadow = None  # the trainer's --replica-refresh action
    ch.ensure_shadow(student)
    new = ch._shadow[0][0].mlp.gate_proj.weight
    assert not torch.allclose(old, new)
    surv_old, surv_new = old != 0, new != 0
    # Same wanda criterion on doubled weights keeps the same mask, doubled values.
    assert torch.equal(surv_old, surv_new)
    assert torch.allclose(new[surv_new], 2.0 * old[surv_old], atol=1e-6)
