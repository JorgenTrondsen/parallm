"""Rails for the window-parallel TF distillation branch
(``distill_step`` with ``sync_phase='window-parallel'``, Gate 2 trainer).

* Chain rail: with ``student_forcing_prob=1.0`` the teacher-forced per-window
  chain reproduces the DEPLOYED window-parallel forward exactly (the sf path
  carries the exact per-track tensors, so this is bit-for-bit up to detach).
* Grad rail: one distill step delivers gradients to a mid-window attention,
  the window MLPs, the first layer's attention and the final layer's MLP.
* Schedule guard: a schedule not ending at the final layer raises.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks
from pt_converter.train.distill import DistillConfig, distill_step
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


# g2-style windows on 8 layers: [0], [1,2], [3,4], [5,6], [7].
G2_SCHEDULE = (0, 2, 4, 6, 7)


def _build(n_tracks=2, sync_after_layers=G2_SCHEDULE):
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(11)
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=2, text_config_attr="config"
    )
    student = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=list(sync_after_layers), track_group=None,
    )
    student.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    student.set_sync_phase("window-parallel")
    teacher = HookedTeacher(
        text_model=dense, lm_head=lm_head,
        sync_layer_indices=list(sync_after_layers),
    )
    return cfg, student, teacher


def _batch(cfg, seq=16):
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    return {
        "input_ids": ids,
        "attention_mask": torch.ones((1, seq), dtype=torch.long),
        "labels": ids.clone(),
    }


def _dcfg(**kw):
    base = dict(
        sync_layer_indices=G2_SCHEDULE,
        lambda_block=1.0, lambda_kl=0.0, lambda_ce=0.0,
        normalize_block_mse=True, block_mse_clamp=10.0,
        sync_phase="window-parallel",
    )
    base.update(kw)
    return DistillConfig(**base)


def test_sf1_chain_reproduces_deployed_forward():
    # THE rail: under full student forcing the TF per-window chain must equal
    # the deployed window-parallel forward — same carry, same base, same syncs.
    cfg, student, teacher = _build()
    student.train()
    batch = _batch(cfg)

    recorded: list[torch.Tensor] = []
    real = student.sync_module.forward

    def _record(h_list, block_start):
        out = real(h_list, block_start)
        recorded.append(out.detach().clone())
        return out

    student.sync_module.forward = _record
    try:
        out = distill_step(
            student, teacher, batch, _dcfg(),
            compute_klce_metrics=False, student_forcing_prob=1.0,
        )
    finally:
        student.sync_module.forward = real
    assert torch.isfinite(out["block_mse"]).all() and out["block_mse"].item() > 0.0

    student.eval()
    with torch.no_grad():
        _, sync_h = student(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            return_sync_hiddens=True, return_hidden_pre_lm_head=True,
        )
    # Last TF sync = the final window's post-MLP sync = the deployed forward's
    # tapped hidden at the final layer.
    assert torch.allclose(recorded[-1], sync_h[7], atol=1e-6, rtol=0)


def test_grads_reach_all_sublayers():
    cfg, student, teacher = _build()
    student.train()
    out = distill_step(
        student, teacher, _batch(cfg), _dcfg(),
        compute_klce_metrics=False, student_forcing_prob=0.0,
    )
    assert torch.isfinite(out["block_mse"]).all() and out["block_mse"].item() > 0.0

    layers0 = student.text_models[0].layers

    def _has_grad(module):
        return any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in module.parameters()
        )

    first = layers0[0]
    first_mixer = first.linear_attn if first.layer_type == "linear_attention" else first.self_attn
    assert _has_grad(first_mixer), "first-layer attention got no gradient"
    # Window [3,4]: layer 3 full-attention (parallel read), layer 4 MLP (reads R).
    assert _has_grad(layers0[3].self_attn), "mid-window attention got no gradient"
    assert _has_grad(layers0[4].mlp), "window MLP got no gradient"
    assert _has_grad(layers0[7].mlp), "final-layer MLP got no gradient"


def test_per_window_relmse_taps():
    cfg, student, teacher = _build()
    student.train()
    out = distill_step(
        student, teacher, _batch(cfg), _dcfg(),
        compute_klce_metrics=False, student_forcing_prob=0.0,
        track_layer_relmse=True,
    )
    # One tap per window, keyed by the window-end layer.
    assert set(out["layer_relmse"].keys()) == set(G2_SCHEDULE)
    for v in out["layer_relmse"].values():
        assert torch.isfinite(v).all()


def test_activation_checkpoint_parity():
    # The grad-carrying full forward (logit-MSE objective) runs the WP stack with
    # per-sublayer checkpointing at training shapes — must match the plain path
    # in both forward value and gradient.
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(11)
    dense = Qwen3_5TextModel(cfg).eval()
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=2, text_config_attr="config"
    )
    outs = {}
    for ac in (False, True):
        pt = PTWrappedModel(
            text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
            sync_after_layers=list(G2_SCHEDULE), track_group=None,
            activation_checkpoint=ac,
        )
        pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
        pt.set_sync_phase("window-parallel")
        pt.eval()  # grads stay on; only dropout-style mode flags differ
        ids = torch.randint(
            0, cfg.vocab_size, (1, 16), generator=torch.Generator().manual_seed(3)
        )
        h, _ = pt(
            input_ids=ids, attention_mask=torch.ones((1, 16), dtype=torch.long),
            return_hidden_pre_lm_head=True,
        )
        h.square().mean().backward()
        outs[ac] = (
            h.detach(),
            pt.text_models[0].layers[3].self_attn.q_proj.weight.grad.clone(),
        )
    (h0, g0), (h1, g1) = outs[False], outs[True]
    assert torch.allclose(h0, h1, atol=1e-6, rtol=0)
    assert torch.allclose(g0, g1, atol=1e-6, rtol=0)


def test_schedule_must_end_at_final_layer():
    cfg, student, teacher = _build(sync_after_layers=(0, 2, 4, 6))
    teacher.remove_hooks()
    _, _, teacher = _build()  # hooks incl. layer 7; schedule under test is in cfg
    student.train()
    try:
        distill_step(
            student, teacher, _batch(cfg),
            _dcfg(sync_layer_indices=(0, 2, 4, 6)),
            compute_klce_metrics=False, student_forcing_prob=0.0,
        )
        raise AssertionError("expected ValueError for schedule not ending at the final layer")
    except ValueError:
        pass
