"""Unit tests for the dense healing trainer (``train/heal_dense.py``).

Tiny stock ``Qwen3_5TextModel`` (CPU, no FSDP, no dist):

* TF-loss rail: student == teacher weights ⇒ per-window TF loss ≈ 0 in
  ``serial`` mode (the teacher IS the serial function) and > 0 in
  ``parallel-attn:g2`` (the healing signal exists).
* Grad flow: ``heal_step`` puts gradients on every decoder-layer param and on
  nothing frozen; the validation reader (``do_backward=False``) leaves none.
* Two AdamW steps on a fixed batch reduce the TF window loss (finite, no NaN).
"""
from __future__ import annotations

import copy

import torch
from torch import nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.eval.dense_parallel import build_windows
from pt_converter.train.heal_dense import HealConfig, heal_step
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


def _build(mode="parallel-attn", group=2):
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    teacher_model = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)
    student = copy.deepcopy(teacher_model)
    hcfg = HealConfig(windows=build_windows(8, group), mode=mode)
    teacher = HookedTeacher(teacher_model, lm_head, hcfg.capture_indices)
    return student, lm_head, teacher, hcfg, cfg


def _batch(cfg, seq=16):
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (2, seq))
    mask = torch.ones((2, seq), dtype=torch.long)
    return ids, mask


def test_capture_indices_are_window_ends():
    hcfg = HealConfig(windows=build_windows(8, 2))
    assert hcfg.capture_indices == [0, 2, 4, 6, 7]


def test_tf_loss_rail_serial_zero_parallel_positive():
    # Same weights as the teacher: the serial function IS the teacher ⇒ TF ≈ 0;
    # the parallel-attn rewiring differs ⇒ TF > 0 (the healing signal).
    student, lm_head, teacher, _, cfg = _build(mode="serial")
    ids, mask = _batch(cfg)
    with torch.no_grad():
        m_serial = heal_step(
            student, lm_head, teacher, HealConfig(windows=build_windows(8, 2), mode="serial"),
            ids, mask, do_backward=False,
        )
        m_par = heal_step(
            student, lm_head, teacher, HealConfig(windows=build_windows(8, 2)),
            ids, mask, do_backward=False,
        )
    assert m_serial["tf_window_mse"] < 1e-8
    assert m_serial["fr_logit_mse"] < 1e-8
    assert m_par["tf_window_mse"] > 1e-4
    assert m_par["fr_logit_mse"] > 1e-6


def test_grad_flow_layers_only():
    student, lm_head, teacher, hcfg, cfg = _build()
    for prm in lm_head.parameters():
        prm.requires_grad = False
    for prm in student.embed_tokens.parameters():
        prm.requires_grad = False
    for prm in student.norm.parameters():
        prm.requires_grad = False
    ids, mask = _batch(cfg)
    metrics = heal_step(student, lm_head, teacher, hcfg, ids, mask)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    for i, layer in enumerate(student.layers):
        grads = [p_.grad for p_ in layer.parameters() if p_.requires_grad]
        assert grads and all(g is not None for g in grads), f"layer {i} missing grads"
        assert all(torch.isfinite(g).all() for g in grads if g is not None), f"layer {i} nan grad"
    assert all(p_.grad is None for p_ in student.embed_tokens.parameters())
    assert all(p_.grad is None for p_ in lm_head.parameters())


def test_validation_reader_leaves_no_grads():
    student, lm_head, teacher, hcfg, cfg = _build()
    ids, mask = _batch(cfg)
    with torch.no_grad():
        heal_step(student, lm_head, teacher, hcfg, ids, mask, do_backward=False)
    assert all(p_.grad is None for p_ in student.parameters())


def test_two_steps_reduce_tf_loss():
    student, lm_head, teacher, hcfg, cfg = _build()
    ids, mask = _batch(cfg)
    optim = torch.optim.AdamW(
        [p_ for p_ in student.parameters() if p_.requires_grad], lr=3e-4, weight_decay=0.0
    )
    losses = []
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        metrics = heal_step(student, lm_head, teacher, hcfg, ids, mask)
        losses.append(metrics["tf_window_mse"])
        optim.step()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < losses[0]
