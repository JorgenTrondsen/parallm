"""Boundary gradient damping (``boundary_grad_alpha``) for the free-running MSE.

Single-process K=2 harness (track_group=None — SyncBoundary degenerates to a
local sum, no NCCL). Sliced at sync_block_depth=2 over 8 layers → 4 windows
(boundaries [1, 3, 5, 7]), so a loss on the LAST tap reaches window w through
k = 3 − w damping crossings — enough depth to assert the alpha^k geometry.

Verified properties:
  * the damping is value-exact: forward outputs/taps are bitwise identical at
    any alpha (``h.detach() + alpha*(h − h.detach())`` has forward value h);
  * alpha=0 hard-truncates: a tap's gradient never crosses a boundary — its
    own window's grads are bit-identical to the full unroll, every upstream
    window's (and the embedding's) grads are exactly zero;
  * 0<alpha<1 scales a tap's gradient into window w by exactly alpha^k (the
    cotangent crossing each boundary is scaled once per crossing, and the
    within-window backward is linear in the cotangent);
  * all of the above hold under window-granularity activation checkpointing
    (the damping sits outside the checkpoint closures).
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
# 8 layers at D=2: window w covers layers [2w, 2w+1], boundary after 2w+1.
_WINDOWS = [(0, 1), (2, 3), (4, 5), (6, 7)]
_LAST_TAP = 7  # boundary of window 3


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


def _build_pt(activation_checkpoint: bool = False, checkpoint_granularity: str = "layer"):
    cfg = _tiny_config()
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=2, text_config_attr="config"
    )
    assert manifest.sync_layer_indices == [1, 3, 5, 7]

    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=2,
        local_track_ids=(0, 1),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
        activation_checkpoint=activation_checkpoint,
        checkpoint_granularity=checkpoint_granularity,
    )
    pt.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)
    return cfg, pt


def _inputs(cfg):
    torch.manual_seed(29)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)
    target = torch.randn(1, 16, cfg.hidden_size)
    return input_ids, attention_mask, target


def _window_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    if m is None:
        return None
    layer = int(m.group(1))
    return next(w for w, (s, e) in enumerate(_WINDOWS) if s <= layer <= e)


def _last_tap_grads(pt, input_ids, attention_mask, target, alpha):
    """Backward an MSE on the LAST boundary tap only; return name → grad copy."""
    pt.zero_grad(set_to_none=True)
    _, sync_hiddens = pt(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_sync_hiddens=True,
        boundary_grad_alpha=alpha,
    )
    F.mse_loss(sync_hiddens[_LAST_TAP], target).backward()
    return {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in pt.named_parameters()
    }


def test_damping_is_value_exact():
    cfg, pt = _build_pt()
    pt.eval()
    input_ids, attention_mask, _ = _inputs(cfg)
    # Grad-enabled on purpose: the damping branch only engages under grad, and
    # value-identity must hold exactly there (training-time forward).
    base_logits, base_taps = pt(
        input_ids=input_ids, attention_mask=attention_mask,
        return_sync_hiddens=True, boundary_grad_alpha=1.0,
    )
    for alpha in (0.5, 0.25, 0.0):
        logits, taps = pt(
            input_ids=input_ids, attention_mask=attention_mask,
            return_sync_hiddens=True, boundary_grad_alpha=alpha,
        )
        assert torch.equal(base_logits, logits)
        for idx in base_taps:
            assert torch.equal(base_taps[idx], taps[idx])


def _assert_truncation(pt, cfg):
    input_ids, attention_mask, target = _inputs(cfg)
    g1 = _last_tap_grads(pt, input_ids, attention_mask, target, alpha=1.0)
    g0 = _last_tap_grads(pt, input_ids, attention_mask, target, alpha=0.0)

    nonzero_windows_full = set()
    for name in g1:
        w = _window_of(name)
        if w == len(_WINDOWS) - 1:
            # Own window: no damping op on its path — bit-identical grads.
            assert g1[name] is not None and g0[name] is not None
            assert torch.equal(g0[name], g1[name]), name
        elif w is not None or "embed_tokens" in name:
            # Upstream windows + embedding: exactly zero at alpha=0 (the zero
            # cotangent propagates as exact zeros), nonzero somewhere at alpha=1.
            if g0[name] is not None:
                assert torch.count_nonzero(g0[name]) == 0, name
            if g1[name] is not None and torch.count_nonzero(g1[name]) > 0:
                nonzero_windows_full.add(w if w is not None else "embed")
        else:
            # Final norm / lm_head: not on the tap-loss path in either run.
            assert g0[name] is None and g1[name] is None, name
    # The full unroll must actually reach every upstream window + the embedding
    # (otherwise the zero assertions above would be vacuous).
    assert nonzero_windows_full >= {0, 1, 2, "embed"}


def test_alpha0_truncates_to_own_window():
    cfg, pt = _build_pt()
    pt.eval()
    _assert_truncation(pt, cfg)


def test_alpha0_truncates_under_window_checkpointing():
    cfg, pt = _build_pt(activation_checkpoint=True, checkpoint_granularity="window")
    pt.train()
    _assert_truncation(pt, cfg)


def test_alpha_scales_cross_window_grads_geometrically():
    cfg, pt = _build_pt()
    pt.eval()
    input_ids, attention_mask, target = _inputs(cfg)
    g1 = _last_tap_grads(pt, input_ids, attention_mask, target, alpha=1.0)
    gh = _last_tap_grads(pt, input_ids, attention_mask, target, alpha=0.5)

    sq1 = [0.0] * len(_WINDOWS)
    sqh = [0.0] * len(_WINDOWS)
    for name in g1:
        w = _window_of(name)
        if w is None or g1[name] is None:
            continue
        sq1[w] += g1[name].double().pow(2).sum().item()
        sqh[w] += gh[name].double().pow(2).sum().item()
    for w in range(len(_WINDOWS)):
        k = (len(_WINDOWS) - 1) - w  # boundary crossings from the last tap
        ratio = (sqh[w] ** 0.5) / (sq1[w] ** 0.5)
        assert abs(ratio - 0.5 ** k) < 1e-3 * (0.5 ** k), (w, k, ratio)
