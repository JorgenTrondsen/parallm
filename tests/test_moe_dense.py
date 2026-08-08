"""Rails for dense expert evaluation (`parallm.model.moe_dense`).

The load-bearing claim is EQUIVALENCE: dense runs all E experts and relies on the
non-selected routing weights being exactly 0.0 to drop them. If that is wrong the
model still trains — just against a slightly different function — so it has to be
checked in fp32, where there is no bf16 floor to hide a real error inside, and on
GRADIENTS as well as the output (the whole point of the change is the backward).
"""
from __future__ import annotations

import pytest
import torch
from transformers.integrations.moe import (
    ALL_EXPERTS_FUNCTIONS,
    grouped_mm_experts_forward,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from parallm.model.moe_dense import (
    DENSE_WIDTH_MAX,
    IMPL_NAME,
    choose_experts_impl,
    dense_experts_forward,
    set_experts_policy,
)


class _Cfg:
    hidden_size = 96
    intermediate_size = 16
    num_local_experts = 8
    _experts_implementation = "grouped_mm"


def _experts():
    torch.manual_seed(0)
    e = GptOssExperts(_Cfg())
    for p in e.parameters():
        p.data.normal_(0, 0.05)
    e.config = _Cfg()
    e.has_gate = e.has_bias = e.is_transposed = True
    e.is_concatenated = False
    return e


def _routing(tokens=48, top_k=2):
    torch.manual_seed(1)
    h = torch.randn(tokens, _Cfg.hidden_size)
    tw, idx = torch.topk(torch.randn(tokens, _Cfg.num_local_experts), top_k, dim=-1)
    return h, idx, torch.softmax(tw, -1)


def _fwd_bwd(fn, exp, h, idx, tw):
    for p in exp.parameters():
        p.grad = None
    h = h.detach().requires_grad_(True)
    out = fn(exp, h, idx, tw)
    torch.manual_seed(7)  # a non-uniform upstream grad, so errors cannot cancel
    out.backward(torch.randn_like(out))
    grads = {n: p.grad.clone() for n, p in exp.named_parameters()}
    grads["hidden"] = h.grad.clone()
    return out.detach(), grads


def test_dense_matches_grouped_mm_on_output_and_every_gradient():
    exp = _experts()
    h, idx, tw = _routing()
    o_ref, g_ref = _fwd_bwd(grouped_mm_experts_forward, exp, h, idx, tw)
    o_new, g_new = _fwd_bwd(dense_experts_forward, exp, h, idx, tw)

    assert torch.allclose(o_new, o_ref, atol=1e-5, rtol=1e-4)
    for name, ref in g_ref.items():
        rel = ((g_new[name] - ref).norm() / ref.norm()).item()
        assert rel < 1e-5, f"grad {name} diverged: relL2 {rel:.2e}"


def test_unselected_experts_contribute_exactly_zero():
    """The equivalence rests entirely on `gated * 0.0 == 0.0`, so perturbing an
    expert no token routed to must not move the output by a single bit."""
    exp = _experts()
    h, idx, tw = _routing()
    unused = sorted(set(range(_Cfg.num_local_experts)) - set(idx.flatten().tolist()))
    if not unused:  # routing happened to cover every expert; force one open
        idx = idx.clamp(max=_Cfg.num_local_experts - 2)
        unused = [_Cfg.num_local_experts - 1]

    before = dense_experts_forward(exp, h, idx, tw).clone()
    with torch.no_grad():
        exp.down_proj[unused[0]].add_(1e3)
        exp.down_proj_bias[unused[0]].add_(1e3)
    after = dense_experts_forward(exp, h, idx, tw)
    assert torch.equal(before, after)


def test_ep_sentinel_indices_do_not_clobber_a_real_slot():
    """Expert-parallel routing hands in out-of-range ids with weight 0. Clamping them
    lands on a REAL expert, so the scatter must add (0) rather than overwrite."""
    exp = _experts()
    h, idx, tw = _routing()
    sent_idx = idx.clone()
    sent_tw = tw.clone()
    sent_idx[:, 1] = _Cfg.num_local_experts + 5  # sentinel
    sent_tw[:, 1] = 0.0
    dropped_idx = idx.clone()
    dropped_tw = tw.clone()
    dropped_tw[:, 1] = 0.0  # same math, no sentinel

    assert torch.allclose(
        dense_experts_forward(exp, h, sent_idx, sent_tw),
        dense_experts_forward(exp, h, dropped_idx, dropped_tw),
    )


def test_registered_under_its_name_so_a_config_can_select_it():
    assert ALL_EXPERTS_FUNCTIONS.get_interface(IMPL_NAME, None) is dense_experts_forward


def test_dense_mlp_half_holds_a_single_graph():
    """Half the point of the change: no sort, no histc, no data-dependent offsets, so
    `--compile` gets ONE graph over the whole MLP half instead of fragments around an
    opaque expert kernel. `fullgraph=True` is what makes that a rail rather than a hope.
    """
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    from parallm.model.seam import seam_mlp

    cfg = GptOssConfig(
        hidden_size=64, intermediate_size=8, num_local_experts=4,
        num_experts_per_tok=2, num_attention_heads=1, num_key_value_heads=1,
        head_dim=16, num_hidden_layers=1, layer_types=["full_attention"],
    )
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = IMPL_NAME
    layer = GptOssDecoderLayer(cfg, 0)
    for p in layer.parameters():
        p.data.normal_(0, 0.02)

    torch._dynamo.reset()
    compiled = torch.compile(seam_mlp, fullgraph=True, dynamic=False)
    x = torch.randn(1, 8, 64, requires_grad=True)
    compiled(layer, x).sum().backward()
    assert x.grad is not None


def test_auto_policy_switches_on_per_expert_width():
    try:
        set_experts_policy("auto")
        assert choose_experts_impl(32, DENSE_WIDTH_MAX) == IMPL_NAME
        assert choose_experts_impl(32, DENSE_WIDTH_MAX + 1) == "grouped_mm"
        set_experts_policy("dense")
        assert choose_experts_impl(32, DENSE_WIDTH_MAX + 1) == IMPL_NAME
        set_experts_policy("grouped_mm")
        assert choose_experts_impl(32, 1) == "grouped_mm"
    finally:
        set_experts_policy("auto")

    with pytest.raises(ValueError):
        set_experts_policy("sparse")


def test_gpt_oss_track_config_stamps_the_chosen_implementation():
    """The track builder is what actually reaches a training run, so rail the wiring
    and not just the policy function."""
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from parallm.model.tracks.gpt_oss import build_per_track_text_config

    cfg = GptOssConfig(
        hidden_size=2880, intermediate_size=2880, num_local_experts=32,
        num_experts_per_tok=4, num_attention_heads=64, num_key_value_heads=8,
        head_dim=64, num_hidden_layers=2, layer_types=["full_attention"] * 2,
    )
    try:
        set_experts_policy("dense")
        assert build_per_track_text_config(cfg, 64)._experts_implementation == IMPL_NAME
        set_experts_policy("grouped_mm")
        assert build_per_track_text_config(cfg, 64)._experts_implementation == "grouped_mm"
    finally:
        set_experts_policy("auto")
