"""Unit tests for the per-slab low-rank reparameterization (``model/lowrank_slice.py``).

Tiny stock ``Qwen3_5TextModel`` (CPU, fp32, no dist), n_tracks=4:

* Slab geometry rail: ``slabs_for_spec`` reproduces every SlicerSpec's
  ``slice()`` exactly (the factor sets ARE the per-track copies).
* Round-trip rail: factored model → ``assembled_state_dict`` → strict-load
  into the stock model ⇒ identical forward (the saved ckpt IS the deployed
  function).
* Factored bands have rank ≤ r; ``svd_factor`` is the best rank-r approx;
  grads flow to the factors; ``heal_step`` (serial) trains the factored
  student against the dense teacher.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.eval.dense_parallel import build_windows
from pt_converter.model.lowrank_slice import (
    LowRankSlabLinear,
    _DenseBand,
    _FactorBand,
    apply_lowrank_slicing,
    assembled_state_dict,
    slabs_for_spec,
    svd_factor,
)
from pt_converter.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    KVReplicatedColwise,
    Replicated,
    Rowwise,
)
from pt_converter.train.heal_dense import HealConfig, heal_step
from pt_converter.train.teacher import HookedTeacher

N_TRACKS = 4
RANK = 8


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


def _build_models(rank=RANK):
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    factored = copy.deepcopy(dense)
    stats = apply_lowrank_slicing(factored, cfg, rank, n_tracks=N_TRACKS)
    return dense, factored, stats, cfg


def _batch(cfg, seq=16):
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (2, seq))
    mask = torch.ones((2, seq), dtype=torch.long)
    return ids, mask


# ---------------------------------------------------------------- slab geometry


def test_slabs_match_slicer_specs():
    n = N_TRACKS
    w = torch.randn(24 * n, 40)

    for spec in (Colwise(), Rowwise(dim=0)):
        dim, slabs = slabs_for_spec(spec, tuple(w.shape), n)
        for t in range(n):
            start, size = slabs[t]
            assert torch.equal(w.narrow(dim, start, size), spec.slice(w, t, n))

    spec = GatedQColwise(num_heads=n, head_dim=12)
    wq = torch.randn(n * 2 * 12, 40)
    dim, slabs = slabs_for_spec(spec, tuple(wq.shape), n)
    for t in range(n):
        start, size = slabs[t]
        assert torch.equal(wq.narrow(dim, start, size), spec.slice(wq, t, n))

    spec = KVReplicatedColwise(num_kv_heads=2, sync=False)
    wkv = torch.randn(2 * 10, 40)
    dim, slabs = slabs_for_spec(spec, tuple(wkv.shape), n)
    assert len(slabs) == 2  # one band per kv-GROUP, shared by its tracks
    for t in range(n):
        g = t // (n // 2)
        start, size = slabs[g]
        assert torch.equal(wkv.narrow(dim, start, size), spec.slice(wkv, t, n))

    spec = FusedSegmentColwise(segments=(8 * n, 8 * n, 16 * n))
    wf = torch.randn(32 * n, 40)
    dim, slabs = slabs_for_spec(spec, tuple(wf.shape), n)
    assert len(slabs) == 3 * n
    for t in range(n):
        per_track = torch.cat([wf.narrow(dim, *slabs[s * n + t]) for s in range(3)], dim=dim)
        assert torch.equal(per_track, spec.slice(wf, t, n))

    assert slabs_for_spec(Replicated(), (16, 16), n) is None


# ------------------------------------------------------------------- round trip


def test_assembled_state_dict_roundtrip():
    dense, factored, stats, cfg = _build_models()
    assert stats["modules"] > 0 and stats["bands_factored"] > 0
    assert stats["params_after"] < stats["params_before"]

    sd = assembled_state_dict(factored, factored.state_dict())
    reloaded = Qwen3_5TextModel(cfg).eval()
    reloaded.load_state_dict(sd, strict=True)  # stock keys, nothing left over

    ids, mask = _batch(cfg)
    with torch.no_grad():
        h_f = factored(input_ids=ids, attention_mask=mask).last_hidden_state
        h_r = reloaded(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(h_f, h_r, atol=1e-5, rtol=1e-4)


def test_high_rank_leaves_model_untouched():
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    factored = copy.deepcopy(dense)
    # rank >= every band's min dim ⇒ nothing factors, modules stay nn.Linear.
    stats = apply_lowrank_slicing(factored, cfg, 64, n_tracks=N_TRACKS)
    assert stats["modules"] == 0
    assert not any(isinstance(m, LowRankSlabLinear) for m in factored.modules())
    ids, mask = _batch(cfg)
    with torch.no_grad():
        h_d = dense(input_ids=ids, attention_mask=mask).last_hidden_state
        h_f = factored(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.equal(h_d, h_f)


def test_tiny_bands_stay_dense_linear():
    # in_proj_b/a slabs are 1 row wide — never factorable, module left alone.
    _, factored, _, _ = _build_models()
    for layer in factored.layers:
        if getattr(layer, "layer_type", None) == "linear_attention" or hasattr(layer, "linear_attn"):
            assert isinstance(layer.linear_attn.in_proj_b, nn.Linear)
            assert isinstance(layer.linear_attn.in_proj_a, nn.Linear)
            assert isinstance(layer.linear_attn.in_proj_qkv, LowRankSlabLinear)


# ------------------------------------------------------------------ factor math


def test_factored_bands_have_rank_at_most_r():
    _, factored, _, _ = _build_models()
    mod = next(m for m in factored.modules() if isinstance(m, LowRankSlabLinear))
    for band in mod.bands:
        if isinstance(band, _FactorBand):
            assert torch.linalg.matrix_rank(band.materialize().float()) <= RANK


def test_svd_factor_is_best_rank_r():
    torch.manual_seed(3)
    w = torch.randn(24, 40)
    A, B = svd_factor(w, 5)
    err = (w - A @ B).pow(2).sum()
    S = torch.linalg.svdvals(w)
    assert torch.allclose(err, S[5:].pow(2).sum(), rtol=1e-4)
    # balanced factors: comparable scale on both sides
    assert torch.allclose(A.norm(), B.norm(), rtol=0.5)


def test_grads_flow_to_factors():
    _, factored, _, cfg = _build_models()
    factored.train()
    ids, mask = _batch(cfg)
    out = factored(input_ids=ids, attention_mask=mask).last_hidden_state
    out.sum().backward()
    mod = next(m for m in factored.modules() if isinstance(m, LowRankSlabLinear))
    for band in mod.bands:
        if isinstance(band, _FactorBand):
            assert band.A.grad is not None and torch.isfinite(band.A.grad).all()
            assert band.B.grad is not None and torch.isfinite(band.B.grad).all()
        else:
            assert band.W.grad is not None


# ------------------------------------------------------------- heal integration


def test_heal_step_serial_trains_factored_student():
    dense, factored, _, cfg = _build_models()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)
    for prm in lm_head.parameters():
        prm.requires_grad = False
    hcfg = HealConfig(windows=build_windows(8, 2), mode="serial")
    teacher = HookedTeacher(dense, lm_head, hcfg.capture_indices)
    ids, mask = _batch(cfg)

    optim = torch.optim.AdamW(
        [p_ for p_ in factored.layers.parameters() if p_.requires_grad],
        lr=3e-4, weight_decay=0.0,
    )
    losses = []
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        metrics = heal_step(factored, lm_head, teacher, hcfg, ids, mask)
        losses.append(metrics["tf_window_mse"])
        optim.step()
    assert losses[0] > 1e-6  # truncation error is a real healing signal
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < losses[0]
