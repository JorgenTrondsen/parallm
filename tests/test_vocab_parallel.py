"""Vocab-parallel embed + KL/CE parity tests.

Three checks:
  1. ``_kl_ce_vocab_parallel`` with world_size=1 (single shard = full vocab,
     no process group) reproduces the existing dense ``_kl_ce_chunked`` —
     same KL/CE values AND same grads w.r.t. the hidden state and lm_head.
  2. ``VocabParallelEmbedding`` partials, masked to disjoint vocab ranges,
     sum to a plain ``nn.Embedding`` lookup.
  3. A 2-rank (gloo, CPU) run: the all-reduced KL/CE equals the dense value,
     the per-shard lm_head grad equals the matching rows of the dense grad,
     and the summed hidden grad equals the dense hidden grad — i.e. the
     vocab-parallel reductions stitch back to the dense result.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from pt_converter.model.vocab_parallel import VocabParallelEmbedding, vocab_range
from pt_converter.train.distill import _kl_ce_chunked, _kl_ce_vocab_parallel


def _tiny_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=2,
        layer_types=["linear_attention"] * 3 + ["full_attention"]
        + ["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4, vocab_size=128, rms_norm_eps=1e-6,
    )


def test_vp_model_embed_matches_legacy_and_gather_roundtrips():
    """world=1 PTWrappedModel(vocab_parallel=True): embed == legacy embed, and
    gather reconstructs the full embed/lm_head that was sharded on load."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from pt_converter.model.pt_model import PTWrappedModel
    from pt_converter.slicer.convert import slice_model_to_tracks

    cfg = _tiny_config()
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, std=0.02)
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=2, sync_block_depth=4, text_config_attr="config"
    )
    full_embed = tracks[0]["embed_tokens.weight"]
    full_lm_head = tracks[0]["lm_head.weight"]

    common = dict(text_config=cfg, n_tracks=2, local_track_ids=(0, 1),
                  sync_after_layers=manifest.sync_layer_indices, track_group=None)
    legacy = PTWrappedModel(**common).eval()
    legacy.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=False)

    vp = PTWrappedModel(**common, vocab_parallel=True, vp_world_size=1, vp_rank=0).eval()
    vp.load_track_state_dicts({0: tracks[0], 1: tracks[1]}, strict=True)
    vp.load_vocab_parallel_weights(full_embed, full_lm_head)

    assert vp.text_models[0].embed_tokens is None  # not hosted per-track in VP
    assert vp.lm_head.weight.shape == (cfg.vocab_size, cfg.hidden_size)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        assert torch.allclose(vp.embed(input_ids), legacy.embed(input_ids), atol=1e-6)
        h, _ = vp(input_ids=input_ids, return_hidden_pre_lm_head=True)
    assert h.shape == (1, 10, cfg.hidden_size) and torch.isfinite(h).all()

    embed_g, lm_g = vp.gather_vocab_parallel_weights()  # world=1 → full == loaded
    assert torch.allclose(embed_g, full_embed, atol=1e-6)
    assert torch.allclose(lm_g, full_lm_head, atol=1e-6)


def _fixture(seed=0, B=2, T=12, D=16, V=40):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, T, D, generator=g)
    lm_w = torch.randn(V, D, generator=g) * 0.05
    teacher_logits = torch.randn(B, T, V, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g)
    labels[0, 3] = -100  # exercise ignore_index
    attn = torch.ones(B, T)
    attn[1, -2:] = 0  # exercise padding mask
    return hidden, lm_w, teacher_logits, labels, attn


@pytest.mark.parametrize("lambda_logit_mse", [0.0, 0.3])
@pytest.mark.parametrize("kl_temperature", [1.0, 2.0])
@pytest.mark.parametrize("chunk_size", [64, 5])
def test_vp_klce_world1_matches_dense(kl_temperature, chunk_size, lambda_logit_mse):
    hidden0, lm_w, teacher_logits, labels, attn = _fixture()
    V, D = lm_w.shape
    kw = dict(lambda_kl=1.0, lambda_ce=0.5, lambda_logit_mse=lambda_logit_mse,
              kl_temperature=kl_temperature, chunk_size=chunk_size)

    # Dense reference. The helper now RETURNS the grad w.r.t. hidden; the caller
    # drives the backward (one traversal, combinable with other graph-rooted losses).
    h_dense = hidden0.clone().requires_grad_(True)
    lm_dense = nn.Linear(D, V, bias=False)
    lm_dense.weight.data.copy_(lm_w)
    kl_d, ce_d, lm_d, grad_h_d = _kl_ce_chunked(
        h_dense, lm_dense, teacher_logits, labels, attn, **kw
    )
    h_dense.backward(grad_h_d)

    # Vocab-parallel, single shard (world_size=1, no group).
    h_vp = hidden0.clone().requires_grad_(True)
    lm_vp = nn.Linear(D, V, bias=False)
    lm_vp.weight.data.copy_(lm_w)
    kl_v, ce_v, lm_v, grad_h_v = _kl_ce_vocab_parallel(
        h_vp, lm_vp, 0, V, teacher_logits, labels, attn,
        group=None, world_size=1, vocab_size=V, **kw,
    )
    h_vp.backward(grad_h_v)

    assert torch.allclose(kl_d, kl_v, atol=1e-5, rtol=1e-4), (kl_d, kl_v)
    assert torch.allclose(ce_d, ce_v, atol=1e-5, rtol=1e-4), (ce_d, ce_v)
    assert torch.allclose(lm_d, lm_v, atol=1e-5, rtol=1e-4), (lm_d, lm_v)
    assert torch.allclose(h_dense.grad, h_vp.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(lm_dense.weight.grad, lm_vp.weight.grad, atol=1e-5, rtol=1e-4)

    # logit-MSE value parity against the standalone dense losses.logit_mse formula.
    from pt_converter.train.losses import logit_mse
    student_logits = hidden0 @ lm_w.t()
    expected_lm = logit_mse(student_logits, teacher_logits, attn, center=True)
    if lambda_logit_mse == 0.0:
        assert lm_v.detach().item() == 0.0
    else:
        assert torch.allclose(lm_v, expected_lm, atol=1e-5, rtol=1e-4), (lm_v, expected_lm)


def test_vp_embedding_partials_sum_to_dense():
    V, D, world = 40, 16, 4
    g = torch.Generator().manual_seed(1)
    full = torch.randn(V, D, generator=g)
    dense = nn.Embedding(V, D)
    dense.weight.data.copy_(full)
    input_ids = torch.randint(0, V, (2, 9), generator=g)

    summed = torch.zeros(2, 9, D)
    for r in range(world):
        lo, hi = vocab_range(V, world, r)
        vpe = VocabParallelEmbedding(V, D, lo, hi)
        vpe.weight.data.copy_(full[lo:hi])
        summed = summed + vpe(input_ids)

    assert torch.allclose(summed, dense(input_ids), atol=1e-6)


# ----- 2-rank distributed parity (gloo / CPU) -----

def _worker(rank, world_size, port, payload, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    hidden0, lm_w, teacher_logits, labels, attn = payload
    V, D = lm_w.shape
    lo, hi = vocab_range(V, world_size, rank)

    h = hidden0.clone().requires_grad_(True)
    lm = nn.Linear(D, hi - lo, bias=False)
    lm.weight.data.copy_(lm_w[lo:hi])
    # teacher_logits is now passed as this rank's vocab shard (B,T,Vs). The
    # centered logit-MSE's per-token mean is over the FULL vocab, so this also
    # exercises the cross-shard all-reduce of Σ_v d and Σ_v d².
    kl, ce, lmse, grad_h = _kl_ce_vocab_parallel(
        h, lm, lo, hi, teacher_logits[:, :, lo:hi], labels, attn,
        lambda_kl=1.0, lambda_ce=0.5, lambda_logit_mse=0.3, vocab_size=V,
        kl_temperature=2.0, chunk_size=7,
        group=dist.group.WORLD, world_size=world_size,
    )
    h.backward(grad_h)
    # Sum hidden grad across ranks → should equal the dense hidden grad.
    h_grad = h.grad.clone()
    dist.all_reduce(h_grad, op=dist.ReduceOp.SUM)
    results[rank] = {
        "kl": kl.item(),
        "ce": ce.item(),
        "lmse": lmse.item(),
        "lm_grad": lm.weight.grad.clone(),
        "h_grad_summed": h_grad,
        "lo": lo, "hi": hi,
    }
    dist.destroy_process_group()


def test_vp_klce_two_rank_matches_dense():
    import torch.multiprocessing as mp

    payload = _fixture(seed=2, V=40)
    hidden0, lm_w, teacher_logits, labels, attn = payload
    V, D = lm_w.shape
    kw = dict(lambda_kl=1.0, lambda_ce=0.5, lambda_logit_mse=0.3,
              kl_temperature=2.0, chunk_size=7)

    # Dense reference.
    h_dense = hidden0.clone().requires_grad_(True)
    lm_dense = nn.Linear(D, V, bias=False)
    lm_dense.weight.data.copy_(lm_w)
    kl_d, ce_d, lm_d, grad_h_d = _kl_ce_chunked(h_dense, lm_dense, teacher_logits, labels, attn, **kw)
    h_dense.backward(grad_h_d)

    world_size = 2
    mgr = mp.Manager()
    results = mgr.dict()
    # find a free port
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_worker, args=(world_size, port, payload, results), nprocs=world_size, join=True)

    for r in range(world_size):
        res = results[r]
        assert abs(res["kl"] - kl_d.item()) < 1e-4, (res["kl"], kl_d.item())
        assert abs(res["ce"] - ce_d.item()) < 1e-4, (res["ce"], ce_d.item())
        # cross-shard centered logit-MSE == dense (full-vocab) logit-MSE
        assert abs(res["lmse"] - lm_d.item()) < 1e-4, (res["lmse"], lm_d.item())
        # per-shard lm_head grad == dense grad rows for this shard
        assert torch.allclose(
            res["lm_grad"], lm_dense.weight.grad[res["lo"]:res["hi"]], atol=1e-5, rtol=1e-4
        )
        # summed hidden grad == dense hidden grad
        assert torch.allclose(res["h_grad_summed"], h_dense.grad, atol=1e-5, rtol=1e-4)
