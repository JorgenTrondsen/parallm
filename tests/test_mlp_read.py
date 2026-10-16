"""MLP read base (`parallm.model.mlp_read`): a local estimate of the unsynced MLP sum that
each track's next token mixer reads at a post-attn-only layer.

The anchors are exact: a FULL-width base makes every track read the true residual, so a
post-attn walk with it must reproduce the `exact` schedule — which holds only if each track
subtracts precisely the base columns it already carries (a double count, or a wrong track
index, breaks it). No base must leave the walk bit-identical; a half-width base must differ
from both. Tiny Qwen3, fp32, one process, N=4.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

torch.set_default_dtype(torch.float32)

N = 4


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Tiny Qwen3, N=4, bases on layers 0-2 at full width (128) and half (64)."""
    import importlib.util
    import json
    import pathlib

    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    from parallm.slicer.convert import slice_model_to_tracks

    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=N, num_key_value_heads=2, head_dim=16, vocab_size=128,
                      max_position_embeddings=64, rms_norm_eps=1e-6)
    torch.manual_seed(42)
    dense = Qwen3Model(cfg).eval()
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name.endswith("layernorm.weight"):
                p.normal_(mean=1.0, std=0.2)   # a non-trivial γ, which the fold must carry
    shards, _ = slice_model_to_tracks(dense, n_tracks=N, sync_block_depth=1,
                                      text_config_attr="config")
    d = tmp_path_factory.mktemp("mlp_read")
    for t, sd in enumerate(shards):
        save_file({k: v.contiguous() for k, v in sd.items()}, str(d / f"track_{t}.safetensors"))
    g = torch.Generator().manual_seed(5)
    save_file({f"layer.{i}": torch.rand(N, cfg.intermediate_size // N, generator=g) for i in range(4)},
              str(d / "score.safetensors"))
    spec = importlib.util.spec_from_file_location(
        "build_trunc_estimator",
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_trunc_estimator.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    full, half = cfg.intermediate_size, cfg.intermediate_size // 2
    for r in (full, half):
        (d / f"alloc_{r}.json").write_text(json.dumps({"r": {str(i): r for i in range(3)}}))
        assert builder.main(["--checkpoint-dir", str(d), "--score", str(d / "score.safetensors"),
                             "--alloc", str(d / f"alloc_{r}.json"),
                             "--out", str(d / f"base_r{r}.safetensors")]) == 0
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(head.weight, std=0.5)
    return cfg, shards, head, d, full, half


def _model(cfg, shards, head, phase, **kw):
    from parallm.model.pt_model import PTWrappedModel

    pt = PTWrappedModel(text_config=cfg, n_tracks=kw.pop("n_tracks", N),
                        local_track_ids=kw.pop("local_track_ids", tuple(range(N))),
                        sync_after_layers=list(range(cfg.num_hidden_layers)), track_group=None, **kw)
    if not kw:
        pt.load_track_state_dicts({t: shards[t] for t in range(N)}, strict=False)
    pt.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    pt.lm_head.load_state_dict(head.state_dict())
    pt.set_sync_phase(phase)
    return pt.eval()


def _base(path, layers=None, **kw):
    from parallm.model.mlp_read import load_mlp_read_base

    return load_mlp_read_base(str(path), layers, tuple(range(N)), N, "cpu", torch.float32, **kw)


def _logits(pt, base=None):
    pt.set_mlp_read(base)
    ids = torch.randint(0, pt.lm_head.out_features, (2, 12), generator=torch.manual_seed(9))
    with torch.no_grad():
        return pt(input_ids=ids)[0]


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _stats(cfg, layers, seed=11):
    """``x_rms`` (Wanda on gate/up) and an orthonormal ``y_U`` (the fold's norm sketch)."""
    H = cfg.hidden_size
    g = torch.Generator().manual_seed(seed)
    out = {}
    for li in layers:
        q, _ = torch.linalg.qr(torch.randn(H, H, generator=g))
        out[f"layers.{li}.y_U"] = q.contiguous()
        out[f"layers.{li}.x_rms"] = torch.rand(H, generator=g) + 0.1
    return out


def test_full_width_base_reproduces_exact(setup):
    cfg, shards, head, d, full, half = setup
    exact = _logits(_model(cfg, shards, head, "exact"))
    pa = _model(cfg, shards, head, "post-attn")
    plain = _logits(pa)
    got = _logits(pa, _base(d / f"base_r{full}.safetensors"))
    assert _rel(plain, exact) > 1e-3          # the read is stale without the base
    assert _rel(got, exact) < 1e-5


def test_no_base_is_bit_identical_and_half_width_is_neither(setup):
    cfg, shards, head, d, full, half = setup
    pa = _model(cfg, shards, head, "post-attn")
    plain = _logits(pa)
    assert torch.equal(_logits(pa, None), plain)
    exact = _logits(_model(cfg, shards, head, "exact"))
    mid = _logits(pa, _base(d / f"base_r{half}.safetensors"))
    assert _rel(mid, plain) > 1e-4 and _rel(mid, exact) > 1e-4


def test_guards(setup):
    cfg, shards, head, d, full, half = setup
    base = _base(d / f"base_r{half}.safetensors")
    with pytest.raises(ValueError, match="NO post-MLP sync"):
        _model(cfg, shards, head, "exact").set_mlp_read(base)
    with pytest.raises(ValueError, match="post-attn sync"):
        _model(cfg, shards, head, "post-mlp").set_mlp_read(base)
    merged = _model(cfg, shards, head, "post-attn", n_tracks=1, local_track_ids=(0,),
                    merge_group=N, exec_groups=N // 2)
    with pytest.raises(ValueError, match="looped"):
        merged.set_mlp_read(base)
    with pytest.raises(ValueError, match="no base for layers"):
        _base(d / f"base_r{half}.safetensors", [3])


# ----- compression: Wanda / N:M pruning of gate/up -----

def _layer(setup, li=1):
    from safetensors.torch import load_file

    cfg, shards, head, d, full, half = setup
    base = load_file(str(d / f"base_r{half}.safetensors"))
    return (*(base[f"layer.{li}.{k}"] for k in ("gate", "up", "down", "index")), _stats(cfg, [li]))


def test_wanda_masks_each_channel_row_by_x_rms(setup):
    from parallm.model.mlp_read import compress_layer
    from parallm.model.replica import wanda_prune_weight

    g, u, dn, idx, stats = _layer(setup)
    t, _ = compress_layer(g, u, dn, idx, stats, 1, {"gu": ("s", 0.5)})
    H, r = g.shape
    for name, w0 in (("gate", g), ("up", u)):
        z = (t[name] == 0).sum(0)          # zeros per channel (column of [H, r])
        assert torch.equal(z, torch.full_like(z, H // 2))
        assert torch.equal(t[name], wanda_prune_weight(w0.float().T, 0.5, stats["layers.1.x_rms"]).T)
    assert torch.equal(t["down"], dn.float())


def test_nm_keeps_n_of_every_m_by_score(setup):
    from parallm.model.mlp_read import _bits, compress_layer

    g, u, dn, idx, stats = _layer(setup)
    t, _ = compress_layer(g, u, dn, idx, stats, 1, {"gu": ("nm", 1, 4)})
    nz = (t["gate"].T != 0).view(g.shape[1], -1, 4).sum(-1)    # per channel row, per group of 4
    assert torch.equal(nz, torch.ones_like(nz))
    # the survivor in each group is the top |w|·x_rms
    w = g.float().T.view(g.shape[1], -1, 4)
    score = w.abs() * stats["layers.1.x_rms"].view(1, -1, 4)
    assert torch.equal((t["gate"].T != 0).view_as(w), score == score.max(-1, keepdim=True).values)
    assert _bits(("nm", 2, 4)) == (32 + 3) / 4 and _bits(("nm", 1, 8)) == (16 + 3) / 8


def test_sequential_pruning_and_its_load_time_path(setup, tmp_path):
    from parallm.model.mlp_read import _prune, compress_layer, parse_compress_spec

    g, u, dn, idx, stats = _layer(setup)
    x = stats["layers.1.x_rms"]
    w = g.float().T
    seq = _prune(w, ("then", [("nm", 3, 4), ("s", 0.35)]), x)
    z = (seq == 0).float().mean(1)
    assert torch.allclose(z, torch.full_like(z, round((0.25 + 0.35 * 0.75) * w.shape[1]) / w.shape[1]))
    assert ((seq != 0) <= (_prune(w, ("nm", 3, 4), x) != 0)).all()   # never revives a 3:4 drop
    t, led = compress_layer(g, u, dn, idx, stats, 1, parse_compress_spec("gu:3:4+s35"))
    zf = float((t["gate"] == 0).float().mean())
    # a combined mask is priced as a bitmap at its realised sparsity
    assert led == 2 * g.numel() * (1 + 16 * (1 - zf)) / 8 + dn.numel() * 2 + idx.numel() * 4
    cfg, shards, head, d, full, half = setup
    allst = _stats(cfg, range(3))
    save_file(allst, str(tmp_path / "stats.safetensors"))
    live = _base(d / f"base_r{half}.safetensors", compress="gu:3:4+s35",
                 stats=str(tmp_path / "stats.safetensors"))
    t, _ = compress_layer(g, u, dn, idx, allst, 1, parse_compress_spec("gu:3:4+s35"))
    assert live.spec == "gu:3:4+s35" and torch.equal(live.g1, t["gate"])
    assert _rel(_logits(_model(cfg, shards, head, "post-attn"), live),
                _logits(_model(cfg, shards, head, "post-attn"), _base(d / f"base_r{half}.safetensors"))) > 1e-4


def test_ledger_prices_a_bitmap_plus_survivors():
    from parallm.model.mlp_read import _bytes

    w = torch.empty(5120, 8192, device="meta")
    per_layer = lambda f: 3 * _bytes(w, f)
    assert abs(32 * per_layer(None) / 2**30 - 7.50) < 0.005
    assert abs(32 * per_layer(0.5) / 2**30 - 4.22) < 0.005
    assert abs(32 * per_layer(0.75) / 2**30 - 2.34) < 0.005


def test_spec_parsing():
    from parallm.model.mlp_read import parse_compress_spec

    assert parse_compress_spec("gu:s50") == {"gu": ("s", 0.5)}
    assert parse_compress_spec("gu:2:4") == {"gu": ("nm", 2, 4)}
    assert parse_compress_spec("gu:3:4+s35") == {"gu": ("then", [("nm", 3, 4), ("s", 0.35)])}
    for bad in ("", "gu:50", "gu:s0", "d:s50", "gu:s50,gu:s60", "gu:4:4", "gu:0:4",
                "gu:s35+3:4", "gu:3:4+s0", "gu:3:4&s35"):
        with pytest.raises(ValueError):
            parse_compress_spec(bad)


# ----- the fold: the base's down projection folded into each track's next q/k/v -----

def test_fold_with_the_exact_scale_is_the_unfolded_read(setup):
    cfg, shards, head, d, full, half = setup
    pa = _model(cfg, shards, head, "post-attn")
    ref = _logits(pa, _base(d / f"base_r{half}.safetensors"))
    folded = _base(d / f"base_r{half}.safetensors")
    assert any(p_.requires_grad for p_ in pa.parameters())
    folded.bind_fold(pa, "exact")
    assert all(b_.grad_fn is None for b_ in folded.buffers())   # no graph kept alive per (layer, track)
    assert _rel(_logits(pa, folded), ref) < 1e-5
    # r0 drops the hidden-space term: the down projection is gone, one track's F is priced
    r0 = _base(d / f"base_r{half}.safetensors")
    r0.bind_fold(pa, "r0")
    assert not any(n.startswith("d") for n, _ in r0.named_buffers())
    # bf16 ledger: gate + up stay, down [64, 64] → one track's F [64, q 16 + k 16 + v 16]
    assert r0.ledger_bytes == 3 * (2 * 64 * 64 + 64 * 48) * 2
    assert 1e-7 < _rel(_logits(pa, r0), ref) < 0.05   # an approximation, and a small one


def test_fold_norm_sketch_full_rank_is_exact(setup, tmp_path):
    cfg, shards, head, d, full, half = setup
    pa = _model(cfg, shards, head, "post-attn")
    ref = _logits(pa, _base(d / f"base_r{half}.safetensors"))
    st = str(tmp_path / "stats.safetensors")
    save_file(_stats(cfg, range(3)), st)   # orthonormal y_U [64, 64]
    full_r = _base(d / f"base_r{half}.safetensors")
    full_r.bind_fold(pa, f"r{cfg.hidden_size}", stats=st)
    assert _rel(_logits(pa, full_r), ref) < 1e-5
    r0 = _base(d / f"base_r{half}.safetensors")
    r0.bind_fold(pa, "r0")
    r8 = _base(d / f"base_r{half}.safetensors")
    r8.bind_fold(pa, "r8", stats=st)
    assert r8.ledger_bytes == r0.ledger_bytes + 3 * (64 * 8 + 64 * 8) * 2   # D·Y + Y per layer
    with pytest.raises(ValueError, match="y_U"):
        _base(d / f"base_r{half}.safetensors").bind_fold(pa, "r8")
    with pytest.raises(ValueError, match="exact\\|r<M>"):
        _base(d / f"base_r{half}.safetensors").bind_fold(pa, "rx")


def test_fold_norm_constant_closes_the_sketch_and_checks_provenance(setup, tmp_path):
    cfg, shards, head, d, full, half = setup
    pa = _model(cfg, shards, head, "post-attn")
    st = str(tmp_path / "stats.safetensors")
    save_file(_stats(cfg, range(3)), st)
    cal = _base(d / f"base_r{half}.safetensors")
    cal.start_norm_calibration(st, [2, 64])
    _logits(pa, cal)                                  # the unfolded deployment walk
    c = cal.finish_norm_calibration()
    for li in range(3):
        assert abs(c[(li, 64)]) < 1e-5 * (1 + abs(c[(li, 2)]))   # full-rank sketch misses nothing
        assert c[(li, 2)] > 0                                    # a rank-2 one under-states the RMS
    cpath = str(tmp_path / "c.safetensors")
    save_file({f"layer.{li}.c_r2": torch.tensor(c[(li, 2)]) for li in range(3)}, cpath,
              metadata={"source": f"base_r{half}.safetensors", "spec": ""})

    def fold(norm, **kw):
        b = _base(d / f"base_r{half}.safetensors")
        b.bind_fold(pa, norm, stats=st, **kw)
        return _logits(pa, b)

    exact = fold("exact")
    assert _rel(fold("r2+c", norm_c=cpath), exact) < _rel(fold("r2"), exact)
    save_file({f"layer.{li}.c_r2": torch.tensor(0.0) for li in range(3)}, cpath,
              metadata={"source": f"base_r{half}.safetensors", "spec": "gu:s50"})
    with pytest.raises(ValueError, match="calibrated on"):
        fold("r2+c", norm_c=cpath)
    with pytest.raises(ValueError, match="norm_c"):
        fold("r2+c")
