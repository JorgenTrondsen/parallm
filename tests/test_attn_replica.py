"""Attention replica rails (`parallm.model.attn_replica`, `PTWrappedModel.set_attn_replica`).

The claims under test: an EXACT replica read is the dense attention, so every track's MLP
reads the same thing and the column-sliced shards sum to the DENSE MLP; the read stays OUT of
the residual, which keeps carrying each track's exact heads; folding ``o_proj`` into the
tracks' own gate/up reproduces that read while storing no ``o_proj`` at all; and the pricing
is the one the frontier table quotes.

Every test was mutation-checked while writing it (let the read into the residual, give a
track another track's fold, drop the ``inv_s`` scaling) — each mutation turns one red.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from parallm.model.attn_replica import (
    CodedLinear,
    _parse,
    build_replica,
    fold_spec,
    norm_c_key,
    replica_memory,
)
from parallm.model.replica_code import (
    PAIRED,
    SCATTER_BLOCK,
    _decode_vals,
    _pair,
    _unpair,
    coded_bits,
    decode,
    encode,
)
from parallm.model.replica_pack import unpack_sparse_weight_device
from parallm.model.pt_model import PTWrappedModel
from parallm.model.replica import wanda_prune_weight
from parallm.slicer.convert import slice_model_to_tracks

N_TRACKS = 4
NUM_KV = 2
LAYERS = 4
# Layer 0 exact, the rest post-MLP only: 1-3 then hold the shared R the replica needs (no
# post-attn sync, and each one's predecessor does sync post-MLP). The shape of the real cell.
CELL = "0:exact,1-3:post-mlp"
REP = (1, 2, 3)


def _cfg() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=64, intermediate_size=32, num_hidden_layers=LAYERS,
        num_attention_heads=N_TRACKS, num_key_value_heads=NUM_KV, head_dim=16,
        vocab_size=128, max_position_embeddings=64, rms_norm_eps=1e-6,
    )


def _dense(seed: int = 42):
    cfg = _cfg()
    torch.manual_seed(seed)
    dense = Qwen3Model(cfg).eval()
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                p.normal_(mean=1.0, std=0.05)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    return cfg, dense


def _pt(dense, cfg, phase: str = CELL):
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=1, text_config_attr="config")
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=N_TRACKS, local_track_ids=tuple(range(N_TRACKS)),
        sync_after_layers=manifest.sync_layer_indices, track_group=None,
    ).eval()
    pt.load_track_state_dicts({t: tracks[t] for t in range(N_TRACKS)}, strict=True)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())
    pt.set_sync_phase(phase)
    return pt


def _rep(cfg, dense, spec="exact", bases=None, layers=REP):
    return build_replica(cfg, layers, dense.state_dict(), spec, bases)


def _ids(cfg, n=12):
    torch.manual_seed(5)
    return torch.randint(0, cfg.vocab_size, (1, n))


def _relmse(a, b):
    return float((a - b).pow(2).sum() / b.pow(2).sum().clamp(min=1e-12))


def _o_bases(cfg, r=1, seed=3):
    """A deliberately WRONG rank-r output basis for `o`, so the replica's estimate is far
    from the truth and any leak of it into the residual is visible."""
    g = torch.Generator().manual_seed(seed)
    H = cfg.hidden_size
    return {f"layers.{i}.o.U": torch.linalg.qr(torch.randn(H, H, generator=g))[0][:, :r]
            for i in REP}


def _o_rms(cfg, seed=23):
    g = torch.Generator().manual_seed(seed)
    O = cfg.num_attention_heads * cfg.head_dim
    return {f"layers.{i}.o_rms": 10 * torch.rand(O, generator=g) + 0.1 for i in REP}


def _x_rms(cfg, seed=31):
    """The per-channel ``‖x_j‖`` a pruned q/k/v is scored with."""
    g = torch.Generator().manual_seed(seed)
    return {f"layers.{i}.x_rms": 10 * torch.rand(cfg.hidden_size, generator=g) + 0.1
            for i in REP}


def _q_bases(cfg, r=8, seed=7):
    """A q basis + latent RMS, so ``q:RANK/sPCT`` can be built in the unit-test model."""
    g = torch.Generator().manual_seed(seed)
    Q = cfg.num_attention_heads * cfg.head_dim
    out = {}
    for i in REP:
        out[f"layers.{i}.q.U"] = torch.linalg.qr(torch.randn(Q, Q, generator=g))[0][:, :r]
        out[f"layers.{i}.q.z_rms"] = torch.sort(
            torch.rand(r, generator=g) + 0.1, descending=True).values
    return out


def _heads_from_weights(pt, input_ids, R, li):
    """Every track's attention output at ``li`` recomputed from the shared ``R`` with that
    track's OWN module — what a node holding copies computes. No collective."""
    tm0 = pt.text_models[0]
    with torch.no_grad():
        h = pt.embed(input_ids)
        pos, text_pos = tm0._resolve_position_ids(h, None)
        masks = pt._adapter.build_masks(tm0.config, h, None, text_pos)
        pe = tm0.rotary_emb(h, pos)
        mask = masks[tm0.config.layer_types[li]]
        return {
            t: tm.layers[li].self_attn(
                hidden_states=tm.layers[li].input_layernorm(R), attention_mask=mask,
                position_ids=text_pos, past_key_values=None, position_embeddings=pe)[0]
            for t, tm in enumerate(pt.text_models)
        }


def _scaffold(pt, ids, li):
    tm0 = pt.text_models[0]
    h = pt.embed(ids)
    pos, text_pos = tm0._resolve_position_ids(h, None)
    masks = pt._adapter.build_masks(tm0.config, h, None, text_pos)
    return tm0.rotary_emb(h, pos), masks[tm0.config.layer_types[li]]


def _mlp_delta(layer, x):
    return layer.mlp(layer.post_attention_layernorm(x))


# --------------------------------------------------------------------------- #
# 1. The spec and its price
# --------------------------------------------------------------------------- #

def test_specs_are_parsed_and_refused():
    ranks, fracs, fold, code = _parse("q:1024/s50,k:s60,v:s60,o:fold/s60,norm:r32+c")
    assert ranks == {"q": 1024} and fracs == {"q": 0.5, "k": 0.6, "v": 0.6} and not code
    assert (fold.frac, fold.norm, fold.rank, fold.bias) == (0.6, "r32+c", 32, True)
    assert fold.estimator == "r32" == norm_c_key(7, fold.estimator).rsplit(".", 1)[1]
    assert _parse("exact") == ({}, {}, None, False) and fold_spec("q:8,k:s50") is None
    assert _parse("q:1024/s50,code")[3] is True and _parse("k:s50")[3] is False
    assert _parse("o:fold/s60,norm:r0,code")[3] is True
    for bad_code in ("q:8,code", "exact,code", "k:s50,code,code", "code:1", "o:fold,norm:r0,code"):
        with pytest.raises(ValueError):
            _parse(bad_code)
    e = fold_spec("o:fold,norm:exact")
    assert e.frac is None and e.norm == "exact" and e.rank == 0
    assert fold_spec("o:fold,norm:r0").norm == "r0" and not fold_spec("o:fold,norm:r0").bias
    for bad in ("o:fold", "norm:r8", "o:fold,norm:x", "o:fold,norm:8", "o:fold,norm:r-1",
                "q:fold,norm:r8", "o:fold,o:s50,norm:r8", "o:8,o:fold,norm:r8",
                "o:fold/60,norm:r8", "o:fold,norm:r8,norm:r8", "o:fold/s0,norm:r8",
                "o:fold/s100,norm:r8", "o:fold,norm:", "o:fold,norm:c", "o:fold,norm:exact+c",
                "q:0", "q:s0", "q:s100", "z:8", "q:8,q:s50", "q:/s50"):
        with pytest.raises(ValueError):
            _parse(bad)
    cfg, dense = _dense()
    with pytest.raises(ValueError, match="needs bases"):
        _rep(cfg, dense, "o:fold,norm:r2")
    with pytest.raises(ValueError, match="o_rms"):
        _rep(cfg, dense, "o:fold/s50,norm:r1", _o_bases(cfg))
    with pytest.raises(ValueError, match="exceeds"):
        _rep(cfg, dense, "o:fold,norm:r2", _o_bases(cfg, r=1))
    with pytest.raises(ValueError, match="norm_c.r0"):
        _rep(cfg, dense, "o:fold,norm:r0+c", _o_rms(cfg))
    rep = _rep(cfg, dense, "o:fold,norm:r0")  # a dense fold reading rms(R) needs no bases
    assert all(isinstance(rep.attn[str(i)].o_proj, nn.Identity) for i in REP)
    assert not any(n.endswith("o_proj.weight") for n, _ in rep.named_parameters())


def test_memory_matches_the_32b_accounting():
    """The frontier table, pinned: a changed pricing formula must change a number here."""
    c = SimpleNamespace(hidden_size=5120, num_attention_heads=64, num_key_value_heads=8,
                        head_dim=128, intermediate_size=25600)
    assert replica_memory(c, "exact", 32)["gib"] == pytest.approx(5.625, abs=2e-3)
    base = replica_memory(c, "q:320,o:640", 32)
    assert base["gib"] == pytest.approx(1.387, abs=2e-3)
    # Net of the track's own kv head: the replica need not hold a second copy of it.
    assert base["gib"] - base["gib_net"] == pytest.approx(0.625 / 8, abs=1e-9)
    # Pruned: one bitmap bit per weight plus the bf16 survivors, 1 + 16(1-f).
    both = replica_memory(c, "q:320,o:640,k:s50,v:s50", 32)
    assert both["gib"] == pytest.approx(base["gib"] - 0.625 + 0.625 * 9 / 16, abs=1e-9)
    # The kv dedup survives pruning (row-separable), not truncation.
    assert both["gib"] - both["gib_net"] == pytest.approx(0.625 * 9 / 16 / 8, abs=1e-9)
    trunc = replica_memory(c, "q:320,o:640,k:256,v:256", 32)
    assert trunc["gib_net"] == trunc["gib"]
    # Coded: 1 + 12(1-f) where pruned; the unpruned q/o factors stay at 16.
    assert replica_memory(c, "q:320,o:640,k:s50,v:s50,code", 32)["gib"] == pytest.approx(
        base["gib"] - 0.625 + 0.625 * 7 / 16, abs=1e-9)
    # Rank, then density: the rank-r factors pay the pruned bits per weight.
    assert replica_memory(c, "q:1024/s60,o:1024/s60", 32)["gib"] == pytest.approx(
        5.625 - 5.0 + 1.625 * 7.4 / 16, abs=2e-3)
    # THE FRONTIER ARM: q rank-1024/50%, k/v full rank 60%, o folded and pruned 60%,
    # rank-32 norm sketch. 0.952 GiB at macro4 0.7628 — the teacher and the exact replica.
    best = replica_memory(c, "q:1024/s50,k:s60,v:s60,o:fold/s60,norm:r32+c", 32, n_tracks=64)
    assert best["gib"] == pytest.approx(0.952, abs=5e-4)
    assert best["gib_fold"] == pytest.approx(0.181, abs=5e-4)
    assert best["gib_sketch"] == pytest.approx(0.025, abs=5e-4)
    assert best["kv_per_token"] == 128 * 1024
    assert best["gib"] - best["gib_net"] == pytest.approx(0.289 / 8, abs=5e-4)  # k+v at s60
    coded = replica_memory(c, "q:1024/s50,k:s60,v:s60,o:fold/s60,norm:r32+c,code", 32,
                           n_tracks=64)
    assert coded["gib"] == pytest.approx(0.749, abs=5e-4)
    assert coded["gib_net"] == pytest.approx(0.721, abs=5e-4)
    assert coded["gib_sketch"] == best["gib_sketch"]  # dense, never coded
    assert coded["gib_fold"] == pytest.approx(best["gib_fold"] * 5.8 / 7.4, abs=5e-4)
    # The fold is [I/N, q_dim] PER TRACK, so it cannot be priced without knowing N. The
    # trap this closes: `n_tracks` is 1 on a merged track, so a default would be silent.
    assert replica_memory(c, "o:fold,norm:r0", 32, n_tracks=128)["gib_fold"] == pytest.approx(
        best["gib_fold"] / 2 * 16 / 7.4, abs=2e-3)
    with pytest.raises(ValueError, match="needs n_tracks"):
        replica_memory(c, "o:fold,norm:r0", 32)


def test_the_codec_is_lossless_and_only_the_exponent_is_coded():
    """`decode(encode(w))` is bit-exact under both coders, and only the exponent is coded."""
    g = torch.Generator().manual_seed(11)
    w = wanda_prune_weight(torch.randn(256, 512, generator=g), 0.5,
                           torch.rand(512, generator=g) + 0.1).to(torch.bfloat16)
    nnz = int((w != 0).sum())
    field, huff = encode(w, "field"), encode(w, "huffman")
    assert torch.equal(decode(field), w) and torch.equal(decode(huff), w)
    assert "mask" in field and int(field["ebits"]) <= 4 and int(huff["ebits"]) == PAIRED
    pre = 32 * -(-w.numel() // SCATTER_BLOCK)
    assert pre == 32 * field["pre"].numel() and pre <= 0.02 * w.numel()
    assert coded_bits(field) == w.numel() + nnz * (8 + int(field["ebits"])) + pre
    assert coded_bits(huff) < coded_bits(field) < w.numel() + 16 * nnz
    assert huff["hi"].numel() == field["hi"].numel() == nnz

    d = (torch.randn(64, 64, generator=g) * torch.exp(
        torch.randn(64, 1, generator=g) * 3)).to(torch.bfloat16)
    pd, hd = encode(d, "field"), encode(d, "huffman")
    assert "mask" not in pd and int(pd["ebits"]) == 8 and int(hd["ebits"]) == 0
    assert torch.equal(decode(pd), d) and torch.equal(decode(hd), d)

    for n in (2048, 2049, 3):  # odd tail
        off = torch.randint(0, 13, (n,), generator=g, dtype=torch.uint8)
        assert torch.equal(_unpair(torch.from_numpy(_pair(off.numpy())), n), off)

    for p, plane in ((field, "hi"), (field, "epack"), (huff, "hi"), (huff, "epack")):
        bad = dict(p, **{plane: p[plane].clone()})  # mutation: one flipped bit
        bad[plane][0] ^= 1
        assert not torch.equal(decode(bad), w)
    with pytest.raises(ValueError, match="bf16"):
        encode(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="entirely zero"):
        encode(torch.zeros(4, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="coder must be"):
        encode(w, "lzma")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_fused_scatter_kernel_matches_the_torch_oracle():
    """Planes on the device, header on the host, fused kernel equal to the torch path."""
    g = torch.Generator().manual_seed(23)
    for shape, frac in (((128, 256), 0.5), ((77, 611), 0.6), ((3, SCATTER_BLOCK + 7), 0.4)):
        w = wanda_prune_weight(torch.randn(*shape, generator=g), frac,
                               torch.rand(shape[1], generator=g) + 0.1).to(torch.bfloat16)
        p = encode(w.cuda())
        assert all(p[k].is_cuda for k in ("hi", "epack", "mask", "pre"))
        assert not any(p[k].is_cuda for k in ("shape", "ebase", "ebits", "nnz"))
        oracle = unpack_sparse_weight_device(
            {"shape": p["shape"], "mask": p["mask"], "vals": _decode_vals(p)}, torch.bfloat16)
        assert torch.equal(decode(p), oracle) and torch.equal(decode(p).cpu(), w), shape
        bad = dict(p, pre=p["pre"].clone())  # mutation: a wrong prefix
        bad["pre"][-1] += 1
        assert not torch.equal(decode(bad).cpu(), w), shape


def test_wanda_pruning_is_row_separable_so_a_node_rebuilds_its_own_kv_head():
    """Wanda thresholds each row alone, so a node regenerates its own kv head's pruned rows."""
    g = torch.Generator().manual_seed(3)
    W = torch.randn(128, 64, generator=g)
    rms = torch.rand(64, generator=g) + 0.1
    whole = wanda_prune_weight(W, 0.6, rms)
    for lo in (0, 32, 96):
        rows = slice(lo, lo + 32)
        assert torch.equal(whole[rows], wanda_prune_weight(W[rows], 0.6, rms))
    assert not torch.equal(whole[0:32], wanda_prune_weight(W[0:32], 0.6, rms.flip(0)))


def test_a_coded_replica_is_bit_identical_to_the_dense_one():
    """`code` changes the bytes only: projections, both q factors, the fold and the read."""
    cfg, dense = _dense()
    sd = {k: v.to(torch.bfloat16) for k, v in dense.state_dict().items()}
    bases = {**_o_rms(cfg), **_x_rms(cfg), **_q_bases(cfg)}
    arm = "q:8/s50,k:s50,v:s50,o:fold/s50,norm:r0"
    plain = build_replica(cfg, REP, sd, arm, bases)
    coded = build_replica(cfg, REP, sd, arm + ",code", bases)
    assert coded.code and not plain.code
    for i in REP:
        for name in ("q_proj", "k_proj", "v_proj"):
            a, b = getattr(plain.attn[str(i)], name), getattr(coded.attn[str(i)], name)
            for pm, cm in (zip(a, b) if isinstance(a, nn.Sequential) else [(a, b)]):
                assert isinstance(cm, CodedLinear) and torch.equal(cm.weight, pm.weight)
    with pytest.raises(ValueError, match="stores bf16"):
        _pt(dense, cfg).set_attn_replica(build_replica(cfg, REP, sd, arm + ",code", bases))
    pt = _pt(dense, cfg).to(torch.bfloat16)
    pt.set_attn_replica(coded)
    pt.set_attn_replica(plain)
    for i in REP:
        assert isinstance(coded._fold[i][0], list)
        for c, p in zip(coded._fold_at(i), plain._fold_at(i)):
            assert torch.equal(c, p)

    ids = _ids(cfg)
    with torch.no_grad():
        R = pt(input_ids=ids, return_sync_hiddens=True)[1][REP[0]]
        pe, mask = _scaffold(pt, ids, REP[1])
        for a, b in zip(plain.read(REP[1], R, pe, mask), coded.read(REP[1], R, pe, mask)):
            assert torch.equal(a, b)

    quoted = replica_memory(cfg, arm + ",code", len(REP), n_tracks=N_TRACKS)["gib"] * 2**30
    assert coded.stored_bytes() <= quoted and coded.stored_bytes() < plain.stored_bytes()
    with pytest.raises(RuntimeError, match="eval-only"):
        coded.attn[str(REP[0])].k_proj(torch.zeros(1, cfg.hidden_size, dtype=torch.bfloat16))


# --------------------------------------------------------------------------- #
# 2. The read: shared, and out of the residual
# --------------------------------------------------------------------------- #

def test_an_exact_replica_read_is_the_dense_attention():
    cfg, dense = _dense()
    ids = _ids(cfg)
    with torch.no_grad():
        want = _pt(dense, cfg, "exact")(input_ids=ids)[0]
        pt = _pt(dense, cfg)
        pt.set_attn_replica(_rep(cfg, dense))
        got = pt(input_ids=ids)[0]
        plain = _pt(dense, cfg)(input_ids=ids)[0]
    assert _relmse(got, want) < 1e-8
    assert _relmse(plain, want) > 1e-6, "power: the replica must change the post-mlp walk"


def test_one_shared_read_makes_the_sliced_mlps_the_dense_mlp():
    """Why the replica's read is structurally unlike plain SPD: identical inputs make the
    column-sliced shards sum to the dense MLP exactly; per-track reads do not."""
    cfg, dense = _dense()
    ids, li = _ids(cfg), 2
    pt = _pt(dense, cfg)
    with torch.no_grad():
        R = pt(input_ids=ids, return_sync_hiddens=True)[1][li - 1]
        heads = _heads_from_weights(pt, ids, R, li)
        x = R + sum(heads.values())
        want = _mlp_delta(dense.layers[li], x)
        shared = sum(_mlp_delta(tm.layers[li], x) for tm in pt.text_models)
        own = sum(_mlp_delta(tm.layers[li], R + heads[k]) for k, tm in enumerate(pt.text_models))
    assert _relmse(shared, want) < 1e-9
    assert _relmse(own, want) > 1e-3


def test_the_read_keeps_the_replica_out_of_the_residual():
    """With a badly wrong replica the synced state must still be ``R + Σa_k + Σd_k(R+Â)``:
    the MLPs read the estimate, the carry holds each track's EXACT attention."""
    cfg, dense = _dense()
    ids, li = _ids(cfg), 2
    pt = _pt(dense, cfg)
    rep = _rep(cfg, dense, "o:1", _o_bases(cfg))
    pt.set_attn_replica(rep)
    with torch.no_grad():
        states = pt(input_ids=ids, return_sync_hiddens=True)[1]
        R = states[li - 1]
        pe, mask = _scaffold(pt, ids, li)
        x_hat = R + rep(li, R, pe, mask)
        heads = _heads_from_weights(pt, ids, R, li)
        d = sum(_mlp_delta(tm.layers[li], x_hat) for tm in pt.text_models)
    assert _relmse(states[li], R + sum(heads.values()) + d) < 1e-8
    # The leak this guards against: carrying the replica sums R + Â + Σd instead.
    assert _relmse(states[li], x_hat + d) > 1e-3


def test_the_replica_is_refused_where_the_shared_R_identity_fails():
    cfg, dense = _dense()
    with pytest.raises(ValueError, match="NO post-attn sync"):
        _pt(dense, cfg, "post-attn").set_attn_replica(_rep(cfg, dense))
    with pytest.raises(ValueError, match="post-MLP sync BEFORE"):
        _pt(dense, cfg, "0-1:post-attn,2-3:post-mlp").set_attn_replica(
            _rep(cfg, dense, layers=(2, 3)))
    with pytest.raises(ValueError, match="outside 1"):
        _pt(dense, cfg).set_attn_replica(_rep(cfg, dense, layers=(0, 1)))
    pt = _pt(dense, cfg)
    pt.set_attn_replica(_rep(cfg, dense))
    pt.set_attn_replica(None)
    assert pt._attn_replica is None and not pt._attn_replica_layers


# --------------------------------------------------------------------------- #
# 3. The fold
# --------------------------------------------------------------------------- #

def test_a_folded_o_with_the_exact_scalar_is_the_exact_read():
    """`o:fold,norm:exact`: no o_proj stored, gate_t(γ⊙(R + W_o a)) computed as
    gate_t(γ⊙R) + (gate_t·diag(γ)·W_o)·a — the exact read to rounding. `norm:r0` (the scalar
    from R alone) is a different read, so the scalar path has teeth."""
    cfg, dense = _dense()
    ids = _ids(cfg)
    with torch.no_grad():
        pt = _pt(dense, cfg)
        pt.set_attn_replica(_rep(cfg, dense))
        want = pt(input_ids=ids)[0]
        rep = _rep(cfg, dense, "o:fold,norm:exact")
        folded = _pt(dense, cfg)
        folded.set_attn_replica(rep)
        got = folded(input_ids=ids)[0]
        zero = _pt(dense, cfg)
        zero.set_attn_replica(_rep(cfg, dense, "o:fold,norm:r0"))
        r_only = zero(input_ids=ids)[0]
    O = cfg.num_attention_heads * cfg.head_dim
    for i in REP:
        G, U = rep._fold[i]
        assert G.shape == U.shape == (N_TRACKS, cfg.intermediate_size // N_TRACKS, O)
    assert _relmse(got, want) < 1e-8
    assert _relmse(r_only, want) > 1e-6, "power: the scalar from R alone must change the read"


def test_the_bind_rail_refuses_a_fold_built_from_a_foreign_o_proj():
    """⚠ The fold is ``gate_t·diag(γ)·W_o`` with W_o the TEACHER's. It is right only if each
    track's head really is the teacher's columns ``t·hd:(t+1)·hd`` — misorder them and every
    track reads another head's output through its own gate, which is a plausible number, not
    a crash. `bind_fold`'s provenance rail is what makes it a crash."""
    cfg, dense = _dense()
    pt = _pt(dense, cfg)
    rep = _rep(cfg, dense, "o:fold,norm:r0")
    hd = cfg.head_dim
    # Swap two heads' column blocks in the W_o the fold is built from.
    W = rep._fold_src[REP[0]].clone()
    W[:, :hd], W[:, hd:2 * hd] = W[:, hd:2 * hd].clone(), W[:, :hd].clone()
    rep._fold_src[REP[0]] = W
    with pytest.raises(RuntimeError, match="own o_proj slice"):
        pt.set_attn_replica(rep)


def test_a_pruned_fold_keeps_each_rows_top_scored_weights_exact():
    cfg, dense = _dense()
    pt = _pt(dense, cfg)
    bases = _o_rms(cfg)
    rep = _rep(cfg, dense, "o:fold/s50,norm:r0", bases)
    pt.set_attn_replica(rep)
    sd = dense.state_dict()
    for i in REP:
        W_o, gamma = (sd[f"layers.{i}.self_attn.o_proj.weight"],
                      sd[f"layers.{i}.post_attention_layernorm.weight"])
        G, _ = rep._fold[i]
        for k, tm in enumerate(pt.text_models):
            gate = tm.layers[i].mlp.gate_proj.weight
            want = wanda_prune_weight((gate.float() * gamma.float()) @ W_o.float(), 0.5,
                                      bases[f"layers.{i}.o_rms"])
            assert torch.equal(G[k], want.to(G.dtype))
            assert ((G[k] == 0).sum(1) >= G.shape[-1] // 2).all()


def test_a_pruned_fold_is_not_the_exact_read_but_is_still_the_folded_form():
    """Pruning the fold is per TRACK — each track's read differs, unlike a pruned o, which
    leaves one shared error. The measured curve was free to 70%, but the arithmetic must
    still be exactly ``down(act((gate(γ⊙R) + a·Gᵀ)·inv_s) · …)``."""
    from parallm.model.seam import FoldRead, fold_mlp

    cfg, dense = _dense()
    ids, li = _ids(cfg), 2
    pt = _pt(dense, cfg)
    rep = _rep(cfg, dense, "o:fold/s50,norm:r0", _o_rms(cfg))
    pt.set_attn_replica(rep)
    with torch.no_grad():
        R = pt(input_ids=ids, return_sync_hiddens=True)[1][li - 1]
        pe, mask = _scaffold(pt, ids, li)
        fr = rep.read(li, R, pe, mask)
        assert isinstance(fr, FoldRead)
        # norm:r0 means the scalar is rms(R) alone — no sketch module was built.
        assert str(li) not in rep.sketch
        inv = torch.rsqrt(R.float().pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        assert torch.equal(fr.inv_s, inv)
        layer = pt.text_models[0].layers[li]
        one = fr.track(0)
        g = layer.mlp.gate_proj(layer.post_attention_layernorm.weight * R).float() \
            + fr.a @ one.gate.T.float()
        u = layer.mlp.up_proj(layer.post_attention_layernorm.weight * R).float() \
            + fr.a @ one.up.T.float()
        want = layer.mlp.down_proj(layer.mlp.act_fn((g * inv).float()) * (u * inv).float())
        got = fold_mlp(layer.mlp, layer.post_attention_layernorm, one)
    assert _relmse(got, want) < 1e-10


def test_a_fold_is_refused_on_the_merged_walk():
    """`_run_batched_stack` has no replica branch, so a merged model would DROP it silently
    and score a different network. The refusal is the whole point."""
    cfg, dense = _dense()
    pt = _pt(dense, cfg)
    pt.merge_group = 2  # what a merged convert hands PTWrappedModel
    with pytest.raises(ValueError, match="looped walk only"):
        pt.set_attn_replica(_rep(cfg, dense, "o:fold,norm:r0"))
