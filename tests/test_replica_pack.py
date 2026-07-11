"""Rails for the packed replica format (``model/replica_pack.py``).

The packed on-disk form must reconstruct the ``model.replica`` degrade path
bit-exactly (else the cheap file is not the same model), and must actually be
cheap (the packed bytes match the bits/weight budget).
"""
from __future__ import annotations

import torch

torch.set_default_dtype(torch.float32)

from parallm.model.replica import fake_quant_weight, wanda_prune_weight
from parallm.model.replica_pack import (
    _bits_per_weight,
    load_replica_pool,
    pack_sparse_weight,
    save_replica_pool,
    unpack_sparse_weight,
    unpack_sparse_weight_device,
)


def test_device_unpack_bit_exact():
    torch.manual_seed(4)
    w = torch.randn(32, 48, dtype=torch.bfloat16)
    norms = torch.rand(48) + 0.1
    for bits in (None, 4, 8):
        p = pack_sparse_weight(w, 0.5, norms, bits=bits)
        ref = unpack_sparse_weight(p, dtype=torch.bfloat16)
        assert torch.equal(unpack_sparse_weight_device(p), ref), bits
        out = torch.empty(32, 48, dtype=torch.bfloat16)
        assert torch.equal(unpack_sparse_weight_device(p, out=out), ref), bits


def test_wanda_pack_roundtrip_bit_exact():
    torch.manual_seed(0)
    w = torch.randn(32, 48, dtype=torch.bfloat16)
    norms = torch.rand(48) + 0.1
    ref = wanda_prune_weight(w, 0.5, norms)
    got = unpack_sparse_weight(pack_sparse_weight(w, 0.5, norms), dtype=torch.bfloat16)
    assert torch.equal(got, ref)


def test_qwanda_pack_roundtrip_bit_exact():
    torch.manual_seed(1)
    w = torch.randn(32, 48, dtype=torch.bfloat16)
    norms = torch.rand(48) + 0.1
    ref = wanda_prune_weight(fake_quant_weight(w, 4), 0.5, norms)  # the degrade path for bits=4
    got = unpack_sparse_weight(pack_sparse_weight(w, 0.5, norms, bits=4), dtype=torch.bfloat16)
    assert torch.equal(got, ref)


def test_packed_bytes_hit_the_budget():
    torch.manual_seed(2)
    # Realistic in_features so the O(1/in) per-row scale overhead is negligible.
    w = torch.randn(128, 2048, dtype=torch.bfloat16)
    norms = torch.rand(2048) + 0.1
    for bits, frac in [(None, 0.5), (4, 0.5), (4, 0.55)]:
        p = pack_sparse_weight(w, frac, norms, bits=bits)
        stored = sum(t.numel() * t.element_size() for t in p.values())
        budget_bytes = _bits_per_weight(frac, bits) * w.numel() / 8
        # within 15% (row scales + int32 shape are the O(1) slack on a tiny matrix)
        assert stored <= budget_bytes * 1.15, (bits, frac, stored, budget_bytes)


def test_save_load_pool_roundtrip(tmp_path):
    torch.manual_seed(3)
    norms = torch.rand(24) + 0.1
    pool = {}
    for t in range(2):
        w = torch.randn(16, 24, dtype=torch.bfloat16)
        pool[(t, 0, "mlp.gate_proj")] = pack_sparse_weight(w, 0.5, norms, bits=4)
    path = str(tmp_path / "replicas.safetensors")
    save_replica_pool(path, pool, {"config": "qwanda:4:0.5"})
    back, meta = load_replica_pool(path)
    assert meta["config"] == "qwanda:4:0.5"
    assert set(back) == set(pool)
    for key in pool:
        a = unpack_sparse_weight(pool[key])
        b = unpack_sparse_weight(back[key])
        assert torch.equal(a, b)
