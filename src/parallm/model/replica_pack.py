"""Packed on-disk format for the cheap replica pool.

``degrade_track_layers`` returns bf16 layers with zeros — a *simulation* of the
sparse copy, the same size as dense. To actually realize the memory frontier
(qwanda:4:0.5 ≈ 3 bits/weight) the copies must be stored **packed**: a survivor
bitmap (1 bit/weight) plus either int-``bits`` survivor codes + per-row scales
(qwanda) or bf16 survivor values (wanda). This module packs one degraded Linear
weight, and reassembles all 16 tracks' packed pools into / out of a single
``.safetensors`` file.

The pack↔unpack round-trip reproduces the ``model.replica`` degrade path
bit-exactly (rail: ``tests/test_replica_pack.py``), so a packed pool is
interchangeable with the simulated one — it is just the cheap on-disk form.
"""
from __future__ import annotations

import numpy as np
import torch

from parallm.model.replica import (
    block_wanda_prune_weight,
    wanda_prune_weight,
)


def _bits_per_weight(frac: float, bits: "int | None") -> float:
    """Amortized storage cost of one degraded weight: 1-bit survivor bitmap +
    (1−frac) survivors at ``bits`` (qwanda) or 16 bits (bf16 wanda). Row scales
    are O(1/in_features) and ignored."""
    value_bits = bits if bits is not None else 16
    return 1.0 + (1.0 - frac) * value_bits


def pack_sparse_weight(w: torch.Tensor, frac: float, in_norms: torch.Tensor, *,
                       bits: "int | None" = None, block: "int | None" = None) -> dict:
    """Pack one degraded Linear weight into cheap tensors.

    Mirrors ``model.replica.degrade`` exactly: ``bits`` set ⇒ quantize per-row
    (absmax, qmax = 2^(bits−1)−1) FIRST, then wanda-prune the quantized values;
    else wanda-prune the raw weight. Returns a dict of small tensors (all
    safetensors-storable): ``shape``, ``mask`` (packed bitmap), and either
    ``vals`` (bf16 survivors) or ``codes`` (int survivors, nibble-packed for
    bits==4) + ``scale`` (fp32 per-row) + ``bits``.
    """
    out, inf = int(w.shape[0]), int(w.shape[1])
    if bits is not None:
        qmax = float(2 ** (bits - 1) - 1)
        wf = w.float()
        scale = (wf.abs().amax(dim=1) / qmax).clamp(min=1e-12)          # [out]
        codes = torch.round(wf / scale[:, None]).clamp(-qmax - 1, qmax)  # [out, in] integer-valued
        wq = (codes * scale[:, None]).to(w.dtype)                        # == fake_quant_weight(w, bits)
    else:
        scale = codes = None
        wq = w
    w_deg = (block_wanda_prune_weight(wq, frac, in_norms, block) if block is not None
             else wanda_prune_weight(wq, frac, in_norms))
    mask = w_deg != 0
    packed = {
        "shape": torch.tensor([out, inf], dtype=torch.int32),
        "mask": torch.from_numpy(np.packbits(mask.cpu().numpy().reshape(-1))),
    }
    if bits is not None:
        surv = (codes[mask] + (2 ** (bits - 1))).to(torch.uint8)  # signed→unsigned [0, 2^bits-1]
        packed["codes"] = _pack_nibbles(surv) if bits == 4 else surv
        packed["scale"] = scale.to(torch.float32)
        packed["bits"] = torch.tensor([bits], dtype=torch.int32)
    else:
        packed["vals"] = w_deg[mask].to(w.dtype)
    return packed


_FIXED_EXPERT_FIELDS = ("shape", "mask", "scale", "bits")  # equal size across experts


def pack_expert_slab(w: torch.Tensor, frac: float, expert_in_norms: torch.Tensor, *,
                     bits: "int | None" = None, block: "int | None" = None) -> dict:
    """Pack a fused MoE expert slab ``[E, out, in]`` — one ``pack_sparse_weight`` per
    expert with that expert's own input norms ``expert_in_norms[E, in]``.

    The mask/scale/shape/bits fields are equal size across experts and stack on a
    leading E dim; the survivor values (``codes``/``vals``) are RAGGED — after int
    quantization a weight can round to exactly zero, so the survivor count varies
    per expert — so they are concatenated with a per-expert length vector ``vlen``.
    """
    if w.ndim != 3:
        raise ValueError(f"pack_expert_slab expects [E, out, in], got {tuple(w.shape)}")
    E = w.shape[0]
    if expert_in_norms.shape != (E, w.shape[2]):
        raise ValueError(f"expert_in_norms {tuple(expert_in_norms.shape)} != [E={E}, in={w.shape[2]}]")
    packs = [pack_sparse_weight(w[e], frac, expert_in_norms[e], bits=bits, block=block) for e in range(E)]
    vfield = "codes" if "codes" in packs[0] else "vals"

    out = {"num_experts": torch.tensor([E], dtype=torch.int32),
           "vals_are_codes": torch.tensor([vfield == "codes"], dtype=torch.int32)}
    for field in _FIXED_EXPERT_FIELDS:
        if field in packs[0]:
            out[field] = torch.stack([p[field] for p in packs], dim=0)
    out["vlen"] = torch.tensor([p[vfield].shape[0] for p in packs], dtype=torch.int64)
    out[vfield] = torch.cat([p[vfield] for p in packs], dim=0)
    return out


def unpack_expert_slab(packed: dict, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Inverse of ``pack_expert_slab`` → the dense degraded slab ``[E, out, in]``."""
    E = int(packed["num_experts"].item())
    vfield = "codes" if int(packed["vals_are_codes"].item()) else "vals"
    lens = packed["vlen"].tolist()
    off = 0
    mats = []
    for e in range(E):
        pe = {f: packed[f][e] for f in _FIXED_EXPERT_FIELDS if f in packed}
        pe[vfield] = packed[vfield][off:off + lens[e]]
        off += lens[e]
        mats.append(unpack_sparse_weight(pe, dtype))
    return torch.stack(mats, dim=0)


def unpack_sparse_weight(packed: dict, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Inverse of ``pack_sparse_weight`` → the dense degraded weight (``dtype``)."""
    out, inf = (int(x) for x in packed["shape"].tolist())
    mask = torch.from_numpy(
        np.unpackbits(packed["mask"].cpu().numpy())[: out * inf].astype(bool)
    ).reshape(out, inf)
    w = torch.zeros(out, inf, dtype=dtype)
    nnz = int(mask.sum())
    if "vals" in packed:
        w[mask] = packed["vals"].to(dtype)
    else:
        bits = int(packed["bits"].item())
        codes = _unpack_nibbles(packed["codes"], nnz) if bits == 4 else packed["codes"]
        signed = codes.to(torch.float32) - float(2 ** (bits - 1))
        rows = torch.nonzero(mask, as_tuple=False)[:, 0]
        w[mask] = (signed * packed["scale"][rows].float()).to(dtype)
    return w


def _pack_nibbles(u4: torch.Tensor) -> torch.Tensor:
    """Pack a 1-D uint8 tensor of 4-bit values (0..15) two-per-byte."""
    a = u4.numpy().astype(np.uint8)
    if a.size % 2:
        a = np.append(a, np.uint8(0))
    return torch.from_numpy((a[0::2] << 4) | a[1::2])


def _unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of ``_pack_nibbles``; return the first ``n`` values."""
    b = packed.numpy().astype(np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b >> 4
    out[1::2] = b & 0x0F
    return torch.from_numpy(out[:n].copy())


def save_replica_pool(path: str, pool: "dict[tuple[int, int, str], dict]",
                      metadata: "dict[str, str]") -> None:
    """Write all tracks' packed Linears to one safetensors file.

    ``pool`` maps ``(track_id, layer_idx, rel)`` → ``pack_sparse_weight`` output;
    fields are flattened to keys ``"{track}|{layer}|{rel}|{field}"``. Config
    (frac / bits / sync-depth / selection) goes in the file metadata.
    """
    from safetensors.torch import save_file

    flat: dict[str, torch.Tensor] = {}
    for (t, li, rel), fields in pool.items():
        for field, tensor in fields.items():
            flat[f"{t}|{li}|{rel}|{field}"] = tensor.contiguous()
    save_file(flat, path, metadata=metadata)


def load_replica_pool(path: str) -> "tuple[dict[tuple[int, int, str], dict], dict]":
    """Inverse of ``save_replica_pool`` → ``(pool, metadata)``."""
    from safetensors import safe_open

    pool: dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            t, li, rel, field = key.split("|")
            pool.setdefault((int(t), int(li), rel), {})[field] = f.get_tensor(key)
    return pool, metadata
