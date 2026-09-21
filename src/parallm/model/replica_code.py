"""Lossless entropy coding of a pruned bf16 replica weight.

Sign+mantissa stream raw (one uint8 per survivor); only the exponent is coded — Wanda leaves
survivors in a narrow exponent band. `decode(encode(w))` is bit-exact, so this is not
quantization. CUDA decodes through one fused scatter kernel; the torch path is the oracle.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import torch

from parallm.entropy_codec import (
    build_decode_lut,
    build_table,
    decode_blocks_cpu,
    decode_blocks_gpu,
    encode_blocks,
)
from parallm.model.replica_pack import unpack_sparse_weight_device

# `field` stores the exponent offset in a fixed WIDTHS-wide field, `huffman` entropy-codes it,
# and `auto` keeps the smaller — so a coded plane never costs more than the field.
CODERS = ("auto", "huffman", "field")
WIDTHS = (1, 2, 4, 8)
# `ebits`: > 0 is a field of that width, 0 huffman, PAIRED huffman over two nibbles per byte.
PAIRED = -1
# One lane decodes a block serially, so a small block buys parallelism for a few % of bits.
BLOCK_BYTES = 512
# Output weights per fused-scatter program (a power of two for `tl.cumsum`); one stored int32
# survivor prefix per block.
SCATTER_BLOCK = 2048
# The CPU huffman decoder is bit-serial; past this a round-trip check takes minutes.
_CPU_DECODE_LIMIT = 1 << 20


def _pack_field(off: np.ndarray, bits: int) -> np.ndarray:
    if bits == 8:
        return off
    per = 8 // bits
    a = np.append(off, np.zeros((-off.size) % per, np.uint8))
    out = np.zeros(a.size // per, np.uint8)
    for k in range(per):
        out |= a[k::per] << (8 - bits * (k + 1))
    return out


def _unpack_field(b: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    shifts = torch.arange(8 - bits, -1, -bits, dtype=torch.uint8, device=b.device)
    return ((b.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(-1)[:n]


def _pair(off: np.ndarray) -> np.ndarray:
    a = np.append(off, np.zeros(off.size % 2, np.uint8))
    return ((a[0::2] << 4) | a[1::2]).astype(np.uint8)


def _unpair(p: torch.Tensor, n: int) -> torch.Tensor:
    return torch.stack((p >> 4, p & 0xF), -1).reshape(-1)[:n]


def _exp_field(off: np.ndarray, device) -> dict:
    bits = next(b for b in WIDTHS if int(off.max()) >> b == 0)
    return {"ebits": torch.tensor([bits], dtype=torch.int32),
            "epack": torch.from_numpy(_pack_field(off, bits)).to(device)}


def _exp_huffman(off: np.ndarray, device) -> dict:
    # A paired plane halves the block too, so each lane's chain halves rather than the lanes.
    paired = int(off.max()) < 16
    sym = _pair(off) if paired else off
    lengths = build_table(np.bincount(sym, minlength=256))
    blob, offsets = encode_blocks(sym, lengths, BLOCK_BYTES >> paired)
    return {"ebits": torch.tensor([PAIRED if paired else 0], dtype=torch.int32),
            "elen": torch.from_numpy(lengths),
            "epack": torch.from_numpy(blob).to(device),
            "eoff": torch.from_numpy(offsets).to(device),
            "lut": torch.from_numpy(build_decode_lut(lengths)).to(device)}


def _decode_exp(p: dict, n: int) -> torch.Tensor:
    bits = int(p["ebits"].item())
    if bits > 0:
        return _unpack_field(p["epack"], bits, n)
    paired = bits == PAIRED
    ns, bb = ((n + 1) // 2 if paired else n), BLOCK_BYTES >> paired
    if p["epack"].is_cuda:
        out = decode_blocks_gpu(p["epack"], p["eoff"], p["lut"], ns, bb)
    else:
        out = torch.from_numpy(decode_blocks_cpu(
            p["epack"].numpy(), p["eoff"].numpy(), p["elen"].numpy(), ns, bb))
    return _unpair(out, n) if paired else out


def _prefix(mask: np.ndarray) -> np.ndarray:
    """Survivors before each SCATTER_BLOCK chunk (int32: torch has no CUDA uint32 arithmetic)."""
    c = np.add.reduceat(mask.astype(np.int64), np.arange(0, mask.size, SCATTER_BLOCK))
    return (np.cumsum(c) - c).astype(np.int32)


def encode(w: torch.Tensor, coder: str = "auto") -> dict:
    """Pack a bf16 weight (zeros = pruned); raises unless it round-trips exactly on its device."""
    if w.dtype != torch.bfloat16:
        raise ValueError(f"replica_code stores bf16, got {w.dtype}")
    if coder not in CODERS:
        raise ValueError(f"replica_code coder must be one of {CODERS}, got {coder!r}")
    mask = w != 0
    dense = bool(mask.all())
    vals = (w.reshape(-1) if dense else w[mask]).contiguous().cpu()
    if vals.numel() == 0:
        raise ValueError("replica_code: the weight is entirely zero")
    if coder != "field" and w.device.type == "cpu" and vals.numel() > _CPU_DECODE_LIMIT:
        raise ValueError(f"replica_code: {vals.numel()} survivors are too many to check on the "
                         f"CPU — build on a GPU, or pass coder='field'")
    u = vals.view(torch.int16).numpy().view(np.uint16)
    hi = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.uint8)   # sign<<7 | mantissa
    exp = ((u >> 7) & 0xFF).astype(np.uint8)
    base = int(exp.min())
    # Planes on w's device; the header stays on the host so a decode's .item() costs no sync.
    packed = {
        "shape": torch.tensor(list(w.shape), dtype=torch.int32),
        "hi": torch.from_numpy(hi).to(w.device),
        "ebase": torch.tensor([base], dtype=torch.int32),
        "nnz": torch.tensor([vals.numel()], dtype=torch.int64),
        **min((f(exp - np.uint8(base), w.device) for name, f in
               (("field", _exp_field), ("huffman", _exp_huffman)) if coder in ("auto", name)),
              key=coded_bits),
    }
    if not dense:
        flat = mask.cpu().numpy().reshape(-1)
        packed["mask"] = torch.from_numpy(np.packbits(flat)).to(w.device)
        packed["pre"] = torch.from_numpy(_prefix(flat)).to(w.device)
    if not torch.equal(decode(packed), w):
        raise ValueError(f"replica_code: {tuple(w.shape)} did not round-trip bit-exactly")
    return packed


@cache
def _scatter_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fused_scatter_kernel(mask_ptr, hi_ptr, exp_ptr, pre_ptr, out_ptr,
                             n_weights, ebase, BLOCK: tl.constexpr):
        """Assemble one BLOCK of output weights from the planes; zero where pruned."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        ok = offs < n_weights
        byte = tl.load(mask_ptr + (offs >> 3), mask=ok, other=0).to(tl.int32)
        bit = (byte >> (7 - (offs & 7).to(tl.int32))) & 1           # packbits is MSB-first
        idx = tl.load(pre_ptr + pid).to(tl.int32) + tl.cumsum(bit, 0) - bit
        live = ok & (bit == 1)
        h = tl.load(hi_ptr + idx, mask=live, other=0).to(tl.int32)
        e = tl.load(exp_ptr + idx, mask=live, other=0).to(tl.int32) + ebase
        v = (((h & 0x80) | (e >> 1)) << 8) | ((e << 7) & 0xFF) | (h & 0x7F)
        tl.store(out_ptr + offs, tl.where(bit == 1, v, 0).to(tl.int16), mask=ok)

    return fused_scatter_kernel


def _decode_vals(p: dict) -> torch.Tensor:
    """The survivors as bf16, built in uint8: little-endian bf16 is [e0|m6..m0][s|e7..e1]."""
    hi = p["hi"]
    e = _decode_exp(p, int(p["nnz"].item())) + int(p["ebase"].item())
    return torch.stack(((e << 7) | (hi & 0x7F), (hi & 0x80) | (e >> 1)), -1
                       ).reshape(-1).view(torch.bfloat16)


def decode(packed: dict, out: "torch.Tensor | None" = None) -> torch.Tensor:
    """Inverse of `encode`, on the planes' device; ``out`` is an optional preallocated buffer."""
    shape = tuple(packed["shape"].tolist())
    if "mask" not in packed:
        vals = _decode_vals(packed).reshape(shape)
        return vals if out is None else out.copy_(vals)
    if not packed["hi"].is_cuda:
        return unpack_sparse_weight_device(
            {"shape": packed["shape"], "mask": packed["mask"], "vals": _decode_vals(packed)},
            torch.bfloat16, out)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=packed["hi"].device)
    n, e = out.numel(), _decode_exp(packed, int(packed["nnz"].item()))
    _scatter_kernel()[(-(-n // SCATTER_BLOCK),)](
        packed["mask"], packed["hi"], e, packed["pre"], out.view(torch.int16), n,
        int(packed["ebase"].item()), BLOCK=SCATTER_BLOCK, num_warps=4)
    return out


def coded_bits(packed: dict) -> int:
    """Stored bits, counted from the tensors; the decode LUT is derived, so it is not counted."""
    return sum(packed[k].numel() * packed[k].element_size() * 8
               for k in ("hi", "epack", "mask", "elen", "eoff", "pre") if k in packed)
