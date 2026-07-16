"""Static-table entropy codec for the replica pool's CODES plane.

Measured on the real pools (2026-07-12): int4 survivor codes carry 2.97 bits
of entropy per 4-bit symbol (74.3%; identically 5.93/8 at byte granularity —
the nibbles are iid), int8 codes 6.82/8. The survivor bitmap is at the entropy
limit (density 0.4995, byte entropy 7.997/8) and the per-row scales are ~0.3%
of pool bytes, so ONLY the codes plane is coded; mask/scale stream raw. A
static canonical Huffman code over byte symbols (the distribution is a stable
quantized-Gaussian, shared pool-wide) lands within ~1% of that bound: ~21-26%
fewer PCIe/DRAM bytes, losslessly — decoded bytes are bit-identical, so the
GEMV path is untouched.

Framing: the plane is split into fixed RAW-size blocks (default 8 KB), each
coded independently against the one shared table and starting byte-aligned in
the blob; ``offsets`` holds exact per-block compressed byte starts
(``uint32[n_blocks+1]``), so staging buffers size exactly and every block
decodes independently — one decoder lane per block, thousands in flight. The
blob carries 4 zero pad bytes past ``offsets[-1]`` so the last block's 3-byte
peek window never reads out of bounds.

Code lengths are capped at 12 bits (package-merge, optimal under the cap):
the decode LUT is 4096 × uint16 ``(symbol << 4) | length`` — L1-resident on
the GPU — and a code spans at most 3 bytes at any bit alignment. Bitstream is
MSB-first. Decoding runs in the streamed ring's pump path (eager, during the
sync stall — the GPU is idle there), never inside a CUDA-graph capture.
"""
from __future__ import annotations

import numpy as np
import torch

MAX_CODE_LEN = 12
DEFAULT_BLOCK_BYTES = 8192


def build_table(hist: np.ndarray, max_len: int = MAX_CODE_LEN) -> np.ndarray:
    """Optimal length-limited (package-merge) Huffman code lengths for a byte
    histogram: ``uint8[256]``, 0 = symbol absent."""
    freq = np.asarray(hist, dtype=np.int64)
    assert freq.shape == (256,) and freq.min() >= 0
    syms = np.nonzero(freq)[0]
    lengths = np.zeros(256, dtype=np.uint8)
    if len(syms) == 0:
        raise ValueError("empty histogram")
    if len(syms) == 1:
        lengths[syms[0]] = 1
        return lengths
    # Boundary package-merge: pair-and-merge max_len-1 times; every item kept
    # in the cheapest 2(n-1) of the final list adds one bit to its symbols.
    singles = sorted((int(freq[s]), (int(s),)) for s in syms)
    level = singles
    for _ in range(max_len - 1):
        pairs = [(level[i][0] + level[i + 1][0], level[i][1] + level[i + 1][1])
                 for i in range(0, len(level) - 1, 2)]
        level = sorted(singles + pairs)
    for _w, ss in level[: 2 * (len(syms) - 1)]:
        for s in ss:
            lengths[s] += 1
    kraft = (2.0 ** -lengths[syms].astype(np.float64)).sum()
    assert abs(kraft - 1.0) < 1e-9, kraft  # complete code
    return lengths


def _canonical_codes(lengths: np.ndarray) -> np.ndarray:
    """Canonical code values (MSB-first) for the given lengths: uint16[256]."""
    codes = np.zeros(256, dtype=np.uint16)
    code = 0
    prev_len = 0
    for l in range(1, MAX_CODE_LEN + 1):
        code <<= l - prev_len
        prev_len = l
        for s in np.nonzero(lengths == l)[0]:
            codes[s] = code
            code += 1
    return codes


def build_decode_lut(lengths: np.ndarray) -> np.ndarray:
    """12-bit-peek decode LUT: uint16[4096] of ``(symbol << 4) | length``."""
    codes = _canonical_codes(lengths)
    lut = np.zeros(1 << MAX_CODE_LEN, dtype=np.uint16)
    for s in np.nonzero(lengths)[0]:
        l = int(lengths[s])
        base = int(codes[s]) << (MAX_CODE_LEN - l)
        lut[base: base + (1 << (MAX_CODE_LEN - l))] = (s << 4) | l
    return lut


def encode_blocks(data: np.ndarray, lengths: np.ndarray,
                  block_bytes: int = DEFAULT_BLOCK_BYTES):
    """Encode a byte plane against the shared table. Returns ``(blob uint8,
    offsets uint32[n_blocks+1])``; blob has 4 pad bytes past ``offsets[-1]``."""
    data = np.ascontiguousarray(data, dtype=np.uint8)
    n = data.size
    if n == 0:
        return np.zeros(4, np.uint8), np.zeros(1, np.uint32)
    code_lut = _canonical_codes(lengths)
    ls = lengths[data].astype(np.int64)
    assert ls.min() > 0, "symbol outside the table"
    cs = np.cumsum(ls)
    starts = cs - ls  # bit offset of each symbol in the unframed stream

    n_blocks = -(-n // block_bytes)
    first = np.arange(n_blocks) * block_bytes
    counts = np.minimum(first + block_bytes, n) - first
    block_bit_base = starts[first]
    block_bits = cs[np.minimum(first + block_bytes, n) - 1] - block_bit_base
    offsets = np.zeros(n_blocks + 1, dtype=np.uint32)
    # Blocks start 4-byte aligned (<= 3 waste bytes each, ~0.05% at 8 KB): the
    # GPU decoder refills its bit-buffer with single aligned uint32 pulls.
    offsets[1:] = np.cumsum((((block_bits + 7) >> 3) + 3) & ~np.int64(3))

    # Per-symbol output bit position: block byte-aligned base + within-block.
    out_bit = (starts - np.repeat(block_bit_base, counts)
               + np.repeat(offsets[:-1].astype(np.int64) * 8, counts))
    # Each ≤12-bit code at alignment ≤7 fits a 4-byte big-endian window.
    # 8 pad bytes: the GPU decoder's bit-buffer may pull up to 8 bytes past
    # the last block's compressed end.
    v = code_lut[data].astype(np.uint32) << (32 - (out_bit & 7) - ls).astype(np.uint32)
    B = out_bit >> 3
    blob = np.zeros(int(offsets[-1]) + 8, dtype=np.uint8)
    for k in range(4):
        np.bitwise_or.at(blob, B + k, ((v >> (8 * (3 - k))) & 0xFF).astype(np.uint8))
    return blob, offsets


def decode_blocks_cpu(blob: np.ndarray, offsets: np.ndarray,
                      lengths: np.ndarray, raw_len: int,
                      block_bytes: int = DEFAULT_BLOCK_BYTES) -> np.ndarray:
    """Bit-serial reference decoder (tests / small planes)."""
    lut = build_decode_lut(lengths)
    out = np.empty(raw_len, dtype=np.uint8)
    mem = blob.tobytes()
    for b in range(len(offsets) - 1):
        bitpos = int(offsets[b]) * 8
        for i in range(b * block_bytes, min((b + 1) * block_bytes, raw_len)):
            byte_i = bitpos >> 3
            win = int.from_bytes(mem[byte_i: byte_i + 3], "big")
            e = int(lut[(win >> (12 - (bitpos & 7))) & 0xFFF])
            out[i] = e >> 4
            bitpos += e & 0xF
    return out


# --------------------------------------------------------------------------- #
# GPU decoder: one lane per block, LANES blocks per program in lockstep, with
# a 64-bit register bit-buffer per lane. Blocks start 4-byte aligned, so the
# buffer refills with ONE conditional aligned uint32 pull per ~5.6 symbols —
# the per-symbol dependency chain is just the 8 KB L1-resident LUT gather.
# Runs eagerly on the ring's pump path during the sync stall; never captured.
# --------------------------------------------------------------------------- #
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def huff_decode_batched_kernel(blob_ptr, woff_ptr, ooff_ptr, nsym_ptr,
                                   lut_ptr, out_ptr, n_blocks, BB,
                                   LANES: tl.constexpr):
        # Batched variant: per-block indirection tables (word offset into the
        # staging blob, byte offset into the output arena, symbol count) let
        # ONE launch decode many planes — small per-plane launches starve the
        # GPU of warps (measured 7.8 vs 26 GB/s).
        blk = tl.program_id(0) * LANES + tl.arange(0, LANES)
        ok = blk < n_blocks
        nw = tl.load(woff_ptr + blk, mask=ok, other=0).to(tl.int64)
        ob = tl.load(ooff_ptr + blk, mask=ok, other=0).to(tl.int64)
        ns = tl.load(nsym_ptr + blk, mask=ok, other=0)
        buf = tl.zeros((LANES,), dtype=tl.uint64)  # upcoming bits, MSB-aligned
        nbits = tl.zeros((LANES,), dtype=tl.int32)
        for i in range(0, BB):
            m = i < ns
            need = m & (nbits <= 32)
            w = tl.load(blob_ptr + nw, mask=need, other=0).to(tl.uint64)
            b32 = (((w & 0xFF) << 24) | ((w & 0xFF00) << 8)
                   | ((w >> 8) & 0xFF00) | ((w >> 24) & 0xFF))
            buf = buf | (b32 << tl.maximum(32 - nbits, 0).to(tl.uint64))
            nbits = tl.where(need, nbits + 32, nbits)
            nw = tl.where(need, nw + 1, nw)
            e = tl.load(lut_ptr + (buf >> 52).to(tl.int32), mask=m, other=0).to(tl.uint32)
            tl.store(out_ptr + ob + i, (e >> 4).to(tl.uint8), mask=m)
            l = e & 0xF
            buf = buf << l.to(tl.uint64)
            nbits = nbits - l.to(tl.int32)

    @triton.jit
    def huff_decode_kernel(blob_ptr, off_ptr, lut_ptr, out_ptr,
                           raw_len, BB, LANES: tl.constexpr):
        blk = tl.program_id(0) * LANES + tl.arange(0, LANES)
        base_out = blk.to(tl.int64) * BB
        rem = raw_len - base_out  # symbols this lane decodes (≤0 = idle lane)
        nw = (tl.load(off_ptr + blk, mask=rem > 0, other=0) >> 2).to(tl.int64)
        buf = tl.zeros((LANES,), dtype=tl.uint64)  # upcoming bits, MSB-aligned
        nbits = tl.zeros((LANES,), dtype=tl.int32)
        for i in range(0, BB):
            m = i < rem
            # One conditional word pull keeps >= 12 bits buffered (consume is
            # <= 12/iter). May run <= 8 bytes past the block's compressed end
            # (peeked, never consumed) — encode_blocks pads the blob for it.
            need = m & (nbits <= 32)
            w = tl.load(blob_ptr + nw, mask=need, other=0).to(tl.uint64)
            b32 = (((w & 0xFF) << 24) | ((w & 0xFF00) << 8)  # LE word ->
                   | ((w >> 8) & 0xFF00) | ((w >> 24) & 0xFF))  # MSB-first bits
            buf = buf | (b32 << tl.maximum(32 - nbits, 0).to(tl.uint64))
            nbits = tl.where(need, nbits + 32, nbits)
            nw = tl.where(need, nw + 1, nw)
            e = tl.load(lut_ptr + (buf >> 52).to(tl.int32), mask=m, other=0).to(tl.uint32)
            tl.store(out_ptr + base_out + i, (e >> 4).to(tl.uint8), mask=m)
            l = e & 0xF
            buf = buf << l.to(tl.uint64)
            nbits = nbits - l.to(tl.int32)

    return huff_decode_kernel, huff_decode_batched_kernel


_kernels = None
_DECODE_LANES = 64


def _get_kernels():
    global _kernels
    if _kernels is None:
        _kernels = _kernel()
    return _kernels


def decode_batched_gpu(staging_u32: torch.Tensor, word_off: torch.Tensor,
                       out_off: torch.Tensor, nsym: torch.Tensor,
                       lut: torch.Tensor, out_arena: torch.Tensor,
                       block_bytes: int = DEFAULT_BLOCK_BYTES) -> None:
    """One launch over many planes on the CURRENT stream: block ``b`` decodes
    ``nsym[b]`` symbols from staging word ``word_off[b]`` into
    ``out_arena[out_off[b]:]``. Tables are static (built at load), so this is
    safe to enqueue from the ring's pump path."""
    import triton

    _, batched = _get_kernels()
    n_blocks = word_off.numel()
    grid = (triton.cdiv(n_blocks, _DECODE_LANES),)
    batched[grid](staging_u32, word_off, out_off, nsym, lut, out_arena,
                  n_blocks, block_bytes, LANES=_DECODE_LANES, num_warps=2)


def decode_blocks_gpu(blob: torch.Tensor, offsets: torch.Tensor,
                      lut: torch.Tensor, raw_len: int,
                      block_bytes: int = DEFAULT_BLOCK_BYTES,
                      out: "torch.Tensor | None" = None) -> torch.Tensor:
    """Decode on the CURRENT stream into ``out`` (allocated if None). All
    tensors device-resident: blob/out uint8, offsets uint32, lut uint16."""
    import triton

    single, _ = _get_kernels()
    if out is None:
        out = torch.empty(raw_len, dtype=torch.uint8, device=blob.device)
    n_blocks = -(-raw_len // block_bytes)
    grid = (triton.cdiv(n_blocks, _DECODE_LANES),)
    # word view: blocks start 4-byte aligned and the pad keeps numel % 4 == 0
    single[grid](blob.view(torch.uint32), offsets, lut, out,
                 raw_len, block_bytes,
                 LANES=_DECODE_LANES, num_warps=2)
    return out
