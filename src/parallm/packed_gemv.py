"""Fused packed sparse-int GEMV: y = W·x computed directly from the pool's
packed bits (survivor bitmap + nibble/int8 codes / bf16 values + row scales),
never materializing the dense weight.

This is the decode-regime kernel the deployment account assumes: per token the
engine reads the ~3 bits/weight pool instead of writing/reading a ~16× dense
expansion (the profiled unpack was ~90% of v1's per-token compute). Weights are
rounded to bf16 in-kernel so the math matches the unpack path (which stored
bf16 weights) up to accumulation order.

Row addressing (v2): survivors are stored compactly in mask order, so random
access needs prefix popcounts. ``_RelEntry`` precomputes ``block_start`` — the
absolute index of each row's first survivor PER ``GEMV_BLOCK_I``-wide bitmap
block (derived from the bitmap at load, ~2% extra device bytes, no pool-format
change). That kills the loop-carried value-index dependency the v1 kernel
serialized on: every block iteration is independent, so loads pipeline.
Within a block, the bitmap is loaded byte-wise (8× fewer mask loads than v1's
per-lane byte reloads) and value offsets come from a per-byte popcount cumsum
plus a 3-step within-byte cumsum — ~2× fewer integer ops/weight than the v1
full-width cumsum, which was the measured throughput ceiling. Accumulation
stays entirely within one program in a fixed order: deterministic, no atomics
(the shadow carry must stay bit-identical across ranks).

Launch economy (measured to matter as much as the bytes): programs cover
``BLOCK_O`` rows each for memory-level parallelism, and every rel computes all
N tracks in ONE launch — shared-carry inputs pass ``x_tstride=0`` (each track
reads the same x), per-track inputs pass a stride into a stacked ``[N, M, in]``
x. The engine's batched shadow forward (``PackedShadow.blinear``) is the only
caller; large-M inputs (prefill/eval) fall back to dense-unpack + matmul there,
where the packed byte reads would block tensor-core GEMMs.
"""
from __future__ import annotations

import torch

GEMV_BLOCK_I = 256  # bitmap block width; block_start layout depends on it


def _kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.libdevice import popc

    @triton.jit
    def packed_gemv_kernel(
        x_ptr, y_ptr, mask_ptr, val_ptr, scale_ptr, bstart_ptr,
        M, IN, OUT, NB, X_TSTRIDE,
        BITS: tl.constexpr, OFFSET: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_O: tl.constexpr,
    ):
        o_blocks = tl.cdiv(OUT, BLOCK_O)
        t = tl.program_id(0) // o_blocks
        rows = (tl.program_id(0) % o_blocks) * BLOCK_O + tl.arange(0, BLOCK_O)
        m = tl.program_id(1)
        r_ok = rows < OUT
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        if BITS > 0:
            scale = tl.load(scale_ptr + t * OUT + rows, mask=r_ok, other=0.0)
        mask_base = mask_ptr + t * (OUT * (IN // 8))
        x_base = x_ptr + t * X_TSTRIDE + m * IN
        NBY: tl.constexpr = BLOCK_I // 8
        for jb in range(0, IN, BLOCK_I):
            # Independent iterations: this block's value index comes from the
            # precomputed table, not the previous iteration's popcount.
            vidx = tl.load(bstart_ptr + (t * OUT + rows) * NB + jb // BLOCK_I,
                           mask=r_ok, other=0)
            bytes_off = jb // 8 + tl.arange(0, NBY)
            mb = tl.load(mask_base + rows[:, None] * (IN // 8) + bytes_off[None, :],
                         mask=r_ok[:, None] & (bytes_off < IN // 8)[None, :],
                         other=0).to(tl.int32)                       # [O, NBY]
            bits = (mb[:, :, None] >> (7 - tl.arange(0, 8))[None, None, :]) & 1
            on = bits != 0                                           # [O, NBY, 8]
            bpc = popc(mb)                                           # hw byte popcounts
            vi = (vidx[:, None, None]
                  + (tl.cumsum(bpc, axis=1) - bpc)[:, :, None]       # bytes before
                  + tl.cumsum(bits, axis=2) - bits)                  # bits before
            if BITS == 0:
                w = tl.load(val_ptr + vi, mask=on, other=0.0).to(tl.float32)
            else:
                if BITS == 4:
                    byte = tl.load(val_ptr + (vi >> 1), mask=on, other=0).to(tl.int32)
                    code = tl.where((vi & 1) == 0, byte >> 4, byte & 0x0F)
                else:
                    code = tl.load(val_ptr + vi, mask=on, other=0).to(tl.int32)
                w = (code.to(tl.float32) - OFFSET) * scale[:, None, None]
                # match the unpack path, which stored bf16 weights
                w = w.to(tl.bfloat16).to(tl.float32)
            w = tl.where(on, w, 0.0)
            lanes = jb + tl.arange(0, BLOCK_I)
            xb = tl.load(x_base + lanes, mask=lanes < IN, other=0.0).to(tl.float32)
            xb = tl.reshape(xb, (NBY, 8))
            acc += tl.sum(tl.sum(w * xb[None, :, :], axis=2), axis=1)
        tl.store(y_ptr + (t * M + m) * OUT + rows, acc.to(tl.bfloat16), mask=r_ok)

    return packed_gemv_kernel


_packed_gemv_kernel = None


def _launch(x, mask, values, scale, block_start, n_tracks, M, out_features,
            in_features, bits, x_tstride) -> torch.Tensor:
    global _packed_gemv_kernel
    if _packed_gemv_kernel is None:
        _packed_gemv_kernel = _kernel()
    import triton

    y = torch.empty(n_tracks, M, out_features, dtype=torch.bfloat16, device=x.device)
    nb = triton.cdiv(in_features, GEMV_BLOCK_I)
    BLOCK_O = 2  # swept with num_warps/GEMV_BLOCK_I on the real 27B pool
    grid = (n_tracks * triton.cdiv(out_features, BLOCK_O), M)
    _packed_gemv_kernel[grid](
        x, y, mask, values, scale if scale is not None else values, block_start,
        M, in_features, out_features, nb, x_tstride,
        BITS=bits or 0, OFFSET=float(1 << (bits - 1)) if bits else 0.0,
        BLOCK_I=GEMV_BLOCK_I, BLOCK_O=BLOCK_O, num_warps=1,
    )
    return y


def packed_gemv(x: torch.Tensor, *, mask: torch.Tensor, values: torch.Tensor,
                scale: "torch.Tensor | None", block_start: torch.Tensor,
                out_features: int, in_features: int,
                bits: "int | None") -> torch.Tensor:
    """One track's packed Linear applied to ``x [M, in]`` → bf16 ``[M, out]``.

    ``mask``/``scale``/``block_start`` are this track's slices; ``values`` is
    the ENTRY's whole flat value stream (``block_start`` holds absolute
    indices into it, per row per ``GEMV_BLOCK_I``-wide bitmap block)."""
    return _launch(x, mask, values, scale, block_start, 1, x.shape[0],
                   out_features, in_features, bits, 0)[0]
