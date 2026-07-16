"""Entropy codec rails: length-limited canonical Huffman over byte symbols.

Rail 1 — the code is complete and cap-respecting (Kraft equality, len ≤ 12).
Rail 2 — encode → decode is the identity, byte-exact: the codec feeds decoded
bytes straight into the packed-GEMV path, so anything short of bit-exactness
is a correctness bug, not a quality tradeoff.
Rail 3 — the GPU decoder ≡ the CPU reference on multi-block planes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from parallm.entropy_codec import (MAX_CODE_LEN, build_decode_lut, build_table,
                                   decode_blocks_cpu, decode_blocks_gpu,
                                   encode_blocks)


def _pool_like(n: int, seed: int = 0) -> np.ndarray:
    """Nibble-packed quantized-Gaussian codes — the measured pool distribution
    (int4 symbol entropy ~2.97/4, byte entropy ~5.93/8)."""
    rng = np.random.default_rng(seed)
    q = np.clip(np.round(rng.normal(0, 2.2, size=2 * n)), -7, 7).astype(np.int64) + 8
    q = q[q != 8][: 2 * n] if (q == 8).any() else q[: 2 * n]  # survivors only
    while q.size < 2 * n:
        q = np.concatenate([q, q])[: 2 * n]
    return ((q[0::2] << 4) | q[1::2]).astype(np.uint8)


def _roundtrip(data: np.ndarray, block_bytes: int = 8192) -> float:
    hist = np.bincount(data, minlength=256)
    lengths = build_table(hist)
    blob, offsets = encode_blocks(data, lengths, block_bytes)
    got = decode_blocks_cpu(blob, offsets, lengths, data.size, block_bytes)
    assert np.array_equal(got, data)
    return offsets[-1] / max(data.size, 1)


def test_table_cap_and_kraft():
    # Extreme skew would want >12-bit codes without the cap.
    hist = np.zeros(256, np.int64)
    hist[:32] = 1
    hist[200] = 10**9
    lengths = build_table(hist)  # Kraft equality asserted inside
    assert lengths.max() <= MAX_CODE_LEN
    assert lengths[200] == 1
    assert (lengths[np.nonzero(hist)[0]] > 0).all()
    lut = build_decode_lut(lengths)
    assert (lut != 0).all()  # complete code fills every 12-bit prefix


def test_cpu_roundtrip_shapes_and_skew():
    rng = np.random.default_rng(1)
    for data in [
        rng.integers(0, 256, size=50_000, dtype=np.uint8),   # incompressible
        _pool_like(30_000),                                   # the real shape
        _pool_like(8192),                                     # exactly 1 block
        _pool_like(8193),                                     # block + 1 tail
        _pool_like(100),                                      # sub-block
        np.full(5000, 7, np.uint8),                           # single symbol
    ]:
        _roundtrip(data)


def test_ratio_near_entropy():
    data = _pool_like(400_000)
    hist = np.bincount(data, minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    h_bytes = float(-(p * np.log2(p)).sum()) / 8
    ratio = _roundtrip(data)
    assert h_bytes < ratio < h_bytes + 0.02  # within ~2% + block padding


def test_gpu_decode_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    data = _pool_like(3_000_000, seed=3)  # ~367 blocks
    hist = np.bincount(data, minlength=256)
    lengths = build_table(hist)
    blob, offsets = encode_blocks(data, lengths)
    lut = build_decode_lut(lengths)
    got = decode_blocks_gpu(
        torch.from_numpy(blob).cuda(),
        torch.from_numpy(offsets).cuda(),
        torch.from_numpy(lut).cuda(),
        data.size,
    )
    assert np.array_equal(got.cpu().numpy(), data)
