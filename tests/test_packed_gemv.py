"""GPU rail for the fused packed GEMV: the kernel must match the dense-unpack
path (unpack_sparse_weight → F.linear) for every pool format, up to bf16
accumulation order (both paths round weights to bf16; only the summation
order differs)."""
from __future__ import annotations

import pytest
import torch

from parallm.model.replica_pack import pack_sparse_weight, unpack_sparse_weight

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda_only
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_kernel_matches_dense_path(bits):
    from parallm.engine import _RelEntry
    from parallm.packed_gemv import packed_gemv

    torch.manual_seed(0)
    N, o, i = 3, 48, 520  # i % 8 == 0 but not a BLOCK_I multiple
    packs, dense = [], []
    for _ in range(N):
        w = torch.randn(o, i, dtype=torch.bfloat16)
        norms = torch.rand(i) + 0.1
        packs.append(pack_sparse_weight(w, 0.5, norms, bits=bits))
        dense.append(unpack_sparse_weight(packs[-1], torch.bfloat16).cuda())
    e = _RelEntry(packs, "cuda")

    for M in (1, 5, 33):
        x = torch.randn(M, i, dtype=torch.bfloat16, device="cuda")
        for t in range(N):
            ref = torch.nn.functional.linear(x, dense[t]).float()
            got = packed_gemv(
                x, mask=e.mask[t], values=e.values,
                scale=e.scale[t] if e.scale is not None else None,
                block_start=e.block_start[t],
                out_features=o, in_features=i, bits=bits,
            ).float()
            rel_err = (got - ref).norm() / ref.norm().clamp(min=1e-6)
            assert rel_err < 2e-2, (bits, M, t, rel_err.item())


@cuda_only
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_m_tiled_matches_per_row(bits):
    """The M-tiled kernel (one weight-block load serving M_TILE chunk
    positions) against the M=1 path row by row. The verify-chunk speed fix
    must not change what a position computes."""
    from parallm.engine import _RelEntry
    from parallm.packed_gemv import packed_gemv

    torch.manual_seed(2)
    o, i = 48, 520
    w = torch.randn(o, i, dtype=torch.bfloat16)
    norms = torch.rand(i) + 0.1
    p = pack_sparse_weight(w, 0.5, norms, bits=bits)
    e = _RelEntry([p], "cuda")
    kw = dict(mask=e.mask[0], values=e.values,
              scale=e.scale[0] if e.scale is not None else None,
              block_start=e.block_start[0], out_features=o, in_features=i,
              bits=bits)

    for M in (2, 17, 33):
        x = torch.randn(M, i, dtype=torch.bfloat16, device="cuda")
        tiled = packed_gemv(x, **kw)
        per_row = torch.cat([packed_gemv(x[m:m + 1], **kw) for m in range(M)])
        # tl.dot products are exact (bf16xbf16 in f32); only the accumulation
        # tree differs from the M=1 path -> last-ulp class, same as the
        # kernel-vs-dense rail above.
        torch.testing.assert_close(tiled.float(), per_row.float(),
                                   rtol=1e-2, atol=1e-2)


@cuda_only
def test_packed_linear_dense_fallback():
    """Kernel coverage at M past blinear's dense-fallback threshold."""
    from parallm.engine import _RelEntry
    from parallm.packed_gemv import packed_gemv

    torch.manual_seed(1)
    o, i = 32, 256
    w = torch.randn(o, i, dtype=torch.bfloat16)
    norms = torch.rand(i) + 0.1
    p = pack_sparse_weight(w, 0.5, norms, bits=4)
    dense = unpack_sparse_weight(p, torch.bfloat16).cuda()
    e = _RelEntry([p], "cuda")

    x = torch.randn(100, i, dtype=torch.bfloat16, device="cuda")
    ref = torch.nn.functional.linear(x, dense).float()
    # kernel row-by-row on the same input (past the module's fallback threshold,
    # so exercise the kernel directly at M=100 for coverage)
    got = packed_gemv(x, mask=e.mask[0], values=e.values, scale=e.scale[0],
                      block_start=e.block_start[0], out_features=o, in_features=i,
                      bits=4).float()
    rel_err = (got - ref).norm() / ref.norm()
    assert rel_err < 2e-2, rel_err.item()
