"""Offline entropy repack: replicas.safetensors -> replicas.ent.safetensors.

Losslessly Huffman-codes the pool's CODES plane (the only compressible plane:
int4 codes carry ~2.97/4 bits of entropy, int8 ~6.82/8; the survivor bitmap is
AT the entropy limit and scales are ~0.3% of bytes) against ONE pool-wide
static table, framed in fixed raw-size blocks with exact per-block offsets —
the format ``parallm.entropy_codec`` decodes on the GPU during the sync
stalls. Each (layer, rel) plane is assembled in ENGINE order
(``engine.assemble_codes``) before coding, so the artifact's decoded bytes are
bit-identical to what the streamed ring would have copied raw.

The artifact holds only blob/offsets/raw_len per (layer, rel) plus the table;
mask/scale/shape keep streaming raw from the original pool file.

    python scripts/repack_replicas_entropy.py \\
        --replicas convert_out/qwen3_6_27b_n8_tracks/replicas.safetensors \\
        [--out .../replicas.ent.safetensors] [--block-bytes 8192] [--procs N] \\
        [--bench-decode]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallm.engine import assemble_codes  # noqa: E402
from parallm.entropy_codec import (build_table, encode_blocks)  # noqa: E402
from parallm.model.replica_pack import load_replica_pool  # noqa: E402

_POOL = None  # set before fork; workers read it copy-on-write


def _packs(key):
    li, rel = key
    n_tracks = max(k[0] for k in _POOL) + 1
    return [_POOL[(t, li, rel)] for t in range(n_tracks)]


def _hist_one(key):
    torch.set_num_threads(1)
    plane, _ = assemble_codes(_packs(key))
    return np.bincount(plane.numpy(), minlength=256)


def _encode_one(args):
    key, lengths, block_bytes = args
    torch.set_num_threads(1)
    plane, _ = assemble_codes(_packs(key))
    data = plane.numpy()
    blob, offsets = encode_blocks(data, lengths, block_bytes)
    return key, blob, offsets, data.size


def main() -> int:
    global _POOL
    p = argparse.ArgumentParser()
    p.add_argument("--replicas", required=True)
    p.add_argument("--out", default=None,
                   help="default: <replicas dir>/replicas.ent.safetensors")
    p.add_argument("--block-bytes", type=int, default=8192)
    p.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument("--bench-decode", action="store_true",
                   help="time the GPU decoder on the largest plane after writing")
    args = p.parse_args()
    out_path = args.out or os.path.join(os.path.dirname(args.replicas),
                                        "replicas.ent.safetensors")

    t0 = time.time()
    _POOL, meta = load_replica_pool(args.replicas)
    keys = sorted({(li, rel) for (_t, li, rel) in _POOL})
    sample = _POOL[next(iter(_POOL))]
    if "vals" not in sample and "vlen" not in sample:
        pass
    elif "vals" in sample:
        raise SystemExit(f"[error] pool {meta.get('config')} stores bf16 vals — "
                         "nothing to entropy-code (int codes pools only)")
    else:
        raise SystemExit("[error] MoE expert-slab pools not supported yet")
    print(f"[repack] {len(keys)} (layer, rel) planes, config={meta.get('config')}, "
          f"procs={args.procs}", flush=True)

    with Pool(args.procs) as pool:
        hist = sum(pool.imap_unordered(_hist_one, keys, chunksize=4))
    lengths = build_table(hist)
    pf = hist / hist.sum()
    ent = float(-(pf[pf > 0] * np.log2(pf[pf > 0])).sum())
    avg = float((hist * lengths).sum() / hist.sum())
    print(f"[repack] byte entropy {ent:.3f}/8, huffman {avg:.3f}/8 "
          f"(+{(avg / ent - 1) * 100:.2f}% vs bound)", flush=True)

    tensors, raw_total, blob_total = {}, 0, 0
    with Pool(args.procs) as pool:
        it = pool.imap_unordered(
            _encode_one, [(k, lengths, args.block_bytes) for k in keys],
            chunksize=2)
        for key, blob, offsets, raw_len in it:
            li, rel = key
            tensors[f"{li}|{rel}|blob"] = torch.from_numpy(blob)
            tensors[f"{li}|{rel}|offsets"] = torch.from_numpy(offsets)
            tensors[f"{li}|{rel}|raw_len"] = torch.tensor([raw_len], dtype=torch.int64)
            raw_total += raw_len
            blob_total += int(offsets[-1]) + offsets.nbytes
    tensors["__table__|lengths"] = torch.from_numpy(lengths)

    from safetensors.torch import save_file
    save_file(tensors, out_path, metadata={
        "codec": "huff12",
        "block_bytes": str(args.block_bytes),
        "source_config": str(meta.get("config")),
    })
    other = sum(t.nbytes for e in _POOL.values() for f, t in e.items()
                if f != "codes")
    print(f"[repack] codes {raw_total / 2**30:.2f} GB -> {blob_total / 2**30:.2f} GB "
          f"(ratio {blob_total / raw_total:.3f}) | pool "
          f"{(raw_total + other) / 2**30:.2f} -> {(blob_total + other) / 2**30:.2f} GB "
          f"| {time.time() - t0:.0f}s -> {out_path}", flush=True)

    if args.bench_decode:
        from parallm.entropy_codec import build_decode_lut, decode_blocks_gpu
        li, rel = max(keys, key=lambda k: tensors[f"{k[0]}|{k[1]}|raw_len"].item())
        blob = tensors[f"{li}|{rel}|blob"].cuda()
        offsets = tensors[f"{li}|{rel}|offsets"].cuda()
        raw_len = int(tensors[f"{li}|{rel}|raw_len"].item())
        lut = torch.from_numpy(build_decode_lut(lengths)).cuda()
        out = torch.empty(raw_len, dtype=torch.uint8, device="cuda")
        for _ in range(3):
            decode_blocks_gpu(blob, offsets, lut, raw_len, args.block_bytes, out=out)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            decode_blocks_gpu(blob, offsets, lut, raw_len, args.block_bytes, out=out)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 10
        ref, _ = assemble_codes(_packs((li, rel)))
        assert torch.equal(out.cpu(), ref), "GPU decode != source plane"
        print(f"[bench] decode {raw_len / 2**20:.0f} MB plane: "
              f"{raw_len / 2**30 / dt:.1f} GB/s (bit-exact)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
