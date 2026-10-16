"""Price lossless entropy coding (`parallm.model.replica_code`) of an MLP read base, on the CPU.

Coding is bit-exact (`replica_code.encode` refuses anything that does not round-trip), so it
never changes accuracy; this only counts the planes it would store — the same planes
`encode` builds (sign+mantissa byte per survivor, the smaller of a Huffman or fixed-width
exponent code, plus the survivor bitmap and block prefixes for a pruned weight), without the
GPU round-trip check. Per node, one track:

    python scripts/mlp_base_code_size.py --base <base>.safetensors \\
        [--compress gu:s20 --stats mlp_stats.safetensors] [--fold <convert dir>]

``--fold`` prices the folded base (`MlpReadBase.bind_fold`): gate/up + ONE track's
``F = D diag(γ) W_qkvᵀ`` from that convert's track 0, instead of ``down``.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from safetensors import safe_open

from parallm.model import replica_code as rc
from parallm.model.mlp_read import compress_layer, parse_compress_spec


def coded_bits(w: torch.Tensor) -> int:
    w = w.to(torch.bfloat16).cpu()
    mask = w != 0
    dense = bool(mask.all())
    vals = (w.reshape(-1) if dense else w[mask]).contiguous()
    u = vals.view(torch.int16).numpy().view(np.uint16)
    exp = ((u >> 7) & 0xFF).astype(np.uint8)
    off = exp - np.uint8(int(exp.min()))
    planes = {"hi": torch.empty(vals.numel(), dtype=torch.uint8)}
    planes.update(min((rc._exp_field(off, "cpu"), rc._exp_huffman(off, "cpu")),
                      key=lambda p: rc.coded_bits({**planes, **p})))
    if not dense:
        flat = mask.numpy().reshape(-1)
        planes["mask"] = torch.from_numpy(np.packbits(flat))
        planes["pre"] = torch.from_numpy(rc._prefix(flat))
    return rc.coded_bits(planes)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--compress", default=None)
    p.add_argument("--stats", default=None)
    p.add_argument("--fold", default=None, metavar="CONVERT_DIR")
    args = p.parse_args(argv)
    spec = parse_compress_spec(args.compress) if args.compress else None
    stats_f = safe_open(args.stats, "pt") if spec else None
    track = safe_open(f"{args.fold}/track_0.safetensors", "pt") if args.fold else None
    coded = 0.0
    with safe_open(args.base, "pt") as f:
        layers = sorted({int(k.split(".")[1]) for k in f.keys() if k.startswith("layer.")})
        for li in layers:
            g, u, d = (f.get_tensor(f"layer.{li}.{k}") for k in ("gate", "up", "down"))
            index = f.get_tensor(f"layer.{li}.index")
            t = {"gate": g, "up": u, "down": d}
            if spec:
                st = {k: stats_f.get_tensor(k) for k in stats_f.keys() if k.startswith(f"layers.{li}.")}
                t, _ = compress_layer(g, u, d, index, st, li, spec)
            if track is not None:
                j = li + 1
                W = torch.cat([track.get_tensor(f"layers.{j}.self_attn.{x}_proj.weight")
                               for x in ("q", "k", "v")]).float()
                gam = track.get_tensor(f"layers.{j}.input_layernorm.weight").float()
                t["down"] = (t["down"].float() * gam) @ W.T
            for v in t.values():
                coded += coded_bits(v)
            print(f"[layer {li}] coded so far {coded / 8 / 2**30:.3f} GiB", flush=True)
    print(f"[code] {args.base.split('/')[-1]} {args.compress or 'dense'}{' +fold' if args.fold else ''}: "
          f"coded {coded / 8 / 2**30:.3f} GiB over {len(layers)} layers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
