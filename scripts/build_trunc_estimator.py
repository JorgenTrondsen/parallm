"""Build a per-layer EXACT-TRUNCATION read base: layer i's top-r_i true MLP channels.

CPU only, no torchrun. ``--score`` ranks every (track, channel) of a layer (``layer.{i}`` =
``[n_tracks, channels]``, `scripts/trace_read_channels.py summarize`'s
``fisher_score.safetensors``); ``--alloc`` (`scripts/alloc_read.py`) gives each read layer its
r. Layer i of the base holds, for the top-r_i channels S of the flattened grid,

    g = gate_proj.weight[S].T   (H, r)
    u = up_proj.weight[S].T     (H, r)
    d = down_proj.weight[:, S].T (r, H)

so `parallm.model.mlp_read` computes `ghat = down(silu(x@g) * (x@u))` = `Σ_{i∈S} d^(i)` — a partial
sum of the TRUE MLP output. Nothing is fitted.

⚠ Channels are addressed as (track_id, local index), never as a dense index, so nothing here
depends on the slicer's channel ordering.

    python scripts/build_trunc_estimator.py --checkpoint-dir <convert> \\
        --score fisher_score.safetensors --alloc alloc.json --out <base>.safetensors
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--score", required=True, help="layer.{i} [n_tracks, channels], higher = kept first")
    p.add_argument("--alloc", required=True, help="JSON with 'r': {layer: r} (scripts/alloc_read.py)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    score = load_file(args.score)
    with open(args.alloc) as f:
        alloc = {int(k): int(v) for k, v in json.load(f)["r"].items()}
    layers = sorted(alloc)
    if missing := [li for li in layers if f"layer.{li}" not in score]:
        raise SystemExit(f"--alloc layers {missing} have no scores in {args.score}")
    NT, INTER = score[f"layer.{layers[0]}"].shape
    print(f"[init] {len(layers)} layers, {NT} tracks x {INTER} channels, Σr={sum(alloc.values())}")

    # order[li] = (track_id, local_idx) by DESCENDING score, the top alloc[li].
    order = {}
    for li in layers:
        idx = torch.argsort(score[f"layer.{li}"].reshape(-1), descending=True)[:alloc[li]]
        order[li] = [(int(i) // INTER, int(i) % INTER) for i in idx]

    # want[tid][li] = (local indices, destination columns) for this track at this layer
    want: dict[int, dict[int, tuple[list[int], list[int]]]] = {t: {} for t in range(NT)}
    for li in layers:
        for pos, (tid, loc) in enumerate(order[li]):
            a, b = want[tid].setdefault(li, ([], []))
            a.append(loc)
            b.append(pos)

    buf: dict[int, dict[str, torch.Tensor]] = {}
    for tid in range(NT):
        with safe_open(os.path.join(args.checkpoint_dir, f"track_{tid}.safetensors"), "pt") as f:
            for li, (loc, pos) in want[tid].items():
                g = f.get_tensor(f"layers.{li}.mlp.gate_proj.weight")     # (INTER, H)
                u = f.get_tensor(f"layers.{li}.mlp.up_proj.weight")       # (INTER, H)
                d = f.get_tensor(f"layers.{li}.mlp.down_proj.weight")     # (H, INTER)
                if li not in buf:
                    H, r = g.shape[1], alloc[li]
                    buf[li] = {"gate": torch.zeros(H, r, dtype=g.dtype),
                               "up": torch.zeros(H, r, dtype=u.dtype),
                               "down": torch.zeros(r, H, dtype=d.dtype)}
                loc, pos = torch.tensor(loc), torch.tensor(pos)
                buf[li]["gate"][:, pos] = g[loc].transpose(0, 1)
                buf[li]["up"][:, pos] = u[loc].transpose(0, 1)
                buf[li]["down"][pos, :] = d[:, loc].transpose(0, 1)
        if (tid + 1) % 8 == 0:
            print(f"[read] track {tid + 1}/{NT}", flush=True)

    out = {}
    for li in layers:
        for k in ("gate", "up", "down"):
            out[f"layer.{li}.{k}"] = buf[li][k].contiguous()
        # ⛔ The index is load-bearing: column j holds channel `chan` of track `track`, and it is
        # the only thing that tells each track which base columns it already carries exactly.
        out[f"layer.{li}.index"] = torch.tensor(order[li], dtype=torch.int32).contiguous()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out, args.out, metadata={"n_tracks": str(NT), "n_channels": str(INTER),
                                       "score": os.path.basename(args.score),
                                       "alloc": json.dumps({str(li): alloc[li] for li in layers})})
    nb = sum(v.numel() * v.element_size() for v in out.values())
    print(f"[out] Σr={sum(alloc.values())}  {len(layers)} layers  {nb / 2**30:.2f} GiB  -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
