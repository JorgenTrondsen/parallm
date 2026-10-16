"""Per-layer r for the local MLP read — or keep that layer's MLP sync — from a read trace.

Every read layer i (0-62) either keeps its post-MLP sync (no read, no bytes, one more
collective) or drops it and reads its MLP sum through a top-r base, r from the trace ladder
(r = 0: post-attn with no read). Bytes are proportional to Σ r (every layer has the same
dims); the cost is the predicted answer-letter KL of the whole config, cross-layer terms
included, from `scripts/trace_read_channels.py`'s ``curves.pt`` (cross-fitted Fisher ranking).

For each count N of kept MLP syncs, λ is scanned: per layer the r minimising
``r + λ·K_i(r)``, and the N layers where that minimum is largest keep their sync instead.
The joint KL is not monotone in λ, so every λ on the grid is priced. Either

    --channels C   the lowest joint KL with Σ r ≤ C, or
    --budget K     the smallest Σ r with joint KL ≤ K (trace units).

    python scripts/alloc_read.py --curves D/curves.pt --channels 14336 --syncs 16-28 \\
        [--emit 19 --out alloc.json]
"""
from __future__ import annotations

import argparse
import json

import torch

from parallm.eval.read_cost import config_kl
from parallm.utils.layers import format_layers, parse_layers

SYNC = None


def layer_costs(M: torch.Tensor, p: torch.Tensor, ladder) -> torch.Tensor:
    """``C [layers, len(ladder)]``: mean predicted KL of reading ONLY that layer at each r."""
    C = torch.empty(M.shape[2], len(ladder), dtype=torch.float64)
    for k in range(M.shape[2]):
        for j, r in enumerate(ladder):
            C[k, j] = config_kl(M[:, :, k:k + 1], p, {0: r}, ladder).mean()
    return C


def choose_n(C: torch.Tensor, ladder, lam: float, n_sync: int) -> list:
    """Per layer the r minimising ``r + λ·C``; the ``n_sync`` layers where that minimum is largest
    keep their sync instead (a sync saves exactly that much)."""
    best, j = (torch.tensor(ladder, dtype=torch.float64) + lam * C).min(1)
    picks = [ladder[int(x)] for x in j]
    for k in torch.argsort(best, descending=True)[:n_sync].tolist():
        picks[k] = SYNC
    return picks


def solve_n(C, M, p, ladder, budget: float, n_sync: int, grid: int = 400) -> list:
    """The smallest Σ r with exactly ``n_sync`` MLP syncs kept whose JOINT KL fits ``budget``."""
    best = None
    for lam in torch.logspace(0, 12, grid, dtype=torch.float64).tolist():
        picks = choose_n(C, ladder, lam, n_sync)
        if float(config_kl(M, p, dict(enumerate(picks)), ladder).mean()) <= budget:
            size = sum(r for r in picks if r)
            if best is None or size < best[0]:
                best = (size, picks)
    if best is None:
        raise ValueError(f"budget unreachable with {n_sync} syncs")
    return best[1]


def solve_channels(C, M, p, ladder, channels: int, n_sync: int, grid: int = 600) -> list:
    """The lowest JOINT KL with exactly ``n_sync`` MLP syncs kept and Σ r ≤ ``channels``."""
    best = None
    for lam in torch.logspace(0, 12, grid, dtype=torch.float64).tolist():
        picks = choose_n(C, ladder, lam, n_sync)
        if sum(r for r in picks if r) <= channels:
            kl = float(config_kl(M, p, dict(enumerate(picks)), ladder).mean())
            if best is None or kl < best[0]:
                best = (kl, picks)
    if best is None:
        raise ValueError(f"no config fits {channels} channels with {n_sync} syncs")
    return best[1]


def schedule(read_layers, picks) -> str:
    """``--sync-phase``: dropped MLP syncs are post-attn."""
    post = [i for i, r in zip(read_layers, picks) if r is not SYNC]
    return f"exact,{format_layers(post)}:post-attn" if post else "exact"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curves", required=True)
    p.add_argument("--channels", type=int, default=None, help="Σ r cap over all read layers")
    p.add_argument("--budget", type=float, default=None, help="joint letter KL, the trace's units")
    p.add_argument("--syncs", required=True, metavar="RANGE", help="counts of MLP syncs kept")
    p.add_argument("--bytes-per-channel", type=float, default=7700.0,
                   help="coded, per node, after fold + gate/up pruning + code (for the GiB column)")
    p.add_argument("--emit", type=int, default=None, help="write the config with this many syncs")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if (args.channels is None) == (args.budget is None):
        raise SystemExit("give exactly one of --channels / --budget")

    cv = torch.load(args.curves)
    ladder, read_layers = tuple(cv["ladder"]), list(cv["read_layers"])
    M, prob = cv["fisher_xfit"], cv["p"]
    C = layer_costs(M, prob, ladder)
    gib = args.bytes_per_channel / 2 ** 30
    print(f"{'syncs':>5} {'Σr':>8} {'GiB':>6} {'joint KL':>10}  synced layers")
    found = {}
    for n in parse_layers(args.syncs):
        if args.channels is not None:
            picks = solve_channels(C, M, prob, ladder, args.channels, n)
        else:
            picks = solve_n(C, M, prob, ladder, args.budget, n)
        total = sum(r for r in picks if r)
        joint = float(config_kl(M, prob, dict(enumerate(picks)), ladder).mean())
        print(f"{n:5d} {total:8d} {total * gib:6.3f} {joint:10.4f}  "
              f"{format_layers([i for i, r in zip(read_layers, picks) if r is SYNC]) or '-'}", flush=True)
        found[n] = (total, picks, joint)
    if args.emit is not None:
        if args.emit not in found:
            raise SystemExit(f"--emit {args.emit} is not among --syncs {args.syncs}")
        total, picks, joint = found[args.emit]
        alloc = {"channels": args.channels, "budget": args.budget, "joint_kl": joint, "sum_r": total,
                 "sync_phase": schedule(read_layers, picks),
                 "r": {str(i): r for i, r in zip(read_layers, picks) if r}}
        with open(args.out, "w") as f:
            json.dump(alloc, f, indent=1)
        print(f"[emit] {args.emit} syncs, Σr {total}, joint KL {joint:.4f}\n"
              f"  --sync-phase {alloc['sync_phase']}\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
