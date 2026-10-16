"""Answer-letter KL of eval arms against a reference arm, per item, from the dumps' ``ll``.

The softmax over a multiple-choice item's per-option loglikelihoods is the answer distribution
the accuracy is the argmax of; KL(ref ‖ arm) on it is the one signal that tracked WHICH math
items an arm flips (letter KL AUC .962 vs ≤ .58 for every hidden-state tap). Reported on all
docs and split at ``--cut`` (doc_id < cut is what a trace or selection may have seen):

    python scripts/letter_kl.py --cut 300 exact=exact.json r2048=MR031_r2048.json ...

The first arm is the reference. Needs dumps written with ``ll`` (`downstream.slim_samples`).
"""
from __future__ import annotations

import argparse
import json
import math


def load_ll(path: str, task: str) -> "dict[int, tuple[list[float], float]]":
    d = json.load(open(path))
    if "samples" not in d:      # {target: results}, as the harness writes it
        d = d[next(iter(d))]
    recs = d["samples"][task]
    out = {}
    for r in recs:
        # `ll` (slim dumps), or the full `filtered_resps` of a dump that kept them.
        ll = r.get("ll") or [float(o[0]) for o in r.get("filtered_resps") or []]
        if not ll:
            raise SystemExit(f"{path} has no per-option `ll` (dumped before slim_samples kept it)")
        out[int(r["doc_id"])] = (ll, float(r["acc"]))
    return out


def letter_kl(ref: "list[float]", arm: "list[float]") -> float:
    """KL(softmax(ref) ‖ softmax(arm)) over the options."""
    def logsm(v):
        m = max(v)
        z = m + math.log(sum(math.exp(x - m) for x in v))
        return [x - z for x in v]
    lp, lq = logsm(ref), logsm(arm)
    return sum(math.exp(a) * (a - b) for a, b in zip(lp, lq))


def _mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut", type=int, default=300)
    p.add_argument("--task", default="mmlu_math_mc")
    p.add_argument("arms", nargs="+", metavar="NAME=PATH")
    args = p.parse_args(argv)
    arms = dict(a.split("=", 1) for a in args.arms)
    names = list(arms)
    D = {k: load_ll(v, args.task) for k, v in arms.items()}
    ref = names[0]
    for lo, hi, tag in ((0, None, "all docs"), (0, args.cut, f"doc_id<{args.cut}"),
                        (args.cut, None, f"held-out doc_id>={args.cut}")):
        ids = sorted(i for i in D[ref] if i >= lo and (hi is None or i < hi)
                     and all(i in D[k] for k in names))
        print(f"\n== {tag} (n={len(ids)}) — KL({ref} ‖ arm) on the answer letters ==")
        kl = {k: [letter_kl(D[ref][i][0], D[k][i][0]) for i in ids] for k in names[1:]}
        for k in names[1:]:
            m, se = _mean_se(kl[k])
            acc = sum(D[k][i][1] for i in ids) / len(ids)
            lost = sum(D[ref][i][1] > D[k][i][1] for i in ids)
            won = sum(D[ref][i][1] < D[k][i][1] for i in ids)
            print(f"  {k:14s} KL {m:.5f} SE {se:.5f}  acc {acc:.4f}  lost {lost:3d} won {won:3d}")
        for x in range(1, len(names)):
            for y in range(x + 1, len(names)):
                diff = [b - a for a, b in zip(kl[names[x]], kl[names[y]])]
                m, se = _mean_se(diff)
                z = m / se if se else 0.0
                print(f"  {names[y]:>14s} − {names[x]:<14s} ΔKL {m:+.5f} SE {se:.5f} z {z:+.2f} "
                      f"p {math.erfc(abs(z) / 2 ** .5):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
