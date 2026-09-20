#!/usr/bin/env python3
"""PAIRED per-document comparison of two macro dumps.

⚠⚠ WHY THIS EXISTS. A frozen-artifact eval is BIT-EXACT on re-run, and that was being read as
"the number is precise". It is not: it means REPRODUCIBLE. Every arm scores the SAME 200
documents per task, so two arms differing by 0.010 may be separated by a handful of flipped
answers — 0.010 macro is 10 questions out of 1000. Comparing point estimates against the
marginal SE (~0.016 at limit 200) is the wrong test and throws away the pairing.

The right test is McNemar on the DISCORDANT pairs: of the documents where the two arms
disagree, how lopsided is the split? Concordant documents carry no information about the
difference and correctly drop out, which is why this resolves far smaller effects than the
marginal SE suggests.

Per task: b = A right / B wrong, c = A wrong / B right; diff = (c - b)/n.
SE of a paired proportion difference = sqrt(b + c - (c-b)^2/n) / n.
Macro is the unweighted mean of 5 task accuracies, so its SE is sqrt(Σ SE_t^2)/5.
"""
import json
import sys
from math import erf, sqrt

# The macro this repo reports, read from its ONE definition rather than re-listed:
# `codemmlu_fim` left the set on 2026-09-12 and a second copy of the list here would
# silently keep scoring the retired 5-task macro against 4-task dumps.
from parallm.eval.downstream import DEFAULT_TASKS

TASKS = DEFAULT_TASKS.split(",")


def per_doc(path: str, tasks: list[str] = TASKS) -> dict[str, dict[int, float]]:
    s = json.load(open(path))["student"]["samples"]
    out = {}
    for t in tasks:
        out[t] = {r["doc_id"]: float(r.get("acc", 0.0)) for r in s[t]}
    return out


def mcnemar(A: dict, B: dict, tasks: list[str] = TASKS):
    """``(rows, macro_diff, macro_SE, z, p)`` for two per-doc dicts over `tasks`.

    Factored out of `main` so a rescore over a different task set uses THIS implementation
    rather than a second copy — one set of maths, one place to be wrong. That is what `tasks`
    is a parameter for: dropping `codemmlu_fim` must not mean re-deriving the paired SE.
    """
    tot_d, var, rows = 0.0, 0.0, []
    for t in tasks:
        ids = sorted(set(A[t]) & set(B[t]))
        assert len(ids) == len(A[t]) == len(B[t]), f"{t}: doc_id mismatch — arms not comparable"
        b = sum(1 for i in ids if A[t][i] > B[t][i])
        c = sum(1 for i in ids if A[t][i] < B[t][i])
        n = len(ids)
        d = (c - b) / n
        se = sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
        rows.append((t, n, b, c, d, se))
        tot_d += d
        var += se ** 2
    md, mse = tot_d / len(tasks), sqrt(var) / len(tasks)
    z = md / mse if mse else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return rows, md, mse, z, p


def main(pa: str, pb: str, na: str, nb: str) -> None:
    rows, md, mse, z, p = mcnemar(per_doc(pa), per_doc(pb))

    print(f"\nPAIRED  A={na}   B={nb}   (positive => B better)")
    print(f"{'task':<16} {'n':>4} {'A>B':>4} {'A<B':>4} {'disc':>5} {'diff':>8} {'SE':>7}")
    for t, n, b, c, d, se in rows:
        print(f"{t:<16} {n:>4} {b:>4} {c:>4} {b + c:>5} {d:>+8.4f} {se:>7.4f}")
    print(f"{'MACRO':<16} {'':>4} {'':>4} {'':>4} "
          f"{sum(r[2] + r[3] for r in rows):>5} {md:>+8.4f} {mse:>7.4f}")
    print(f"  z = {z:+.2f}   p = {p:.3f}   "
          f"{'SIGNIFICANT' if p < 0.05 else 'NOT significant at 0.05'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
