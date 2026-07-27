"""Downstream (lm-eval) scoring for the student.

Collapses an lm-eval ``simple_evaluate`` results dict into the program's reported
numbers. Pure dict arithmetic, so it is unit-testable on CPU without the
``[eval]`` extra or a GPU — keep this module free of top-level ``torch`` /
``lm_eval`` imports.
"""
from __future__ import annotations

# The program's downstream convention: acc_norm where the task reports a
# length-normalized accuracy, plain acc otherwise. NOT a global acc_norm→acc
# priority — arc_easy/arc_challenge report acc_norm too, and taking it would
# give a different number than every result recorded in logs/.
MACRO_TASK_METRIC = {
    "hellaswag": "acc_norm", "piqa": "acc_norm",
    "arc_easy": "acc", "arc_challenge": "acc", "winogrande": "acc",
}


def _pick(metrics: dict, *names: str) -> "float | None":
    """First of ``names`` present, ignoring lm-eval's ``,<filter>`` key suffix."""
    for name in names:
        for key, val in metrics.items():
            if key.split(",")[0] == name and isinstance(val, (int, float)):
                return float(val)
    return None


def macro_metrics(results: "dict | None") -> "dict[str, float]":
    """Each task's convention metric, keyed by task name.

    ``results`` is ``None`` on every rank but global rank 0 (``simple_evaluate``
    returns nothing elsewhere), which yields ``{}``.
    """
    table = (results or {}).get("results", {})
    picked = {
        task: _pick(m, MACRO_TASK_METRIC.get(task, "acc"), "acc_norm")
        for task, m in table.items()
    }
    return {task: val for task, val in picked.items() if val is not None}


def macro_score(results: "dict | None") -> float:
    """Mean of ``macro_metrics`` — the number checkpoints are selected on.

    Averages over whatever tasks scored, so a subset run reports that subset's
    macro; 0.0 when nothing scored. Higher is better.
    """
    vals = macro_metrics(results).values()
    return sum(vals) / len(vals) if vals else 0.0
