"""Downstream (lm-eval) scoring helpers for the student.

``make_student_forward_fn`` adapts a ``PTWrappedModel`` to the ``PTLM`` forward
contract (owner rank returns full logits, peers ``None``); ``scripts/eval_lm_harness.py``
uses it to score downstream retention. ``aggregate_downstream_score`` collapses an
lm-eval ``simple_evaluate`` results dict into a single scalar (acc_norm→acc, mean
over tasks) — pure dict arithmetic, unit-testable on CPU without the ``[eval]``
extra or a GPU.
"""
from __future__ import annotations


def _pick_metric(metrics: dict, names: "list[str]") -> "float | None":
    """First metric in ``names`` present in ``metrics`` (ignoring lm-eval's
    ``,filter`` key suffix, e.g. ``"acc_norm,none"``)."""
    for name in names:
        for key, val in metrics.items():
            if key.split(",")[0] == name and isinstance(val, (int, float)):
                return float(val)
    return None


def aggregate_downstream_score(
    results: "dict | None",
    tasks: "list[str] | None" = None,
    metric_priority: "tuple[str, ...]" = ("acc_norm", "acc"),
) -> float:
    """Mean accuracy across ``tasks`` from an lm-eval ``simple_evaluate`` dict.

    For each task we take ``acc_norm`` if present else ``acc`` (lm-eval keys carry
    a ``,filter`` suffix that we strip). Tasks absent from the results — or with no
    matching metric — are skipped. Returns 0.0 when nothing scored. Higher is
    better. Pure dict arithmetic, so it is unit-testable without GPUs or lm_eval.
    """
    table = (results or {}).get("results", {})
    if tasks is None:
        tasks = list(table.keys())
    scores = []
    for task in tasks:
        metrics = table.get(task)
        if not metrics:
            continue
        val = _pick_metric(metrics, list(metric_priority))
        if val is not None:
            scores.append(val)
    return sum(scores) / len(scores) if scores else 0.0


def make_student_forward_fn(student):
    """``ForwardFn`` for ``PTLM``: the owner rank produces full ``(B, T, V)``
    logits, peers return ``None`` (the legacy ``PTLM`` contract). The cross-track
    ``SyncBoundary`` collectives inside the forward fire on every rank, so the run
    stays in lockstep regardless.
    """

    def _fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> "torch.Tensor | None":
        logits, _ = student(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=False,
        )
        return logits

    return _fn
