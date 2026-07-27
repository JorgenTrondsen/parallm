"""Unit tests for the downstream scorer (eval/downstream.py).

Only the pure piece (no lm_eval, no GPUs): the metric aggregation that turns an
lm-eval ``simple_evaluate`` dict into the reported macro.
"""
import pytest

from parallm.eval.downstream import macro_metrics, macro_score


def _results(**task_metrics):
    """Build a minimal lm-eval-shaped results dict."""
    return {"results": dict(task_metrics)}


def _ledger(hellaswag, arc_easy, arc_challenge, winogrande, piqa):
    """The five-task table as lm-eval emits it, from logs/qwen3_5/*_eval_lim200.log.
    Each argument is (acc, acc_norm); winogrande reports no acc_norm at all."""
    return _results(
        hellaswag={"acc,none": hellaswag[0], "acc_norm,none": hellaswag[1]},
        arc_easy={"acc,none": arc_easy[0], "acc_norm,none": arc_easy[1]},
        arc_challenge={"acc,none": arc_challenge[0], "acc_norm,none": arc_challenge[1]},
        winogrande={"acc,none": winogrande},
        piqa={"acc,none": piqa[0], "acc_norm,none": piqa[1]},
    )


def test_macro_reproduces_the_122b_ledger():
    """Pins the convention against real recorded runs — acc_norm for
    hellaswag/piqa, acc for arc_*/winogrande, and the ``,none`` suffix stripped."""
    # moe_122b_exact_eval_lim200.log — the dense-equivalent baseline.
    exact = macro_score(_ledger((0.6950, 0.7900), (0.8350, 0.8100),
                                (0.6400, 0.6550), 0.7550, (0.7950, 0.8400)))
    # moe_122b_d1b_heal_eval_lim200.log — 95.85% retention on record.
    d1b = macro_score(_ledger((0.5900, 0.7400), (0.8000, 0.8050),
                              (0.5850, 0.5600), 0.7350, (0.8000, 0.8400)))
    assert exact == pytest.approx(0.7720)
    assert d1b == pytest.approx(0.7400)
    assert d1b / exact == pytest.approx(0.9585, abs=1e-4)


def test_per_task_metrics_are_the_ones_averaged():
    # arc_challenge must contribute acc (0.6400), not acc_norm (0.6550) — a
    # global acc_norm→acc priority would silently report a different number.
    m = macro_metrics(_ledger((0.6950, 0.7900), (0.8350, 0.8100),
                              (0.6400, 0.6550), 0.7550, (0.7950, 0.8400)))
    assert m == {"hellaswag": 0.7900, "arc_easy": 0.8350, "arc_challenge": 0.6400,
                 "winogrande": 0.7550, "piqa": 0.8400}


def test_macro_averages_over_the_subset_that_scored():
    res = _results(arc_challenge={"acc,none": 0.60}, winogrande={"acc,none": 0.70})
    assert macro_score(res) == pytest.approx(0.65)


def test_macro_off_rank0_is_zero():
    # simple_evaluate returns None on every rank but global rank 0; those ranks
    # take the broadcast value instead of this one.
    assert macro_score(None) == 0.0
    assert macro_score({}) == 0.0
    assert macro_score(_results()) == 0.0
    assert macro_metrics(None) == {}


def test_tasks_outside_the_five_fall_back():
    res = _results(mmlu={"acc,none": 0.55}, some_task={"acc_norm,none": 0.45})
    assert macro_score(res) == pytest.approx(0.50)
