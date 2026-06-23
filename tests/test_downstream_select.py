"""Unit tests for the in-loop downstream-selection scorer (eval/downstream.py).

Only the pure pieces (no lm_eval, no GPUs): the metric aggregation that turns an
lm-eval ``simple_evaluate`` dict into a single selection scalar, and the
non-vocab-parallel passthrough of the logit gather.
"""
from types import SimpleNamespace

import pytest
import torch

from pt_converter.eval.downstream import (
    aggregate_downstream_score,
    gather_full_logits,
)


def _results(**task_metrics):
    """Build a minimal lm-eval-shaped results dict."""
    return {"results": dict(task_metrics)}


def test_prefers_acc_norm_over_acc_and_strips_filter_suffix():
    # lm-eval keys carry a ",<filter>" suffix; acc_norm should win over acc.
    res = _results(arc_challenge={"acc,none": 0.30, "acc_norm,none": 0.50})
    assert aggregate_downstream_score(res, ["arc_challenge"]) == 0.50


def test_falls_back_to_acc_when_no_acc_norm():
    res = _results(winogrande={"acc,none": 0.62})
    assert aggregate_downstream_score(res, ["winogrande"]) == 0.62


def test_means_over_multiple_tasks():
    res = _results(
        arc_challenge={"acc_norm,none": 0.40},
        winogrande={"acc,none": 0.60},
    )
    assert aggregate_downstream_score(res, ["arc_challenge", "winogrande"]) == pytest.approx(0.50)


def test_missing_task_is_skipped():
    res = _results(arc_challenge={"acc_norm,none": 0.40})
    # winogrande absent → averaged over the one task that scored.
    assert aggregate_downstream_score(res, ["arc_challenge", "winogrande"]) == 0.40


def test_tasks_none_uses_all_present():
    res = _results(
        a={"acc,none": 0.2},
        b={"acc,none": 0.4},
    )
    assert aggregate_downstream_score(res) == pytest.approx(0.30)


def test_empty_or_none_results_is_zero():
    assert aggregate_downstream_score(None) == 0.0
    assert aggregate_downstream_score({}) == 0.0
    assert aggregate_downstream_score(_results(), ["arc_challenge"]) == 0.0


def test_gather_full_logits_non_vocab_parallel_passthrough():
    # A student without vocab parallelism returns its logits unchanged (the
    # owner already produced full (B, T, V) logits).
    student = SimpleNamespace(vocab_parallel=False, vp_world_size=1)
    logits = torch.randn(2, 3, 5)
    assert gather_full_logits(student, logits) is logits


def test_gather_full_logits_single_shard_passthrough():
    # vocab_parallel flagged but world_size == 1 (single shard == full vocab).
    student = SimpleNamespace(vocab_parallel=True, vp_world_size=1)
    logits = torch.randn(1, 4, 7)
    assert gather_full_logits(student, logits) is logits
