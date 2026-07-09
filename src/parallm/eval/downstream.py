"""In-loop downstream (lm-eval) scoring for checkpoint selection.

Training selects ``best/`` and early-stops on ``val_kl`` today, but the project's
own findings show KL/ppl can read ~85% recovered while hard-reasoning *downstream*
retention is ~22-33% — the proxy hides the failure. This module runs a small, fast
multiple-choice subset of lm-evaluation-harness *inside* the training loop so the
selection metric can be the thing we actually care about (downstream accuracy),
not the misleading proxy.

It reuses the ``PTLM`` adapter from ``eval/lm_eval_adapter.py`` (so scoring,
batching, and the distributed lockstep-forward contract are shared with the
standalone ``scripts/eval_lm_harness.py``). The one new piece is **vocab-parallel
support**: during training the student's ``lm_head`` is sharded over the vocab dim
across ranks, so its forward returns only a ``(B, T, Vs)`` logit slice. We
all-gather those slices into full ``(B, T, V)`` logits so the loglikelihood scorer
sees the same thing the standalone (non-vocab-parallel) eval does.

Only the pure ``aggregate_downstream_score`` / logit-gather helpers live at module
scope (no ``lm_eval`` import), so they are importable / unit-testable on CPU
without the ``[eval]`` extra. ``run_downstream_eval`` imports ``lm_eval`` lazily.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


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


def gather_full_logits(student, local_logits: torch.Tensor) -> torch.Tensor:
    """All-gather a vocab-parallel ``(B, T, Vs)`` logit shard into full ``(B, T, V)``.

    Mirrors ``PTWrappedModel.gather_vocab_parallel_weights``: pad each rank's shard
    to the uniform width ``ceil(V/world)``, all-gather in rank order, concatenate
    along the vocab dim, and trim to ``V``. Returns the FULL logits on **every**
    rank (the all-gather is collective, so it must run on every rank for lockstep).
    Non-vocab-parallel / single-shard students return ``local_logits`` unchanged.
    """
    if not getattr(student, "vocab_parallel", False) or student.vp_world_size <= 1:
        return local_logits
    V = student.text_config.vocab_size
    ws = student.vp_world_size
    shard = math.ceil(V / ws)
    B, T, Vs = local_logits.shape
    padded = local_logits.new_zeros((B, T, shard))
    padded[..., :Vs] = local_logits
    out = [torch.empty_like(padded) for _ in range(ws)]
    if dist.is_initialized() and student.vp_group is not None:
        dist.all_gather(out, padded.contiguous(), group=student.vp_group)
    else:
        out = [padded]
    return torch.cat(out, dim=-1)[..., :V].contiguous()


def make_student_forward_fn(student):
    """``ForwardFn`` for ``PTLM`` that returns full ``(B, T, V)`` logits.

    Vocab-parallel: gather the per-rank shards into full logits on every rank.
    Non-vocab-parallel: the owner rank produces full logits, peers ``None`` (the
    legacy ``PTLM`` contract). Either way the cross-track ``SyncBoundary``
    collectives inside the forward fire on every rank, so the run stays in lockstep.
    """
    vp = getattr(student, "vocab_parallel", False) and student.vp_world_size > 1

    def _fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> "torch.Tensor | None":
        if vp:
            hidden, _ = student(
                input_ids=input_ids, attention_mask=attention_mask,
                return_sync_hiddens=False, return_hidden_pre_lm_head=True,
            )
            return gather_full_logits(student, student.lm_head(hidden))
        logits, _ = student(
            input_ids=input_ids, attention_mask=attention_mask, return_sync_hiddens=False,
        )
        return logits

    return _fn


def run_downstream_eval(
    student,
    tokenizer,
    *,
    tasks: "list[str]",
    limit: "int | None",
    batch_size: int,
    max_length: int,
    num_fewshot: "int | None",
    seed: int,
    rank: int,
    forward_fn=None,
) -> dict:
    """Run a small lm-eval pass on the student; return a broadcast aggregate score.

    Every rank runs identical lm-eval code (same seeds ⇒ same request stream) so
    the in-forward collectives line up; only rank 0 holds real loglikelihoods
    (vocab-parallel gathers full logits on every rank, but we still score on rank 0
    only and broadcast, matching the standalone eval's owner/peer pattern). The
    returned ``score`` is identical on every rank — safe to drive the (collective)
    checkpoint-save and early-stop decisions with.

    Returns ``{"score": float, "per_task": {task: acc}}`` (``per_task`` populated
    on rank 0 only; ``score`` broadcast to all ranks).
    """
    # Lazy: lm_eval (and the PTLM adapter that imports it) is the `[eval]` extra.
    from lm_eval import simple_evaluate

    from parallm.eval.lm_eval_adapter import PTLM

    device = torch.cuda.current_device()
    lm = PTLM(
        forward_fn=forward_fn if forward_fn is not None else make_student_forward_fn(student),
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        is_owner_rank=(rank == 0),
    )
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )

    per_task: dict[str, float] = {}
    score = 0.0
    if rank == 0:
        score = aggregate_downstream_score(results, tasks)
        table = (results or {}).get("results", {})
        for task in tasks:
            val = _pick_metric(table.get(task, {}), ["acc_norm", "acc"])
            if val is not None:
                per_task[task] = val

    # Broadcast rank-0's score so every rank takes the same selection decision
    # (the save + early-stop paths are collective / loop-controlling).
    score_t = torch.tensor([score], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.broadcast(score_t, src=0)
    return {"score": score_t.item(), "per_task": per_task}
