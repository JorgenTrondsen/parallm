"""Block draft-verify greedy decode (lever 2, deployment semantics).

A small same-tokenizer DRAFT proposes ``k`` tokens; ONE verifier pass over
``prefix + proposals`` accepts the agreeing prefix and appends the verifier's
own next token after the last accepted proposal (the bonus/correction token).
Greedy everywhere ⇒ by induction every emitted token is the verifier's greedy
next-token given its prefix, so the emitted sequence is EXACTLY the verifier's
greedy decode — quality == verifier by construction. What varies is the
amortization factor τ = emitted tokens per verify event; sync events per
decoded token = (events per verify pass) / τ. The offline probe
(``probe_draft_accept.py``) measures τ teacher-forced on data text; this module
measures it on the verifier's OWN trajectory, which is the deployed number.

The loop is rank-agnostic: both callables must return identical values on
every rank of a distributed verifier (the probe script wraps them with
broadcasts so the sequence stays in lockstep while the PT student's internal
collectives run on all ranks).
"""
from __future__ import annotations

from typing import Callable

import torch

# (seq (1, T) -> proposals (k,)) and (seq (1, T), proposals (k,) -> emitted (1, <=k+1))
ProposeFn = Callable[[torch.Tensor, int], torch.Tensor]
VerifyStepFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def greedy_verify_step_factory(verifier_logits) -> VerifyStepFn:
    """Wrap a full-sequence logits fn ``ids (1, T) -> (1, T, V)`` as a verify
    step. Position ``P-1+i`` of the joint pass predicts token index ``P+i``, so
    the argmaxes at positions ``P-1 .. P+k-1`` are the verifier's greedy targets
    for the k proposal slots plus the bonus slot."""

    def _step(seq: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        ids = torch.cat([seq, proposals.view(1, -1)], dim=1)
        logits = verifier_logits(ids)
        p_len, k = seq.shape[1], proposals.numel()
        targets = logits[0, p_len - 1 : p_len + k].argmax(-1)  # (k+1,)
        accepted = int((targets[:k] == proposals).cumprod(dim=0).sum())
        return targets[: accepted + 1].view(1, -1)

    return _step


def naive_greedy_propose_factory(draft) -> ProposeFn:
    """Draft proposer by full re-forward per token — no KV-cache correctness
    surface, identical semantics. Deployment would cache; τ does not care
    (draft compute is not a verify event)."""

    @torch.no_grad()
    def _propose(seq: torch.Tensor, k: int) -> torch.Tensor:
        ids = seq
        toks = []
        for _ in range(k):
            tok = draft(input_ids=ids).logits[:, -1].argmax(-1)  # (1,)
            toks.append(tok)
            ids = torch.cat([ids, tok.view(1, 1)], dim=1)
        return torch.cat(toks, dim=0)

    return _propose


@torch.no_grad()
def draft_verify_generate(
    draft_propose: ProposeFn,
    verify_step: VerifyStepFn,
    prompt_ids: torch.Tensor,
    *,
    k: int,
    max_new: int,
    eos_token_id: "int | None" = None,
) -> "tuple[torch.Tensor, int, int]":
    """Block draft-verify greedy decode from ``prompt_ids`` (1, P).

    Returns ``(seq, tokens, events)``: the full sequence incl. prompt, the
    number of emitted tokens, and the number of verify events. Stops after
    ``max_new`` tokens (the final event may overshoot by up to k; overshoot is
    kept — it was bought with the same event) or at ``eos_token_id``.
    """
    seq = prompt_ids
    tokens = events = 0
    while tokens < max_new:
        proposals = draft_propose(seq, k)
        emitted = verify_step(seq, proposals)
        events += 1
        if eos_token_id is not None:
            hits = (emitted[0] == eos_token_id).nonzero()
            if hits.numel():
                emitted = emitted[:, : int(hits[0]) + 1]
                seq = torch.cat([seq, emitted], dim=1)
                tokens += emitted.shape[1]
                break
        seq = torch.cat([seq, emitted], dim=1)
        tokens += emitted.shape[1]
    return seq, tokens, events
