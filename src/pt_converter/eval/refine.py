"""Jacobi / iterative-refinement forward: fresh cross-track content at every
sublayer from a handful of bulk exchanges instead of per-layer syncs.

The scheme: every track runs the FULL layer stack comm-free (pass 0), caching
each sublayer's per-track residual contribution (token-mixer output ``Y`` and
MLP output ``M``). ONE bulk exchange — a single all-reduce of the stacked
``(2L, B, T, H)`` local contribution sums — then delivers Σ-across-tracks of
every sublayer's contribution. The stack is then re-run for ``iters``
refinement passes; pass ``k`` reconstructs every layer input from the
pass-``k−1`` exchanged sums, so every layer reads a full-scale, full-rank
approximate residual (no partial-residual regime anywhere). Cost per token:
``iters+1`` sync EVENTS (vs 32 at D=1) at ``iters+1``× compute.

Sublayer indexing: ``j = 2i`` is layer ``i``'s token mixer (contribution
computed on the layer input), ``j = 2i+1`` its MLP (computed on the post-attn
point). ``S[j]`` = Σ over all tracks of sublayer ``j``'s contribution.

Two carry rules for the refinement passes:

* ``own-fresh`` — each track carries its own residual; at sublayer ``j`` it
  adds its OWN fresh contribution plus the cached ``S_prev[j] − own_prev[j]``
  (others' previous-pass content), then overwrites the own-cache slot.
* ``shared`` — every sublayer computes on the pass-``k−1`` reconstructed trunk
  ``R_j = emb + Σ_{j'<j} S_prev[j']``; all tracks see identical inputs.

Provable exactness (both carries): the input to sublayer 0 is the embedding
(exact every pass), so by induction pass ``k`` makes the first ``k+1`` sublayer
sums exact; at ``iters = 2L−1`` the final trunk equals the fully-synced
per-sublayer forward — on a freshly sliced (untrained) model, the DENSE
forward. The downstream-vs-``iters`` curve therefore measures pure fixed-point
convergence speed, with no training confound.

Evaluated under lm-harness loglikelihood = full-sequence scoring = PREFILL
semantics, where whole-sequence refinement (all positions iterate together) is
the natural deployment mode. Token-by-token decode (past tokens' per-track
KV/GDN state frozen at their final pass) is a related but distinct scheme —
out of scope here.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist

from pt_converter.eval.intervention import _sum_track_deltas
from pt_converter.model.cross_head_estimator import seam_mlp, seam_token_mixer
from pt_converter.model.pt_model import PTWrappedModel

_CARRIES = ("own-fresh", "shared")


def _exchange_all_layers(
    stack: torch.Tensor, group: "dist.ProcessGroup | None"
) -> torch.Tensor:
    """ONE bulk all-reduce of the stacked per-sublayer local K-track sums.

    ``stack``: ``(2L, B, T, H)`` in the model dtype. This is the whole
    exchange — one NCCL call per refinement pass regardless of depth.
    ``group=None`` (single-process multi-track) skips the collective.
    """
    if group is not None and dist.is_initialized():
        stack = stack.contiguous()
        dist.all_reduce(stack, op=dist.ReduceOp.SUM, group=group)
    return stack


class RefineSpec:
    """One refinement configuration. Duck-types ``Channel.name`` so the probe's
    per-channel loop and results table apply unchanged."""

    def __init__(
        self,
        iters: int,
        carry: str = "own-fresh",
        base_sync_indices: "tuple[int, ...] | None" = None,
    ):
        if carry not in _CARRIES:
            raise ValueError(f"unknown refine carry {carry!r}; expected one of {_CARRIES}")
        if iters < 0:
            raise ValueError(f"iters must be >= 0, got {iters}")
        self.iters = iters
        self.carry = carry
        self.base_sync_indices = tuple(base_sync_indices) if base_sync_indices else None
        b = f"+b{len(self.base_sync_indices)}" if self.base_sync_indices else ""
        self.name = f"refine:{carry}:x{iters}{b}"


@torch.no_grad()
def refine_forward(
    student: PTWrappedModel,
    input_ids: torch.Tensor,
    attention_mask: "torch.Tensor | None",
    *,
    iters: int,
    carry: str = "own-fresh",
    base_sync_indices: "Sequence[int] | None" = None,
) -> torch.Tensor:
    """The iterative-refinement forward. Returns the post-final-norm hidden
    ``(B, T, H)`` (pre-``lm_head``), identical on every rank.

    Pass structure: pass 0 (comm-free per-track, optionally with real boundary
    syncs at ``base_sync_indices`` — the hybrid arm, +1 event each), exchange,
    then ``iters`` refinement passes each ending in one exchange. Every rank
    must call it in lockstep: exactly ``iters+1`` bulk exchanges (plus the base
    syncs) in fixed order, no rank-dependent branching.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    if carry not in _CARRIES:
        raise ValueError(f"unknown refine carry {carry!r}; expected one of {_CARRIES}")

    emb = student.embed(input_ids)
    tm0 = student.text_models[0]
    position_ids, text_position_ids = tm0._resolve_position_ids(emb, None)
    causal_mask = create_causal_mask(
        config=tm0.config, inputs_embeds=emb, attention_mask=attention_mask,
        past_key_values=None, position_ids=text_position_ids,
    )
    linear_attn_mask = (
        None if (attention_mask is not None and torch.all(attention_mask == 1)) else attention_mask
    )
    position_embeddings = tm0.rotary_emb(emb, position_ids)
    L = len(tm0.layers)
    K = len(student.text_models)
    group = student.sync_module.track_group
    # The last layer's sync is subsumed by the final trunk combine.
    base_set = set(base_sync_indices or ()) - {L - 1}

    def _mask(i: int):
        return linear_attn_mask if tm0.config.layer_types[i] == "linear_attention" else causal_mask

    # ---- pass 0: comm-free per-track run, caching contributions ----
    sums = torch.zeros((2 * L, *emb.shape), dtype=emb.dtype, device=emb.device)
    own = [torch.zeros_like(sums) for _ in range(K)] if carry == "own-fresh" else None
    h = [emb for _ in range(K)]
    block_start = emb
    for i in range(L):
        layer_mask = _mask(i)
        for k, tm in enumerate(student.text_models):
            ha, y, _ = seam_token_mixer(
                tm.layers[i], h[k], position_embeddings, layer_mask, text_position_ids
            )
            sums[2 * i] += y
            if own is not None:
                own[k][2 * i] = y
            nh = seam_mlp(tm.layers[i], ha, None)
            sums[2 * i + 1] += nh - ha
            if own is not None:
                own[k][2 * i + 1] = nh - ha
            h[k] = nh
        if i in base_set:
            base = block_start + _sum_track_deltas([hk - block_start for hk in h], group)
            h = [base for _ in range(K)]
            block_start = base
    S_prev = _exchange_all_layers(sums, group)

    # ---- refinement passes 1..iters ----
    for _ in range(iters):
        sums = torch.zeros_like(S_prev)
        if carry == "shared":
            R = emb
            for i in range(L):
                layer_mask = _mask(i)
                for tm in student.text_models:
                    _, y, _ = seam_token_mixer(
                        tm.layers[i], R, position_embeddings, layer_mask, text_position_ids
                    )
                    sums[2 * i] += y
                R_mid = R + S_prev[2 * i]
                for tm in student.text_models:
                    sums[2 * i + 1] += seam_mlp(tm.layers[i], R_mid, None) - R_mid
                R = R_mid + S_prev[2 * i + 1]
        else:  # own-fresh
            h = [emb for _ in range(K)]
            for i in range(L):
                layer_mask = _mask(i)
                for k, tm in enumerate(student.text_models):
                    ha, y, _ = seam_token_mixer(
                        tm.layers[i], h[k], position_embeddings, layer_mask, text_position_ids
                    )
                    sums[2 * i] += y
                    h_mid = ha + (S_prev[2 * i] - own[k][2 * i])
                    own[k][2 * i] = y
                    nh = seam_mlp(tm.layers[i], h_mid, None)
                    m = nh - h_mid
                    sums[2 * i + 1] += m
                    h[k] = nh + (S_prev[2 * i + 1] - own[k][2 * i + 1])
                    own[k][2 * i + 1] = m
        S_prev = _exchange_all_layers(sums, group)

    # fp32 accumulation of the 2L-term trunk sum (bf16 drift guard at scale).
    trunk = emb + S_prev.float().sum(dim=0).to(emb.dtype)
    return tm0.norm(trunk)


def refine_intervention_forward(student, input_ids, attention_mask, sync_indices, channel):
    """Adapter matching the probe's ``fwd(student, ids, attn, sync_indices,
    channel)`` call signature. ``sync_indices`` is ignored — refinement has no
    boundary schedule; pass-0 base syncs come from the ``RefineSpec``."""
    return refine_forward(
        student, input_ids, attention_mask,
        iters=channel.iters, carry=channel.carry,
        base_sync_indices=channel.base_sync_indices,
    )
