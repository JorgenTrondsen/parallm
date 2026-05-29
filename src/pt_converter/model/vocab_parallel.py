"""Vocab-parallel (tensor-parallel over the vocabulary) primitives.

The PT student's only full-vocab-sized tensors are ``embed_tokens`` and
``lm_head`` (each ``[V, H]``) plus the fp32 KL/CE expansion over the vocab.
In the original layout all of these lived on the single track-0 owner rank,
which made that rank the binding memory constraint (and the serial bottleneck
for the KL/CE phase). Sharding the vocab dimension ``V`` across all ranks
spreads those costs evenly:

  - each rank ``r`` owns rows ``[v_lo : v_hi)`` of ``embed_tokens`` / ``lm_head``;
  - the input embedding is each rank's slice-lookup summed across ranks (the
    same all-reduce the existing ``SyncBoundary`` already performs);
  - the output KL+CE is a vocab-parallel softmax (global log-sum-exp via a
    couple of small all-reduces) — see ``train/distill.py``.

This is a natural fit for this codebase because every rank already holds the
full post-sync hidden state and the full teacher logits, so no extra
gathers of the (large) hidden / teacher tensors are needed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def vocab_range(vocab_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Half-open vocab row range ``[v_lo, v_hi)`` owned by ``rank``.

    Uses a uniform shard width ``Vs = ceil(V / world_size)`` so the offset of
    rank ``r`` is simply ``r * Vs`` (gather/scatter can pad to ``Vs`` and trim).
    The last rank(s) may own a smaller (or empty, for very small V) slice; the
    vocab-parallel reductions sum over the owned rows only, so unequal widths
    are fine. With ``world_size == 1`` this is the full vocab.
    """
    shard = math.ceil(vocab_size / world_size)
    v_lo = min(rank * shard, vocab_size)
    v_hi = min(v_lo + shard, vocab_size)
    return v_lo, v_hi


class VocabParallelEmbedding(nn.Module):
    """Embedding holding only vocab rows ``[v_lo, v_hi)`` of a ``[V, H]`` table.

    ``forward`` returns this rank's *partial* embedding: tokens whose id falls
    in ``[v_lo, v_hi)`` are looked up locally, all other positions are zeroed.
    Summing the partials across ranks (one all-reduce, done by the caller via
    ``SyncBoundary``) reconstructs the full embedding on every rank.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        v_lo: int,
        v_hi: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.v_lo = v_lo
        self.v_hi = v_hi
        self.num_local = v_hi - v_lo
        # Only the shard that owns the pad row applies padding_idx semantics;
        # other shards never see that id in their range so it is a no-op there.
        self.local_padding_idx = (
            padding_idx - v_lo if (padding_idx is not None and v_lo <= padding_idx < v_hi) else None
        )
        self.weight = nn.Parameter(torch.empty(self.num_local, embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Mask to this shard's vocab range; out-of-range ids are looked up at
        # local index 0 then zeroed, so they contribute nothing to the sum and
        # leak no gradient into row 0.
        in_range = (input_ids >= self.v_lo) & (input_ids < self.v_hi)
        local_ids = (input_ids - self.v_lo).clamp_(0, self.num_local - 1)
        emb = F.embedding(local_ids, self.weight, padding_idx=self.local_padding_idx)
        return emb * in_range.unsqueeze(-1).to(emb.dtype)
