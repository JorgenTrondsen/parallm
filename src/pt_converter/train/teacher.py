"""Frozen dense teacher with hooks at the PT sync layer indices.

Loads the original Qwen3.5 dense text model, registers forward hooks on
the decoder layers at every sync boundary, and exposes a `forward()` that
returns (final_logits, {layer_idx: hidden_state}).

The teacher's hidden states must be captured *after* the layer's residual
addition completes (i.e., the layer's output). HF's `Qwen3_5DecoderLayer.forward`
returns the post-residual hidden state directly, so a forward hook on the
DecoderLayer module captures exactly what we want.

Batch-sharded mode (``shard_group`` + ``shard_world_size`` > 1): in the
training loop every rank holds the IDENTICAL batch, so running the full
teacher forward on every rank computes the same thing world_size times.
Sharding splits the batch across the ranks — each runs the forward on its
``ceil(B / world)`` slice — and one all-gather per captured layer rebuilds
the full-batch hidden on every rank. The gathered tensors are bit-identical
everywhere (all_gather is deterministic), so everything downstream is
unchanged; the per-rank teacher compute drops to ~1/world of the dense
forward at the cost of one (B, T, H) all-gather per capture.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn


class HookedTeacher:
    """Wraps a loaded dense text decoder + lm_head and captures hidden states at sync points.

    Args:
        shard_group / shard_world_size / shard_rank: enable the batch-sharded
            forward (see module docstring). The default (world 1) keeps the
            legacy full-batch path — single-process callers (tests, eval
            scripts) are unaffected. ``shard_group=None`` with a world > 1
            uses the default process group.
    """

    def __init__(
        self,
        text_model: nn.Module,
        lm_head: nn.Linear,
        sync_layer_indices: Iterable[int],
        shard_group: "dist.ProcessGroup | None" = None,
        shard_world_size: int = 1,
        shard_rank: int = 0,
    ):
        self.text_model = text_model
        self.lm_head = lm_head
        self.sync_indices = list(sync_layer_indices)
        self.shard_group = shard_group
        self.shard_world_size = shard_world_size
        self.shard_rank = shard_rank
        self._captures: dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._install_hooks()

    def _install_hooks(self):
        layers = list(self.text_model.layers)
        for idx in self.sync_indices:
            layer = layers[idx]

            def make_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    # Qwen3_5DecoderLayer returns a Tensor (post-residual hidden state).
                    out = output if isinstance(output, torch.Tensor) else output[0]
                    self._captures[layer_idx] = out
                return _hook

            self._handles.append(layer.register_forward_hook(make_hook(idx)))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        need_logits: bool = True,
    ) -> tuple["torch.Tensor | None", dict[int, torch.Tensor]]:
        """Teacher forward → ``(logits_or_None, {layer_idx: hidden})``.

        ``need_logits=False`` skips the lm_head matmul (and, in sharded mode,
        the final-hidden all-gather) and returns ``logits=None`` — for steps
        where the logit objective is off and kl/ce aren't being logged.
        """
        if self.shard_world_size > 1:
            return self._forward_batch_sharded(input_ids, attention_mask, need_logits)
        # Install a fresh dict the hooks write into this forward.
        self._captures = {}
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = self.lm_head(outputs.last_hidden_state) if need_logits else None
        # Hand the caller SOLE ownership of the captures and drop our reference,
        # so the caller can free each hidden state (pop/del) as it is consumed
        # — otherwise the 8 (B,T,H) bf16 tensors stay resident for the whole
        # step (grows ×B, the headroom we want back for larger batch sizes).
        captures = self._captures
        self._captures = {}
        return logits, captures

    @torch.no_grad()
    def _forward_batch_sharded(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        need_logits: bool,
    ) -> tuple["torch.Tensor | None", dict[int, torch.Tensor]]:
        """Batch-sharded forward: each rank computes B/world, gathers per capture.

        The batch is padded to ``shard * world`` rows by repeating row 0 (the
        padded rows' outputs are trimmed away after each gather), so the
        collective shapes are uniform even when world doesn't divide B. Every
        rank calls this on the SAME full batch and the gathers run in the fixed
        ``sorted(sync_indices)`` order, so the collectives stay matched.
        """
        B = input_ids.shape[0]
        ws = self.shard_world_size
        shard = math.ceil(B / ws)
        padded = shard * ws

        def pad(t: torch.Tensor) -> torch.Tensor:
            if padded == B:
                return t
            return torch.cat([t, t[:1].expand(padded - B, *t.shape[1:])], dim=0)

        lo = self.shard_rank * shard
        ids_local = pad(input_ids)[lo:lo + shard]
        mask_local = pad(attention_mask)[lo:lo + shard] if attention_mask is not None else None

        self._captures = {}
        outputs = self.text_model(
            input_ids=ids_local,
            attention_mask=mask_local,
            use_cache=False,
        )
        local_captures = self._captures
        self._captures = {}

        def gather(local: torch.Tensor) -> torch.Tensor:
            out = local.new_empty((padded, *local.shape[1:]))
            dist.all_gather_into_tensor(out, local.contiguous(), group=self.shard_group)
            if padded == B:
                return out
            # clone() so the trimmed result owns exact-size storage — a [:B]
            # view would pin the padded buffer for the capture's whole lifetime.
            return out[:B].clone()

        captures = {}
        for idx in sorted(local_captures):
            # pop: free each rank-local shard capture as soon as it's gathered.
            captures[idx] = gather(local_captures.pop(idx))
        logits = None
        if need_logits:
            # lm_head on the gathered full-batch hidden — identical output
            # (full logits, or this rank's vocab shard when lm_head is sharded)
            # to applying it inside the unsharded forward.
            logits = self.lm_head(gather(outputs.last_hidden_state))
        return logits, captures
