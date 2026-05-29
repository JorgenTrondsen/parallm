"""PTWrappedModel: the user-facing per-rank model.

One PTWrappedModel instance per *rank*. Holds the K = tracks_per_rank tracks
this rank owns (as `nn.ModuleList` of per-track text models) plus (on the
rank that owns track 0) the shared `lm_head`. Exposes a forward that runs
all K tracks lockstep through the layers, driving cross-track sync at the
configured sync points, and returns `(logits, sync_hiddens)` for the
distillation loop.

The model-specific decoder layer assembly is provided by a `ModelAdapter`
(see `pt_converter.adapters`). PTWrappedModel itself holds no model-family
knowledge: it looks up the adapter by `text_config.model_type`, calls its
per-track config builder, and instantiates its `track_text_model_cls`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from torch.utils.checkpoint import checkpoint

from pt_converter.model.sync import SyncBoundary


@dataclass
class PTTrackTextModelConfig:
    """The engine→adapter contract for instantiating a per-track text model.

    Every adapter's `track_text_model_cls` accepts the constructor signature
    `(per_track_text_config, pt_cfg: PTTrackTextModelConfig, sync_module: SyncBoundary)`.
    Fields here are model-agnostic; per-track-specific dim changes live in
    `per_track_text_config` produced by the adapter.
    """

    n_tracks: int
    sync_after_layers: tuple[int, ...]
    track_id: int = 0
    # When False, track 0 does NOT build its full [V,H] embed_tokens — the
    # vocab-parallel path hosts a sharded embedding on the wrapper instead.
    host_embed_tokens: bool = True


class PTWrappedModel(nn.Module):
    LM_HEAD_OWNER_TRACK = 0

    def __init__(
        self,
        text_config,
        *,
        n_tracks: int,
        local_track_ids: Sequence[int],
        sync_after_layers: list[int],
        track_group: "torch.distributed.ProcessGroup | None" = None,
        activation_checkpoint: bool = False,
        compile_layers: bool = False,
        compile_mode: str = "default",
        vocab_parallel: bool = False,
        vp_world_size: int = 1,
        vp_rank: int = 0,
    ):
        super().__init__()
        # Late import: pt_converter.adapters imports its registered adapters,
        # which in turn import model/tracks/<model>.py — those import this
        # module for `PTTrackTextModelConfig`. Importing the registry lazily
        # here breaks the cycle without losing the import-time registration.
        from pt_converter.adapters import get_adapter_for_config

        adapter = get_adapter_for_config(text_config)
        per_track_cfg = adapter.build_per_track_text_config(text_config, n_tracks)
        self.text_config = text_config
        self.per_track_text_config = per_track_cfg
        self.n_tracks = n_tracks
        self.local_track_ids = tuple(local_track_ids)
        self.sync_after_layers = tuple(sync_after_layers)
        self._adapter = adapter  # held for `load_track_state_dicts` remap
        # Per-layer activation checkpointing of the full student forward. Under
        # vocab-parallel (the default) EVERY rank backwards through that forward
        # — `_kl_ce_vocab_parallel` ends in one `hidden.backward(grad_h_accum)`
        # on every rank — so checkpointing trades the held forward graph (the
        # ~25 GB `student_fwd`/`klce` peak) for one recompute pass on every rank,
        # lowering the peak uniformly. (Legacy --no-vocab-parallel: only the
        # lm_head-owner rank backwards; peers run the forward for collective
        # ordering and pay no recompute.) The KL/CE backward is a single
        # `hidden.backward` either way (not per-chunk). This is the lever for
        # fitting larger batch sizes at seq_len=4096 on 40 GB GPUs.
        self._use_checkpoint = activation_checkpoint

        # ----- Vocab-parallel (tensor-parallel over V) setup -----
        # When enabled, embed_tokens + lm_head + the KL/CE softmax are sharded
        # over the vocab dimension across all `vp_world_size` ranks (rank `r`
        # owns rows [v_lo, v_hi)). This decouples those full-vocab-sized tensors
        # from track-0 ownership, balancing memory and parallelizing the KL/CE
        # phase. See `pt_converter.model.vocab_parallel`.
        self.vocab_parallel = vocab_parallel
        self.vp_world_size = vp_world_size
        self.vp_rank = vp_rank
        self.vp_group = track_group  # spans one rank per GPU (== WORLD here)
        if vocab_parallel:
            from pt_converter.model.vocab_parallel import vocab_range
            self.v_lo, self.v_hi = vocab_range(text_config.vocab_size, vp_world_size, vp_rank)
        else:
            self.v_lo, self.v_hi = 0, text_config.vocab_size

        self.sync_module = SyncBoundary(track_group=track_group, n_tracks=n_tracks)
        self.text_models = nn.ModuleList(
            [
                adapter.track_text_model_cls(
                    per_track_cfg,
                    PTTrackTextModelConfig(
                        n_tracks=n_tracks,
                        sync_after_layers=tuple(sync_after_layers),
                        track_id=tid,
                        host_embed_tokens=not vocab_parallel,
                    ),
                    sync_module=self.sync_module,
                )
                for tid in self.local_track_ids
            ]
        )
        # Per-track decoder layers are tiny at high n_tracks (e.g. 1 attention head
        # and a 768-wide MLP per track at n_tracks=16) and run in a sequential
        # Python loop over the K local tracks, so each rank is launch-bound on many
        # small kernels. Compile each layer's forward in place so inductor fuses the
        # tiny kernels and cuts launch overhead. We use `layer.compile(...)` (the
        # in-place nn.Module method) — NOT `torch.compile(layer)`, which wraps the
        # module in an OptimizedModule and renames state_dict keys to
        # `...layers.{i}._orig_mod.*`, breaking load_track_state_dicts/save. The
        # in-place form leaves module identity and state_dict() untouched, so both
        # forward call sites and all checkpoint I/O keep working. The SyncBoundary
        # all-reduce lives in the forward loop (outside the layer), so no collective
        # is captured into a graph. dynamic=False: shapes are static (B, T fixed).
        if compile_layers:
            for tm in self.text_models:
                for layer in tm.layers:
                    layer.compile(mode=compile_mode, dynamic=False)

        if vocab_parallel:
            # Vocab-parallel: every rank holds the embed + lm_head shard for its
            # vocab range [v_lo, v_hi). The embedding is summed across ranks
            # (one all-reduce in `embed`); lm_head produces this rank's logit
            # slice, consumed by the vocab-parallel KL/CE in `train/distill.py`.
            from pt_converter.model.vocab_parallel import VocabParallelEmbedding
            self.vp_embed = VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                self.v_lo,
                self.v_hi,
                padding_idx=getattr(text_config, "pad_token_id", None),
            )
            self.lm_head = nn.Linear(text_config.hidden_size, self.v_hi - self.v_lo, bias=False)
        else:
            # lm_head lives on the owner track only. The final SyncBoundary
            # broadcasts the post-block hidden state to all tracks, so whichever
            # rank hosts the owner already has the correct synced state to
            # project to logits. Peer ranks return logits=None from forward.
            self.vp_embed = None
            if self.LM_HEAD_OWNER_TRACK in self.local_track_ids:
                self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
            else:
                self.lm_head = None

    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Full input embedding (B, S, H), grad-connected, identical on every rank.

        Vocab-parallel: each rank looks up its vocab shard (zeros elsewhere) and
        the partials are summed via one all-reduce. Non-VP (legacy): the owner
        track embeds the full vocab, peers contribute zeros, and the same sum
        broadcasts it to every track. Both go through `sync_module`, so the
        cross-track collective ordering is identical.
        """
        if self.vocab_parallel:
            partial = self.vp_embed(input_ids)
            return self.sync_module([partial], torch.zeros_like(partial))
        embeds_per_track: list[torch.Tensor] = []
        for tm in self.text_models:
            if tm.embed_tokens is not None:
                embeds_per_track.append(tm.embed_tokens(input_ids))
            else:
                B, S = input_ids.shape
                embeds_per_track.append(
                    torch.zeros(
                        B,
                        S,
                        tm.config.hidden_size,
                        device=input_ids.device,
                        dtype=tm.norm.weight.dtype,
                    )
                )
        return self.sync_module(embeds_per_track, torch.zeros_like(embeds_per_track[0]))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_sync_hiddens: bool = False,
        return_hidden_pre_lm_head: bool = False,
    ):
        # Local import: keeps the engine model-family-agnostic at import time.
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

        # 1. Embed (vocab-parallel slice, or legacy owner-only) + cross-track
        #    broadcast via the all-reduce in `embed`.
        h = self.embed(input_ids)

        # 2. Scaffolding (rotary, masks) computed once — every track's per-track
        # config is identical, so reuse the first track's modules.
        tm0 = self.text_models[0]
        position_ids_resolved, text_position_ids = tm0._resolve_position_ids(h, position_ids)
        causal_mask = create_causal_mask(
            config=tm0.config,
            inputs_embeds=h,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = (
            None
            if (attention_mask is not None and torch.all(attention_mask == 1))
            else attention_mask
        )
        position_embeddings = tm0.rotary_emb(h, position_ids_resolved)

        # 3. Lockstep layer iteration with per-block syncs.
        block_start = h
        sync_set = set(self.sync_after_layers)
        sync_hiddens: dict[int, torch.Tensor] = {} if return_sync_hiddens else None
        per_track_h = [block_start for _ in self.text_models]
        use_ckpt = self._use_checkpoint and torch.is_grad_enabled()
        for layer_idx in range(len(tm0.layers)):
            new_h: list[torch.Tensor] = []
            for k, tm in enumerate(self.text_models):
                layer = tm.layers[layer_idx]
                mask = (
                    linear_attn_mask
                    if tm.config.layer_types[layer_idx] == "linear_attention"
                    else causal_mask
                )
                if use_ckpt:
                    out = checkpoint(
                        layer,
                        per_track_h[k],
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        use_cache=False,
                        use_reentrant=False,
                    )
                else:
                    out = layer(
                        per_track_h[k],
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )
                new_h.append(out)
            per_track_h = new_h
            if layer_idx in sync_set:
                h = self.sync_module(per_track_h, block_start)
                block_start = h
                per_track_h = [h for _ in self.text_models]
                if sync_hiddens is not None:
                    sync_hiddens[layer_idx] = h

        h = tm0.norm(h)
        # The caller may want the post-norm hidden state instead of logits so
        # they can chunk the lm_head application themselves (eliminates the
        # held (B, T, V) bf16 logits tensor — ~2 GB at seq=4096). All ranks
        # return `h` regardless of lm_head ownership; this keeps SyncBoundary
        # collective ordering matched (peers must still run the full forward).
        if return_hidden_pre_lm_head:
            return h, sync_hiddens
        # NOTE: in vocab-parallel mode `self.lm_head(h)` is only this rank's
        # vocab SLICE (B, T, v_hi-v_lo), not full logits — callers needing the
        # logit objective must request the hidden state and use the
        # vocab-parallel KL/CE in `train/distill.py`. The legacy path returns
        # full (B, T, V) logits on the owner and None on peers.
        logits = self.lm_head(h) if self.lm_head is not None else None
        return logits, sync_hiddens

    def load_track_state_dicts(
        self,
        track_states: dict[int, dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Load per-track shards into the K local text_models (and lm_head if owned).

        ``track_states`` keys must be exactly ``self.local_track_ids``. Each
        value is the per-track state_dict emitted by
        ``slicer.convert.slice_model_to_tracks`` (top-level keys like
        ``embed_tokens.weight``, ``norm.weight``, ``lm_head.weight``, and
        ``layers.{i}.<adapter-prefix>.*``). We rewrite into the namespaces
        ``text_models.{k}.embed_tokens.weight`` / ``.norm.weight`` /
        ``.layers.{i}.*`` and the rank-shared ``lm_head.weight``.
        """
        provided = set(track_states.keys())
        expected = set(self.local_track_ids)
        if provided != expected:
            raise ValueError(
                f"load_track_state_dicts expected track ids {sorted(expected)}, "
                f"got {sorted(provided)}"
            )

        remapped: dict[str, torch.Tensor] = {}
        for k, tid in enumerate(self.local_track_ids):
            track_state = track_states[tid]
            prefix = f"text_models.{k}."
            for key, val in track_state.items():
                if key in ("embed_tokens.weight", "lm_head.weight") and self.vocab_parallel:
                    # Vocab-parallel: embed + lm_head are sharded onto the
                    # wrapper's vp_embed / lm_head via load_vocab_parallel_weights,
                    # not into the per-track text models. Skip the full tensors
                    # here (they only appear in the track-0 shard).
                    continue
                if key == "embed_tokens.weight":
                    remapped[prefix + "embed_tokens.weight"] = val
                elif key == "norm.weight":
                    remapped[prefix + "norm.weight"] = val
                elif key == "lm_head.weight":
                    # Owner only; routed to the rank-shared head.
                    remapped["lm_head.weight"] = val
                elif key.startswith("layers."):
                    remapped[prefix + key] = val
                else:
                    remapped[prefix + key] = val

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if strict:
            # rotary buffer keys (e.g. text_models.{k}.rotary_emb.inv_freq) may
            # legitimately be missing — they're computed at init.
            missing_critical = [k for k in missing if "rotary_emb" not in k]
            if self.vocab_parallel:
                # vp_embed.weight / lm_head.weight are loaded separately via
                # load_vocab_parallel_weights, so they're expected-missing here.
                missing_critical = [
                    k for k in missing_critical
                    if not (k.startswith("vp_embed.") or k.startswith("lm_head."))
                ]
            if missing_critical or unexpected:
                raise RuntimeError(
                    f"load_track_state_dicts mismatch:\n  missing={missing_critical}\n  unexpected={unexpected}"
                )

    def load_vocab_parallel_weights(
        self,
        full_embed_weight: torch.Tensor,
        full_lm_head_weight: torch.Tensor,
    ) -> None:
        """Slice the full `[V, H]` embed + lm_head into this rank's vocab shard.

        Every rank calls this with the *full* tensors (read from the track-0
        shard, where the slicer stores them as `OwnerOnly`); each keeps rows
        `[v_lo, v_hi)`. Keeps the on-disk checkpoint format unchanged.
        """
        if not self.vocab_parallel:
            raise RuntimeError("load_vocab_parallel_weights called on a non-vocab-parallel model")
        with torch.no_grad():
            self.vp_embed.weight.copy_(full_embed_weight[self.v_lo:self.v_hi].to(self.vp_embed.weight.dtype))
            self.lm_head.weight.copy_(full_lm_head_weight[self.v_lo:self.v_hi].to(self.lm_head.weight.dtype))

    def gather_vocab_parallel_weights(self) -> "tuple[torch.Tensor, torch.Tensor] | None":
        """All-gather the embed + lm_head vocab shards → full `[V, H]` on rank 0.

        Returns `(embed_weight, lm_head_weight)` on rank 0 (both full vocab) and
        `None` on peers, so `save_checkpoint` can write the unchanged on-disk
        format (full tensors in the track-0 shard). Shards are zero-padded to a
        uniform width for the collective, then trimmed back to `V`.
        """
        if not self.vocab_parallel:
            raise RuntimeError("gather_vocab_parallel_weights called on a non-vocab-parallel model")
        import torch.distributed as dist
        V = self.text_config.vocab_size
        shard = math.ceil(V / self.vp_world_size)
        H = self.text_config.hidden_size

        def _gather(local: torch.Tensor) -> "torch.Tensor | None":
            padded = local.new_zeros((shard, H))
            padded[: local.shape[0]] = local
            if dist.is_initialized() and self.vp_group is not None:
                out = [torch.empty_like(padded) for _ in range(self.vp_world_size)]
                dist.all_gather(out, padded, group=self.vp_group)
            else:
                out = [padded]
            if self.vp_rank != 0:
                return None
            return torch.cat(out, dim=0)[:V].contiguous()

        embed_full = _gather(self.vp_embed.weight.detach())
        lm_head_full = _gather(self.lm_head.weight.detach())
        if self.vp_rank != 0:
            return None
        return embed_full, lm_head_full
