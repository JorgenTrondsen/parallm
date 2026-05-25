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
        # Per-layer activation checkpointing. Effective on every rank when the
        # flag is set: peer ranks (K>1) never backward through the full forward
        # (no recompute cost at all), and the lm_head rank's KL+CE backward is
        # now a single `hidden.backward(grad_h_accum)` call (one recompute pass
        # through the layers, not N_chunks). Needed to fit n_tracks=16 at
        # seq_len=4096 on 40 GB GPUs.
        self._use_checkpoint = activation_checkpoint

        self.sync_module = SyncBoundary(track_group=track_group, n_tracks=n_tracks)
        self.text_models = nn.ModuleList(
            [
                adapter.track_text_model_cls(
                    per_track_cfg,
                    PTTrackTextModelConfig(
                        n_tracks=n_tracks,
                        sync_after_layers=tuple(sync_after_layers),
                        track_id=tid,
                    ),
                    sync_module=self.sync_module,
                )
                for tid in self.local_track_ids
            ]
        )
        # lm_head lives on the owner track only. The final SyncBoundary
        # broadcasts the post-block hidden state to all tracks, so whichever
        # rank hosts the owner already has the correct synced state to
        # project to logits. Peer ranks return logits=None from forward.
        if self.LM_HEAD_OWNER_TRACK in self.local_track_ids:
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        else:
            self.lm_head = None

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

        # 1. Embed (owner only) + cross-track broadcast via the zero-padded sync.
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
        h = self.sync_module(embeds_per_track, torch.zeros_like(embeds_per_track[0]))

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
            if missing_critical or unexpected:
                raise RuntimeError(
                    f"load_track_state_dicts mismatch:\n  missing={missing_critical}\n  unexpected={unexpected}"
                )
