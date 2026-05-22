"""PTWrappedModel: the user-facing per-rank model.

One PTWrappedModel instance per *rank*. Holds the rank's track of the text
decoder plus a (replicated) `lm_head`. Exposes a forward that returns
`(logits, sync_hiddens)` for use by the distillation loop.

The model-specific decoder layer assembly is provided by a `ModelAdapter`
(see `pt_converter.adapters`). PTWrappedModel itself holds no model-family
knowledge: it looks up the adapter by `text_config.model_type`, calls its
per-track config builder, and instantiates its `track_text_model_cls`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

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
        track_id: int,
        sync_after_layers: list[int],
        track_group: "torch.distributed.ProcessGroup | None" = None,
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
        self.track_id = track_id
        self._adapter = adapter  # held for `load_track_state_dict` remap

        sync_module = SyncBoundary(track_group=track_group, n_tracks=n_tracks)
        self.text_model = adapter.track_text_model_cls(
            per_track_cfg,
            PTTrackTextModelConfig(
                n_tracks=n_tracks,
                sync_after_layers=tuple(sync_after_layers),
                track_id=track_id,
            ),
            sync_module=sync_module,
        )
        # lm_head lives on the owner track only. The final SyncBoundary
        # broadcasts the post-block hidden state to all tracks, so the owner
        # already has the correct synced state to project to logits. Peer
        # tracks return logits=None from forward.
        if track_id == self.LM_HEAD_OWNER_TRACK:
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_sync_hiddens: bool = False,
    ):
        hidden_states, sync_hiddens = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_sync_hiddens=return_sync_hiddens,
        )
        logits = self.lm_head(hidden_states) if self.lm_head is not None else None
        return logits, sync_hiddens

    def load_track_state_dict(self, track_state: dict[str, torch.Tensor], strict: bool = True):
        """Load a per-track state_dict produced by `slicer.convert.slice_model_to_tracks`.

        Slicer keys look like (model-agnostic top level + model-specific layer
        sub-prefixes declared by the adapter):
            embed_tokens.weight, norm.weight, lm_head.weight,
            layers.{i}.<adapter.state_dict_layer_prefixes[k]>.*

        We rewrite them to this module's namespace:
            text_model.embed_tokens.weight, text_model.norm.weight, lm_head.weight,
            text_model.layers.{i}.<...>

        `embed_tokens.weight` and `lm_head.weight` are present only on the owner
        track's slicer output and only constructed on the owner track here, so
        on peer tracks both sides simply omit them — `load_state_dict` sees them
        neither in the input nor in `self.state_dict()` and raises nothing.
        """
        remapped: dict[str, torch.Tensor] = {}
        # Top-level keys are model-agnostic.
        top_level_remap = {
            "embed_tokens.weight": "text_model.embed_tokens.weight",
            "norm.weight": "text_model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for key, val in track_state.items():
            if key in top_level_remap:
                remapped[top_level_remap[key]] = val
            elif key.startswith("layers."):
                # Every `layers.{i}.*` key is hosted under `text_model.layers.{i}.*`.
                # The adapter's `state_dict_layer_prefixes` is informational here —
                # the slicer only emits keys under those known prefixes anyway —
                # but we keep the contract explicit so future custom remaps can
                # live in one place.
                remapped[f"text_model.{key}"] = val
            else:
                remapped[key] = val

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if strict:
            # rotary buffer keys (e.g. text_model.rotary_emb.inv_freq) may legitimately
            # be missing — they're computed at init. Filter those out before erroring.
            missing_critical = [k for k in missing if "rotary_emb" not in k]
            if missing_critical or unexpected:
                raise RuntimeError(
                    f"load_track_state_dict mismatch:\n  missing={missing_critical}\n  unexpected={unexpected}"
                )
        return missing, unexpected
