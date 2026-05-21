"""PTWrappedModel: the user-facing per-rank model.

One PTWrappedModel instance per *rank*. Holds the rank's track of the text
decoder plus a (replicated) `lm_head`. Exposes a forward that returns
`(logits, sync_hiddens)` for use by the distillation loop.

This class is intentionally thin: the heavy lifting lives in
`PTTrackTextModel`. PTWrappedModel adds the LM head and provides a
convenience `load_track_state_dict` that consumes the slicer output.
"""
from __future__ import annotations

import torch
from torch import nn

from pt_converter.model.sync import SyncBoundary
from pt_converter.model.tracks.qwen3_5 import (
    PTTrackTextModel,
    PTTrackTextModelConfig,
    build_per_track_text_config,
)


class PTWrappedModel(nn.Module):
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
        per_track_cfg = build_per_track_text_config(text_config, n_tracks)
        self.text_config = text_config
        self.per_track_text_config = per_track_cfg
        self.n_tracks = n_tracks
        self.track_id = track_id

        sync_module = SyncBoundary(track_group=track_group, n_tracks=n_tracks)
        self.text_model = PTTrackTextModel(
            per_track_cfg,
            PTTrackTextModelConfig(
                n_tracks=n_tracks,
                sync_after_layers=tuple(sync_after_layers),
                track_id=track_id,
            ),
            sync_module=sync_module,
        )
        # LM head kept replicated across tracks for the first iteration.
        # Output of the final norm is summed-and-broadcast (post final sync), so
        # logits are computed identically on every rank.
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

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
        logits = self.lm_head(hidden_states)
        return logits, sync_hiddens

    def load_track_state_dict(self, track_state: dict[str, torch.Tensor], strict: bool = True):
        """Load a per-track state_dict produced by `slicer.convert.slice_model_to_tracks`.

        Slicer keys look like:
            embed_tokens.weight, norm.weight, lm_head.weight,
            layers.{i}.input_layernorm.weight,
            layers.{i}.self_attn.q_proj.weight, ...
            layers.{i}.linear_attn.in_proj_qkv.weight, ...
            layers.{i}.mlp.gate_proj.weight, ...

        We rewrite them to this module's namespace:
            text_model.embed_tokens.weight, text_model.norm.weight, lm_head.weight,
            text_model.layers.{i}.<...>
        """
        remapped: dict[str, torch.Tensor] = {}
        for key, val in track_state.items():
            if key == "embed_tokens.weight":
                remapped["text_model.embed_tokens.weight"] = val
            elif key == "norm.weight":
                remapped["text_model.norm.weight"] = val
            elif key == "lm_head.weight":
                remapped["lm_head.weight"] = val
            elif key.startswith("layers."):
                remapped[f"text_model.{key}"] = val
            else:
                remapped[key] = val

        # The rotary_emb has no learnable weights but registers buffers; we tolerate
        # missing keys for it. We DO want to be strict about everything else.
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
