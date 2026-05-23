"""Per-track Qwen3.5 text model.

We construct a standard `Qwen3_5DecoderLayer` per layer, but using a *modified*
text config whose head counts and intermediate_size have been divided by
`n_tracks`. The HF modules pick up the smaller shapes automatically, so a
track's `Qwen3_5Attention` ends up with (num_attention_heads // n_tracks) q-heads,
its `Qwen3_5GatedDeltaNet` with proportionally fewer linear k/v heads, and its
`Qwen3_5MLP` with intermediate_size // n_tracks.

The hidden_size, vocab_size, head_dim, and norms remain full size — the
residual stream is full hidden_size on every track.

Cross-track sync is driven from `PTWrappedModel.forward`, which runs all K
local tracks lockstep and calls `SyncBoundary` once per sync point. This
module's own `forward` is a plain layer-stack forward (no sync), kept only
for direct introspection / debugging of a single track's layer stack — the
training loop does not invoke it.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    create_causal_mask,
)

from pt_converter.model.sync import SyncBoundary
from pt_converter.model.pt_model import PTTrackTextModelConfig


def build_per_track_text_config(text_config, n_tracks: int):
    """Deep-copy `text_config` and apply the KV-replicated per-track sizing.

    Under the KV-replicated rule:
      - num_attention_heads //= n_tracks (each track holds 1 or more q-heads)
      - num_key_value_heads = 1 (each track holds exactly one replicated kv-head
        from its kv-group; the dense num_kv must satisfy `n_tracks % num_kv == 0`)
      - linear_num_*_heads //= n_tracks (no replication needed; SSM heads divide)
      - intermediate_size //= n_tracks
    """
    cfg = copy.deepcopy(text_config)
    if cfg.num_attention_heads % n_tracks != 0:
        raise ValueError(f"num_attention_heads {cfg.num_attention_heads} not divisible by {n_tracks}")
    if n_tracks % cfg.num_key_value_heads != 0:
        raise ValueError(
            f"n_tracks {n_tracks} must be a multiple of num_key_value_heads {cfg.num_key_value_heads} "
            f"(KV-replicated rule)"
        )
    if cfg.linear_num_key_heads % n_tracks != 0:
        raise ValueError(f"linear_num_key_heads {cfg.linear_num_key_heads} not divisible by {n_tracks}")
    if cfg.linear_num_value_heads % n_tracks != 0:
        raise ValueError(f"linear_num_value_heads {cfg.linear_num_value_heads} not divisible by {n_tracks}")
    if cfg.intermediate_size % n_tracks != 0:
        raise ValueError(f"intermediate_size {cfg.intermediate_size} not divisible by {n_tracks}")
    cfg.num_attention_heads //= n_tracks
    cfg.num_key_value_heads = 1  # one replicated kv-head per track
    cfg.linear_num_key_heads //= n_tracks
    cfg.linear_num_value_heads //= n_tracks
    cfg.intermediate_size //= n_tracks
    return cfg


class PTTrackTextModel(nn.Module):
    """One track's text decoder. Holds layers, norm, rotary, and (on the owner
    track only) `embed_tokens`.

    `embed_tokens` lives on track 0 only. The cross-track broadcast of the
    embedding and the per-block syncs are driven externally by
    `PTWrappedModel.forward`, not from this module's `forward`.

    The final RMSNorm (`norm`) is held by every track but only ever applied
    to the synced hidden state (via `PTWrappedModel.forward`).
    """

    EMBED_OWNER_TRACK = 0

    def __init__(
        self,
        per_track_text_config,
        pt_cfg: PTTrackTextModelConfig,
        sync_module: SyncBoundary,
    ):
        super().__init__()
        self.config = per_track_text_config
        self.pt_cfg = pt_cfg

        if pt_cfg.track_id == self.EMBED_OWNER_TRACK:
            self.embed_tokens = nn.Embedding(
                per_track_text_config.vocab_size,
                per_track_text_config.hidden_size,
                getattr(per_track_text_config, "pad_token_id", None),
            )
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(per_track_text_config, layer_idx)
                for layer_idx in range(per_track_text_config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(
            per_track_text_config.hidden_size, eps=per_track_text_config.rms_norm_eps
        )
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=per_track_text_config)
        # Held only so adapter-contract consumers can inspect it; this module
        # does not invoke sync_module from its own forward.
        self.sync_module = sync_module

    def _resolve_position_ids(self, inputs_embeds, position_ids):
        # Mirrors Qwen3_5TextModel.forward. The 4-dim form is for multimodal mrope;
        # for pure text fine-tuning, position_ids will typically be None.
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        return position_ids, text_position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ):
        """Plain per-track layer-stack forward, no cross-track sync.

        Not used by `PTWrappedModel.forward` (the training path) — kept for
        direct inspection of a single track's stack. Owner-only embed lookup;
        callers handling peer tracks must pass `inputs_embeds` directly.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "This track has no embed_tokens (non-owner); pass inputs_embeds explicitly."
                )
            inputs_embeds = self.embed_tokens(input_ids)

        position_ids, text_position_ids = self._resolve_position_ids(inputs_embeds, position_ids)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None
        else:
            linear_attn_mask = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_mask = (
                linear_attn_mask
                if self.config.layer_types[layer_idx] == "linear_attention"
                else causal_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )

        return self.norm(hidden_states)
