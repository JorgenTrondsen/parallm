"""Per-track Qwen3.5 text model.

We construct a standard `Qwen3_5DecoderLayer` per layer, but using a *modified*
text config whose head counts and intermediate_size have been divided by
`n_tracks`. The HF modules pick up the smaller shapes automatically, so a
track's `Qwen3_5Attention` ends up with (num_attention_heads // n_tracks) q-heads,
its `Qwen3_5GatedDeltaNet` with proportionally fewer linear k/v heads, and its
`Qwen3_5MLP` with intermediate_size // n_tracks.

The hidden_size, vocab_size, head_dim, and norms remain full size — the
residual stream is full hidden_size on every track.

The forward mirrors `Qwen3_5TextModel.forward` and inserts a SyncBoundary
after each layer index in `sync_after_layers`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    create_causal_mask,
)

from pt_converter.model.sync import SyncBoundary


def build_per_track_text_config(text_config, n_tracks: int):
    """Deep-copy `text_config` and divide every sliceable dim by `n_tracks`."""
    cfg = copy.deepcopy(text_config)
    if cfg.num_attention_heads % n_tracks != 0:
        raise ValueError(f"num_attention_heads {cfg.num_attention_heads} not divisible by {n_tracks}")
    if cfg.num_key_value_heads % n_tracks != 0:
        raise ValueError(f"num_key_value_heads {cfg.num_key_value_heads} not divisible by {n_tracks}")
    if cfg.linear_num_key_heads % n_tracks != 0:
        raise ValueError(f"linear_num_key_heads {cfg.linear_num_key_heads} not divisible by {n_tracks}")
    if cfg.linear_num_value_heads % n_tracks != 0:
        raise ValueError(f"linear_num_value_heads {cfg.linear_num_value_heads} not divisible by {n_tracks}")
    if cfg.intermediate_size % n_tracks != 0:
        raise ValueError(f"intermediate_size {cfg.intermediate_size} not divisible by {n_tracks}")
    cfg.num_attention_heads //= n_tracks
    cfg.num_key_value_heads //= n_tracks
    cfg.linear_num_key_heads //= n_tracks
    cfg.linear_num_value_heads //= n_tracks
    cfg.intermediate_size //= n_tracks
    return cfg


@dataclass
class PTTrackTextModelConfig:
    n_tracks: int
    sync_after_layers: tuple[int, ...]
    track_id: int = 0


class PTTrackTextModel(nn.Module):
    """One track's text decoder. Holds embed_tokens, layers, norm, and rotary.

    The vocab/embedding/norm are full-size and replicated across tracks. Only
    the decoder layers are per-track sliced.
    """

    def __init__(
        self,
        per_track_text_config,
        pt_cfg: PTTrackTextModelConfig,
        sync_module: SyncBoundary,
    ):
        super().__init__()
        self.config = per_track_text_config
        self.pt_cfg = pt_cfg
        self.sync_after = set(pt_cfg.sync_after_layers)

        self.embed_tokens = nn.Embedding(
            per_track_text_config.vocab_size,
            per_track_text_config.hidden_size,
            getattr(per_track_text_config, "pad_token_id", None),
        )
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
        return_sync_hiddens: bool = False,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_ids, text_position_ids = self._resolve_position_ids(inputs_embeds, position_ids)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        # For training (no cache, full attention_mask), the linear-attn mask path
        # collapses to None when attention_mask is all-ones. Mirror that here.
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None
        else:
            linear_attn_mask = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        sync_hiddens: list[torch.Tensor] = []
        h_pre_block = hidden_states  # snapshot for delta-sync
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
            if layer_idx in self.sync_after:
                hidden_states = self.sync_module(hidden_states, h_pre_block)
                if return_sync_hiddens:
                    sync_hiddens.append(hidden_states)
                h_pre_block = hidden_states

        hidden_states = self.norm(hidden_states)
        return hidden_states, sync_hiddens
