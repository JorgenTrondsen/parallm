"""Model-agnostic per-track text model.

Every supported family's per-track model is the same object — a stack of standard
HF decoder layers built from a *modified* config (head counts and MLP width divided
by `n_tracks`) — and differs only in **which HF module classes** it instantiates.
`PTTrackTextModelBase` holds the whole body; each family supplies four class
attributes (decoder layer / RMSNorm / rotary / causal-mask builder) and a tiny
`build_per_track_text_config` that divides its own MLP-width fields on top of the
shared attention sizing in `apply_common_per_track_sizing`.

The hidden_size, vocab_size, head_dim, and norms stay full size — the residual
stream is full hidden_size on every track. Cross-track sync is driven externally by
`PTWrappedModel.forward`; this module's own `forward` is a plain single-track
layer-stack forward (no sync) kept only for introspection.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from parallm.model.sync import SyncBoundary
from parallm.model.pt_model import PTTrackTextModelConfig


def apply_common_per_track_sizing(text_config, n_tracks: int):
    """Deep-copy `text_config` and apply the KV-replicated *attention* sizing shared
    by every family. Callers then divide their own MLP-width fields.

      - num_attention_heads //= n_tracks (each track holds >=1 q-head)
      - num_key_value_heads = 1 (one replicated kv-head per kv-group; dense num_kv
        must satisfy `n_tracks % num_kv == 0`)
      - linear_num_*_heads //= n_tracks (SSM heads divide, no replication)

    Also forces SDPA: the per-track full-attention layers are built standalone (no
    PreTrainedModel to resolve a backend), so eager would materialize a full (T, T)
    score matrix per head. Only affects full-attention layers; masks built via
    `create_causal_mask(config=...)` follow this setting.
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
    cfg.num_attention_heads //= n_tracks
    cfg.num_key_value_heads = 1
    cfg.linear_num_key_heads //= n_tracks
    cfg.linear_num_value_heads //= n_tracks
    cfg._attn_implementation = "sdpa"
    return cfg


class PTTrackTextModelBase(nn.Module):
    """One track's text decoder. Subclasses set the four HF-module class attributes.

    `embed_tokens` lives on track 0 only (the cross-track embedding broadcast and
    per-block syncs are driven by `PTWrappedModel.forward`). The final RMSNorm is
    held on every track but only ever applied to the synced hidden state.
    """

    EMBED_OWNER_TRACK = 0

    # Subclasses override with the HF module family:
    DECODER_LAYER_CLS: type
    RMSNORM_CLS: type
    ROTARY_CLS: type
    CREATE_CAUSAL_MASK: staticmethod  # (config, inputs_embeds, attention_mask, past_key_values, position_ids)

    def __init__(
        self,
        per_track_text_config,
        pt_cfg: PTTrackTextModelConfig,
        sync_module: SyncBoundary,
    ):
        super().__init__()
        self.config = per_track_text_config
        self.pt_cfg = pt_cfg

        if pt_cfg.track_id == self.EMBED_OWNER_TRACK and pt_cfg.host_embed_tokens:
            self.embed_tokens = nn.Embedding(
                per_track_text_config.vocab_size,
                per_track_text_config.hidden_size,
                getattr(per_track_text_config, "pad_token_id", None),
            )
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                self.DECODER_LAYER_CLS(per_track_text_config, layer_idx)
                for layer_idx in range(per_track_text_config.num_hidden_layers)
            ]
        )
        self.norm = self.RMSNORM_CLS(
            per_track_text_config.hidden_size, eps=per_track_text_config.rms_norm_eps
        )
        self.rotary_emb = self.ROTARY_CLS(config=per_track_text_config)
        # Held only so adapter-contract consumers can inspect it; not invoked here.
        self.sync_module = sync_module

    def _resolve_position_ids(self, inputs_embeds, position_ids):
        # Mirrors the HF TextModel.forward. The 4-dim form is for multimodal mrope;
        # for pure text fine-tuning, position_ids is typically None.
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
        """Plain per-track layer-stack forward, no cross-track sync (introspection only).

        Not used by `PTWrappedModel.forward` (the training path). Owner-only embed
        lookup; callers handling peer tracks must pass `inputs_embeds` directly.
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

        causal_mask = self.CREATE_CAUSAL_MASK(
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
