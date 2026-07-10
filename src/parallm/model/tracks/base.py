"""Model-agnostic per-track text model.

Every supported family's per-track model is the same object — a stack of standard
HF decoder layers built from a *modified* config (head counts and MLP width divided
by `n_tracks`) — and differs only in **which HF module classes** it instantiates.
`PTTrackTextModelBase` holds the whole body; each family supplies three class
attributes (decoder layer / RMSNorm / rotary) and a tiny
`build_per_track_text_config` that divides its own MLP-width fields on top of the
shared attention sizing in `apply_common_per_track_sizing`.

The hidden_size, vocab_size, head_dim, and norms stay full size — the residual
stream is full hidden_size on every track. The per-track model exposes no forward
of its own: `PTWrappedModel.forward` drives the layers/norm/rotary directly and
owns all cross-track sync.
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
