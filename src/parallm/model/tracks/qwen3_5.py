"""Per-track Qwen3.5 (dense) text model — the shared body lives in `tracks.base`.

This family divides `intermediate_size` on top of the common attention sizing.
"""
from __future__ import annotations

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    create_causal_mask,
)

from parallm.model.tracks.base import PTTrackTextModelBase, apply_common_per_track_sizing


class PTTrackTextModel(PTTrackTextModelBase):
    DECODER_LAYER_CLS = Qwen3_5DecoderLayer
    RMSNORM_CLS = Qwen3_5RMSNorm
    ROTARY_CLS = Qwen3_5TextRotaryEmbedding
    CREATE_CAUSAL_MASK = staticmethod(create_causal_mask)


def build_per_track_text_config(text_config, n_tracks: int):
    cfg = apply_common_per_track_sizing(text_config, n_tracks)
    if cfg.intermediate_size % n_tracks != 0:
        raise ValueError(f"intermediate_size {cfg.intermediate_size} not divisible by {n_tracks}")
    cfg.intermediate_size //= n_tracks
    return cfg
