"""Per-track Qwen3.5-MoE text model — the shared body lives in `tracks.base`.

Same structure as the dense family but with the `Qwen3_5Moe*` module classes. The
per-track config divides the expert width (`moe_intermediate_size`) and the
shared-expert width, keeping `num_experts` / `num_experts_per_tok` intact so the
(replicated) router picks the same top-k on every track; each track's
`Qwen3_5MoeExperts` then builds `[E, 2*(I/N), H]` / `[E, H, I/N]` slabs and its MoE
forward yields a partial sum completed by the cross-track all-reduce.
"""
from __future__ import annotations

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)

from parallm.model.tracks.base import PTTrackTextModelBase, apply_common_per_track_sizing


class PTTrackTextModel(PTTrackTextModelBase):
    DECODER_LAYER_CLS = Qwen3_5MoeDecoderLayer
    RMSNORM_CLS = Qwen3_5MoeRMSNorm
    ROTARY_CLS = Qwen3_5MoeTextRotaryEmbedding


def build_per_track_text_config(text_config, n_tracks: int):
    cfg = apply_common_per_track_sizing(text_config, n_tracks)
    for field in ("moe_intermediate_size", "shared_expert_intermediate_size"):
        val = getattr(cfg, field)
        if val % n_tracks != 0:
            raise ValueError(f"{field} {val} not divisible by {n_tracks}")
        setattr(cfg, field, val // n_tracks)
    return cfg
