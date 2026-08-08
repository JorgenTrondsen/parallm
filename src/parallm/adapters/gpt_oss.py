"""gpt-oss adapter: registers the gpt-oss slicer specs + per-track model.

Covers openai/gpt-oss-20b and -120b — same architecture, differing only in layer
count and expert count, so one adapter serves both. This is the single source of
truth for "is gpt-oss PT-supported."

Note gpt-oss ships a FLAT config (`model_type: gpt_oss`, no `text_config`) and its
dense model class is `GptOssModel`, not `...TextModel`.
"""
from __future__ import annotations

from typing import Any

from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel

from parallm.adapters import ModelAdapter, register_model_adapter
from parallm.model.tracks.gpt_oss import (
    PTTrackTextModel,
    build_per_track_text_config,
)
from parallm.slicer.gpt_oss import (
    VALID_LAYER_TYPES,
    build_masks,
    decoder_layer_specs,
    top_level_specs,
)
from parallm.utils.max_tracks import ConstraintSet


def _gpt_oss_constraints(cfg: Any) -> ConstraintSet:
    # `intermediate_size` is the per-expert width and MUST divide N: the expert slab's
    # gate/up lanes are interleaved, so an uneven (odd-width) chunk would split a
    # gate/up pair. num_local_experts is replicated (the router runs in full on every
    # track), so it imposes no constraint. For 20b/120b (2880 wide, 64 q heads, 8 kv)
    # this yields N in {64, 32, 16, 8}.
    return ConstraintSet(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        divides=(int(cfg.intermediate_size),),
    )


GPT_OSS_ADAPTER = ModelAdapter(
    model_type="gpt_oss",
    layer_specs=decoder_layer_specs,
    top_level_specs=top_level_specs,
    valid_layer_types=VALID_LAYER_TYPES,
    track_text_model_cls=PTTrackTextModel,
    build_per_track_text_config=build_per_track_text_config,
    build_masks=build_masks,
    full_text_model_cls=GptOssModel,
    constraints=_gpt_oss_constraints,
    # Merging (concatenated slabs) IS supported and is the fast path — see the
    # `tracks/gpt_oss.py` docstring for why widening experts beats stacking tracks.
    supports_merged_tracks=True,
    # The batched fold in `parallm.model.batched` is not: `engine._batched_mlp` is a
    # dense SwiGLU, and under G > 1 each stream carries its own residual, so the
    # shared router picks a DIFFERENT top-k per stream and there is no single
    # `grouped_mm` to issue. `train_cli` therefore expresses F as the merge width
    # instead of as `exec_groups` for this family.
    supports_batched_exec=False,
)

register_model_adapter(GPT_OSS_ADAPTER)
