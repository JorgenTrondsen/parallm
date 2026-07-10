"""Qwen3.5-MoE adapter: registers the MoE slicer specs + per-track MoE model.

The MoE model reuses the dense Qwen3.5 attention slicing verbatim (imported inside
`slicer.qwen3_5_moe`) and only swaps the dense MLP for the sparse MoE block. This
is the single source of truth for "is Qwen3.5-MoE PT-supported."
"""
from __future__ import annotations

from typing import Any

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from parallm.adapters import ModelAdapter, register_model_adapter
from parallm.model.tracks.qwen3_5_moe import (
    PTTrackTextModel,
    build_per_track_text_config,
)
from parallm.slicer.qwen3_5_moe import moe_decoder_layer_specs, top_level_specs
from parallm.utils.max_tracks import ConstraintSet


def _qwen3_5_moe_get_layer_types(text_cfg: Any) -> list[str]:
    return list(text_cfg.layer_types)


def _qwen3_5_moe_constraints(cfg: Any) -> ConstraintSet:
    # Same attention constraints as dense; MLP is MoE so N must divide the expert
    # width and the shared-expert width. num_experts is replicated (the router runs
    # in full on every track) so it imposes no divisibility constraint.
    return ConstraintSet(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        divides=(
            int(cfg.linear_num_key_heads),
            int(cfg.linear_num_value_heads),
            int(cfg.moe_intermediate_size),
            int(cfg.shared_expert_intermediate_size),
        ),
    )


QWEN3_5_MOE_ADAPTER = ModelAdapter(
    model_type="qwen3_5_moe_text",
    layer_specs=moe_decoder_layer_specs,
    top_level_specs=top_level_specs,
    get_layer_types=_qwen3_5_moe_get_layer_types,
    valid_layer_types=("full_attention", "linear_attention"),
    track_text_model_cls=PTTrackTextModel,
    build_per_track_text_config=build_per_track_text_config,
    state_dict_layer_prefixes=(
        "self_attn",
        "linear_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
    ),
    full_text_model_cls=Qwen3_5MoeTextModel,
    constraints=_qwen3_5_moe_constraints,
)

register_model_adapter(QWEN3_5_MOE_ADAPTER)
