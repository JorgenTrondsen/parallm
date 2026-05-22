"""Qwen3.5 adapter: glues the existing qwen3_5 slicer specs and per-track text model
into the model-agnostic `ModelAdapter` shape and registers it.

This is the single source of truth for "is Qwen3.5 PT-supported." Nothing outside
this file should import from `pt_converter.slicer.qwen3_5` or
`pt_converter.model.tracks.qwen3_5`.
"""
from __future__ import annotations

from typing import Any

from pt_converter.adapters import ModelAdapter, register_model_adapter
from pt_converter.model.tracks.qwen3_5 import (
    PTTrackTextModel,
    build_per_track_text_config,
)
from pt_converter.slicer.qwen3_5 import decoder_layer_specs, top_level_specs


def _qwen3_5_get_layer_types(text_cfg: Any) -> list[str]:
    # Qwen3.5 ships explicit hybrid layer_types in the config.
    return list(text_cfg.layer_types)


QWEN3_5_ADAPTER = ModelAdapter(
    model_type="qwen3_5_text",
    layer_specs=decoder_layer_specs,
    top_level_specs=top_level_specs,
    get_layer_types=_qwen3_5_get_layer_types,
    valid_layer_types=("full_attention", "linear_attention"),
    track_text_model_cls=PTTrackTextModel,
    build_per_track_text_config=build_per_track_text_config,
    # Sub-module prefixes that appear under `layers.{i}.<prefix>.*` in slicer output.
    state_dict_layer_prefixes=(
        "self_attn",
        "linear_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
    ),
)

register_model_adapter(QWEN3_5_ADAPTER)
