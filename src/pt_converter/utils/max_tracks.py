"""Detect the maximum natural track count for a given model config.

The track count is bounded by the gcd of every dimension we need to slice.
For a dense transformer this is the gcd of head counts (and any per-head MLP
slicing factor). For Qwen3.5 the binding dimension is `num_key_value_heads`.

This module is model-agnostic: each model_type registers a function that
returns the list of dimensions to take the gcd over.
"""
from __future__ import annotations

from math import gcd
from functools import reduce
from typing import Callable

_DIM_PROVIDERS: dict[str, Callable[[object], list[int]]] = {}


def register_dims(model_type: str):
    """Decorator: register the list-of-sliceable-dims provider for a model_type.

    The function receives the (sub-)config that actually carries the dims and
    returns the integer dimensions that must each be divisible by N.
    """

    def _wrap(fn: Callable[[object], list[int]]):
        _DIM_PROVIDERS[model_type] = fn
        return fn

    return _wrap


def _gcd_all(values: list[int]) -> int:
    if not values:
        raise ValueError("No dimensions provided to gcd_all")
    return reduce(gcd, values)


def max_tracks_for_config(config) -> int:
    """Return the maximum N such that every sliceable dim is divisible by N.

    `config` may be a top-level HF config (with a `.text_config`) or the text
    config itself. We try the top-level model_type first, then fall back to
    `config.text_config.model_type` if present.
    """
    model_type = getattr(config, "model_type", None)
    if model_type in _DIM_PROVIDERS:
        dims = _DIM_PROVIDERS[model_type](config)
        return _gcd_all(dims)

    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        text_model_type = getattr(text_cfg, "model_type", None)
        if text_model_type in _DIM_PROVIDERS:
            dims = _DIM_PROVIDERS[text_model_type](text_cfg)
            return _gcd_all(dims)

    raise NotImplementedError(
        f"No max-tracks provider registered for model_type={model_type!r}. "
        f"Register one via @register_dims in pt_converter.utils.max_tracks."
    )


@register_dims("qwen3_5_text")
def _qwen3_5_dims(cfg) -> list[int]:
    return [
        int(cfg.num_attention_heads),
        int(cfg.num_key_value_heads),
        int(cfg.linear_num_key_heads),
        int(cfg.linear_num_value_heads),
    ]
