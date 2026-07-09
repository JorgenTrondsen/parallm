"""Model-adapter registry.

A `ModelAdapter` packages everything model-specific that the PT engine needs:
slicer specs, per-track text-model class, per-track config builder, and the
state-dict layer prefixes used by the loader. The engine (`slicer/convert.py`
and `model/pt_model.py`) holds no model knowledge and looks up the right
adapter via `get_adapter_for_config(config)`.

To add a new model, write a new module under `src/parallm/adapters/`
that builds a `ModelAdapter` and calls `register_model_adapter(adapter)`.
Add a one-line import to this file so the registration runs at package
import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from parallm.slicer.base import LayerSpec


@dataclass(frozen=True)
class ModelAdapter:
    """All callbacks the PT engine needs to convert and run one model family.

    All callbacks operate on a *text config* (the sub-config containing
    `num_hidden_layers`, `num_attention_heads`, etc.). The registry resolves
    the right adapter from a top-level config by checking `config.model_type`
    first and falling back to `config.text_config.model_type`.
    """

    model_type: str
    # ----- Slicer side -----
    layer_specs: Callable[[Any, str], LayerSpec]
    top_level_specs: Callable[[Any], LayerSpec]
    get_layer_types: Callable[[Any], list[str]]
    valid_layer_types: tuple[str, ...]
    # ----- Per-track model side -----
    track_text_model_cls: type
    build_per_track_text_config: Callable[[Any, int], Any]
    # ----- State-dict remap -----
    # Sub-module prefixes (under `layers.{i}.*`) that the slicer emits and
    # that need to be re-routed under `text_models.{k}.layers.{i}.*` at load time.
    state_dict_layer_prefixes: tuple[str, ...]


_REGISTRY: dict[str, ModelAdapter] = {}


def register_model_adapter(adapter: ModelAdapter) -> ModelAdapter:
    """Idempotent registration: re-registering the same model_type overwrites
    the previous adapter (useful for tests and for swapping experimental
    adapters in a notebook)."""
    _REGISTRY[adapter.model_type] = adapter
    return adapter


def get_adapter_for_config(config) -> ModelAdapter:
    """Resolve an adapter by config.model_type, falling back to text_config.model_type."""
    model_type = getattr(config, "model_type", None)
    if model_type in _REGISTRY:
        return _REGISTRY[model_type]
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        text_model_type = getattr(text_cfg, "model_type", None)
        if text_model_type in _REGISTRY:
            return _REGISTRY[text_model_type]
    registered = sorted(_REGISTRY.keys())
    raise KeyError(
        f"No ModelAdapter registered for model_type={model_type!r} "
        f"(text_config.model_type={getattr(text_cfg, 'model_type', None)!r}). "
        f"Registered adapters: {registered}"
    )


def list_registered() -> list[str]:
    """Inspect the registry; useful for CLI help and error messages."""
    return sorted(_REGISTRY.keys())


# Import-time adapter registrations. Add one line per shipped adapter.
from parallm.adapters import qwen3_5 as _qwen3_5  # noqa: E402,F401
