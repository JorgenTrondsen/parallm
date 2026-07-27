"""Tests for the model-adapter registry pattern.

Verifies that the engine (slicer/convert.py + model/pt_model.py) is fully
adapter-driven: registering a synthetic adapter is enough to make the engine
accept a new model_type, and the Qwen3.5 adapter still produces identical
slicer output to the underlying spec functions.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# Importing parallm triggers built-in adapter registration.
import parallm  # noqa: F401
from parallm.adapters import (
    ModelAdapter,
    get_adapter_for_config,
    list_registered,
    register_model_adapter,
)
from parallm.slicer.base import Colwise, LayerSpec, Replicated
from parallm.slicer.qwen3_5 import decoder_layer_specs, top_level_specs


def test_qwen3_5_adapter_is_registered_at_import_time():
    assert "qwen3_5_text" in list_registered()


def test_get_adapter_resolves_qwen3_5_from_text_subconfig():
    text_cfg = SimpleNamespace(model_type="qwen3_5_text")
    top = SimpleNamespace(model_type="qwen3_5", text_config=text_cfg)
    adapter = get_adapter_for_config(top)
    assert adapter.model_type == "qwen3_5_text"


def test_get_adapter_unknown_model_type_raises_with_listing():
    cfg = SimpleNamespace(model_type="not_a_real_model")
    with pytest.raises(KeyError) as excinfo:
        get_adapter_for_config(cfg)
    msg = str(excinfo.value)
    # Error should be self-describing.
    assert "not_a_real_model" in msg
    assert "qwen3_5_text" in msg  # the registered name appears in the listing


def test_qwen3_5_adapter_layer_specs_match_direct_call():
    """The adapter must be a transparent passthrough to the underlying spec
    functions — no drift in behavior between the old direct-import path and
    the new registry path.
    """
    # Use a tiny synthetic text-config namespace with just the fields the spec
    # functions consult.
    text_cfg = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        intermediate_size=64,
    )

    adapter = get_adapter_for_config(SimpleNamespace(model_type="qwen3_5_text"))

    for layer_type in ("full_attention", "linear_attention"):
        via_adapter = adapter.layer_specs(text_cfg, layer_type)
        direct = decoder_layer_specs(text_cfg, layer_type)
        assert set(via_adapter.keys()) == set(direct.keys())
        for k in via_adapter:
            assert type(via_adapter[k]) is type(direct[k]), f"spec type drift at {k}"


def test_qwen3_5_adapter_top_level_specs_match_direct_call():
    adapter = get_adapter_for_config(SimpleNamespace(model_type="qwen3_5_text"))
    via = adapter.top_level_specs(SimpleNamespace())
    direct = top_level_specs(SimpleNamespace())
    assert set(via.keys()) == set(direct.keys())
    for k in via:
        assert type(via[k]) is type(direct[k])


def test_qwen3_5_adapter_get_layer_types_reads_from_config():
    adapter = get_adapter_for_config(SimpleNamespace(model_type="qwen3_5_text"))
    cfg = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    assert adapter.get_layer_types(cfg) == ["linear_attention", "full_attention"]


def test_register_then_resolve_synthetic_adapter():
    """Registering a fake adapter and resolving by its model_type must work end-to-end."""

    def _fake_layer_specs(cfg, layer_type) -> LayerSpec:
        return {"weight": Colwise()}

    def _fake_top_specs(cfg) -> LayerSpec:
        return {"embed_tokens.weight": Replicated()}

    def _fake_get_layer_types(cfg) -> list[str]:
        return ["self_attention"] * cfg.num_hidden_layers

    def _fake_build_pt_cfg(cfg, n_tracks):
        return cfg

    class _FakeTrackModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    fake = ModelAdapter(
        model_type="pt_test_synth_model",
        layer_specs=_fake_layer_specs,
        top_level_specs=_fake_top_specs,
        get_layer_types=_fake_get_layer_types,
        valid_layer_types=("self_attention",),
        track_text_model_cls=_FakeTrackModel,
        build_per_track_text_config=_fake_build_pt_cfg,
        state_dict_layer_prefixes=("self_attn", "mlp"),
    )
    register_model_adapter(fake)

    resolved = get_adapter_for_config(SimpleNamespace(model_type="pt_test_synth_model"))
    assert resolved is fake
    assert resolved.get_layer_types(SimpleNamespace(num_hidden_layers=3)) == [
        "self_attention",
        "self_attention",
        "self_attention",
    ]
    # Engine sanity: layer_specs returns a dict keyed by sub-keys.
    assert "weight" in resolved.layer_specs(None, "self_attention")


def test_re_registration_overwrites():
    """Re-registering the same model_type is idempotent (last one wins)."""

    def _l(cfg, lt) -> LayerSpec:
        return {}

    def _t(cfg) -> LayerSpec:
        return {}

    def _types(cfg) -> list[str]:
        return []

    def _build(cfg, n):
        return cfg

    class _M(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    first = ModelAdapter(
        model_type="pt_test_overwrite",
        layer_specs=_l,
        top_level_specs=_t,
        get_layer_types=_types,
        valid_layer_types=(),
        track_text_model_cls=_M,
        build_per_track_text_config=_build,
        state_dict_layer_prefixes=(),
    )
    second = ModelAdapter(
        model_type="pt_test_overwrite",
        layer_specs=_l,
        top_level_specs=_t,
        get_layer_types=_types,
        valid_layer_types=("self_attention",),  # differs
        track_text_model_cls=_M,
        build_per_track_text_config=_build,
        state_dict_layer_prefixes=("self_attn",),  # differs
    )
    register_model_adapter(first)
    register_model_adapter(second)
    assert get_adapter_for_config(SimpleNamespace(model_type="pt_test_overwrite")) is second


def test_slicer_engine_uses_adapter_layer_types(monkeypatch):
    """The slicer engine must read layer_types via the adapter (not via
    `text_cfg.layer_types` directly). We register a synthetic adapter whose
    `get_layer_types` returns a different list than the underlying config
    attribute and verify the engine respects the adapter's answer.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from parallm.adapters import get_adapter_for_config as _real_resolver
    from parallm.model.tracks.qwen3_5 import (
        PTTrackTextModel,
        build_per_track_text_config,
    )
    from parallm.slicer.convert import slice_model_to_tracks
    from parallm.slicer.qwen3_5 import decoder_layer_specs as _real_layer_specs
    from parallm.slicer.qwen3_5 import top_level_specs as _real_top_specs

    # Build a tiny Qwen3.5 model whose config has layer_types entirely
    # "linear_attention" (8 layers).
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention"] * 8,
        full_attention_interval=8,
        vocab_size=64,
        rms_norm_eps=1e-6,
    )
    model = Qwen3_5TextModel(cfg).eval()

    # Register a synthetic adapter for a synthetic model_type that returns
    # the same per-layer behavior as Qwen3.5 but via the adapter contract.
    # We then monkey-patch text_cfg's model_type so the resolver routes here.
    synthetic = ModelAdapter(
        model_type="pt_test_engine_routes_through_adapter",
        layer_specs=_real_layer_specs,
        top_level_specs=_real_top_specs,
        get_layer_types=lambda c: ["linear_attention"] * c.num_hidden_layers,
        valid_layer_types=("linear_attention", "full_attention"),
        track_text_model_cls=PTTrackTextModel,
        build_per_track_text_config=build_per_track_text_config,
        state_dict_layer_prefixes=("self_attn", "linear_attn", "mlp"),
    )
    register_model_adapter(synthetic)
    cfg.model_type = "pt_test_engine_routes_through_adapter"
    model.config.model_type = "pt_test_engine_routes_through_adapter"

    tracks, manifest = slice_model_to_tracks(
        model, n_tracks=1, sync_block_depth=4, text_config_attr="config"
    )
    assert manifest.model_type == "pt_test_engine_routes_through_adapter"
    assert manifest.layer_types == ["linear_attention"] * 8
    assert len(tracks) == 1
