"""Apply SlicerSpecs to a dense HF model and emit N per-track state_dicts + a manifest.

This is the conversion entrypoint. Given a loaded HF `Qwen3_5ForCausalLM`
(or any registered model_type), it produces:

- `tracks`: list[dict[str, Tensor]] — one state_dict per track, keyed with
  the same parameter names the per-track model modules expect.
- `PTManifest`: shapes, layer-type map, sync schedule, model_type.

No device movement is done here. Caller is responsible for moving slices to
the right device / saving to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from parallm.slicer.base import LayerSpec, OwnerOnly, SlicerSpec


@dataclass
class PTManifest:
    model_type: str
    n_tracks: int
    num_layers: int
    layer_types: list[str]  # one of "full_attention" | "linear_attention"
    per_track_param_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Maps top-level param name -> owner track id for params that live on a
    # single track only (e.g. embed_tokens, lm_head). Absence means "every
    # track holds this param" (replicated or sliced).
    top_level_owners: dict[str, int] = field(default_factory=dict)
    # Sync layer indices (after which an all-reduce fires). The schedule is no
    # longer chosen at conversion time — the train script places it dynamically
    # on the most sensitive layers and writes the chosen indices here when it
    # saves a checkpoint. `None` in a fresh convert output (weights are
    # schedule-independent, so the slicer needn't pick one).
    sync_layer_indices: list[int] | None = None


def resolve_param_specs(adapter: "Any", text_cfg: Any) -> dict[str, SlicerSpec]:
    """Map canonical per-track parameter names → SlicerSpec for the whole model.

    Canonical names match the keys the slicer engine emits into per-track
    state_dicts and records in ``PTManifest.per_track_param_shapes``:

      - top-level: ``embed_tokens.weight``, ``norm.weight``, ``lm_head.weight``
      - per-layer: ``layers.{i}.<sub_key>``

    This function is the model-agnostic spec resolver used by ``slice_model_to_tracks``
    and by the training-time replication-group planner in
    ``parallm.train.sync_grads``. It does NOT load or slice any weights.
    """
    out: dict[str, SlicerSpec] = {}
    for canonical, spec in adapter.top_level_specs(text_cfg).items():
        out[canonical] = spec
    layer_types: list[str] = adapter.get_layer_types(text_cfg)
    for layer_idx, layer_type in enumerate(layer_types):
        for sub_key, spec in adapter.layer_specs(text_cfg, layer_type).items():
            out[f"layers.{layer_idx}.{sub_key}"] = spec
    return out


def _resolve_sync_schedule(
    num_layers: int,
    block_depth: int,
    layer_types: list[str] | None = None,
    align_full_attention: bool = False,
) -> list[int]:
    """Return layer indices after which a sync boundary fires.

    Uniform mode (default): block depth D means sync after every D layers.
    Indices are 0-based and point to the *last* layer of a block, so the last
    block is included.

    Full-attention-aligned mode (``align_full_attention``): place a sync
    immediately *before* every full-attention layer (i.e. after layer ``f-1``)
    so each full-attention layer reads a synced — exactly decomposable —
    residual, plus a final sync after the last layer so the post-stack norm
    reads a synced hidden. Gaps longer than ``block_depth`` are subdivided
    (counting back from each mandatory point) so no window exceeds D layers.
    Needs ``layer_types``. Does NOT require D to evenly divide ``num_layers``.
    For D >= the full-attention interval this collapses to one sync per
    interval (you cannot align every full-attention layer with fewer syncs).
    """
    if align_full_attention:
        if layer_types is None:
            raise ValueError("align_full_attention requires layer_types")
        full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        # Mandatory syncs: before each full-attention layer (after f-1; skip
        # f-1 < 0, i.e. a leading full-attention layer reads the already-synced
        # embedding) and after the final layer (for the post-stack norm).
        mandatory = sorted({f - 1 for f in full if f - 1 >= 0} | {num_layers - 1})
        syncs = set(mandatory)
        prev = -1
        for b in mandatory:
            x = b - block_depth
            while x > prev:
                syncs.add(x)
                x -= block_depth
            prev = b
        return sorted(syncs)

    if num_layers % block_depth != 0:
        raise ValueError(
            f"sync_block_depth {block_depth} does not evenly divide num_layers {num_layers}; "
            f"non-uniform schedules are not supported yet."
        )
    return [i for i in range(block_depth - 1, num_layers, block_depth)]


def _apply_layer_specs(
    source_state: dict[str, torch.Tensor],
    prefix: str,
    specs: LayerSpec,
    n_tracks: int,
    manifest: PTManifest,
) -> list[dict[str, torch.Tensor]]:
    """Apply `specs` to params under `prefix.*` in `source_state`, return per-track dicts."""
    out: list[dict[str, torch.Tensor]] = [{} for _ in range(n_tracks)]
    for sub_key, spec in specs.items():
        full_key = f"{prefix}.{sub_key}" if prefix else sub_key
        if full_key not in source_state:
            raise KeyError(f"Source state missing expected key {full_key!r}")
        weight = source_state[full_key]
        for t in range(n_tracks):
            sliced = spec.slice(weight, t, n_tracks)
            out[t][sub_key] = sliced
            # record shape once
            if t == 0:
                manifest.per_track_param_shapes[full_key] = tuple(sliced.shape)
    return out


def slice_model_to_tracks(
    model: torch.nn.Module,
    *,
    n_tracks: int,
    sync_block_depth: int | None = None,
    text_config_attr: str = "config.text_config",
    align_full_attention: bool = False,
) -> tuple[list[dict[str, torch.Tensor]], PTManifest]:
    """Convert `model` into `n_tracks` per-track state_dicts plus a PTManifest.

    Caller passes a fully-loaded HF model. We do not assume the model is on
    any particular device; slicing happens in-place on whatever tensors the
    state_dict returns.

    `text_config_attr` says where to find the sub-config that has
    `num_attention_heads` / `num_key_value_heads` / etc. For Qwen3.5 this
    is `config.text_config`; for plain text models it can be `config`.

    The sync schedule is **not** chosen here by default: the per-track weights
    are schedule-independent, so conversion leaves `manifest.sync_layer_indices`
    as `None` and the train script places the boundaries dynamically (on the most
    sensitive layers) at startup. `sync_block_depth` (optionally with
    `align_full_attention`) is retained only as a convenience for tests / ad-hoc
    callers that want a fixed uniform (or full-attn-aligned) schedule baked in —
    when given, it pre-populates `sync_layer_indices`.
    """
    # resolve nested attr
    obj: Any = model
    for part in text_config_attr.split("."):
        obj = getattr(obj, part)
    text_cfg = obj

    # Late import to avoid a slicer→adapters→slicer cycle at module load time.
    from parallm.adapters import get_adapter_for_config

    adapter = get_adapter_for_config(text_cfg)
    model_type = adapter.model_type
    num_layers = int(text_cfg.num_hidden_layers)
    layer_types: list[str] = adapter.get_layer_types(text_cfg)
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types len {len(layer_types)} != num_hidden_layers {num_layers}")
    unknown = set(layer_types) - set(adapter.valid_layer_types)
    if unknown:
        raise ValueError(
            f"Adapter {model_type!r} declared valid_layer_types={adapter.valid_layer_types}, "
            f"but config produced unknown types {sorted(unknown)}"
        )

    sync_indices = (
        None
        if sync_block_depth is None
        else _resolve_sync_schedule(
            num_layers, sync_block_depth,
            layer_types=layer_types, align_full_attention=align_full_attention,
        )
    )
    manifest = PTManifest(
        model_type=model_type,
        n_tracks=n_tracks,
        num_layers=num_layers,
        layer_types=layer_types,
        sync_layer_indices=sync_indices,
    )

    source_state = dict(model.state_dict())
    tracks: list[dict[str, torch.Tensor]] = [{} for _ in range(n_tracks)]

    # Top-level: embeddings, final norm, lm_head. These live at
    # `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` for
    # CausalLM wrappers. We probe both naming conventions.
    top_specs = adapter.top_level_specs(text_cfg)
    top_key_candidates = {
        "embed_tokens.weight": [
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "model.model.embed_tokens.weight",
        ],
        "norm.weight": [
            "norm.weight",
            "model.norm.weight",
            "model.model.norm.weight",
        ],
        "lm_head.weight": ["lm_head.weight", "model.lm_head.weight"],
    }
    for canonical, spec in top_specs.items():
        candidates = top_key_candidates.get(canonical, [canonical])
        chosen = next((c for c in candidates if c in source_state), None)
        if chosen is None:
            # lm_head may be tied with embed_tokens; skip if absent.
            if canonical == "lm_head.weight":
                continue
            raise KeyError(f"Cannot find top-level param for {canonical!r}; tried {candidates}")
        weight = source_state[chosen]
        for t in range(n_tracks):
            sliced = spec.slice(weight, t, n_tracks)
            if sliced is None:
                # OwnerOnly: non-owner tracks omit the key entirely.
                continue
            tracks[t][canonical] = sliced
        manifest.per_track_param_shapes[canonical] = tuple(weight.shape)
        if isinstance(spec, OwnerOnly):
            manifest.top_level_owners[canonical] = spec.owner_track

    # Per-layer slicing.
    layers_prefix_candidates = ["layers", "model.layers", "model.model.layers"]
    layers_prefix = next(
        (p for p in layers_prefix_candidates if any(k.startswith(p + ".") for k in source_state)),
        None,
    )
    if layers_prefix is None:
        raise KeyError(f"Could not locate decoder layers in state_dict; tried {layers_prefix_candidates}")

    for layer_idx, layer_type in enumerate(layer_types):
        prefix = f"{layers_prefix}.{layer_idx}"
        specs = adapter.layer_specs(text_cfg, layer_type)
        per_track_layer = _apply_layer_specs(source_state, prefix, specs, n_tracks, manifest)
        for t in range(n_tracks):
            for sub_key, val in per_track_layer[t].items():
                tracks[t][f"layers.{layer_idx}.{sub_key}"] = val

    return tracks, manifest
