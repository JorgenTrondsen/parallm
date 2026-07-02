"""Save/load per-track checkpoints + PTManifest."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors

from pt_converter.slicer.convert import PTManifest


def save_manifest(out_dir: str | Path, manifest: PTManifest) -> Path:
    """Write `manifest.json` into `out_dir`. Returns the manifest path.

    Used both by `save_tracks` (conversion) and by the train script when it
    saves a checkpoint — the latter writes the dynamically-chosen
    `sync_layer_indices` alongside the structural fields so eval reproduces the
    exact schedule the model was trained with.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_dict = asdict(manifest)
    manifest_dict["per_track_param_shapes"] = {
        k: list(v) for k, v in manifest.per_track_param_shapes.items()
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest_dict, indent=2))
    return path


def save_tracks(
    out_dir: str | Path,
    tracks: list[dict[str, torch.Tensor]],
    manifest: PTManifest,
) -> Path:
    """Write `track_{i}.safetensors` per track + `manifest.json`. Returns out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, state in enumerate(tracks):
        # safetensors requires contiguous tensors with no shared storage; we
        # materialize clones so any views from .narrow() get their own buffers.
        materialized = {k: v.detach().contiguous().clone() for k, v in state.items()}
        save_safetensors(materialized, str(out / f"track_{i}.safetensors"))
    save_manifest(out, manifest)
    return out


def load_track(checkpoint_dir: str | Path, track_id: int) -> dict[str, torch.Tensor]:
    return load_safetensors(str(Path(checkpoint_dir) / f"track_{track_id}.safetensors"))


def load_track_keys(
    checkpoint_dir: str | Path, track_id: int, keys: list[str]
) -> dict[str, torch.Tensor]:
    """Load only `keys` from `track_{track_id}.safetensors` (mmap, no full read).

    Used by the vocab-parallel loader so every rank can read just the full
    embed_tokens / lm_head tensors from the track-0 shard without materializing
    the whole shard. Missing keys are silently skipped (e.g. tied lm_head).
    """
    from safetensors import safe_open

    path = str(Path(checkpoint_dir) / f"track_{track_id}.safetensors")
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        present = set(f.keys())
        for k in keys:
            if k in present:
                out[k] = f.get_tensor(k)
    return out


def save_cross_head(out_dir: str | Path, estimator) -> None:
    """Write the cross-head estimator sidecar (``cross_head.safetensors`` +
    ``cross_head.json``).

    Kept OUT of the per-track shards (so ``load_track_state_dicts`` is untouched).
    The module is replicated bit-identically across ranks, so only rank 0 needs to
    call this.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().contiguous().clone().cpu() for k, v in estimator.state_dict().items()}
    save_safetensors(sd, str(out / "cross_head.safetensors"))
    (out / "cross_head.json").write_text(json.dumps(estimator.config_dict(), indent=2))


def load_cross_head(checkpoint_dir: str | Path):
    """Rebuild the cross-head estimator from a checkpoint's sidecar, or ``None`` if
    the checkpoint has none. Returns an eval-ready ``CrossHeadEstimator`` on CPU."""
    cfg_path = Path(checkpoint_dir) / "cross_head.json"
    if not cfg_path.exists():
        return None
    from pt_converter.model.cross_head_estimator import CrossHeadEstimator

    cfg = json.loads(cfg_path.read_text())
    est = CrossHeadEstimator(**cfg)
    est.load_state_dict(load_safetensors(str(Path(checkpoint_dir) / "cross_head.safetensors")))
    return est


def load_manifest(checkpoint_dir: str | Path) -> PTManifest:
    data = json.loads((Path(checkpoint_dir) / "manifest.json").read_text())
    # The cadence descriptors `sync_block_depth` / `sync_schedule` were dropped
    # when schedule selection moved from conversion to the train script; pop them
    # so manifests written by the old converter still load.
    data.pop("sync_block_depth", None)
    data.pop("sync_schedule", None)
    shapes = {k: tuple(v) for k, v in data.pop("per_track_param_shapes", {}).items()}
    # `top_level_owners` was added later; old manifests omit it.
    top_level_owners = data.pop("top_level_owners", {})
    return PTManifest(
        **data,
        per_track_param_shapes=shapes,
        top_level_owners=top_level_owners,
    )
