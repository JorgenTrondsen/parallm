"""Save/load per-track checkpoints + PTManifest."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors

from pt_converter.slicer.convert import PTManifest


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
    manifest_dict = asdict(manifest)
    manifest_dict["per_track_param_shapes"] = {
        k: list(v) for k, v in manifest.per_track_param_shapes.items()
    }
    (out / "manifest.json").write_text(json.dumps(manifest_dict, indent=2))
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


def load_manifest(checkpoint_dir: str | Path) -> PTManifest:
    data = json.loads((Path(checkpoint_dir) / "manifest.json").read_text())
    shapes = {k: tuple(v) for k, v in data.pop("per_track_param_shapes").items()}
    # `top_level_owners` was added later; old manifests omit it.
    top_level_owners = data.pop("top_level_owners", {})
    return PTManifest(
        **data,
        per_track_param_shapes=shapes,
        top_level_owners=top_level_owners,
    )
