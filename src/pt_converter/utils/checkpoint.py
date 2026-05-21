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


def load_manifest(checkpoint_dir: str | Path) -> PTManifest:
    data = json.loads((Path(checkpoint_dir) / "manifest.json").read_text())
    shapes = {k: tuple(v) for k, v in data.pop("per_track_param_shapes").items()}
    return PTManifest(**data, per_track_param_shapes=shapes)
