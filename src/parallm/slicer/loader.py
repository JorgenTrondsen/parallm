"""Stream a HF checkpoint's *text* weights to a bf16 state_dict for the slicer.

One loader for every source we convert — plain-bf16, NVFP4/FP8 mixed-precision,
VLM or text-only. It reads the safetensors shards, dequantizes each weight to
bf16 (NVFP4 uint8 / FP8 float8 via compressed-tensors' tested decoders; everything
else a passthrough cast), remaps the key to the slicer's canonical name, and drops
the vision tower, MTP head, and scale siblings. The result feeds
``slice_model_to_tracks`` via ``StateDictModel`` (no need to instantiate the model).
"""
from __future__ import annotations

import json
import os
import re

import torch
from safetensors import safe_open

from compressed_tensors.compressors.nvfp4 import unpack_fp4_from_uint8
from compressed_tensors.quantization import dequantize

_SCALE_SIBLINGS = (".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale")

# Decoder layers appear as `model.language_model.layers.{i}.*` (VLM) or
# `model.layers.{i}.*` (text-only); both remap to `layers.{i}.*`.
_LAYER = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")
_TOP = {
    "model.language_model.embed_tokens.weight": "embed_tokens.weight",
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.language_model.norm.weight": "norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def dequant_weight(w: torch.Tensor, sib: "dict[str, torch.Tensor]") -> torch.Tensor:
    """One weight → bf16. NVFP4 (uint8) and FP8 (float8_e4m3fn) via the tested
    library decoder / native cast; everything else is a plain bf16 passthrough."""
    if w.dtype == torch.uint8:  # NVFP4, W4A16 group-16 — mirror NVFP4PackedCompressor
        m, n = w.shape
        up = unpack_fp4_from_uint8(w, m, n * 2)
        return dequantize(
            x_q=up, scale=sib["weight_scale"].to(up.dtype),
            global_scale=1.0 / sib["weight_scale_2"].float(), dtype=torch.bfloat16,
        )
    if w.dtype == torch.float8_e4m3fn:  # FP8 per-tensor mixer
        return (w.float() * sib["weight_scale"].float()).to(torch.bfloat16)
    return w.to(torch.bfloat16)


def slicer_key(k: str) -> "str | None":
    """Map a checkpoint key to the slicer's canonical name, or None to drop
    (vision tower, MTP head, scale siblings)."""
    if any(k.endswith(s) for s in _SCALE_SIBLINGS):
        return None
    m = _LAYER.match(k)
    if m:
        return f"layers.{m.group(1)}.{m.group(2)}"
    return _TOP.get(k)


def stream_text_state_dict(hf_model: str, *, drop_lm_head: bool = False) -> "dict[str, torch.Tensor]":
    """Read the checkpoint's text weights → bf16 state_dict keyed for the slicer.
    Dequantizes NVFP4/FP8, drops vision + MTP + scale siblings (+ lm_head if asked)."""
    idx = os.path.join(hf_model, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        by_shard: dict[str, "list[str] | None"] = {}
        for k, shard in wm.items():
            by_shard.setdefault(shard, []).append(k)  # type: ignore[union-attr]
    else:
        by_shard = {"model.safetensors": None}  # single-file: read all keys

    sd: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(hf_model, shard), framework="pt") as f:
            present = set(f.keys())
            for k in (ks if ks is not None else list(present)):
                key = slicer_key(k)
                if key is None or (drop_lm_head and key == "lm_head.weight"):
                    continue
                base = k.rsplit(".", 1)[0]
                sib = {s: f.get_tensor(f"{base}.{s}")
                       for s in ("weight_scale", "weight_scale_2")
                       if f"{base}.{s}" in present}
                sd[key] = dequant_weight(f.get_tensor(k), sib)
    return sd


class StateDictModel:
    """Duck-typed stand-in for ``slice_model_to_tracks``: exposes ``.config`` and
    ``.state_dict()`` so the slicer never instantiates the (huge) model."""

    def __init__(self, sd: "dict[str, torch.Tensor]", config):
        self._sd = sd
        self.config = config

    def state_dict(self):
        return self._sd
