"""Slice a ModelOpt-NVFP4 Qwen3.5-family VLM (e.g. nvidia/Qwen3.6-27B-NVFP4) into N tracks.

The checkpoint is a mixed-precision VLM: MLPs are NVFP4 (packed uint8 + FP8
group-16 scales + global scale), mixers are FP8 (per-tensor), the rest (norms,
GDN A_log/dt_bias/conv, embeddings) are unquantized. We dequantize the LANGUAGE
model to bf16 (reusing compressed-tensors' tested NVFP4 decoder), drop the vision
tower + MTP head, and feed the text state_dict straight to the existing slicer.

Runs on CPU; the dequantized 27B is ~54 GB in host RAM (fine on this box).

    python scripts/convert_nvfp4_27b.py \
        --hf-model ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/<sha> \
        --out-dir convert_out/qwen3_6_27b_n8_tracks --n-tracks 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig

from compressed_tensors.compressors.nvfp4 import unpack_fp4_from_uint8
from compressed_tensors.quantization import dequantize

from parallm.slicer.convert import slice_model_to_tracks
from parallm.utils.checkpoint import save_tracks
from parallm.utils.max_tracks import max_tracks_for_config

# ponytail: dequant inlined here; extract to a module if step-2 calibration needs
# a *running* model (that path re-loads weights into an instantiated model).
_SCALE_SIBLINGS = (".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale")


def _dequant(w: torch.Tensor, sib: "dict[str, torch.Tensor]") -> torch.Tensor:
    """One weight → bf16. NVFP4 (uint8) and FP8 (float8_e4m3fn) via the tested
    library decoder / native cast; everything else is a plain bf16 passthrough."""
    if w.dtype == torch.uint8:  # NVFP4, W4A16 group-16 — mirror NVFP4PackedCompressor.decompress
        m, n = w.shape
        up = unpack_fp4_from_uint8(w, m, n * 2)
        return dequantize(
            x_q=up, scale=sib["weight_scale"].to(up.dtype),
            global_scale=1.0 / sib["weight_scale_2"].float(), dtype=torch.bfloat16,
        )
    if w.dtype == torch.float8_e4m3fn:  # FP8 per-tensor mixer
        return (w.float() * sib["weight_scale"].float()).to(torch.bfloat16)
    return w.to(torch.bfloat16)


class _StateDictModel:
    """Duck-typed stand-in for slice_model_to_tracks: it only needs ``.config``
    and ``.state_dict()`` — no need to instantiate the 54 GB model."""

    def __init__(self, sd: "dict[str, torch.Tensor]", config):
        self._sd = sd
        self.config = config

    def state_dict(self):
        return self._sd


_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def _slicer_key(k: str) -> "str | None":
    """Map a checkpoint key to the slicer's canonical name, or None to drop
    (vision tower, MTP head, scale siblings)."""
    if any(k.endswith(s) for s in _SCALE_SIBLINGS):
        return None
    m = _LAYER.match(k)
    if m:
        return f"layers.{m.group(1)}.{m.group(2)}"
    return {
        "model.language_model.embed_tokens.weight": "embed_tokens.weight",
        "model.language_model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }.get(k)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Path to the NVFP4 checkpoint dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-tracks", type=int, default=None, help="Defaults to max-tracks")
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.hf_model)
    tcfg = cfg.text_config
    max_n = max_tracks_for_config(tcfg)
    n = args.n_tracks if args.n_tracks is not None else max_n
    if n > max_n:
        print(f"[error] n_tracks={n} exceeds max-tracks={max_n} for this model", file=sys.stderr)
        return 2

    wm = json.load(open(os.path.join(args.hf_model, "model.safetensors.index.json")))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in wm.items():
        by_shard.setdefault(shard, []).append(k)

    print(f"[info] dequantizing text model to bf16 (n_tracks={n}); dropping vision + MTP…", flush=True)
    text_sd: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(args.hf_model, shard), framework="pt") as f:
            present = set(f.keys())
            for k in ks:
                key = _slicer_key(k)
                if key is None:
                    continue
                base = k.rsplit(".", 1)[0]
                sib = {s: f.get_tensor(f"{base}.{s}")
                       for s in ("weight_scale", "weight_scale_2")
                       if f"{base}.{s}" in present}
                text_sd[key] = _dequant(f.get_tensor(k), sib)
        print(f"[dequant] {shard}: {len(text_sd)} text weights so far", flush=True)

    print(f"[info] slicing {len(text_sd)} text weights into {n} tracks…", flush=True)
    tracks, manifest = slice_model_to_tracks(
        _StateDictModel(text_sd, tcfg), n_tracks=n, text_config_attr="config",
    )
    save_tracks(args.out_dir, tracks, manifest)
    print(f"[ok] wrote {len(tracks)} track shards + manifest to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
