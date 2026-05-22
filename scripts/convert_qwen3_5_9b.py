"""One-shot conversion: load Qwen3.5-9B, slice into N tracks, save to disk.

Usage:
    python scripts/convert_qwen3_5_9b.py \
        --hf-model /home2/jtr020/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
        --out-dir /path/to/pt_tracks \
        --n-tracks 4 \
        --sync-block-depth 4

Runs on CPU (or a single GPU if --device cuda) — the slicer doesn't need
distributed setup. ~9B bf16 weights = ~18GB host RAM; fine on this node.
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from pt_converter.slicer.convert import slice_model_to_tracks
from pt_converter.utils.checkpoint import save_tracks
from pt_converter.utils.max_tracks import max_tracks_for_config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Path or HF id of the dense source model")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-tracks", type=int, default=None, help="Defaults to max-tracks for the model")
    p.add_argument("--sync-block-depth", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.hf_model)
    max_n = max_tracks_for_config(cfg)
    n_tracks = args.n_tracks if args.n_tracks is not None else max_n
    if n_tracks > max_n:
        print(f"[error] requested n_tracks={n_tracks} exceeds max-tracks={max_n}", file=sys.stderr)
        return 2

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[info] loading {args.hf_model} on {args.device} as {args.dtype}…", flush=True)

    # AutoModelForCausalLM for Qwen3.5 returns the text-only causal model whose
    # `.config` is already the text config (Qwen3_5TextConfig), so we point the
    # slicer at `config` directly rather than `config.text_config`.
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()

    print(f"[info] slicing into n_tracks={n_tracks}, sync_block_depth={args.sync_block_depth}", flush=True)
    tracks, manifest = slice_model_to_tracks(
        model,
        n_tracks=n_tracks,
        sync_block_depth=args.sync_block_depth,
        text_config_attr="config",
    )

    print(f"[info] writing {len(tracks)} track shards to {args.out_dir}", flush=True)
    save_tracks(args.out_dir, tracks, manifest)
    print("[ok] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
