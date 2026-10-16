"""Calibrate the per-layer `+c` of a folded MLP read's ``r<M>+c`` norm (`MlpReadBase.bind_fold`).

A folded read never forms the estimate ``z`` in hidden space, so the next layer's RMSNorm
scale comes from a rank-M sketch ``z_M``; ``c`` adds back, per layer, the mean of
``mean_H((carry+z)²) − mean_H((carry+z_M)²)`` — the energy the sketch misses. It depends on
the base (truncation and pruning change ``z``), so it is measured on THAT base and spec, with
the UNFOLDED read in the deployment walk, on calibration text (never the eval's):

    torchrun --standalone --nproc-per-node=8 scripts/calib_fold_norm.py \\
        --hf-model <teacher> --checkpoint-dir convert_out/qwen3/32b_n64_even \\
        --sync-phase "exact,0-31:post-attn" --read-mlp-base <base> \\
        [--read-mlp-compress SPEC] --read-mlp-stats mlp_stats.safetensors --ranks 8,32 --out c.safetensors
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from parallm.dist.groups import build_groups
from parallm.model.merge import plan_track_layout
from parallm.model.mlp_read import load_mlp_read_base
from parallm.model.pt_model import PTWrappedModel
from parallm.train.data import (
    DEFAULT_MIXTURE,
    CalibrationDataConfig,
    PackedTokenStream,
    preset_sources,
)
from parallm.utils.checkpoint import load_manifest, load_track


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--sync-phase", required=True)
    p.add_argument("--read-mlp-base", required=True)
    p.add_argument("--read-mlp-compress", default=None)
    p.add_argument("--read-mlp-stats", required=True)
    p.add_argument("--ranks", default="8,32", help="sketch ranks M to calibrate, comma list")
    p.add_argument("--out", required=True)
    p.add_argument("--data-preset", default=DEFAULT_MIXTURE)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank = dist.get_rank()
    dev = torch.cuda.current_device()
    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    plan = plan_track_layout(manifest.n_tracks, dist.get_world_size(), 1, allow_merge=False)
    cfg = AutoConfig.from_pretrained(args.hf_model)
    text_cfg = getattr(cfg, "text_config", cfg)
    student = PTWrappedModel(
        text_config=text_cfg, n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=list(range(manifest.num_layers)),
        track_group=layout.track_group, fuse_size=plan.fuse_size,
    )
    student.load_track_state_dicts(
        {t: load_track(args.checkpoint_dir, t) for t in layout.local_track_ids}, strict=True)
    student.set_sync_phase(args.sync_phase)
    student = student.to(torch.bfloat16).to(dev).eval()
    base = load_mlp_read_base(args.read_mlp_base, None, student.local_track_ids,
                              student.global_tracks, dev, compress=args.read_mlp_compress,
                              stats=args.read_mlp_stats)
    student.set_mlp_read(base)
    ranks = [int(m) for m in args.ranks.split(",") if m.strip()]
    base.start_norm_calibration(args.read_mlp_stats, ranks)

    tok = AutoTokenizer.from_pretrained(args.hf_model)
    loader = DataLoader(PackedTokenStream(tok, CalibrationDataConfig(
        sources=preset_sources(args.data_preset), seq_len=args.seq_len, seed=args.seed,
    )), batch_size=1, num_workers=0)
    for n, batch in enumerate(loader):
        if n >= args.batches:
            break
        ids = batch["input_ids"].to(dev)
        with torch.no_grad():
            student(input_ids=ids if ids.ndim == 2 else ids.unsqueeze(0))
        if rank == 0:
            print(f"[calib] batch {n + 1}/{args.batches}", flush=True)
    c = base.finish_norm_calibration(layout.track_group)
    if rank == 0:
        out = {f"layer.{li}.c_r{m}": torch.tensor(v, dtype=torch.float32) for (li, m), v in c.items()}
        save_file(out, args.out, metadata={"source": base.source, "spec": base.spec,
                                           "sync_phase": args.sync_phase, "ranks": args.ranks,
                                           "batches": str(args.batches)})
        for m in ranks:
            vals = [c[(li, m)] for li in base.layer_ids]
            print(f"[c] r{m}: per-layer missing mean-square {min(vals):.4g} .. {max(vals):.4g} "
                  f"(layers {base.layer_ids[vals.index(max(vals))]} max)", flush=True)
        print(f"[out] {args.out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
