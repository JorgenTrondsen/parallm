"""torchrun entry point: measure how closely the trained PT student matches the teacher.

Loads a per-rank checkpoint (``track_*.safetensors`` + ``manifest.json``),
rebuilds the sharded teacher exactly as the training script does, and runs
both on a held-out stream. Reports:

  - KL(teacher‖student) (forward) and KL(student‖teacher) (reverse)
  - Top-1 / top-5 prediction agreement and top-5 set IoU
  - Per-sync-boundary hidden-state MSE
  - Student vs teacher perplexity (and gap)

The launch shape mirrors ``scripts/train_qwen3_5_9b.py``. The layout is uniform
(K = n_tracks // world_size); forward output is layout-independent (the
SyncBoundary all-reduce combines all tracks regardless of which rank hosts
them), so it need not match training. Eval uses the legacy full-logits student
path (full lm_head on the owner rank) — it loads a vocab-parallel-trained
checkpoint unchanged (track_0 carries the gathered full embed/lm_head) and
yields full-vocab logits for the fidelity metrics.

Single node, 8 GPUs, n_tracks=16, best checkpoint:

    torchrun --standalone --nproc-per-node=8 scripts/eval_fidelity.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir ./pt_train_out/best \\
        --num-batches 200
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.eval.fidelity import fidelity_step
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path (same as training)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir with track_*.safetensors and manifest.json "
                        "(e.g. ./pt_train_out/best, ./pt_train_out/final, or a step_N dir).")
    p.add_argument("--data-preset", default=DEFAULT_PRESET, choices=preset_names(),
                   help="Streamed eval mixture (DEFAULT). 'qwen-mix' is the training mixture, so "
                        "fidelity is measured on the SAME distribution the student was distilled on "
                        "(directly comparable to the training val_kl). Reads the held-out FRONT slice "
                        "(--skip-docs 0) — the docs the default training val held out. NOTE: qwen-mix's "
                        "the-stack-dedup is GATED (export HF_TOKEN), or pass --data-preset slimpajama / "
                        "--dataset-name. Overridden by --data-source or --dataset-name.")
    p.add_argument("--data-source", action="append", default=None, metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]",
                   help="Custom eval source(s), repeatable; if any given it REPLACES --data-preset.")
    p.add_argument("--dataset-name", default=None,
                   help="Legacy single-dataset override (e.g. 'Salesforce/wikitext' for the old "
                        "WikiText-103 comparator). If set, used instead of --data-preset.")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1",
                   help="Config for --dataset-name (ignored unless --dataset-name is set).")
    p.add_argument("--split", default="validation",
                   help="Split for --dataset-name (ignored unless --dataset-name is set).")
    p.add_argument("--text-key", default="text",
                   help="Text column for --dataset-name (ignored unless --dataset-name is set).")
    p.add_argument("--skip-docs", type=int, default=0,
                   help="Skip the first N docs of the preset/--data-source stream before packing. "
                        "0 reads the held-out front the default training val used; raise to match a "
                        "non-default --val-holdout-docs if you need strict disjointness from training.")
    p.add_argument("--seed", type=int, default=42,
                   help="Mixture interleave seed; keep equal across ranks (matches training).")
    p.add_argument("--num-batches", type=int, default=200,
                   help="Number of packed sequences to evaluate.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=128,
                   help="Seq-chunk size for the vocab-wide fp32 expansion; matches "
                        "--kl-ce-chunk-size in training.")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_count = torch.cuda.device_count()
    if local_rank >= gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} >= visible GPU count {gpu_count}. "
            f"Launch with --nproc-per-node <= #GPUs."
        )
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    # ----- Manifest + uniform layout (forward output is layout-independent). -----
    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    _log(
        rank,
        f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
        f"K={layout.tracks_per_rank} D={manifest.sync_block_depth}",
    )
    _log(rank, f"[init] rank={rank} local_track_ids={layout.local_track_ids}")

    # ----- Teacher (frozen, FSDP-sharded across the world). -----
    cfg = AutoConfig.from_pretrained(args.hf_model)
    _log(rank, "[init] loading frozen dense teacher…")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    text_model = (
        teacher_model.model.language_model
        if hasattr(teacher_model.model, "language_model")
        else teacher_model.model
    )
    teacher = HookedTeacher(
        text_model=text_model,
        lm_head=teacher_model.lm_head,
        sync_layer_indices=manifest.sync_layer_indices,
    )
    teacher_model = teacher_model.to(torch.cuda.current_device())
    wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)

    # ----- Student. Same construction as the train script so the loaded
    # state_dict aligns 1:1 with the safetensors keys. -----
    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=layout.track_group,
    )
    track_states = {tid: load_track(args.checkpoint_dir, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)
    student.eval()

    # ----- Data. PackedTokenStream is reused as-is; switching dataset is just
    # a CalibrationDataConfig change. -----
    # Eval source priority: --data-source > --dataset-name (legacy single) > --data-preset.
    # Default is the qwen-mix training mixture's held-out front slice, so fidelity is
    # measured in-distribution and lines up with the training val_kl.
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if args.data_source:
        data_cfg = CalibrationDataConfig(
            sources=[parse_source_spec(s) for s in args.data_source],
            seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
        )
        data_desc = "custom --data-source"
    elif args.dataset_name:
        data_cfg = CalibrationDataConfig.single(
            dataset_name=args.dataset_name, dataset_config=args.dataset_config,
            split=args.split, text_key=args.text_key, seq_len=args.seq_len, seed=args.seed,
        )
        data_desc = f"{args.dataset_name}/{args.dataset_config}/{args.split}"
    else:
        data_cfg = CalibrationDataConfig(
            sources=preset_sources(args.data_preset),
            seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
        )
        data_desc = f"preset:{args.data_preset} (held-out front, skip_docs={args.skip_docs})"
    _log(rank, f"[data] eval source: {data_desc}")
    ds = PackedTokenStream(tok, data_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    sync_indices = tuple(manifest.sync_layer_indices)
    sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    for batch in loader:
        if n_batches >= args.num_batches:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        m = fidelity_step(student, teacher, batch, sync_indices, args.chunk_size)
        for name, val in m.items():
            if name not in sums:
                sums[name] = torch.zeros((), device=val.device, dtype=torch.float32)
            sums[name] = sums[name] + val.float()
        n_batches += 1

    # ----- Cross-rank reduction. fidelity_step emits zeros from non-owner
    # ranks, so SUM lands on the owner's value. Divide by batch count for the
    # per-token mean over the eval slice. -----
    for name in sums:
        dist.all_reduce(sums[name], op=dist.ReduceOp.SUM)
        sums[name] = sums[name] / max(1, n_batches)

    if rank == 0:
        s_nll = sums["student_nll"].item()
        t_nll = sums["teacher_nll"].item()
        s_ppl = math.exp(s_nll)
        t_ppl = math.exp(t_nll)
        kl_fwd = sums["kl_forward"].item()
        kl_rev = sums["kl_reverse"].item()
        print()
        print(f"===== Fidelity over {n_batches} batches "
              f"({data_desc}, seq_len={args.seq_len}) =====")
        print(f"  perplexity   : student={s_ppl:.4f}  teacher={t_ppl:.4f}  gap={s_ppl - t_ppl:+.4f}")
        print(f"  nll          : student={s_nll:.4f}  teacher={t_nll:.4f}  delta={s_nll - t_nll:+.4f}")
        print(f"  KL forward (t‖s) = {kl_fwd:.4f} nats")
        print(f"  KL reverse (s‖t) = {kl_rev:.4f} nats")
        print(f"  top1_agree   = {sums['top1_agree'].item():.4f}")
        print(f"  top5_agree   = {sums['top5_agree'].item():.4f}  (teacher top-1 ∈ student top-5)")
        print(f"  top5_set_iou = {sums['top5_set_iou'].item():.4f}")
        print(f"  per-sync-boundary block_mse (raw | rel=Σ(s−t)²/Σt²):")
        for layer_idx in sync_indices:
            raw = sums[f"block_mse_l{layer_idx}"].item()
            rel = sums[f"block_relmse_l{layer_idx}"].item()
            print(f"    layer {layer_idx:3d}: raw={raw:.6e}  rel={rel:.6e}")
        print()

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
