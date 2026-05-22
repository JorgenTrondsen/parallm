"""torchrun entrypoint for PT distillation fine-tuning on Qwen3.5-9B.

Launch (16 ranks, n_tracks=16, 2 ranks per GPU on an 8xA100 node):

    torchrun --standalone --nproc-per-node=16 scripts/train_qwen3_5_9b.py \
        --hf-model /path/to/qwen3_5_9b \
        --tracks-dir /path/to/pt_tracks \
        --max-steps 1000 \
        --seq-len 4096 \
        --batch-size 1 \
        --lr 3e-5

We deliberately oversubscribe GPUs (LOCAL_RANK 0..15, set_device(local_rank %
gpu_count)) because the per-track student is small (~562M params at N=16) and
the cross-track all-reduce is a single global op. NCCL handles two-procs-per-
GPU on modern PyTorch.

This script is intentionally minimal: distributed init, build groups, load
sliced student + dense teacher, wrap teacher with FSDP2, run distillation
steps, save periodically.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.data import CalibrationDataConfig, PackedTokenStream
from pt_converter.train.distill import DistillConfig, distill_step
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path")
    p.add_argument("--tracks-dir", required=True, help="Output of convert_qwen3_5_9b.py")
    p.add_argument("--out-dir", default="./pt_train_out")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lambda-block", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=0.5)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    args = p.parse_args()

    # ----- Distributed init -----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Oversubscribe GPUs: when nproc-per-node > visible GPUs, share devices.
    gpu_count = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % gpu_count)
    rank = dist.get_rank()

    # ----- Load manifest and groups -----
    manifest = load_manifest(args.tracks_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    _log(rank, f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} D={manifest.sync_block_depth}")
    _log(rank, f"[init] rank={rank} track_id={layout.track_id} intra_track_rank={layout.intra_track_rank}")

    # ----- Load teacher (dense) -----
    cfg = AutoConfig.from_pretrained(args.hf_model)
    _log(rank, "[init] loading frozen dense teacher…")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    # Use the inner text decoder for hidden-state hooks.
    text_model = teacher_model.model.language_model if hasattr(teacher_model.model, "language_model") else teacher_model.model
    teacher = HookedTeacher(
        text_model=text_model,
        lm_head=teacher_model.lm_head,
        sync_layer_indices=manifest.sync_layer_indices,
    )
    teacher_model = teacher_model.to(torch.cuda.current_device())
    wrap_teacher_with_fsdp(teacher_model)

    # ----- Build per-rank student and load its track shard -----
    _log(rank, f"[init] building PT student for track {layout.track_id}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        track_id=layout.track_id,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=layout.track_group,
    )
    track_state = load_track(args.tracks_dir, layout.track_id)
    student.load_track_state_dict(track_state, strict=True)
    student = student.to(torch.cuda.current_device())
    wrap_student_with_fsdp(student, layout)

    # ----- Data -----
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    ds = PackedTokenStream(tok, CalibrationDataConfig(seq_len=args.seq_len))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    # ----- Optimizer -----
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    distill_cfg = DistillConfig(
        sync_layer_indices=tuple(manifest.sync_layer_indices),
        lambda_block=args.lambda_block,
        lambda_kl=args.lambda_kl,
        lambda_ce=args.lambda_ce,
        kl_temperature=args.kl_temperature,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    student.train()
    for batch in loader:
        if step >= args.max_steps:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        optim.zero_grad(set_to_none=True)
        losses = distill_step(student, teacher, batch, distill_cfg)
        losses["total"].backward()
        optim.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            _log(
                rank,
                f"[step {step}] total={losses['total'].item():.4f} "
                f"block_mse={losses['block_mse'].item():.4f} "
                f"kl={losses['kl'].item():.4f} ce={losses['ce'].item():.4f} "
                f"elapsed={elapsed:.1f}s",
            )

        if step > 0 and step % args.save_every == 0:
            _log(rank, f"[save] step {step}")
            # FSDP2 full-state-dict save for the per-track shard.
            # Each rank within a track group saves its shard; rank-0 of each track
            # writes a combined .safetensors. Left as TODO once FSDP2 save API is finalized.

        step += 1

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
