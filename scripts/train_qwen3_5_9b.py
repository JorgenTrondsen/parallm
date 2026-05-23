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
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.data import CalibrationDataConfig, PackedTokenStream
from pt_converter.train.distill import DistillConfig, distill_step, validate_step
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path")
    p.add_argument("--tracks-dir", required=True, help="Output of convert_qwen3_5_9b.py (manifest source)")
    p.add_argument("--resume-from", default=None,
                   help="Optional checkpoint dir (e.g. an earlier run's best/). Loads track_<id>.safetensors from here instead of --tracks-dir.")
    p.add_argument("--out-dir", default="./pt_train_out")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--save-every", type=int, default=0,
                   help="0 disables step checkpoints. Use --eval-every to drive best/ saves instead.")
    p.add_argument("--save-final", action="store_true",
                   help="Write final/ when training exits (loop done or early-stopped).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lambda-block", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=0.5)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=0,
                   help="0 disables validation. >0 runs val_batches batches of WT-103 val every N steps.")
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="0 disables early stop. Otherwise stop after N eval windows with no val_ce improvement.")
    p.add_argument("--min-improvement", type=float, default=0.01,
                   help="Min val_ce drop (nats) to count as an improvement.")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear warmup steps. 0 disables the LR schedule entirely.")
    p.add_argument("--lr-min-ratio", type=float, default=0.1,
                   help="Cosine decay floor as a fraction of --lr.")
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
    wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)

    # ----- Build per-rank student and load its track shard -----
    _log(rank, f"[init] building PT student for track {layout.track_id}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        track_id=layout.track_id,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=layout.track_group,
    )
    state_src = args.resume_from if args.resume_from else args.tracks_dir
    if args.resume_from:
        _log(rank, f"[init] resuming track state from {args.resume_from}")
    track_state = load_track(state_src, layout.track_id)
    student.load_track_state_dict(track_state, strict=True)
    student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)

    # ----- Data -----
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    ds = PackedTokenStream(tok, CalibrationDataConfig(seq_len=args.seq_len))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    val_loader = None
    if args.eval_every > 0:
        val_ds = PackedTokenStream(
            tok, CalibrationDataConfig(split="validation", seq_len=args.seq_len)
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # ----- Optimizer + LR schedule -----
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    scheduler = None
    if args.warmup_steps > 0:
        warmup = args.warmup_steps
        total = args.max_steps
        min_ratio = args.lr_min_ratio

        def lr_lambda(s: int) -> float:
            if s < warmup:
                return (s + 1) / max(1, warmup)
            progress = (s - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    distill_cfg = DistillConfig(
        sync_layer_indices=tuple(manifest.sync_layer_indices),
        lambda_block=args.lambda_block,
        lambda_kl=args.lambda_kl,
        lambda_ce=args.lambda_ce,
        kl_temperature=args.kl_temperature,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str):
        """Per-rank track save into <out-dir>/<name>/."""
        ck_dir = Path(args.out_dir) / name
        if rank == 0:
            ck_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(args.tracks_dir) / "manifest.json",
                ck_dir / "manifest.json",
            )
        dist.barrier()
        track_state = {
            k: v.detach().contiguous().clone().cpu()
            for k, v in student.state_dict().items()
        }
        save_safetensors(
            track_state,
            str(ck_dir / f"track_{layout.track_id}.safetensors"),
        )
        dist.barrier()

    def run_eval() -> float:
        """Run val_batches of validation; return mean val CE (same on all ranks)."""
        assert val_loader is not None
        student.eval()
        ce_sum = torch.zeros((), device=torch.cuda.current_device())
        n = 0
        for vb in val_loader:
            if n >= args.val_batches:
                break
            vb = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in vb.items()}
            if vb["input_ids"].ndim == 1:
                vb = {k: v.unsqueeze(0) for k, v in vb.items()}
            out = validate_step(student, vb)
            ce_sum = ce_sum + out["ce"]
            n += 1
        student.train()
        # On peer tracks every per-batch CE was 0; owner's sum is the real
        # total. all_reduce SUM lets all ranks see the same mean.
        dist.all_reduce(ce_sum, op=dist.ReduceOp.SUM)
        return (ce_sum / max(1, n)).item()

    best_val_ce = float("inf")
    windows_since_best = 0
    stop_now = False

    step = 0
    t0 = time.time()
    student.train()
    for batch in loader:
        if step >= args.max_steps or stop_now:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        optim.zero_grad(set_to_none=True)
        losses = distill_step(student, teacher, batch, distill_cfg)
        losses["total"].backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]
            _log(
                rank,
                f"[step {step}] total={losses['total'].item():.4f} "
                f"block_mse={losses['block_mse'].item():.4f} "
                f"kl={losses['kl'].item():.4f} ce={losses['ce'].item():.4f} "
                f"lr={lr_now:.2e} elapsed={elapsed:.1f}s",
            )

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            _log(rank, f"[save] step {step}")
            save_checkpoint(f"step_{step}")

        if val_loader is not None and step > 0 and step % args.eval_every == 0:
            val_ce = run_eval()
            improved = val_ce < (best_val_ce - args.min_improvement)
            if improved:
                best_val_ce = val_ce
                windows_since_best = 0
                _log(rank, f"[eval step={step}] val_ce={val_ce:.4f} (new best)")
                save_checkpoint("best")
            else:
                windows_since_best += 1
                _log(
                    rank,
                    f"[eval step={step}] val_ce={val_ce:.4f} "
                    f"(no improvement; {windows_since_best}/{args.early_stop_patience})",
                )
            if args.early_stop_patience > 0 and windows_since_best >= args.early_stop_patience:
                _log(rank, f"[early-stop] val_ce hasn't improved in {args.early_stop_patience} windows (best={best_val_ce:.4f})")
                stop_now = True

        step += 1

    if args.save_final:
        _log(rank, "[save] final")
        save_checkpoint("final")

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
