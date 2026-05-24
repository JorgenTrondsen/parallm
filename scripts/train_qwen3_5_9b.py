"""torchrun entrypoint for PT distillation fine-tuning on Qwen3.5-9B.

Launch: one rank per visible GPU. The K = n_tracks / world_size tracks each
rank owns are hosted inside a single PTWrappedModel; cross-track sync is a
(local-sum across K) + (NCCL all-reduce across ranks) — no oversubscribed
NCCL communicators (NCCL ≥ 2.19 rejects those).

Single node, 8 GPUs, n_tracks=16  →  K=2 per rank:

    torchrun --standalone --nproc-per-node=8 scripts/train_qwen3_5_9b.py \
        --hf-model /home2/jtr020/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
        --tracks-dir /path/to/pt_tracks \
        --max-steps 1000 \
        --seq-len 4096 \
        --batch-size 1 \
        --lr 3e-5

Two nodes × 8 GPUs, n_tracks=16  →  K=1 per rank:

    torchrun --nnodes=2 --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR \
        --master-port=$MASTER_PORT --nproc-per-node=8 \
        scripts/train_qwen3_5_9b.py ...

This script is intentionally minimal: distributed init, build groups, load
sliced student + dense teacher, wrap teacher with FSDP2, run distillation
steps, save periodically.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.adapters import get_adapter_for_config
from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.data import CalibrationDataConfig, PackedTokenStream
from pt_converter.train.distill import DistillConfig, distill_step, validate_step
from pt_converter.train.sync_grads import (
    assert_replicated_consistent,
    build_replication_plan,
    compute_global_grad_norm,
    sync_replicated_grads,
)
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
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Global, replication-deduplicated grad-norm clip. <=0 disables clipping.")
    p.add_argument("--save-every", type=int, default=0,
                   help="0 disables step checkpoints. Use --eval-every to drive best/ saves instead.")
    p.add_argument("--save-final", action="store_true",
                   help="Write final/ when training exits (loop done or early-stopped).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lambda-block", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=0.5)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--kl-ce-chunk-size", type=int, default=128,
                   help="Seq-chunk size for the KL+CE pass; caps the per-chunk fp32 "
                        "(B, chunk, V) expansion on the lm_head-owning rank.")
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
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for torch / cuda / python / numpy RNGs. Same on every rank "
                        "(no per-rank randomness in this pipeline). Restored from a "
                        "resumed train_state if present.")
    p.add_argument("--rank0-tracks", type=int, default=None,
                   help="Number of student tracks to host on rank 0 (which also owns "
                        "embed_tokens + lm_head). Default = n_tracks // world_size "
                        "(uniform). Lower values shift the imbalance off rank 0; the "
                        "displaced tracks are distributed as evenly as possible across "
                        "ranks 1..world_size-1, with any remainder going to the earliest "
                        "peer ranks.")
    args = p.parse_args()

    # ----- Distributed init -----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_count = torch.cuda.device_count()
    if local_rank >= gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} >= visible GPU count {gpu_count}. "
            f"Launch with --nproc-per-node <= #GPUs (one rank per GPU); "
            f"K = n_tracks / world_size tracks are hosted per rank."
        )
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    # ----- RNG seed (same on every rank; no per-rank randomness in this pipeline) -----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ----- Load resume training state (if any). Done early so the optimizer can
    # restore from it after construction. The HF streaming dataset has no state
    # of its own — data position is *not* resumed, so loss curves after resume
    # won't be bit-identical to an uninterrupted run.
    resume_state = None
    if args.resume_from:
        train_state_path = Path(args.resume_from) / f"train_state_rank{rank}.pt"
        if train_state_path.exists():
            resume_state = torch.load(
                str(train_state_path),
                map_location=f"cuda:{local_rank}",
                weights_only=False,
            )
            _log(rank, f"[init] resuming training state from {train_state_path}")
        else:
            _log(rank, f"[init] WARNING: --resume-from set but no {train_state_path.name}; optimizer/step will start fresh")

    # ----- Load manifest and groups -----
    manifest = load_manifest(args.tracks_dir)
    tracks_per_rank_list = None
    if args.rank0_tracks is not None:
        world_size = dist.get_world_size()
        k0 = args.rank0_tracks
        remaining = manifest.n_tracks - k0
        peers = world_size - 1
        if k0 <= 0 or peers <= 0 or remaining < peers:
            raise ValueError(
                f"--rank0-tracks={k0} invalid for n_tracks={manifest.n_tracks}, "
                f"world_size={world_size}: each peer must get at least one track."
            )
        base, extra = divmod(remaining, peers)
        # Earliest peers absorb the remainder so the layout is deterministic.
        tracks_per_rank_list = [k0] + [base + (1 if i < extra else 0) for i in range(peers)]
    layout = build_groups(n_tracks=manifest.n_tracks, tracks_per_rank_list=tracks_per_rank_list)
    _log(
        rank,
        f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
        f"K={layout.tracks_per_rank} D={manifest.sync_block_depth}",
    )
    if tracks_per_rank_list is not None:
        _log(rank, f"[init] non-uniform layout K_per_rank={tracks_per_rank_list}")
    _log(rank, f"[init] rank={rank} local_track_ids={layout.local_track_ids}")

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

    # ----- Build per-rank student and load its K track shards -----
    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=layout.track_group,
    )
    state_src = args.resume_from if args.resume_from else args.tracks_dir
    if args.resume_from:
        _log(rank, f"[init] resuming track state from {args.resume_from}")
    track_states = {tid: load_track(state_src, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)

    # ----- Replicated-parameter gradient sync -----
    # Norms (Replicated) and KV projections (KVReplicatedColwise when
    # n_tracks > num_kv_heads) hold bit-identical copies across multiple
    # tracks. Without intervention each copy drifts under its own gradient.
    # We average gradients within each replication group every step so the
    # copies remain identical forever.
    adapter = get_adapter_for_config(cfg.text_config)
    replication_plan = build_replication_plan(
        student, adapter=adapter, text_cfg=cfg.text_config, layout=layout
    )
    assert_replicated_consistent(replication_plan)
    _log(rank, f"[init] replication plan: {len(replication_plan)} groups synced per step")

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
    # foreach=False uses the single-tensor Adam path; lower peak memory than
    # the default _multi_tensor_adam (which stacks intermediates and OOMs on
    # the rank that hosts embed_tokens + lm_head + K student tracks).
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        foreach=False,
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

    # ----- Restore optimizer / scheduler / RNG from resume_state, if any. -----
    if resume_state is not None:
        optim.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["torch_rng"].cpu())
        torch.cuda.set_rng_state(resume_state["cuda_rng"].cpu())

    distill_cfg = DistillConfig(
        sync_layer_indices=tuple(manifest.sync_layer_indices),
        lambda_block=args.lambda_block,
        lambda_kl=args.lambda_kl,
        lambda_ce=args.lambda_ce,
        kl_temperature=args.kl_temperature,
        kl_ce_chunk_size=args.kl_ce_chunk_size,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str):
        """Per-rank K-track save into <out-dir>/<name>/.

        Each rank writes one ``track_{tid}.safetensors`` per local track in
        the slicer's on-disk key format (``embed_tokens.weight``,
        ``norm.weight``, ``layers.{i}.*``, plus ``lm_head.weight`` on the
        owner). Together the world covers all n_tracks shards.
        """
        ck_dir = Path(args.out_dir) / name
        if rank == 0:
            ck_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(args.tracks_dir) / "manifest.json",
                ck_dir / "manifest.json",
            )
        dist.barrier()
        full_sd = student.state_dict()
        for k, tid in enumerate(layout.local_track_ids):
            prefix = f"text_models.{k}."
            track_state: dict[str, torch.Tensor] = {}
            for key, val in full_sd.items():
                if key.startswith(prefix):
                    sub = key[len(prefix):]
                    track_state[sub] = val.detach().contiguous().clone().cpu()
            if tid == PTWrappedModel.LM_HEAD_OWNER_TRACK and "lm_head.weight" in full_sd:
                track_state["lm_head.weight"] = (
                    full_sd["lm_head.weight"].detach().contiguous().clone().cpu()
                )
            save_safetensors(track_state, str(ck_dir / f"track_{tid}.safetensors"))

        # Per-rank training state: optimizer (per-rank because each rank holds
        # different params), plus the shared step / scheduler / best-val / RNG
        # (saved redundantly across ranks, simpler than coordinating one writer).
        train_state = {
            "optimizer": optim.state_dict(),
            "step": step,
            "best_val_ce": best_val_ce,
            "windows_since_best": windows_since_best,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        if scheduler is not None:
            train_state["scheduler"] = scheduler.state_dict()
        torch.save(train_state, str(ck_dir / f"train_state_rank{rank}.pt"))
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

    if resume_state is not None:
        step = int(resume_state["step"])
        best_val_ce = float(resume_state["best_val_ce"])
        windows_since_best = int(resume_state["windows_since_best"])
        _log(rank, f"[init] resumed at step={step} best_val_ce={best_val_ce:.4f} "
                   f"windows_since_best={windows_since_best}")
    else:
        step = 0
        best_val_ce = float("inf")
        windows_since_best = 0
    stop_now = False

    t0 = time.time()
    student.train()
    for batch in loader:
        if step >= args.max_steps or stop_now:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        optim.zero_grad(set_to_none=True)
        # distill_step now backwards each block immediately to bound peak
        # memory; gradients accumulate on student params internally.
        losses = distill_step(student, teacher, batch, distill_cfg)
        sync_replicated_grads(replication_plan)

        # Global, replication-deduplicated grad norm. Same scalar on every rank,
        # so the clip coefficient derived from it scales every replicated copy
        # by the same factor — bit-equality survives the clip. A non-finite norm
        # means NaN/Inf reached the grads (loss spike, bf16 overflow); skip the
        # step rather than poison the weights with a corrupted update.
        total_norm = compute_global_grad_norm(student, replication_plan)
        if not torch.isfinite(total_norm):
            _log(
                rank,
                f"[step {step}] SKIP non-finite grad "
                f"(loss={losses['total'].item()}, grad_norm=nan/inf)",
            )
            optim.zero_grad(set_to_none=True)
            step += 1
            continue

        if args.max_grad_norm > 0:
            clip_coef = (args.max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)
            for p in student.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_coef)

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
                f"grad_norm={total_norm.item():.3e} "
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
