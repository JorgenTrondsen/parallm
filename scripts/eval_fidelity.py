"""torchrun entry point: measure how closely the trained PT student matches the teacher.

Loads a per-rank checkpoint (``track_*.safetensors`` + ``manifest.json``),
rebuilds the sharded teacher exactly as the training script does, and runs
both on a held-out stream. Reports:

  - KL(teacher‖student) (forward) and KL(student‖teacher) (reverse)
  - Top-1 / top-5 prediction agreement and top-5 set IoU
  - Per-sync-boundary hidden-state MSE
  - Student vs teacher perplexity (and gap)

The launch shape mirrors ``scripts/train_qwen3_5_9b.py``. The layout flags
(``--rank0-tracks``) must match what training used so the per-rank track
ownership lines up with the checkpoint's per-rank ``track_*.safetensors``.

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
from pt_converter.train.data import CalibrationDataConfig, PackedTokenStream
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _compute_tracks_per_rank_list(n_tracks: int, world_size: int, rank0_tracks: int) -> list[int]:
    """Same asymmetric-layout derivation as train_qwen3_5_9b.py.

    Replicated here rather than imported because the train script is an entry
    point, not a library module. Keep this in sync with the train script's
    body if the layout policy changes.
    """
    remaining = n_tracks - rank0_tracks
    peers = world_size - 1
    if rank0_tracks <= 0 or peers <= 0 or remaining < peers:
        raise ValueError(
            f"--rank0-tracks={rank0_tracks} invalid for n_tracks={n_tracks}, "
            f"world_size={world_size}: each peer must get at least one track."
        )
    base, extra = divmod(remaining, peers)
    return [rank0_tracks] + [base + (1 if i < extra else 0) for i in range(peers)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path (same as training)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir with track_*.safetensors and manifest.json "
                        "(e.g. ./pt_train_out/best, ./pt_train_out/final, or a step_N dir).")
    p.add_argument("--dataset-name", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="validation")
    p.add_argument("--num-batches", type=int, default=200,
                   help="Number of packed sequences to evaluate.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=128,
                   help="Seq-chunk size for the vocab-wide fp32 expansion; matches "
                        "--kl-ce-chunk-size in training.")
    p.add_argument("--rank0-tracks", type=int, default=None,
                   help="MUST match the layout the checkpoint was trained with. "
                        "Default = n_tracks // world_size (uniform).")
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
    world_size = dist.get_world_size()

    # ----- Manifest + layout. Same policy as train_qwen3_5_9b.py so the
    # per-rank track ownership matches the on-disk track_*.safetensors. -----
    manifest = load_manifest(args.checkpoint_dir)
    tracks_per_rank_list = None
    if args.rank0_tracks is not None:
        tracks_per_rank_list = _compute_tracks_per_rank_list(
            manifest.n_tracks, world_size, args.rank0_tracks
        )
    layout = build_groups(n_tracks=manifest.n_tracks, tracks_per_rank_list=tracks_per_rank_list)
    _log(
        rank,
        f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
        f"K={layout.tracks_per_rank} D={manifest.sync_block_depth}",
    )
    if tracks_per_rank_list is not None:
        _log(rank, f"[init] non-uniform layout K_per_rank={tracks_per_rank_list}")
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
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    ds = PackedTokenStream(
        tok,
        CalibrationDataConfig(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            seq_len=args.seq_len,
        ),
    )
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
              f"({args.dataset_name}/{args.dataset_config}/{args.split}, "
              f"seq_len={args.seq_len}) =====")
        print(f"  perplexity   : student={s_ppl:.4f}  teacher={t_ppl:.4f}  gap={s_ppl - t_ppl:+.4f}")
        print(f"  nll          : student={s_nll:.4f}  teacher={t_nll:.4f}  delta={s_nll - t_nll:+.4f}")
        print(f"  KL forward (t‖s) = {kl_fwd:.4f} nats")
        print(f"  KL reverse (s‖t) = {kl_rev:.4f} nats")
        print(f"  top1_agree   = {sums['top1_agree'].item():.4f}")
        print(f"  top5_agree   = {sums['top5_agree'].item():.4f}  (teacher top-1 ∈ student top-5)")
        print(f"  top5_set_iou = {sums['top5_set_iou'].item():.4f}")
        print(f"  per-sync-boundary block_mse:")
        for layer_idx in sync_indices:
            print(f"    layer {layer_idx:3d}: {sums[f'block_mse_l{layer_idx}'].item():.6e}")
        print()

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
