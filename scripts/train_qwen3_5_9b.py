"""torchrun entrypoint for PT distillation fine-tuning on Qwen3.5-9B.

Launch: one rank per visible GPU. The K = n_tracks / world_size tracks each
rank owns are hosted inside a single PTWrappedModel; cross-track sync is a
(local-sum across K) + (NCCL all-reduce across ranks) — no oversubscribed
NCCL communicators (NCCL ≥ 2.19 rejects those).

Single node, 8 GPUs, n_tracks=16  →  K=2 per rank:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nproc-per-node=8 scripts/train_qwen3_5_9b.py \
        --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
        --tracks-dir convert_out/qwen3_5_9b_n16_d4 \
        --activation-checkpoint \
        --max-steps 1000 \
        --seq-len 4096 \
        --batch-size 1 \
        --lr 3e-5

    Vocab-parallel (default) shards embed_tokens + lm_head + KL/CE over all
    ranks, so the layout is uniform K = n_tracks // world_size (no straggler) and
    no rank specially carries the embed/lm_head. --no-vocab-parallel selects the
    legacy track-0-owner path (full embed/lm_head on rank 0; memory-heavy at
    n_tracks=16 on 40 GB).

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
import re
import shutil
import time
from dataclasses import replace
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
from pt_converter.model.vocab_parallel import vocab_range
from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)
from pt_converter.train.distill import (
    DistillConfig,
    _block_ranges,
    adaptive_weights_from_relmse,
    distill_step,
    student_forcing_schedule,
    validate_step,
)
from pt_converter.train.sync_grads import (
    assert_replicated_consistent,
    build_replication_plan,
    compute_global_grad_norm,
    sync_replicated_grads,
)
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track, load_track_keys
from pt_converter.utils.mem_report import (
    component_breakdown,
    device_mem,
    format_report,
    log_stage,
    print_all_ranks,
)


def _log(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


PROBE_ALPHAS = (1.0, 0.5, 0.25, 0.0)


def run_fr_grad_probe(student, teacher, loader, distill_cfg, manifest, args, rank):
    """Per-window grad-norm probe of the free-running MSE term (--fr-grad-probe).

    Quantifies the unrolled-gradient amplification behind the fr divergence: the
    SAME N microbatches run through an fr-ONLY distill_step (lambda_block/kl/ce
    forced to 0 — the block loop still runs, since it harvests the fr teacher
    targets, but backwards nothing) at each alpha in PROBE_ALPHAS, and the
    resulting per-window parameter grad L2 norms are tabulated. At alpha=1.0
    (legacy full unroll) shallow windows receive deep-tap gradients amplified by
    the product of downstream window Jacobians; damping flattens the profile
    geometrically (~alpha^k per crossing). Replicated norm copies are counted
    once per track on every rank — identically at every alpha, so cross-alpha
    ratios are exact. The per-alpha mean fr_mse footer must be IDENTICAL across
    alphas: the damping is value-exact, so this doubles as a forward-equality
    check in the production dtype.
    """
    n_batches = args.fr_grad_probe
    device = torch.cuda.current_device()
    data_iter = iter(loader)
    batches = []
    for _ in range(n_batches):
        b = next(data_iter)
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        if b["input_ids"].ndim == 1:
            b = {k: v.unsqueeze(0) for k, v in b.items()}
        batches.append(b)

    # Param-name → group index: windows by layer id, then embed / lm_head;
    # whatever remains (per-track final norm) lands in the trailing bucket.
    windows = _block_ranges(manifest.num_layers, tuple(manifest.sync_layer_indices))
    win_of_layer = {l: w for w, (s, e) in enumerate(windows) for l in range(s, e + 1)}
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    group_names = [f"w{w:<2d} L{s}-{e}" for w, (s, e) in enumerate(windows)]
    group_names += ["embed", "lm_head", "final_norm"]
    n_groups = len(group_names)

    def group_of(name: str) -> int:
        m = layer_re.search(name)
        if m:
            return win_of_layer[int(m.group(1))]
        if "embed" in name:           # embed_tokens / vp_embed
            return n_groups - 3
        if "lm_head" in name:
            return n_groups - 2
        return n_groups - 1           # per-track final norm (+ any stragglers)

    probe_cfg = replace(
        distill_cfg,
        lambda_block=0.0, lambda_kl=0.0, lambda_ce=0.0,
        free_running_mse=True, lambda_free_running=1.0,
    )
    norms: dict[float, torch.Tensor] = {}
    fr_means: dict[float, float] = {}
    student.train()
    for alpha in PROBE_ALPHAS:
        student.zero_grad(set_to_none=True)
        cfg_a = replace(probe_cfg, fr_grad_alpha=alpha)
        fr_sum = 0.0
        for i, b in enumerate(batches):
            losses = distill_step(
                student, teacher, b, cfg_a,
                student_forcing_prob=0.0,
                forcing_seed=(args.seed, 0, i),
                loss_scale=1.0 / n_batches,
                compute_klce_metrics=False,
            )
            fr_sum += losses["fr_mse"].item()
        sq = torch.zeros(n_groups, device=device, dtype=torch.float64)
        for name, p in student.named_parameters():
            if p.grad is not None:
                sq[group_of(name)] += p.grad.detach().double().pow(2).sum()
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        norms[alpha] = sq.sqrt()
        fr_means[alpha] = fr_sum / n_batches
        _log(rank, f"[fr-probe] alpha={alpha:g} done (mean fr_mse={fr_means[alpha]:.6f})")
    student.zero_grad(set_to_none=True)

    if rank == 0:
        header = "  ".join(f"a={a:<8g}" for a in PROBE_ALPHAS)
        print(f"[fr-probe] fr-only grad L2 norm per window over {n_batches} "
              f"microbatch(es), taps={probe_cfg.free_running_taps}", flush=True)
        print(f"[fr-probe] {'group':12s} {header}", flush=True)
        for gi, gname in enumerate(group_names):
            vals = "  ".join(f"{norms[a][gi].item():10.3e}" for a in PROBE_ALPHAS)
            print(f"[fr-probe] {gname:12s} {vals}", flush=True)
        footer = ", ".join(f"a={a:g}: {fr_means[a]:.6f}" for a in PROBE_ALPHAS)
        print(f"[fr-probe] mean fr_mse per alpha (must be identical — damping is "
              f"value-exact): {footer}", flush=True)


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
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over this many microbatches before each "
                        "optimizer step (effective batch = --batch-size × this × world). "
                        "Each microbatch's losses are scaled 1/grad-accum so the grads "
                        "AVERAGE rather than sum, giving a less noisy small-batch block-MSE/"
                        "KL signal at fixed memory. --max-steps / --eval-every / the LR "
                        "schedule all count OPTIMIZER steps. 1 (default) = no accumulation, "
                        "bit-identical to the legacy single-microbatch loop.")
    p.add_argument("--lr", type=float, default=3e-5)
    # ----- Training data (streamed + packed; see train/data.py) -----
    p.add_argument("--data-preset", default=DEFAULT_PRESET, choices=preset_names(),
                   help="Training-data mixture. 'qwen-mix' (default) approximates Qwen3.5's "
                        "code/math-heavy distribution (DKYoon/SlimPajama-6B 0.70 + "
                        "open-web-math 0.15 + bigcode/the-stack-dedup 0.15), so the KL/CE "
                        "recovery matches the teacher on code/math, not just Wikipedia. NOTE: "
                        "the-stack-dedup is GATED — accept its terms on the HF dataset page and "
                        "export HF_TOKEN, or use the ungated 'slimpajama'/'wikitext' presets. "
                        "All sources are parquet-native (script datasets unsupported) and stream "
                        "— nominal dataset size is irrelevant (only consumed tokens are fetched). "
                        "Overridden by any --data-source. Point HF_HOME at scratch to save quota.")
    p.add_argument("--data-source", action="append", default=None, metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]",
                   help="Add a custom training source (repeatable). Empty CONFIG/KEY fields keep "
                        "defaults, e.g. 'DKYoon/SlimPajama-6B::text:0.7'. If any --data-source "
                        "is given it REPLACES --data-preset.")
    p.add_argument("--val-dataset-name", default=None,
                   help="Held-out validation source. DEFAULT (unset): a HELD-OUT slice of the "
                        "TRAINING mixture — val mirrors the train sources (same seed) but reads the "
                        "front while the train stream skips the first --val-holdout-docs documents, "
                        "so the two are disjoint and val_kl measures the in-distribution "
                        "generalization gap (directly comparable to the per-step kl/ce). Set this "
                        "to an external dataset (e.g. Salesforce/wikitext) to use a FIXED cross-run "
                        "comparator instead (the legacy default; already disjoint, so no train "
                        "skip). NOTE: leaving the default means val_kl is on the training mixture, "
                        "not comparable to WikiText-103 history.")
    p.add_argument("--val-dataset-config", default="wikitext-103-raw-v1",
                   help="Dataset config for an external --val-dataset-name (ignored in the default "
                        "mirror-the-training-mixture mode).")
    p.add_argument("--val-split", default="validation",
                   help="Split for an external --val-dataset-name (ignored in mirror mode).")
    p.add_argument("--val-text-key", default="text",
                   help="Text column for an external --val-dataset-name (ignored in mirror mode).")
    p.add_argument("--val-holdout-docs", type=int, default=16384,
                   help="Mirror mode only: number of leading mixture documents reserved for the "
                        "held-out val set. The TRAIN stream skips them once at startup (a one-time "
                        "read of that many raw docs) and val reads the front, keeping the two "
                        "disjoint. Must exceed the docs val consumes "
                        "(~val_batches × batch_size × seq_len / tokens-per-doc); the 16384 default "
                        "covers the default eval config down to ~25 tokens/doc — raise it if you "
                        "bump --val-batches/--batch-size on a short-document source. Unused when "
                        "--val-dataset-name is set.")
    p.add_argument("--activation-checkpoint", action="store_true",
                   help="Per-layer activation checkpointing of the full student forward. "
                        "Under vocab-parallel (default) EVERY rank backwards through that "
                        "forward (the KL/CE phase ends in one hidden.backward per rank), so "
                        "this trades the held forward graph — the ~25 GB student_fwd/klce "
                        "peak — for one recompute pass, lowering the peak on every rank. "
                        "(Legacy --no-vocab-parallel: only the lm_head-owner rank backwards, "
                        "so it helps that rank; peers pay no recompute.) Compute ↑, memory ↓ "
                        "— the lever for fitting a larger --batch-size at seq=4096 on 40 GB GPUs.")
    p.add_argument("--checkpoint-granularity", default="window", choices=["window", "layer"],
                   help="Activation-checkpointing granule (only with --activation-checkpoint). "
                        "'window' (default): checkpoint each whole sync window (the D layers "
                        "between boundaries) per track, saving only the SHARED synced window "
                        "input — one (B,T,H) tensor per window instead of the window input "
                        "PLUS every mid-window per-track hidden that per-layer wrapping pins "
                        "(~5 GB less resident at n16/d2 B=5). Math is bit-identical and each "
                        "layer is recomputed exactly once either way; a window's backward "
                        "transiently re-materializes its D·K-layer graph. 'layer': the legacy "
                        "per-layer wrap (smaller recompute transient, more saved tensors). "
                        "At D=1 the two coincide.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile each per-track decoder layer in place "
                        "(default: on; use --no-compile to disable). Inductor fusion of "
                        "the tiny per-track kernels; cuts launch overhead, biggest win on "
                        "the high-K straggler rank. Math unchanged; first steps pay a "
                        "one-time compile warmup.")
    p.add_argument("--vocab-parallel", action=argparse.BooleanOptionalAction, default=True,
                   help="Vocab/tensor-parallel embed_tokens + lm_head + KL/CE across all "
                        "ranks (default: on; --no-vocab-parallel for the legacy track-0-owner "
                        "path). Shards the full-vocab tensors and the KL/CE fp32 softmax over "
                        "the world, balancing memory (rank 0 no longer carries embed+lm_head+"
                        "klce alone) and parallelizing the previously rank-0-serial klce phase. "
                        "On-disk checkpoint format is unchanged (gather-on-save into track 0). "
                        "The layout is uniform K = n_tracks // world_size.")
    p.add_argument("--sync-attention-heads", action=argparse.BooleanOptionalAction, default=False,
                   help="Keep the full-attention head params (k_proj, v_proj, q_norm, "
                        "k_norm) bit-identical across the tracks of each kv-group, the "
                        "legacy behaviour. Default OFF: those copies DIVERGE per track "
                        "(each track gets its own KV head — GQA → per-track MHA), a "
                        "capacity superset that is free on memory and drops the per-step "
                        "kv-group all-reduces. Pass --sync-attention-heads to A/B back to "
                        "the synced/dense-GQA-equivalent path. The residual-stream norms "
                        "(input/post layernorm, final norm, linear-attn norm) are always "
                        "kept synced.")
    p.add_argument("--compile-mode", default="default",
                   help="torch.compile mode. 'default' = inductor fusion only "
                        "(recommended). 'reduce-overhead' (CUDA graphs) is "
                        "experimental here — can conflict with activation-checkpoint "
                        "recompute and needs static memory.")
    p.add_argument("--shard-teacher-fwd", action=argparse.BooleanOptionalAction, default=True,
                   help="Batch-shard the frozen-teacher forward across the world (default: "
                        "on). Every rank holds the IDENTICAL batch, so the legacy path "
                        "computes the same full teacher forward world_size times; sharded, "
                        "each rank forwards only ceil(B/world) rows and one all-gather per "
                        "captured layer rebuilds the full-batch hiddens (bit-identical on "
                        "every rank — training math unchanged). Teacher compute drops "
                        "~world_size-fold; at seq=4096 the gathers are a small fraction of "
                        "the saving. --no-shard-teacher-fwd restores the legacy redundant "
                        "path (A/B / fallback). Rows pad to ceil(B/world)*world, so a B not "
                        "divisible by world wastes the padded slots (B=5 on 8 ranks: 3 "
                        "duplicate rows) — still ~B/world-fold cheaper.")
    p.add_argument("--compile-teacher", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile each frozen-teacher decoder layer (in-place, before "
                        "fully_shard), inference-only. Separate from --compile so it can be "
                        "disabled alone if it conflicts with FSDP2 or the sync-boundary "
                        "forward hooks; --no-compile-teacher falls back to eager teacher.")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Global, replication-deduplicated grad-norm clip. <=0 disables clipping.")
    p.add_argument("--save-every", type=int, default=0,
                   help="Overwrite the rolling latest/ checkpoint every N steps (single dir, "
                        "~52 GB at n=16). 0 disables it; use --eval-every to drive best/ saves instead.")
    p.add_argument("--save-final", action="store_true",
                   help="Write final/ when training exits (loop done or early-stopped).")
    p.add_argument("--best-name", default="best",
                   help="Directory name (under --out-dir) for the best-val_kl checkpoint. "
                        "Default 'best'. Set e.g. 'best_fr' on a warm-start finishing run "
                        "so its improvements don't overwrite the checkpoint it resumed from.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--profile", action="store_true",
                   help="Per-phase CUDA-synced wall-clock breakdown of each distill "
                        "step (teacher_fwd / setup / block_loop / student_fwd / klce / "
                        "data_wait / other), logged per step and as a mean. Cheap "
                        "(just cuda.synchronize + perf_counter); absolute ms stay "
                        "representative of a real run. No effect on training math; "
                        "off = zero overhead.")
    p.add_argument("--profile-trace", action="store_true",
                   help="Additionally capture a torch.profiler kernel trace over a "
                        "short window (Chrome trace + key_averages table on rank 0). "
                        "Implies --profile. NOTE: CUDA activity tracing inflates the "
                        "absolute per-phase ms during its active window — use it for "
                        "kernel-level attribution, not for wall-clock budgeting.")
    p.add_argument("--mem-report", action="store_true",
                   help="Per-rank breakdown of what occupies GPU memory: lifecycle "
                        "deltas at each init stage (baseline / teacher / student / "
                        "optimizer), a measured resident-component table (teacher shard, "
                        "student params, grads, AdamW state — actual bytes/dtype, not "
                        "assumed), the transient per-phase activation peak inside one "
                        "distill step, and the allocated-vs-reserved-vs-device gap "
                        "(CUDA ctx + NCCL). Cheap; off = zero overhead. NOTE: if combined "
                        "with --profile, the final --profile peak reflects steps after "
                        "the --mem-report-step capture (which resets peak stats).")
    p.add_argument("--mem-report-step", type=int, default=3,
                   help="Step on which --mem-report captures per-phase memory and the "
                        "resident-component breakdown. Default 3 (post compile/cuDNN "
                        "warmup, so peaks are steady-state); clamped to < --max-steps, "
                        "falling back to the last step on tiny runs.")
    p.add_argument("--lambda-block", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=0.5)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--student-forcing-prob", type=float, default=0.0,
                   help="Scheduled-sampling probability of feeding a block the student's "
                        "OWN synced hidden (instead of the teacher's) as input, while the MSE "
                        "target stays the teacher hidden. Trains each block to correct from a "
                        "drifted student input toward the teacher output, closing the "
                        "exposure-bias gap between teacher-forced training and free-running "
                        "inference (the cause of the depth-exploding block_mse). 0.0 (default) "
                        "= legacy fully-teacher-forced path; recommended ~0.5. The per-block "
                        "decision is deterministic across ranks (seeded by --seed + step).")
    p.add_argument("--student-forcing-warmup", type=int, default=0,
                   help="Steps over which --student-forcing-prob ramps linearly 0 → prob "
                        "(scheduled-sampling anneal: start teacher-forced, increase student-"
                        "forcing). 0 = constant at --student-forcing-prob from step 0. "
                        "Recommended ~half of --max-steps.")
    p.add_argument("--student-forcing-schedule", default="hold", choices=["hold", "cosine-full"],
                   help="Shape of the student-forcing probability over the run. 'hold' "
                        "(default, legacy): linear ramp 0 → --student-forcing-prob over "
                        "--student-forcing-warmup steps, then HOLD. 'cosine-full': a "
                        "free-running CURRICULUM — cosine ramp 0 → --student-forcing-prob "
                        "across the WHOLE run, approaching the high-forcing regime gently and "
                        "reaching it only near the end. Closes the train(teacher-forced)/"
                        "eval(free-running) gap that drives the depth-exploding block_mse, "
                        "without the unstable long tail of holding at a high prob. Recommended "
                        "with --student-forcing-prob ~0.9 (--student-forcing-warmup is ignored "
                        "in this shape).")
    p.add_argument("--student-forcing-power", type=float, default=1.0,
                   help="Steepness of the 'cosine-full' curriculum (ignored for 'hold'; must be "
                        "> 0). The gap-to-target 0.5*(1+cos(pi*frac)) is raised to this power and "
                        "sf_p = prob*(1 - gap**power): 1.0 (default) is the plain cosine ramp "
                        "(bit-identical to the legacy schedule); >1 closes the gap faster, "
                        "reaching the high-forcing regime EARLIER (the lever on long runs, where "
                        "the plain cosine only reaches high forcing near the end — e.g. at 25%% of "
                        "the run power=3 gives sf_p≈0.34*prob vs 0.15*prob); <1 reaches it later.")
    p.add_argument("--normalize-block-mse", action=argparse.BooleanOptionalAction, default=False,
                   help="Relative (scale-free) block MSE Σ(s−t)²/Σt² per block instead of the "
                        "raw masked mean. The residual-stream norm grows with depth, so the raw "
                        "MSE lets deep layers dominate the gradient (and spike the grad norm); "
                        "normalizing makes each block O(1) so every depth contributes comparably. "
                        "Default off (bit-identical legacy loss). Rescales the block term, so "
                        "--lambda-block becomes a relative weight (1.0 is a fine start).")
    p.add_argument("--block-mse-clamp", type=float, default=10.0,
                   help="Cap the normalized per-block relative MSE at this value (only active "
                        "with --normalize-block-mse). Under student forcing a block can be fed "
                        "the student's own drifted hidden, occasionally blowing the ratio up to "
                        "100+ on one batch — a spike that inflates the gradient and trips the "
                        "--max-grad-norm clip, throttling the whole step. Clamping saturates the "
                        "gradient above the cap (outlier rejection); normal ratios (~0.5–1.5) are "
                        "untouched. <=0 disables the clamp.")
    p.add_argument("--intra-window-mse", action="store_true",
                   help="Supervise EVERY layer inside each sync window, not just the boundary. At "
                        "each within-window layer the synced reconstruction is MSE'd against the "
                        "teacher's hidden at that depth (the forward still feeds each track its "
                        "PARTIAL residual — the taps are sync-for-loss-only), pinning the "
                        "within-window layers to the teacher trajectory. Targets the uniform-D≥2 "
                        "stall, where the mid-window (esp. full-attention) layers run on partial "
                        "residuals. Hooks the teacher at every layer (more captures). Per-window "
                        "loss is averaged over its layers, so --lambda-block keeps its meaning and "
                        "D=1 is bit-identical to the boundary-only path. Off by default.")
    p.add_argument("--free-running-mse", action="store_true",
                   help="Free-running feature matching: relative-MSE the END-TO-END student "
                        "forward's synced hiddens (the same full forward the KL/CE pass uses — "
                        "the student runs on its OWN hiddens throughout) against the teacher "
                        "hiddens at the sync boundaries, with gradients through the WHOLE "
                        "forward. Unlike the block loop (detached at every boundary) this "
                        "trains multi-window error compounding directly — the deep free-running "
                        "relMSE plateau the block loop cannot see. Reuses the already-paid "
                        "student_fwd pass (shares its single backward with KL/CE), so the "
                        "marginal cost is ~zero; retaining the boundary teacher hiddens costs "
                        "~2.5 GB/rank at B=5 D=2 (halve with --free-running-taps deep-half).")
    p.add_argument("--lambda-free-running", type=float, default=1.0,
                   help="Weight on the free-running feature-matching term (multiplied by the "
                        "--free-running-schedule scale). The term is the relative MSE (mean "
                        "over taps, clamped by --block-mse-clamp), so 1.0 is comparable to a "
                        "normalized --lambda-block.")
    p.add_argument("--free-running-schedule", default="constant", choices=["constant", "cosine-full"],
                   help="Per-step scale on --lambda-free-running. 'constant' (default): full "
                        "weight from step 0 (right for warm-started finishing runs). "
                        "'cosine-full': cosine ramp 0 → 1 across the whole run — for "
                        "from-scratch runs, where the early free-running hiddens are garbage "
                        "and the term should phase in as the blocks converge (mirrors the "
                        "--student-forcing-schedule cosine-full curriculum).")
    p.add_argument("--free-running-taps", default="all", choices=["all", "deep-half"],
                   help="Which sync boundaries the free-running MSE supervises. 'deep-half': "
                        "only the deeper half — where the free-running error concentrates — "
                        "halving the retained-teacher-hidden memory.")
    p.add_argument("--fr-grad-alpha", type=float, default=1.0,
                   help="Per-boundary gradient damping of the free-running unroll. During "
                        "the full student forward, the hidden continuing past each sync "
                        "boundary becomes h.detach() + alpha*(h - h.detach()): forward value "
                        "EXACTLY unchanged, gradient across the boundary scaled by alpha, so "
                        "a tap j's gradient into window w shrinks alpha^(j-w). This bounds "
                        "the through-depth Jacobian-product amplification (and the gain-"
                        "raising feedback it rewards) that makes the raw full-unroll term "
                        "diverge (grad-norm creep -> val_kl regression). 1.0 (default) = "
                        "legacy full unroll; 0.0 = hard truncation (each tap trains only its "
                        "own window on the true free-running input — exact DAgger-style "
                        "supervision); ~0.5 keeps short-range cross-window coordination. "
                        "alpha<1 requires --lambda-kl 0 --lambda-ce 0 (the KL/CE backward "
                        "shares the damped graph).")
    p.add_argument("--fr-grad-probe", type=int, default=0, metavar="N",
                   help=">0: instead of training, run the free-running gradient probe after "
                        "init — the SAME N microbatches through an fr-only distill_step at "
                        "each alpha in {1.0, 0.5, 0.25, 0.0}, reporting per-window parameter "
                        "grad norms (rank-0 table) — then exit. Shows the unrolled-gradient "
                        "amplification directly: at alpha=1 the shallow windows' fr-grad "
                        "norms are inflated by the downstream Jacobian products; damping "
                        "flattens the profile geometrically. The printed per-alpha mean "
                        "fr_mse must be IDENTICAL across alphas (the damping is value-exact).")
    p.add_argument("--adaptive-layer-weight", action=argparse.BooleanOptionalAction, default=False,
                   help="Adaptively weight each supervised tap's block-MSE by its OWN running "
                        "relative error. A per-tap EMA of the relative MSE Σ(s−t)²/Σt² is "
                        "maintained; the per-step weight ∝ EMA**power, mean-1 normalized over the "
                        "taps, so gradient budget flows to wherever the student is CURRENTLY worst "
                        "(which the relative metric shows need not be monotone in depth) without "
                        "changing the total block-loss magnitude (lambda_block keeps its meaning). "
                        "The relMSE is read off the SYNCED hidden (identical on every rank), so the "
                        "EMA — and thus the weights — stay in lock-step across ranks. Off by "
                        "default (uniform weights). Pure loss-side: no change to the sync schedule "
                        "/ communication.")
    p.add_argument("--adaptive-layer-weight-ema", type=float, default=0.9,
                   help="EMA decay for the per-tap relative-MSE estimate that drives "
                        "--adaptive-layer-weight (higher = smoother / slower to react).")
    p.add_argument("--adaptive-layer-weight-power", type=float, default=1.0,
                   help="Exponent on the EMA relative MSE before mean-1 normalization "
                        "(--adaptive-layer-weight). >1 sharpens the tilt toward the worst taps, "
                        "<1 softens it. 1.0 = weight directly proportional to relative error.")
    p.add_argument("--kl-ce-chunk-size", type=int, default=512,
                   help="Seq-chunk size for the KL+CE pass; caps the per-chunk fp32 "
                        "(B, chunk, V/world) expansion. Each chunk runs 3 small all-reduces "
                        "+ one autograd.grad and the klce phase is collective/dispatch-bound, "
                        "so larger = fewer chunks = faster (512 → 8 chunks at seq=4096 vs 32 "
                        "at 128, ~4x fewer collectives) at negligible extra memory. The fp32 "
                        "transient grows ×batch, so keep this moderate at large --batch-size.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="0 disables validation. >0 runs val_batches batches of the held-out val set "
                        "(default: a held-out slice of the training mixture; see --val-dataset-name) "
                        "every N steps.")
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="0 disables early stop. Otherwise stop after N eval windows with no val_kl improvement.")
    p.add_argument("--min-improvement", type=float, default=0.01,
                   help="Min val_kl drop to count as an improvement.")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear LR warmup steps. 0 = no warmup phase. A schedule is built "
                        "when this is >0 OR --cosine-decay is set.")
    p.add_argument("--cosine-decay", action=argparse.BooleanOptionalAction, default=False,
                   help="Cosine-decay the LR from --lr down to --lr*--lr-min-ratio over the "
                        "run (after any warmup). Default off (constant LR). Decoupled from "
                        "--warmup-steps, so you can have warmup, decay, both, or neither. "
                        "Recommended on (with a short --warmup-steps) — the legacy constant-LR "
                        "default produced grad-norm spikes / wasted clipped steps.")
    p.add_argument("--lr-min-ratio", type=float, default=0.1,
                   help="Cosine decay floor as a fraction of --lr (only used with --cosine-decay).")
    p.add_argument("--lr-decay-power", type=float, default=1.0,
                   help="Steepness of the LR cosine decay (only used with --cosine-decay; must "
                        "be > 0). The decaying term 0.5*(1+cos(pi*progress)) is raised to this "
                        "power: 1.0 (default) is the plain cosine (bit-identical to the legacy "
                        "schedule); >1 drops the LR faster EARLY then flattens onto the floor "
                        "sooner (e.g. power=3 reaches the floor by ~70%% of the run); <1 decays "
                        "more gently. Same min_ratio floor at the end regardless of power.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for torch / cuda / python / numpy RNGs. Same on every rank "
                        "(no per-rank randomness in this pipeline). Restored from a "
                        "resumed train_state if present.")
    args = p.parse_args()
    # --profile-trace implies the phase timers; the trace is an add-on.
    if args.profile_trace:
        args.profile = True
    if args.fr_grad_alpha != 1.0:
        if not (0.0 <= args.fr_grad_alpha <= 1.0):
            p.error("--fr-grad-alpha must be in [0, 1]")
        if args.lambda_kl != 0.0 or args.lambda_ce != 0.0:
            p.error("--fr-grad-alpha < 1.0 damps every gradient through the full student "
                    "forward; the KL/CE backward shares that graph and would be silently "
                    "damped too. Run with --lambda-kl 0 --lambda-ce 0.")

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
    world_size = dist.get_world_size()

    # ----- Run configuration -----
    # Dump every parsed arg (rank 0) so a run's full config is self-documented in
    # its log — invaluable when comparing runs after the fact. Sorted for a stable
    # diff between logs.
    _log(rank, f"[config] world_size={world_size}")
    for k in sorted(vars(args)):
        _log(rank, f"[config] {k}={getattr(args, k)}")

    # ----- Memory-report lifecycle hook -----
    # Records a device_mem() snapshot at each init stage (for the final report's
    # lifecycle recap) and logs it per rank. No-op unless --mem-report.
    mem_stages: dict[str, dict[str, float]] = {}

    def mem_stage(label: str):
        if not args.mem_report:
            return
        mem_stages[label] = device_mem()
        log_stage(rank, world_size, label)

    if args.mem_report:
        torch.cuda.reset_peak_memory_stats()
        mem_stage("baseline (post set_device)")

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
    # Uniform layout: K = n_tracks // world_size tracks per rank. (The old
    # asymmetric --rank0-tracks layout existed only to keep embed/lm_head off the
    # busiest rank; vocab-parallel shards those across all ranks, so uniform is
    # now always correct.)
    manifest = load_manifest(args.tracks_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    _log(
        rank,
        f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
        f"K={layout.tracks_per_rank} D={manifest.sync_block_depth}",
    )
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
    teacher_model = teacher_model.to(torch.cuda.current_device())

    # Teacher lm_head: vocab-parallel mode gives it a per-rank vocab-row shard
    # [v_lo, v_hi) (matching the student) so each rank computes only its
    # (B,T,Vs) teacher logit slice — 1/world_size of the matmul, and no resident
    # full (B,T,V) logits tensor. The full head is then dropped. The text_model
    # is FSDP-sharded either way (its forward produces the full hidden states
    # every rank needs). Legacy: full lm_head, FSDP-sharded.
    if args.vocab_parallel:
        v_lo, v_hi = vocab_range(cfg.text_config.vocab_size, world_size, rank)
        full_w = teacher_model.lm_head.weight
        shard = torch.nn.Linear(full_w.shape[1], v_hi - v_lo, bias=False).to(
            device=full_w.device, dtype=full_w.dtype
        )
        with torch.no_grad():
            shard.weight.copy_(full_w[v_lo:v_hi])
        shard.weight.requires_grad = False
        teacher_model.lm_head = shard  # drop the full head (frees ~2 GB if untied)
        teacher_lm_head = shard
        fsdp_lm_head = None
    else:
        teacher_lm_head = teacher_model.lm_head
        fsdp_lm_head = teacher_model.lm_head

    # Intra-window per-layer MSE needs a teacher hidden at EVERY layer (not just
    # the sync boundaries); otherwise hook only the boundaries (the targets the
    # block loop consumes). Extra captures cost ~one (B,T,H) bf16 tensor per layer.
    teacher_hook_indices = (
        list(range(manifest.num_layers))
        if args.intra_window_mse
        else manifest.sync_layer_indices
    )
    # Batch-sharded teacher (default): each rank forwards ceil(B/world) rows and
    # the captured hiddens are all-gathered back to the full batch on every rank
    # — identical results, ~world_size-fold less teacher compute per rank.
    shard_teacher = args.shard_teacher_fwd and world_size > 1
    teacher = HookedTeacher(
        text_model=text_model,
        lm_head=teacher_lm_head,
        sync_layer_indices=teacher_hook_indices,
        shard_group=layout.track_group if shard_teacher else None,
        shard_world_size=world_size if shard_teacher else 1,
        shard_rank=rank,
    )
    wrap_teacher_with_fsdp(
        text_model, fsdp_lm_head,
        compile_layers=args.compile_teacher, compile_mode=args.compile_mode,
    )
    mem_stage("teacher loaded (FSDP-sharded)")

    # ----- Build per-rank student and load its K track shards -----
    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=layout.track_group,
        activation_checkpoint=args.activation_checkpoint,
        checkpoint_granularity=args.checkpoint_granularity,
        compile_layers=args.compile,
        compile_mode=args.compile_mode,
        vocab_parallel=args.vocab_parallel,
        vp_world_size=world_size,
        vp_rank=rank,
    )
    state_src = args.resume_from if args.resume_from else args.tracks_dir
    if args.resume_from:
        _log(rank, f"[init] resuming track state from {args.resume_from}")
    track_states = {tid: load_track(state_src, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    if args.vocab_parallel:
        # Every rank reads the FULL [V,H] embed + lm_head from the track-0 shard
        # (the slicer stores them there, OwnerOnly) and keeps its vocab slice.
        # lm_head may be tied to embed_tokens — fall back to embed if absent.
        ht = load_track_keys(state_src, PTWrappedModel.LM_HEAD_OWNER_TRACK,
                             ["embed_tokens.weight", "lm_head.weight"])
        full_embed = ht["embed_tokens.weight"]
        full_lm_head = ht.get("lm_head.weight", full_embed)
        student.load_vocab_parallel_weights(full_embed, full_lm_head)
        _log(rank, f"[init] vocab-parallel: V={cfg.text_config.vocab_size} "
                   f"shard=[{student.v_lo},{student.v_hi}) per rank")
    student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)
    mem_stage("student loaded (K tracks + vocab-parallel embed/lm_head)")

    # ----- Replicated-parameter gradient sync -----
    # The residual-stream norms (input/post layernorm, final norm, linear-attn
    # norm) hold bit-identical copies across tracks and MUST stay synced — we
    # average their gradients within each replication group every step so they
    # remain identical forever. By default the full-attention head params
    # (k_proj/v_proj/q_norm/k_norm) DIVERGE per track (their specs are sync=False),
    # so they're absent from the plan and train independently; --sync-attention-heads
    # (force_sync) restores the legacy bit-identical / dense-GQA-equivalent path.
    adapter = get_adapter_for_config(cfg.text_config)
    replication_plan = build_replication_plan(
        student, adapter=adapter, text_cfg=cfg.text_config, layout=layout,
        force_sync=args.sync_attention_heads,
    )
    assert_replicated_consistent(replication_plan)
    _log(rank, f"[init] replication plan: {len(replication_plan)} groups synced per step "
               f"(attention-head sync: {'ON' if args.sync_attention_heads else 'OFF — diverging'})")

    # ----- Data -----
    # Training mixture: --data-source (repeatable) overrides --data-preset. The
    # fixed seed keeps the interleave identical across ranks (no DistributedSampler;
    # vocab-parallel needs every rank on the same batch).
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if args.data_source:
        train_sources = [parse_source_spec(s) for s in args.data_source]
    else:
        train_sources = preset_sources(args.data_preset)

    # Validation mode:
    #   * default (--val-dataset-name unset): val is a HELD-OUT slice of the
    #     training mixture. Val mirrors the train sources with the same seed but
    #     reads the front (skip_docs=0) while the train stream skips the first
    #     --val-holdout-docs documents, so the two cover disjoint doc ranges and
    #     val_kl measures the in-distribution generalization gap (comparable to
    #     the per-step kl/ce).
    #   * --val-dataset-name set: val is a FIXED external comparator (e.g. WT-103);
    #     it's a different dataset (already disjoint) so the train stream is NOT
    #     skipped.
    val_mirrors_train = args.eval_every > 0 and args.val_dataset_name is None
    train_skip = args.val_holdout_docs if val_mirrors_train else 0
    train_cfg = CalibrationDataConfig(
        sources=train_sources, seq_len=args.seq_len, seed=args.seed, skip_docs=train_skip,
    )
    _log(rank, "[data] train sources: "
               + ", ".join(f"{s.dataset_name}(w={s.weight:g})" for s in train_sources)
               + (f"  [holding out first {train_skip} docs for val]" if train_skip else ""))
    ds = PackedTokenStream(tok, train_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    val_loader = None
    if args.eval_every > 0:
        if val_mirrors_train:
            # Held-out slice of the SAME mixture: identical sources + seed, reading
            # the front of the sequence the train stream skips past (skip_docs=0).
            # Fresh source copies so the two configs don't alias the same list.
            val_cfg = CalibrationDataConfig(
                sources=[replace(s) for s in train_sources],
                seq_len=args.seq_len, seed=args.seed, skip_docs=0,
            )
            _log(rank, f"[data] val: held-out front slice of the training mixture "
                       f"(first {train_skip} docs, mirror sources)")
        else:
            # External FIXED comparator (the legacy WT-103 path).
            val_cfg = CalibrationDataConfig.single(
                dataset_name=args.val_dataset_name,
                dataset_config=args.val_dataset_config,
                split=args.val_split,
                text_key=args.val_text_key,
                seq_len=args.seq_len,
                seed=args.seed,
            )
            _log(rank, f"[data] val: external comparator {args.val_dataset_name}")
        val_ds = PackedTokenStream(tok, val_cfg)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # ----- Optimizer + LR schedule -----
    # foreach=True (multi-tensor AdamW) fuses the per-param update into grouped
    # ops — much faster than single-tensor (optim.step was ~120 ms/step, ~10% at
    # B=1). Its transient stacks intermediates, but optim.step runs AFTER backward
    # (activations freed, ~14.5 GB resident) so its peak stays well below the
    # student_fwd peak under vocab-parallel — it does NOT raise the step peak. The
    # old foreach=False was to avoid OOM on the LEGACY rank-0-heavy layout (rank 0
    # carried embed+lm_head+K tracks at ~36 GB); vocab-parallel balanced that away.
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        foreach=True,
    )
    mem_stage("optimizer constructed (AdamW state lazy)")

    # A schedule is built when warmup OR cosine decay is requested (decoupled):
    # warmup-only holds at peak LR after the ramp; cosine-only decays from step 0;
    # both ramps then decays; neither leaves the LR constant (scheduler stays None).
    scheduler = None
    if args.warmup_steps > 0 or args.cosine_decay:
        warmup = args.warmup_steps
        total = args.max_steps
        min_ratio = args.lr_min_ratio
        cosine = args.cosine_decay
        power = args.lr_decay_power

        def lr_lambda(s: int) -> float:
            if warmup > 0 and s < warmup:
                return (s + 1) / max(1, warmup)
            if not cosine:
                return 1.0
            progress = (s - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            if power == 1.0:  # exact legacy expression (bit-identical default)
                return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            cos_term = 0.5 * (1 + math.cos(math.pi * progress))  # 1 → 0 over the run
            return min_ratio + (1 - min_ratio) * (cos_term ** power)

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
        normalize_block_mse=args.normalize_block_mse,
        block_mse_clamp=(args.block_mse_clamp if args.block_mse_clamp > 0 else None),
        intra_window_mse=args.intra_window_mse,
        free_running_mse=args.free_running_mse,
        lambda_free_running=args.lambda_free_running,
        free_running_taps=args.free_running_taps,
        fr_grad_alpha=args.fr_grad_alpha,
    )

    # ----- Free-running gradient probe (diagnose-then-exit mode) -----
    if args.fr_grad_probe > 0:
        run_fr_grad_probe(student, teacher, loader, distill_cfg, manifest, args, rank)
        teacher.remove_hooks()
        dist.barrier()
        dist.destroy_process_group()
        return 0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(name: str):
        """Per-rank K-track save into <out-dir>/<name>/.

        Each rank writes one ``track_{tid}.safetensors`` per local track in
        the slicer's on-disk key format (``embed_tokens.weight``,
        ``norm.weight``, ``layers.{i}.*``, plus ``lm_head.weight`` on the
        owner). Together the world covers all n_tracks shards. In vocab-parallel
        mode the sharded embed/lm_head are all-gathered to full ``[V,H]`` and
        written into the track-0 shard, so the on-disk format is unchanged.
        """
        ck_dir = Path(args.out_dir) / name
        if rank == 0:
            ck_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(args.tracks_dir) / "manifest.json",
                ck_dir / "manifest.json",
            )
        dist.barrier()
        # Collective: all-gather the vocab shards → full embed/lm_head on rank 0
        # (None on peers). Must run on every rank before the per-track loop.
        vp_full = student.gather_vocab_parallel_weights() if student.vocab_parallel else None
        full_sd = student.state_dict()
        for k, tid in enumerate(layout.local_track_ids):
            prefix = f"text_models.{k}."
            track_state: dict[str, torch.Tensor] = {}
            for key, val in full_sd.items():
                if key.startswith(prefix):
                    sub = key[len(prefix):]
                    track_state[sub] = val.detach().contiguous().clone().cpu()
            if tid == PTWrappedModel.LM_HEAD_OWNER_TRACK:
                if vp_full is not None:  # vocab-parallel: gathered full tensors (rank 0)
                    embed_full, lm_head_full = vp_full
                    track_state["embed_tokens.weight"] = embed_full.detach().contiguous().cpu()
                    track_state["lm_head.weight"] = lm_head_full.detach().contiguous().cpu()
                elif not student.vocab_parallel and "lm_head.weight" in full_sd:
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
            "best_val_kl": best_val_kl,
            "windows_since_best": windows_since_best,
            "relmse_ema": relmse_ema,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        if scheduler is not None:
            train_state["scheduler"] = scheduler.state_dict()
        torch.save(train_state, str(ck_dir / f"train_state_rank{rank}.pt"))
        dist.barrier()

    def run_eval() -> tuple[float, float]:
        """Run val_batches of validation; return (mean val_kl, mean val_ce).

        val_kl drives early stop / best/; val_ce is logged for observability.
        Legacy: only the owner computes non-zero metrics, so we all_reduce SUM
        (peers contribute 0). Vocab-parallel: validate_step already returns the
        GLOBAL metric on every rank, so summing would over-count — skip the
        reduce (every rank runs the same val batches and gets identical scalars).
        """
        assert val_loader is not None
        student.eval()
        sums = torch.zeros(2, device=torch.cuda.current_device())  # [kl, ce]
        n = 0
        for vb in val_loader:
            if n >= args.val_batches:
                break
            vb = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in vb.items()}
            if vb["input_ids"].ndim == 1:
                vb = {k: v.unsqueeze(0) for k, v in vb.items()}
            out = validate_step(
                student, vb, teacher,
                kl_temperature=args.kl_temperature,
                chunk_size=args.kl_ce_chunk_size,
            )
            sums[0] = sums[0] + out["kl"]
            sums[1] = sums[1] + out["ce"]
            n += 1
        student.train()
        if not student.vocab_parallel:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        sums = sums / max(1, n)
        return sums[0].item(), sums[1].item()

    if resume_state is not None:
        step = int(resume_state["step"])
        # .get for back-compat with pre-val_kl checkpoints: the old metric
        # (best_val_ce) is in different units and not comparable, so we just
        # start val_kl tracking fresh on resume from such a checkpoint.
        best_val_kl = float(resume_state.get("best_val_kl", float("inf")))
        windows_since_best = int(resume_state.get("windows_since_best", 0))
        relmse_ema = resume_state.get("relmse_ema", None)
        _log(rank, f"[init] resumed at step={step} best_val_kl={best_val_kl:.4f} "
                   f"windows_since_best={windows_since_best}")
    else:
        step = 0
        best_val_kl = float("inf")
        windows_since_best = 0
        # Per-tap relative-MSE EMA for --adaptive-layer-weight (None until the
        # first step measures it). Identical on every rank (built from synced relMSE).
        relmse_ema = None
    stop_now = False

    # Step on which --mem-report captures per-phase memory + the resident-component
    # breakdown. Clamp into range so tiny --max-steps runs still capture.
    mem_report_step = min(args.mem_report_step, args.max_steps - 1) if args.mem_report else -1

    # ----- Optional profiling -----
    # --profile: pass a fresh `timings` dict into distill_step each step (CUDA-synced
    # phase wall times) and log a per-step + mean breakdown. Cheap; absolute ms stay
    # representative. --profile-trace adds a torch.profiler kernel trace over a tiny
    # fixed window. CRITICAL: the trace is exported only AFTER the loop and after
    # destroy_process_group(), never inside on_trace_ready mid-loop — the slow rank-0
    # key_averages/export would otherwise leave peers blocked in the next collective
    # until the NCCL watchdog (default 600s) aborts the whole run.
    prof = None
    prof_running = False
    prof_dir = None
    TRACE_START = 3          # steps before this run un-traced (clean warmup)
    TRACE_ACTIVE = 2         # number of steps captured in the trace
    phase_keys = ["teacher_fwd", "setup", "block_loop", "student_fwd", "fr_mse", "klce", "bwd_full", "data_wait"]
    phase_totals: dict[str, float] = {}
    profiled_steps = 0
    if args.profile_trace:
        prof_dir = Path(args.out_dir) / "profile"
        if rank == 0:
            prof_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )

    t0 = time.time()
    student.train()
    data_iter = iter(loader)
    while not (step >= args.max_steps or stop_now):
        # Open the trace window at a fixed post-warmup step (symmetric across ranks).
        if prof is not None and not prof_running and step == TRACE_START:
            prof.start()
            prof_running = True
        timings: dict[str, float] | None = {} if args.profile else None
        # --mem-report captures per-phase memory on a single designated step
        # (passing `mem` resets peak stats each phase, so we never pass it every step).
        do_mem_capture = step == mem_report_step
        step_mem: dict[str, dict[str, float]] | None = {} if do_mem_capture else None

        # Student-forcing probability for this OPTIMIZER step (same across the G
        # microbatches; depends only on `step`, so identical on every rank). The
        # per-block draw inside distill_step is seeded by (seed, step, micro) so
        # every rank makes identical teacher/student choices on each microbatch.
        sf_p = student_forcing_schedule(
            step, args.student_forcing_prob, args.student_forcing_warmup,
            args.max_steps, args.student_forcing_schedule,
            power=args.student_forcing_power,
        )
        # Free-running feature-matching scale for this step ('constant' reuses the
        # warmup-0 "hold" shape ⇒ 1.0 every step). Depends only on `step`, so it
        # is identical on every rank.
        fr_scale = student_forcing_schedule(
            step, 1.0, 0, args.max_steps,
            "hold" if args.free_running_schedule == "constant" else "cosine-full",
        )
        # Adaptive per-tap block-MSE weights from the running relMSE EMA (None on
        # the first step → depth-only). The EMA is built from the SYNCED relMSE, so
        # it — and these weights — are identical on every rank.
        adaptive_weights = (
            adaptive_weights_from_relmse(relmse_ema, args.adaptive_layer_weight_power)
            if (args.adaptive_layer_weight and relmse_ema)
            else None
        )
        # With the logit losses OFF, the KL/CE pass (and the teacher logits
        # feeding it) is metrics-only — compute it just on the steps whose
        # losses are printed. Depends only on `step` ⇒ identical on every rank.
        klce_metrics = (
            args.lambda_kl != 0.0
            or args.lambda_ce != 0.0
            or step % args.log_every == 0
        )

        # ----- Gradient accumulation: G microbatches per optimizer step -----
        # Each microbatch's losses are scaled 1/G so the grads ACCUMULATE into the
        # per-step MEAN (no zero_grad between microbatches); the logged losses are
        # the unscaled per-microbatch mean. --max-steps / --eval-every / the LR
        # schedule all count optimizer steps. G=1 ⇒ the legacy single-batch loop.
        G = max(1, args.grad_accum_steps)
        loss_scale = 1.0 / G
        optim.zero_grad(set_to_none=True)
        loss_sums = {"total": 0.0, "block_mse": 0.0, "kl": 0.0, "ce": 0.0, "fr_mse": 0.0}
        relmse_sums: dict[int, torch.Tensor] = {}
        data_wait = 0.0
        micro_count = 0
        t_step = time.perf_counter()
        for micro in range(G):
            # Time the (num_workers=0, inline) data fetch separately from compute.
            t_fetch = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            data_wait += time.perf_counter() - t_fetch
            batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
            if batch["input_ids"].ndim == 1:
                batch = {k: v.unsqueeze(0) for k, v in batch.items()}
            # Pass the mem-capture dict on the FIRST microbatch only (it resets peak stats).
            micro_mem = step_mem if (do_mem_capture and micro == 0) else None
            # distill_step backwards each block immediately to bound peak memory;
            # grads accumulate on student params across blocks AND microbatches.
            losses = distill_step(
                student, teacher, batch, distill_cfg, timings=timings, mem=micro_mem,
                student_forcing_prob=sf_p,
                forcing_seed=(args.seed, step, micro),
                loss_scale=loss_scale,
                adaptive_weights=adaptive_weights,
                track_layer_relmse=args.adaptive_layer_weight,
                free_running_scale=fr_scale,
                compute_klce_metrics=klce_metrics,
            )
            for k in loss_sums:
                loss_sums[k] = loss_sums[k] + losses[k].detach()
            for l, v in losses["layer_relmse"].items():
                relmse_sums[l] = (relmse_sums[l] + v) if l in relmse_sums else v
            micro_count += 1
        if micro_count == 0:
            break  # data stream exhausted before this step consumed anything
        # Per-microbatch mean of the (unscaled) losses, for logging.
        mean_losses = {k: loss_sums[k] / micro_count for k in loss_sums}

        # Fold this step's per-tap relMSE (mean over microbatches) into the EMA.
        # One .tolist() sync per step, only when adaptive weighting is on.
        if args.adaptive_layer_weight and relmse_sums:
            keys = sorted(relmse_sums)
            measured = dict(zip(
                keys,
                (torch.stack([relmse_sums[l] for l in keys]) / micro_count).tolist(),
            ))
            if relmse_ema is None:
                relmse_ema = measured
            else:
                beta = args.adaptive_layer_weight_ema
                relmse_ema = {
                    l: beta * relmse_ema.get(l, measured[l]) + (1.0 - beta) * measured[l]
                    for l in measured
                }

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
                f"(loss={mean_losses['total'].item()}, grad_norm=nan/inf)",
            )
            optim.zero_grad(set_to_none=True)
            if prof is not None:
                prof.step()
            step += 1
            continue

        if args.max_grad_norm > 0:
            clip_coef = (args.max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)
            # One fused multi-tensor mul instead of ~1000 tiny per-param kernels.
            # Same scalar applied to every grad, so it is bit-identical to the
            # per-param loop (replicated-copy bit-equality survives unchanged).
            torch._foreach_mul_(
                [p.grad for p in student.parameters() if p.grad is not None],
                clip_coef,
            )

        optim.step()
        if scheduler is not None:
            scheduler.step()

        # --mem-report: with grads still live (zero_grad runs next iter) and the
        # AdamW state now materialized, attribute the resident footprint to its
        # occupants and pair it with the per-phase peaks captured this step.
        if do_mem_capture:
            mem_stages[f"after optim.step (step {step})"] = device_mem()
            report_lines = format_report(
                rank,
                label=f"MEM REPORT @ step {step}",
                components=component_breakdown(student, teacher, optim),
                device=device_mem(),
                phase_mem=step_mem,
                stages=mem_stages,
            )
            print_all_ranks(rank, world_size, report_lines)
            # Rebuild a clean global peak for any later steps / the --profile peak.
            torch.cuda.reset_peak_memory_stats()

        if args.profile:
            # t_step is set before the microbatch loop and the data fetches happen
            # inside it, so step_total already includes data_wait (a subset). The
            # per-phase `timings` accumulated across all G microbatches inside
            # distill_step (the _phase timer adds, not overwrites).
            timings["data_wait"] = data_wait
            torch.cuda.synchronize()
            step_total = time.perf_counter() - t_step
            accounted = sum(timings.get(k, 0.0) for k in phase_keys)
            other = step_total - accounted
            # Step 0 carries one-time warmup (cuDNN autotune, lazy allocs, first
            # dataset batch); exclude it from the mean.
            if step > 0:
                for k in phase_keys:
                    phase_totals[k] = phase_totals.get(k, 0.0) + timings.get(k, 0.0)
                phase_totals["other"] = phase_totals.get("other", 0.0) + other
                phase_totals["_total"] = phase_totals.get("_total", 0.0) + step_total
                profiled_steps += 1
            breakdown = " ".join(f"{k}={timings.get(k, 0.0) * 1e3:.0f}ms" for k in phase_keys)
            _log(rank, f"[profile step {step}] {breakdown} other={other * 1e3:.0f}ms "
                       f"total={step_total * 1e3:.0f}ms")

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]
            _log(
                rank,
                f"[step {step}] total={mean_losses['total'].item():.4f} "
                f"block_mse={mean_losses['block_mse'].item():.4f} "
                f"fr_mse={mean_losses['fr_mse'].item():.4f} "
                f"kl={mean_losses['kl'].item():.4f} ce={mean_losses['ce'].item():.4f} "
                f"grad_norm={total_norm.item():.3e} "
                f"lr={lr_now:.2e} elapsed={elapsed:.1f}s",
            )

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            _log(rank, f"[save] step {step}")
            # Rolling checkpoint: always overwrite the single latest/ dir in place,
            # so only one periodic checkpoint (~52 GB at n=16) ever exists alongside
            # best/. The saved step number lives inside train_state_rank*.pt, so
            # resume (--resume-from .../latest) recovers it. Overwrites in place
            # rather than writing a temp + rename (no room for a 2nd copy under a
            # tight disk quota); a crash mid-save can corrupt latest/, but best/ is
            # untouched and protects model quality.
            save_checkpoint("latest")

        if val_loader is not None and step > 0 and step % args.eval_every == 0:
            val_kl, val_ce = run_eval()
            improved = val_kl < (best_val_kl - args.min_improvement)
            if improved:
                best_val_kl = val_kl
                windows_since_best = 0
                _log(
                    rank,
                    f"[eval step={step}] val_kl={val_kl:.4f} val_ce={val_ce:.4f} (new best)",
                )
                save_checkpoint(args.best_name)
            else:
                windows_since_best += 1
                _log(
                    rank,
                    f"[eval step={step}] val_kl={val_kl:.4f} val_ce={val_ce:.4f} "
                    f"(no improvement; {windows_since_best}/{args.early_stop_patience})",
                )
            if args.early_stop_patience > 0 and windows_since_best >= args.early_stop_patience:
                _log(rank, f"[early-stop] val_kl hasn't improved in {args.early_stop_patience} windows (best={best_val_kl:.4f})")
                stop_now = True

        # Close the trace window (symmetric across ranks) once it has run its
        # TRACE_ACTIVE steps. Remaining steps run un-traced.
        if prof_running and step >= TRACE_START + TRACE_ACTIVE - 1:
            torch.cuda.synchronize()
            prof.stop()
            prof_running = False
        step += 1

    if prof_running:  # loop ended before the window closed (e.g. tiny --max-steps)
        torch.cuda.synchronize()
        prof.stop()
        prof_running = False

    if args.profile:
        if profiled_steps > 0:
            mean_total = phase_totals.get("_total", 0.0) / profiled_steps
            _log(rank, f"[profile] === mean over {profiled_steps} steps (excl. step 0), rank 0 ===")
            for k in phase_keys + ["other"]:
                mean_ms = phase_totals.get(k, 0.0) / profiled_steps * 1e3
                pct = (phase_totals.get(k, 0.0) / phase_totals["_total"] * 100) if mean_total else 0.0
                _log(rank, f"[profile]   {k:12s} {mean_ms:9.0f} ms  ({pct:4.1f}%)")
            _log(rank, f"[profile]   {'TOTAL':12s} {mean_total * 1e3:9.0f} ms")
        # Peak memory is per-rank; print on every rank.
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[profile] rank {rank} peak CUDA mem allocated: {peak_gb:.2f} GB", flush=True)

    if args.save_final:
        _log(rank, "[save] final")
        save_checkpoint("final")

    teacher.remove_hooks()
    dist.destroy_process_group()

    # Export the kernel trace only after the process group is gone: key_averages()
    # and export_chrome_trace() can take minutes, and with no live PG there is no
    # collective for them to desync and no watchdog to trip. rank 0 only.
    if prof is not None and rank == 0:
        trace_path = prof_dir / "trace_rank0.json"
        prof.export_chrome_trace(str(trace_path))
        print("[profile] === top kernels by self CUDA time ===", flush=True)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
        print(f"[profile] chrome trace written to {trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
