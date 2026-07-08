"""torchrun entrypoint for the dense healing run (Lever-1 Gate-1): self-distill
the serial dense Qwen3.5-9B into the window-parallel ``parallel-attn:g2``
function. See ``src/pt_converter/train/heal_dense.py`` for the objective.

    .venv/bin/torchrun --standalone --nproc-per-node=8 scripts/heal_dense_parallel.py \
        --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
        --out-dir train_out/qwen3_5_9b_dense_heal_g2

Design (differs from the track trainer on purpose — no tracks, no vocab-parallel):
- Student: dense causal model; the 32 decoder layers train (per-layer FSDP2
  ``fully_shard``, bf16 params / fp32 grad reduce); ``embed_tokens`` /
  ``lm_head`` / final ``norm`` are FROZEN and stay replicated (plain bf16) so
  direct submodule calls need no root FSDP wrap and the logit geometry stays
  anchored.
- Teacher: frozen ``HookedTeacher`` capturing the window-END hiddens, wrapped
  by ``wrap_teacher_with_fsdp`` AFTER hook install (the proven order; no
  teacher compile — compiled layers swallow hooks).
- Data parallel: each rank consumes a distinct slice of the packed stream
  (batch index mod world). Validation reader + in-loop lm-eval run the
  IDENTICAL data/requests on every rank so FSDP collectives stay lockstep.
- Selection: in-loop downstream macro (3 tasks, lim ``--downstream-limit``)
  every ``--downstream-eval-every`` steps → ``best/`` in HF format (directly
  loadable by ``scripts/probe_dense_parallel.py --hf-model``). val logit-MSE/KL
  logged as the low-noise trend reader, never for selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_teacher_with_fsdp
from pt_converter.eval.dense_parallel import MODES, build_windows, dense_window_forward
from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    PackedTokenStream,
    preset_names,
    preset_sources,
)
from pt_converter.train.heal_dense import HealConfig, heal_step
from pt_converter.train.teacher import HookedTeacher


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _text_model(causal):
    return causal.model.language_model if hasattr(causal.model, "language_model") else causal.model


def _mean_metrics(metrics: "dict[str, float]", device) -> "dict[str, float]":
    """All-reduce-mean the per-rank metric dict (ranks see different data)."""
    if not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    t = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return dict(zip(keys, t.tolist()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True,
                   help="Student init (the original snapshot, or a healed best/ to warm-start).")
    p.add_argument("--teacher-model", default=None,
                   help="Frozen serial teacher (defaults to --hf-model; MUST be the original "
                        "snapshot when warm-starting the student from a healed checkpoint).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", default="parallel-attn", choices=list(MODES))
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=5001)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr-floor", type=float, default=0.1, help="Cosine floor as a fraction of --lr.")
    p.add_argument("--batch-size", type=int, default=1, help="Per-rank micro-batch.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lambda-logit-mse", type=float, default=0.05)
    p.add_argument("--block-mse-clamp", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--attn-impl", default="sdpa")
    # Low-rank slab reparameterization (co-trained low-rank replica gate):
    # every sliceable Linear becomes per-slab U·V at this rank (SVD init from
    # the loaded weights); 0 disables. Healed-model downstream = exact-copy
    # replica deployment quality at ANY sync count. Use with --mode serial.
    p.add_argument("--lowrank-rank", type=int, default=0)
    p.add_argument("--lowrank-tracks", type=int, default=16,
                   help="Track count defining the slab geometry (matches the conversion).")
    p.add_argument("--downstream-eval-at-start", action="store_true",
                   help="Downstream read before step 1 (the raw SVD-truncation floor).")
    # Data: train stream skips the held-out front; val/probes read the front.
    p.add_argument("--data-preset", default=DEFAULT_PRESET, choices=preset_names())
    p.add_argument("--train-skip-docs", type=int, default=5000)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # Selection / checkpoints.
    p.add_argument("--downstream-tasks", default="arc_challenge,piqa,winogrande")
    p.add_argument("--downstream-eval-every", type=int, default=250)
    p.add_argument("--downstream-limit", type=int, default=200)
    p.add_argument("--downstream-batch-size", type=int, default=4)
    p.add_argument("--downstream-max-length", type=int, default=2048)
    p.add_argument("--save-latest-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # device_id → eager NCCL comm init + clean communicator destroy (without it
    # the final barrier/destroy SIGABRTs at interpreter exit on this stack).
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.hf_model)

    # ----- Student: dense causal model; layers train, embed/norm/lm_head frozen. -----
    _log(rank, f"[init] loading student from {args.hf_model} …")
    student = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    )
    s_text = _text_model(student)
    for mod in (s_text.embed_tokens, s_text.norm, student.lm_head):
        for prm in mod.parameters():
            prm.requires_grad = False
    student.to(device)
    if args.lowrank_rank > 0:
        from pt_converter.model.lowrank_slice import apply_lowrank_slicing

        lr_stats = apply_lowrank_slicing(
            s_text, student.config, args.lowrank_rank, n_tracks=args.lowrank_tracks
        )
        _log(rank, f"[lowrank] rank={args.lowrank_rank} tracks={args.lowrank_tracks} "
                   f"modules={lr_stats['modules']} bands={lr_stats['bands_factored']}F/"
                   f"{lr_stats['bands_dense']}D params {lr_stats['params_before'] / 1e9:.2f}B"
                   f"→{lr_stats['params_after'] / 1e9:.2f}B "
                   f"({100 * lr_stats['params_after'] / max(lr_stats['params_before'], 1):.1f}%)")
        # SVD kernels aren't guaranteed bit-identical across processes; FSDP
        # shards each rank's local values, so make the factors rank0's exactly.
        with torch.no_grad():
            for prm in s_text.layers.parameters():
                dist.broadcast(prm.data, src=0)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # Shard at SUBMODULE granularity: the rewired forward (seam_token_mixer /
    # seam_mlp) calls input_layernorm / mixer / post_attention_layernorm / mlp
    # directly, bypassing layer.forward — so the FSDP unshard hooks must sit on
    # those exact modules (a layer-level wrap never fires and leaves DTensors).
    for layer in s_text.layers:
        mixer = (
            layer.linear_attn
            if getattr(layer, "layer_type", None) == "linear_attention"
            and hasattr(layer, "linear_attn")
            else layer.self_attn
        )
        subs = (layer.input_layernorm, mixer, layer.post_attention_layernorm, layer.mlp)
        n_sub = sum(p_.numel() for m in subs for p_ in m.parameters())
        n_all = sum(p_.numel() for p_ in layer.parameters())
        assert n_sub == n_all, f"layer has {n_all - n_sub} params outside the sharded submodules"
        for sub in subs:
            # Each submodule is its own FSDP "root" (no parent wrap), and roots
            # default to reshard_after_forward=False — 128 roots would keep the
            # WHOLE gathered model resident (~+15GB). Force resharding.
            fully_shard(sub, mp_policy=mp, reshard_after_forward=True)
    student.train()

    windows = build_windows(len(s_text.layers), args.group_size)
    cfg = HealConfig(
        windows=windows, mode=args.mode,
        block_mse_clamp=args.block_mse_clamp, lambda_logit_mse=args.lambda_logit_mse,
    )
    _log(rank, f"[init] mode={args.mode} windows={len(windows)} capture_indices={cfg.capture_indices}")

    # ----- Teacher: hooks BEFORE FSDP wrap (the proven order); never compiled. -----
    teacher_path = args.teacher_model or args.hf_model
    _log(rank, f"[init] loading frozen teacher from {teacher_path} …")
    teacher_causal = AutoModelForCausalLM.from_pretrained(
        teacher_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    ).eval()
    for prm in teacher_causal.parameters():
        prm.requires_grad = False
    t_text = _text_model(teacher_causal)
    teacher = HookedTeacher(t_text, teacher_causal.lm_head, cfg.capture_indices)
    teacher_causal.to(device)
    wrap_teacher_with_fsdp(t_text, teacher_causal.lm_head)

    # ----- Optimizer / schedule (track-trainer conventions). -----
    train_params = [p_ for p_ in student.parameters() if p_.requires_grad]
    optim = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return args.lr_floor + (1 - args.lr_floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)

    # ----- Data: per-rank distinct train slices; identical val front on all ranks. -----
    def _stream(skip_docs: int):
        data_cfg = CalibrationDataConfig(
            sources=preset_sources(args.data_preset),
            seq_len=args.seq_len, seed=args.seed, skip_docs=skip_docs,
        )
        return DataLoader(PackedTokenStream(tok, data_cfg), batch_size=args.batch_size, num_workers=0)

    val_batches = []
    for batch in _stream(0):
        val_batches.append({k: v.to(device) for k, v in batch.items()})
        if len(val_batches) >= args.val_batches:
            break
    _log(rank, f"[data] {len(val_batches)} val batches (front of mixture); train skips "
               f"{args.train_skip_docs} docs, rank-strided mod {world}")

    def _train_batches():
        while True:  # re-open the stream if exhausted (streaming epochs)
            for i, batch in enumerate(_stream(args.train_skip_docs)):
                if i % world == rank:
                    yield {k: v.to(device) for k, v in batch.items()}

    # ----- In-loop downstream eval (all ranks lockstep; rank0 scores). -----
    tasks = [t.strip() for t in args.downstream_tasks.split(",") if t.strip()]

    def _downstream_score() -> "tuple[float, dict]":
        from lm_eval import simple_evaluate
        from pt_converter.eval.downstream import _pick_metric, aggregate_downstream_score
        from pt_converter.eval.lm_eval_adapter import PTLM

        student.eval()

        def _fn(input_ids, attention_mask):
            hidden = dense_window_forward(
                s_text, input_ids, attention_mask, windows=windows, mode=args.mode
            )
            return student.lm_head(hidden)

        lm = PTLM(forward_fn=_fn, tokenizer=tok, max_length=args.downstream_max_length,
                  batch_size=args.downstream_batch_size, device=device,
                  is_owner_rank=(rank == 0))
        res = simple_evaluate(model=lm, tasks=tasks, num_fewshot=0,
                              limit=args.downstream_limit, batch_size=args.downstream_batch_size,
                              random_seed=args.seed, numpy_random_seed=args.seed,
                              torch_random_seed=args.seed, fewshot_random_seed=args.seed)
        score, per_task = 0.0, {}
        if rank == 0:
            score = aggregate_downstream_score(res, tasks)
            table = (res or {}).get("results", {})
            per_task = {t: v for t in tasks
                        if (v := _pick_metric(table.get(t, {}), ["acc_norm", "acc"])) is not None}
        score_t = torch.tensor([score], dtype=torch.float64, device=device)
        dist.broadcast(score_t, src=0)
        student.train()
        return score_t.item(), per_task

    def _save(tag: str, step: int, score: float) -> None:
        sd = get_model_state_dict(
            student, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if rank == 0:
            if args.lowrank_rank > 0:
                from pt_converter.model.lowrank_slice import assembled_state_dict

                sd = assembled_state_dict(student, sd)
            ckpt = out_dir / tag
            ckpt.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(ckpt, state_dict=sd, safe_serialization=True)
            tok.save_pretrained(ckpt)
            (ckpt / "heal_meta.json").write_text(json.dumps(
                {"step": step, "score": score, "mode": args.mode,
                 "group_size": args.group_size, "windows": windows,
                 "lowrank_rank": args.lowrank_rank,
                 "lowrank_tracks": args.lowrank_tracks}, indent=2))
            print(f"[save] {tag}/ at step {step} (score {score:.4f})", flush=True)
        dist.barrier()

    if args.downstream_eval_at_start:
        score, per_task = _downstream_score()
        detail = "  ".join(f"{t}={v:.4f}" for t, v in per_task.items())
        _log(rank, f"[downstream 0] macro={score:.4f}  {detail}")

    # ----- Loop. -----
    best_score, best_step = -1.0, -1
    t0 = time.time()
    train_iter = _train_batches()
    for step in range(1, args.max_steps + 1):
        batch = next(train_iter)
        optim.zero_grad(set_to_none=True)
        metrics = heal_step(
            s_text, student.lm_head, teacher, cfg,
            batch["input_ids"], batch.get("attention_mask"),
        )
        gnorm = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        optim.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            m = _mean_metrics(metrics, device)
            _log(rank, f"[step {step}] tf_mse={m['tf_window_mse']:.4f} "
                       f"(max {m['tf_window_mse_max']:.3f}) "
                       f"fr_lmse={m.get('fr_logit_mse', float('nan')):.4f} "
                       f"fr_kl={m.get('fr_kl', float('nan')):.4f} "
                       f"gnorm={float(gnorm):.2f} lr={sched.get_last_lr()[0]:.2e} "
                       f"{(time.time() - t0) / step:.2f}s/step "
                       f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

        if args.val_every > 0 and step % args.val_every == 0:
            student.eval()
            with torch.no_grad():
                vals: "dict[str, float]" = {}
                for vb in val_batches:
                    vm = heal_step(s_text, student.lm_head, teacher, cfg,
                                   vb["input_ids"], vb.get("attention_mask"), do_backward=False)
                    for k, v in vm.items():
                        vals[k] = vals.get(k, 0.0) + v / len(val_batches)
            student.train()
            _log(rank, f"[val {step}] tf_mse={vals['tf_window_mse']:.4f} "
                       f"fr_lmse={vals['fr_logit_mse']:.4f} fr_kl={vals['fr_kl']:.4f}")

        if args.downstream_eval_every > 0 and step % args.downstream_eval_every == 0:
            score, per_task = _downstream_score()
            detail = "  ".join(f"{t}={v:.4f}" for t, v in per_task.items())
            _log(rank, f"[downstream {step}] macro={score:.4f}  {detail}")
            if score > best_score + 1e-9:
                best_score, best_step = score, step
                _save("best", step, score)

        if args.save_latest_every > 0 and step % args.save_latest_every == 0:
            _save("latest", step, best_score)

    _log(rank, f"[done] best downstream macro {best_score:.4f} at step {best_step} "
               f"({(time.time() - t0) / 60:.0f} min total)")
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
