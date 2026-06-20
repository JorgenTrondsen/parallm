"""torchrun entry point: run lm-evaluation-harness against the PT student and/or the dense teacher.

Default ``--target both`` loads both models, runs lm-eval on each with the
same task list / seeds, and prints side-by-side results — that's how you tell
whether the student "performs the same" as the teacher on the field-standard
benchmarks. Pass ``--target student`` or ``--target teacher`` to evaluate only
one (skips loading the other to save GPU memory).

Distributed pattern: every rank runs identical lm-eval code (same seed → same
request ordering). The student's non-owner ranks emit zero placeholders but
still execute the forward so cross-track SyncBoundary collectives match the
owner rank; only rank 0 prints / saves results. See
``src/pt_converter/eval/lm_eval_adapter.py``.

Single node, 8 GPUs, compare student vs teacher on the default task set:

    torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir ./pt_train_out/best \\
        --output-json ./pt_train_out/lm_eval.json

Smoke run (32 requests per task, fast pipeline check):

    torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir ./pt_train_out/best \\
        --tasks hellaswag,arc_easy \\
        --limit 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.eval.lm_eval_adapter import (
    PTLM,
    is_lm_head_owner,
    make_student_forward_fn,
    make_teacher_forward_fn,
)
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _run_one_target(
    target: str,
    forward_fn,
    is_owner: bool,
    tokenizer,
    tasks: list[str],
    args,
    rank: int,
) -> dict:
    """Single lm-eval pass over the configured task list. Returns the raw
    ``simple_evaluate`` dict (or ``None`` on edge cases)."""
    lm = PTLM(
        forward_fn=forward_fn,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=torch.cuda.current_device(),
        is_owner_rank=is_owner,
    )
    # Defer import: heavy module, only used here.
    from lm_eval import simple_evaluate

    _log(rank, f"[eval] running lm-eval on target={target}…")
    return simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )


def _format_results(results: dict | None) -> str:
    if results is None or "results" not in results:
        return "  (no results)"
    lines = []
    for task_name, metrics in results["results"].items():
        pretty = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if not k.startswith("alias")
        )
        lines.append(f"  {task_name}: {pretty}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["student", "teacher", "both"], default="both",
                   help="Which model(s) to evaluate. Default 'both' loads student + teacher and "
                        "runs lm-eval on each with identical seeds for side-by-side comparison. "
                        "Use 'student' or 'teacher' alone to skip loading the other side.")
    p.add_argument("--hf-model", required=True,
                   help="Dense teacher model path (used for tokenizer in every mode; for teacher "
                        "weights in 'teacher' or 'both' mode).")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir with track_*.safetensors and manifest.json. "
                        "In 'teacher' mode only the manifest is read (for n_tracks / sync layout).")
    p.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,winogrande,piqa",
                   help="Comma-separated lm-eval task names. MMLU is opt-in via --include-mmlu (slow).")
    p.add_argument("--include-mmlu", action="store_true",
                   help="Append `mmlu` to --tasks. Adds ~hours of GPU time on a single node.")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="Override fewshot count for all tasks. Default = each task's standard shot count.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap requests per task. Useful for smoke testing (e.g. --limit 32).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Requests scored per forward. Each forward pays a fixed set of NCCL "
                        "all-reduces (embed + one per sync boundary) regardless of batch size, "
                        "so batching amortizes them and fills the GPU — the biggest eval speedup. "
                        "The owner rank holds a (B, max_len, vocab) bf16 logits tensor, so lower "
                        "this for long-context / loglikelihood-rolling tasks (e.g. wikitext) to "
                        "avoid OOM; the short-context multiple-choice tasks are fine at 8+.")
    p.add_argument("--max-length", type=int, default=4096,
                   help="Tokenizer truncation length for lm-eval contexts.")
    p.add_argument("--output-json", default=None,
                   help="Optional path to dump per-target lm-eval results dict (rank 0 only). "
                        "In 'both' mode the file contains a top-level {student, teacher} dict.")
    p.add_argument("--seed", type=int, default=0,
                   help="lm-eval random/numpy/torch seed. Identical on every rank so request "
                        "ordering and fewshot sampling line up across ranks.")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} >= visible GPU count {torch.cuda.device_count()}."
        )
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    if manifest.sync_layer_indices is None:
        raise SystemExit(
            "[error] checkpoint manifest has no sync_layer_indices — point "
            "--checkpoint-dir at a trained checkpoint (the schedule is placed at "
            "train time, so a raw convert output carries none)."
        )
    sync_layers = list(manifest.sync_layer_indices)
    _log(
        rank,
        f"[init] target={args.target} world={layout.world_size} "
        f"n_tracks={manifest.n_tracks} K={layout.tracks_per_rank}",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    want_student = args.target in ("student", "both")
    want_teacher = args.target in ("teacher", "both")

    student = None
    teacher = None
    teardown_fns: list = []

    if want_student:
        cfg = AutoConfig.from_pretrained(args.hf_model)
        _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
        student = PTWrappedModel(
            text_config=cfg.text_config,
            n_tracks=manifest.n_tracks,
            local_track_ids=layout.local_track_ids,
            sync_after_layers=sync_layers,
            track_group=layout.track_group,
        )
        track_states = {tid: load_track(args.checkpoint_dir, tid) for tid in layout.local_track_ids}
        student.load_track_state_dicts(track_states, strict=True)
        student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
        wrap_student_with_fsdp(student, layout)
        student.eval()

    if want_teacher:
        _log(rank, "[init] loading frozen dense teacher…")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
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
            sync_layer_indices=sync_layers,
        )
        teacher_model = teacher_model.to(torch.cuda.current_device())
        wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)
        teardown_fns.append(teacher.remove_hooks)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.include_mmlu and "mmlu" not in tasks:
        tasks.append("mmlu")
    _log(rank, f"[init] tasks={tasks}")

    all_results: dict[str, dict] = {}
    if want_student:
        all_results["student"] = _run_one_target(
            "student",
            make_student_forward_fn(student),
            is_lm_head_owner(student),
            tokenizer, tasks, args, rank,
        )
    if want_teacher:
        all_results["teacher"] = _run_one_target(
            "teacher",
            make_teacher_forward_fn(teacher),
            True,  # FSDP all-gathers → real logits on every rank
            tokenizer, tasks, args, rank,
        )

    if rank == 0:
        print()
        for tgt, results in all_results.items():
            print(f"===== lm-evaluation-harness ({tgt}) =====")
            print(_format_results(results))
            print()
        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            print(f"full results → {args.output_json}")

    for fn in teardown_fns:
        fn()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
