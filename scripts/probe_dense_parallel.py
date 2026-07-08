"""Lever-1 Gate-0: downstream retention of the DENSE teacher under window-parallel
rewiring (eval/dense_parallel.py). Single GPU, no tracks, no training, no NCCL.

Arms (cross product --modes x --group-sizes; 'serial' runs once as the rail):

    serial                    — must reproduce the teacher's standard-harness score.
    parallel-attn:gG          — the slice-computable target function (exact seam).
    parallel-attn-dropseam:gG — worst-case seam bracket.
    parallel-full:gG          — PaLM-style (no post-attn sync inside the window).

Example (the standard 3-task lim1000 harness):

    .venv/bin/python scripts/probe_dense_parallel.py \
        --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
        --modes serial,parallel-attn,parallel-attn-dropseam,parallel-full \
        --group-sizes 2,4 \
        --downstream-tasks arc_challenge,piqa,winogrande --downstream-limit 1000
"""
from __future__ import annotations

import argparse

import torch

from pt_converter.eval.dense_parallel import MODES, build_windows, dense_window_forward


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path (same as training).")
    p.add_argument("--modes", default="serial,parallel-attn,parallel-attn-dropseam,parallel-full",
                   help=f"Comma-separated subset of {MODES}.")
    p.add_argument("--group-sizes", default="2",
                   help="Comma-separated window sizes for the parallel modes (2 = the D=2 layout; "
                        "add 4 for the D=4 scaling arm). 'serial' ignores this.")
    p.add_argument("--first-parallel-layer", type=int, default=1,
                   help="Layers before this stay singleton/serial (layer 0 is a real boundary in "
                        "the deployed schedule); the final layer is always singleton.")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    # The standard downstream harness (mirrors probe_intervention.py defaults).
    p.add_argument("--downstream-tasks", default="arc_challenge,piqa,winogrande")
    p.add_argument("--downstream-limit", type=int, default=1000)
    p.add_argument("--downstream-batch-size", type=int, default=4)
    p.add_argument("--downstream-max-length", type=int, default=2048)
    p.add_argument("--downstream-num-fewshot", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from lm_eval import simple_evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pt_converter.eval.downstream import _pick_metric, aggregate_downstream_score
    from pt_converter.eval.lm_eval_adapter import PTLM

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    groups = [int(g) for g in args.group_sizes.split(",") if g.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"[error] unknown mode {m!r}; expected a subset of {MODES}")

    device = torch.device(args.device)
    print(f"[init] loading dense model from {args.hf_model} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    ).eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    text_model = (
        model.model.language_model if hasattr(model.model, "language_model") else model.model
    )
    L = len(text_model.layers)

    arms: "list[tuple[str, str, list[list[int]]]]" = []  # (name, mode, windows)
    for m in modes:
        if m == "serial":
            arms.append(("serial", m, build_windows(L, 1)))
            continue
        for g in groups:
            arms.append((f"{m}:g{g}", m, build_windows(L, g, args.first_parallel_layer)))

    tasks = [t.strip() for t in args.downstream_tasks.split(",") if t.strip()]
    results: "list[tuple[str, float, dict]]" = []
    for name, mode, windows in arms:
        sizes = [len(w) for w in windows]
        print(f"[run] arm={name} windows={len(windows)} (sizes {min(sizes)}..{max(sizes)}) …",
              flush=True)

        def _fn(input_ids, attention_mask, _mode=mode, _windows=windows):
            hidden = dense_window_forward(
                text_model, input_ids, attention_mask, windows=_windows, mode=_mode
            )
            return model.lm_head(hidden)

        lm = PTLM(
            forward_fn=_fn, tokenizer=tok,
            max_length=args.downstream_max_length, batch_size=args.downstream_batch_size,
            device=device, is_owner_rank=True,
        )
        res = simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=args.downstream_num_fewshot,
            limit=args.downstream_limit, batch_size=args.downstream_batch_size,
            random_seed=args.seed, numpy_random_seed=args.seed,
            torch_random_seed=args.seed, fewshot_random_seed=args.seed,
        )
        score = aggregate_downstream_score(res, tasks)
        table = (res or {}).get("results", {})
        per_task = {
            t: v for t in tasks
            if (v := _pick_metric(table.get(t, {}), ["acc_norm", "acc"])) is not None
        }
        results.append((name, score, per_task))
        detail = "  ".join(f"{t}={v:.4f}" for t, v in per_task.items())
        print(f"[result] {name}: macro={score:.4f}  {detail}", flush=True)

    print("\n=== dense window-parallel gate ===")
    for name, score, per_task in results:
        detail = "  ".join(f"{t}={v:.4f}" for t, v in per_task.items())
        print(f"  {name:<32} {score:.4f}   {detail}")
    print("  Read: 'serial' is the rail (teacher baseline). parallel-attn:g2 is the "
          "slice-computable D=2 target (exact seam); the deployed slice lands between it "
          "and the dropseam arm. parallel-full removes the post-attn sync inside windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
