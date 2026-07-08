"""Lever-2 gate: offline draft–verify acceptance measurement (token-axis sync
amortization). Single GPU, no tracks, no training.

Deployment being sized: a small same-tokenizer dense DRAFT (replicated per
device, zero sync events) proposes k tokens; the sliced D=1-B student VERIFIES
all k in one pass whose 32 per-layer sync events each carry k tokens (volume is
free, frequency is not). Sync events per decoded token = 32 / τ, where
τ = mean (accepted draft tokens + 1) per verify event. τ ≥ 2 beats the D=2
budget (16 events/token) while keeping the verifier's EXACT output quality.

This gate measures τ offline with the DENSE TEACHER as the verify target (the
D=1-B student tracks the teacher closely; a distributed re-run against the real
student only matters if this read is positive). ``--target-mode`` swaps the
verify target's forward for a dense window-parallel arm (eval/dense_parallel.py)
so the verifier-fidelity sensitivity of the agreement can be bracketed with the
healed dense checkpoints (the trained D=1-B weights were reclaimed for quota):
e.g. ``--target-model train_out/qwen3_5_9b_dense_heal_g2_l15/best
--target-mode parallel-attn`` scores the draft against a 0.647-quality verifier.
Method: teacher-forced greedy
agreement on held-out text from the training mixture — at each position, does
the draft's greedy next-token equal the target's? Block simulation over the
per-sequence agreement indicators then yields τ(k) exactly as deployment would
see it (draft k, accept the agreeing prefix, +1 bonus token from the verifier,
restart from the rejection). Also reports the sampling-mode acceptance
E[Σ_x min(p_draft, p_target)] (distribution overlap) as the temperature-1 analog.

Example:

    .venv/bin/python scripts/probe_draft_accept.py \
        --target-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
        --draft-model Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pt_converter.eval.dense_parallel import MODES, build_windows, dense_window_forward
from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)


def _simulate_blocks(runs: "list[torch.Tensor]", k: int) -> "tuple[float, int, int]":
    """Walk each sequence's agreement indicators as deployment would: draft k,
    accept the agreeing prefix a (≤ k), emit a+1 tokens (verify bonus), restart
    after the rejection. Returns (τ = tokens/event, total tokens, total events)."""
    tokens = events = 0
    for ind in runs:
        n = ind.numel()
        i = 0
        while i < n:
            a = 0
            while a < k and i + a < n and bool(ind[i + a]):
                a += 1
            tokens += a + 1
            events += 1
            i += a + 1
    return (tokens / events if events else 0.0), tokens, events


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-model", required=True, help="Verify target (the dense teacher path).")
    p.add_argument("--draft-model", default="Qwen/Qwen3.5-0.8B",
                   help="Small same-tokenizer draft (HF id or path).")
    p.add_argument("--target-mode", default="hf", choices=("hf",) + MODES,
                   help="Verify target forward: 'hf' = stock forward; any dense_parallel "
                        "mode runs the window-parallel rewiring on the target weights.")
    p.add_argument("--group-size", type=int, default=2,
                   help="Window size for the parallel target modes.")
    p.add_argument("--first-parallel-layer", type=int, default=1)
    p.add_argument("--num-batches", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--block-sizes", default="2,4,8,16")
    p.add_argument("--verify-events", type=int, default=32,
                   help="Sync events per verify pass (32 = the D=1 per-layer count).")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overlap-chunk", type=int, default=256,
                   help="Seq-chunk for the fp32 vocab softmax overlap (memory guard).")
    # Data (mirrors probe_intervention.py: held-out front of the training mixture).
    p.add_argument("--data-preset", default=DEFAULT_PRESET, choices=preset_names())
    p.add_argument("--data-source", action="append", default=None, metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]")
    p.add_argument("--skip-docs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    ks = [int(x) for x in args.block_sizes.split(",") if x.strip()]

    print(f"[init] loading target {args.target_model} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    ).eval().to(device)
    print(f"[init] loading draft {args.draft_model} …", flush=True)
    draft_tok = AutoTokenizer.from_pretrained(args.draft_model)
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=args.attn_impl,
    ).eval().to(device)
    probe = "The quick brown fox jumps over the lazy dog. 東京 3.14159"
    if draft_tok.encode(probe) != tok.encode(probe):
        raise SystemExit("[error] draft tokenizer does not match the target's — spec-dec needs "
                         "an identical vocab. Pick a same-family draft.")

    if args.target_mode == "hf":
        def target_logits(ids: torch.Tensor) -> torch.Tensor:
            return target(input_ids=ids).logits
    else:
        text_model = (
            target.model.language_model
            if hasattr(target.model, "language_model") else target.model
        )
        windows = build_windows(
            len(text_model.layers),
            1 if args.target_mode == "serial" else args.group_size,
            args.first_parallel_layer,
        )
        def target_logits(ids: torch.Tensor) -> torch.Tensor:
            hidden = dense_window_forward(
                text_model, ids, None, windows=windows, mode=args.target_mode
            )
            return target.lm_head(hidden)

    if args.data_source:
        data_cfg = CalibrationDataConfig(
            sources=[parse_source_spec(s) for s in args.data_source],
            seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
        )
    else:
        data_cfg = CalibrationDataConfig(
            sources=preset_sources(args.data_preset),
            seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
        )
    loader = DataLoader(PackedTokenStream(tok, data_cfg), batch_size=args.batch_size, num_workers=0)

    runs: "list[torch.Tensor]" = []  # per-sequence greedy-agreement indicators (CPU)
    overlap_sum, overlap_n = 0.0, 0
    seen = 0
    for batch in loader:
        if seen >= args.num_batches:
            break
        ids = batch["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device, non_blocking=True)
        t_logits = target_logits(ids)
        d_logits = draft(input_ids=ids).logits
        agree = t_logits.argmax(-1) == d_logits.argmax(-1)  # (B, T)
        # Position t proposes token t+1; the last position has no continuation.
        for b in range(agree.shape[0]):
            runs.append(agree[b, :-1].cpu())
        # Sampling-mode acceptance: E_t[Σ_x min(p_draft, p_target)], fp32 in chunks.
        for s in range(0, t_logits.shape[1] - 1, args.overlap_chunk):
            e = min(s + args.overlap_chunk, t_logits.shape[1] - 1)
            pt_ = F.softmax(t_logits[:, s:e].float(), dim=-1)
            pd_ = F.softmax(d_logits[:, s:e].float(), dim=-1)
            overlap_sum += torch.minimum(pt_, pd_).sum(-1).sum().item()
            overlap_n += pt_.shape[0] * pt_.shape[1]
        seen += 1
        if seen % 8 == 0:
            print(f"[data] {seen}/{args.num_batches} batches …", flush=True)

    total = int(sum(r.numel() for r in runs))
    agree_rate = float(sum(r.sum() for r in runs)) / max(total, 1)
    print(f"\n=== draft-accept gate ===")
    mode_tag = args.target_mode if args.target_mode in ("hf", "serial") else \
        f"{args.target_mode}:g{args.group_size}"
    print(f"  draft={args.draft_model}  target={args.target_model}  target-mode={mode_tag}")
    print(f"  positions={total}  greedy-agreement={agree_rate:.4f}  "
          f"sampling-overlap={overlap_sum / max(overlap_n, 1):.4f}")
    print(f"  {'k':>4} {'tau (tok/event)':>16} {'syncs/token':>12}   (verify pass = "
          f"{args.verify_events} events; D=2 budget = 16, D=1 = 32)")
    for k in ks:
        tau, _tokens, _events = _simulate_blocks(runs, k)
        spt = args.verify_events / tau if tau else float("inf")
        print(f"  {k:>4} {tau:>16.3f} {spt:>12.2f}")
    print("  Read: tau >= 2 ⇒ fewer sync events/token than the whole D=2 budget at EXACT "
          "verifier quality; tau >= 3.2 beats 10 events/token. Greedy row is the "
          "deployment-relevant one for loglikelihood-style evals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
