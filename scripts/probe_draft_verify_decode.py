"""Lever-2 deployment probe: block draft-verify greedy decode, measuring τ
(emitted tokens per verify event) on the verifier's OWN generation trajectory.

The offline gate (probe_draft_accept.py) reads τ teacher-forced on data text —
the standard conservative proxy. This probe runs the actual deployed loop:
draft k greedy tokens, ONE verifier pass over prefix+proposals, accept the
agreeing prefix plus the verifier's bonus token, repeat. Emitted text is
bit-exact the verifier's greedy decode; syncs/token = verify-events-per-pass/τ.

Two verifier modes:

  Single GPU (dense weights, hf or window-parallel rewiring):
    .venv/bin/python scripts/probe_draft_verify_decode.py \
        --target-model train_out/qwen3_5_9b_dense_heal_g2_l15/best \
        --target-mode parallel-attn --device cuda:0

  Distributed (the REAL PT student verifier, e.g. the D=1-B checkpoint):
    .venv/bin/torchrun --standalone --nproc_per_node=8 \
        scripts/probe_draft_verify_decode.py \
        --hf-model <teacher path> \
        --pt-checkpoint train_out/qwen3_5_9b_n16_d1_postattn/best \
        --sync-phase post-attn

In distributed mode every rank runs the identical loop; the lm_head-owner rank
computes proposals/acceptance and broadcasts token ids so the PT forward's
collectives stay in lockstep on all ranks.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.eval.dense_parallel import MODES, build_windows, dense_window_forward
from pt_converter.eval.draft_verify import (
    draft_verify_generate,
    greedy_verify_step_factory,
    naive_greedy_propose_factory,
)
from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)


def main() -> int:
    p = argparse.ArgumentParser()
    # Verifier — single-GPU dense mode:
    p.add_argument("--target-model", default=None,
                   help="Dense verifier weights (HF path). Mutually exclusive with --pt-checkpoint.")
    p.add_argument("--target-mode", default="hf", choices=("hf",) + MODES,
                   help="Dense verifier forward: stock hf or a dense_parallel rewiring arm.")
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--first-parallel-layer", type=int, default=1)
    # Verifier — distributed PT mode:
    p.add_argument("--pt-checkpoint", default=None,
                   help="Trained PT checkpoint dir (track_*.safetensors + manifest). Launch via torchrun.")
    p.add_argument("--hf-model", default=None,
                   help="Dense teacher path (tokenizer + config source in PT mode).")
    p.add_argument("--sync-phase", choices=["boundary", "post-attn", "window-parallel"],
                   default="boundary")
    p.add_argument("--sync-indices", default=None,
                   help="Override the manifest sync schedule (raw convert outputs carry none).")
    # Draft + loop:
    p.add_argument("--draft-model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--k", type=int, default=8, help="Draft block length.")
    p.add_argument("--num-prompts", type=int, default=24)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--verify-events", type=int, default=32,
                   help="Sync events per verify pass (32 = the D=1 per-layer count).")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device", default="cuda:0", help="Single-GPU mode only.")
    # Data (held-out front of the training mixture; prompts only).
    p.add_argument("--data-preset", default=DEFAULT_PRESET, choices=preset_names())
    p.add_argument("--data-source", action="append", default=None, metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]")
    p.add_argument("--skip-docs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if (args.target_model is None) == (args.pt_checkpoint is None):
        raise SystemExit("[error] pass exactly one of --target-model / --pt-checkpoint")

    distributed = args.pt_checkpoint is not None
    rank, bcast_src = 0, 0
    if distributed:
        from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp
        from pt_converter.dist.groups import build_groups
        from pt_converter.eval.lm_eval_adapter import is_lm_head_owner
        from pt_converter.model.pt_model import PTWrappedModel
        from pt_converter.utils.checkpoint import load_manifest, load_track

        if args.hf_model is None:
            raise SystemExit("[error] --pt-checkpoint mode needs --hf-model for tokenizer/config")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        device = torch.device(torch.cuda.current_device())

        manifest = load_manifest(args.pt_checkpoint)
        layout = build_groups(n_tracks=manifest.n_tracks)
        if args.sync_indices is not None:
            sync_layers = [int(x) for x in args.sync_indices.split(",") if x.strip() != ""]
        elif manifest.sync_layer_indices is not None:
            sync_layers = list(manifest.sync_layer_indices)
        else:
            raise SystemExit("[error] no sync schedule in manifest; pass --sync-indices")

        tok = AutoTokenizer.from_pretrained(args.hf_model)
        cfg = AutoConfig.from_pretrained(args.hf_model)
        text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
        if rank == 0:
            print(f"[init] PT verifier {args.pt_checkpoint} n_tracks={manifest.n_tracks} "
                  f"K={layout.tracks_per_rank} sync_phase={args.sync_phase}", flush=True)
        student = PTWrappedModel(
            text_config=text_cfg,
            n_tracks=manifest.n_tracks,
            local_track_ids=layout.local_track_ids,
            sync_after_layers=sync_layers,
            track_group=layout.track_group,
        )
        track_states = {tid: load_track(args.pt_checkpoint, tid) for tid in layout.local_track_ids}
        student.load_track_state_dicts(track_states, strict=True)
        student = student.to(device).to(torch.bfloat16)
        if args.sync_phase != "boundary":
            student.set_sync_phase(args.sync_phase)
        wrap_student_with_fsdp(student, layout)
        student.eval()

        # The lm_head-owner rank drives proposals/acceptance; find its global rank.
        flag = torch.tensor(
            [rank if is_lm_head_owner(student) else -1], device=device, dtype=torch.long
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        bcast_src = int(flag.item())

        def verifier_logits(ids: torch.Tensor) -> "torch.Tensor | None":
            logits, _ = student(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                return_sync_hiddens=False,
            )
            return logits

        owner = rank == bcast_src
        draft = None
        if owner:
            print(f"[init] loading draft {args.draft_model} on owner rank {rank} …", flush=True)
            draft = AutoModelForCausalLM.from_pretrained(
                args.draft_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                attn_implementation=args.attn_impl,
            ).eval().to(device)
        base_propose = naive_greedy_propose_factory(draft) if owner else None
        base_verify = greedy_verify_step_factory(verifier_logits)

        def draft_propose(seq: torch.Tensor, k: int) -> torch.Tensor:
            buf = base_propose(seq, k) if owner else \
                torch.empty(k, dtype=torch.long, device=device)
            dist.broadcast(buf, src=bcast_src)
            return buf

        def verify_step(seq: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
            k = proposals.numel()
            # Every rank runs the PT forward (collectives inside); only the
            # owner sees logits and computes acceptance.
            if owner:
                emitted = base_verify(seq, proposals)
                buf = torch.full((k + 2,), -1, dtype=torch.long, device=device)
                buf[0] = emitted.shape[1]
                buf[1 : 1 + emitted.shape[1]] = emitted[0]
            else:
                verifier_logits(torch.cat([seq, proposals.view(1, -1)], dim=1))
                buf = torch.empty(k + 2, dtype=torch.long, device=device)
            dist.broadcast(buf, src=bcast_src)
            return buf[1 : 1 + int(buf[0])].view(1, -1)
    else:
        device = torch.device(args.device)
        print(f"[init] loading dense verifier {args.target_model} …", flush=True)
        tok = AutoTokenizer.from_pretrained(args.target_model)
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation=args.attn_impl,
        ).eval().to(device)
        if args.target_mode == "hf":
            def verifier_logits(ids: torch.Tensor) -> torch.Tensor:
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
            def verifier_logits(ids: torch.Tensor) -> torch.Tensor:
                hidden = dense_window_forward(
                    text_model, ids, None, windows=windows, mode=args.target_mode
                )
                return target.lm_head(hidden)
        print(f"[init] loading draft {args.draft_model} …", flush=True)
        draft = AutoModelForCausalLM.from_pretrained(
            args.draft_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation=args.attn_impl,
        ).eval().to(device)
        draft_propose = naive_greedy_propose_factory(draft)
        verify_step = greedy_verify_step_factory(verifier_logits)

    # --- prompts: held-out front of the training mixture ---
    if args.data_source:
        data_cfg = CalibrationDataConfig(
            sources=[parse_source_spec(s) for s in args.data_source],
            seq_len=args.prompt_len, seed=args.seed, skip_docs=args.skip_docs,
        )
    else:
        data_cfg = CalibrationDataConfig(
            sources=preset_sources(args.data_preset),
            seq_len=args.prompt_len, seed=args.seed, skip_docs=args.skip_docs,
        )
    loader = DataLoader(PackedTokenStream(tok, data_cfg), batch_size=1, num_workers=0)

    tot_tokens = tot_events = 0
    per_prompt: "list[float]" = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_prompts:
                break
            ids = batch["input_ids"]
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            prompt = ids[:, : args.prompt_len].to(device)
            if distributed:  # bulletproof lockstep even if streams ever drift
                dist.broadcast(prompt, src=bcast_src)
            _seq, tokens, events = draft_verify_generate(
                draft_propose, verify_step, prompt,
                k=args.k, max_new=args.max_new, eos_token_id=tok.eos_token_id,
            )
            tot_tokens += tokens
            tot_events += events
            per_prompt.append(tokens / max(events, 1))
            if rank == 0 and (i + 1) % 4 == 0:
                print(f"[gen] {i + 1}/{args.num_prompts} prompts  "
                      f"running tau={tot_tokens / max(tot_events, 1):.3f}", flush=True)

    if rank == 0:
        tau = tot_tokens / max(tot_events, 1)
        spt = args.verify_events / tau if tau else float("inf")
        name = args.pt_checkpoint if distributed else \
            f"{args.target_model} [{args.target_mode}]"
        print(f"\n=== draft-verify decode probe ===")
        print(f"  verifier={name}  draft={args.draft_model}  k={args.k}")
        print(f"  prompts={len(per_prompt)}  emitted={tot_tokens}  verify-events={tot_events}")
        print(f"  ON-GENERATION tau={tau:.3f}  syncs/token={spt:.2f}  "
              f"(verify pass = {args.verify_events} events; D=1 plain decode = "
              f"{args.verify_events}/token)")
        print(f"  per-prompt tau: min={min(per_prompt):.2f}  max={max(per_prompt):.2f}")
        print("  Emitted text is bit-exact the verifier's greedy decode; tau is the "
              "deployed amortization (offline TF read was the data-distribution proxy).")
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
