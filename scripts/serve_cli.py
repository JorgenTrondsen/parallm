"""torchrun entry: single-stream lockstep decode through the sparse-replica engine.

One rank per GPU, K = n_tracks/world tracks per rank; rank 0 owns embed +
lm_head (the head). ``--residency streamed`` keeps the packed pool in pinned
host DRAM and pulls it through a window-granular device ring whose copies
hide behind the sync stalls (composes with CUDA-graphed windows);
``--latency-ms`` simulates the inter-node link by sleeping at every comm
round (embed broadcast + each boundary all-reduce).

    torchrun --standalone --nproc-per-node=8 scripts/serve_cli.py \\
        --tracks-dir convert_out/qwen3_6_27b_n8_tracks \\
        --hf-model <checkpoint dir> \\
        --sync-indices 15,31,47,63 --latency-ms 20 \\
        --prompt "The capital of France is" --max-new-tokens 32
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from parallm.dist.groups import build_groups
from parallm.engine import PackedShadow, generate, generate_draft_verify
from parallm.model.pt_model import PTWrappedModel
from parallm.utils.checkpoint import load_manifest, load_track


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks-dir", required=True)
    p.add_argument("--hf-model", required=True, help="Checkpoint dir (config + tokenizer only)")
    p.add_argument("--replicas", default=None,
                   help="Packed pool (default <tracks-dir>/replicas.safetensors)")
    p.add_argument("--sync-indices", default=None,
                   help="Comma-separated boundary layer indices (falls back to the manifest)")
    p.add_argument("--latency-ms", type=float, default=0.0)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--residency", choices=["resident", "streamed"], default="resident",
                   help="streamed = packed pool in pinned host DRAM, prefetched through "
                        "a device ring behind the sync stalls")
    p.add_argument("--ring-windows", type=int, default=2,
                   help="Ring capacity in sync windows (D layers each); >= 2 and must "
                        "divide the window count (2 = each window's refill hides "
                        "behind one stall; more windows = deeper copy pipeline)")
    p.add_argument("--pool-codec", choices=["raw", "ent"], default="raw",
                   help="ent = stream the entropy-coded codes plane (see "
                        "scripts/repack_replicas_entropy.py) and GPU-decode it "
                        "into the ring during the stalls; ~21-26%% fewer "
                        "PCIe/DRAM bytes, bit-identical. Requires streamed")
    p.add_argument("--ent-file", default=None,
                   help="Entropy artifact (default <tracks-dir>/replicas.ent.safetensors)")
    p.add_argument("--dec-chunks", type=int, default=4,
                   help="Decode launches per window (chunk c decodes while "
                        "chunk c+1's blobs copy — intra-window pipelining)")
    p.add_argument("--draft-k", default="0",
                   help="Block draft-verify decode: the head drafts k tokens "
                        "(--draft-model), one lockstep verify chunk checks "
                        "them — emitted ≡ plain greedy, syncs/token = "
                        "boundaries/τ. 0 = plain per-token decode. Comma list "
                        "sweeps in one process (weights load once): '0,16,32,48'")
    p.add_argument("--draft-model", default=None,
                   help="Same-tokenizer HF checkpoint for the drafter "
                        "(loaded on the head rank only)")
    p.add_argument("--no-fused", action="store_true",
                   help="Disable the packed-GEMV kernel; decode via the dense "
                        "unpack path (the v1 baseline)")
    p.add_argument("--no-cuda-graphs", action="store_true",
                   help="Disable CUDA-graphed decode windows (on by default in "
                        "both residency modes) — the eager debug path")
    p.add_argument("--profile", action="store_true",
                   help="Per-window compute breakdown (device-synced segment walls; "
                        "serializes the compute/stall overlap, so the profiled total "
                        "reads higher than the plain wall)")
    p.add_argument("--torch-profile", action="store_true",
                   help="Wrap decode in torch.profiler and report GPU-busy time vs "
                        "wall (the launch-overhead accounting) + top kernels")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    manifest = load_manifest(args.tracks_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    if args.sync_indices:
        sync = [int(x) for x in args.sync_indices.split(",") if x.strip()]
    elif manifest.sync_layer_indices:
        sync = list(manifest.sync_layer_indices)
    else:
        raise SystemExit("[error] no --sync-indices and the manifest carries no schedule")

    cfg = AutoConfig.from_pretrained(args.hf_model)
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if rank == 0:
        print(f"[init] N={manifest.n_tracks} world={layout.world_size} sync={sync} "
              f"latency={args.latency_ms}ms", flush=True)
    student = PTWrappedModel(
        text_config=text_cfg,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=sync,
        track_group=layout.track_group,
    )
    student.load_track_state_dicts(
        {tid: load_track(args.tracks_dir, tid) for tid in layout.local_track_ids},
        strict=True,
    )
    # bf16 on host FIRST: .to(device) of the fp32 module doubles the device
    # footprint transiently (the old 23.4 GB load peak on the 27B).
    student = student.to(torch.bfloat16).to(device).eval()

    pool_path = args.replicas or os.path.join(args.tracks_dir, "replicas.safetensors")
    ring_layers = window = None
    if args.residency == "streamed":
        num_layers = manifest.num_layers
        bounds = sorted(set(sync) - {num_layers - 1}) + [num_layers - 1]
        window = num_layers // len(bounds)
        if bounds != list(range(window - 1, num_layers, window)):
            raise SystemExit(f"[error] --residency streamed needs a uniform sync "
                             f"schedule (window ring), got boundaries {bounds}")
        ring_layers = args.ring_windows * window
    ent_path = None
    if args.pool_codec == "ent":
        if args.residency != "streamed":
            raise SystemExit("[error] --pool-codec ent requires --residency streamed")
        ent_path = args.ent_file or os.path.join(args.tracks_dir,
                                                 "replicas.ent.safetensors")
    if rank == 0:
        print(f"[init] loading packed pool {pool_path} ({args.residency}"
              + (f", ring={ring_layers} layers" if ring_layers else "")
              + (f", codec=ent {ent_path}" if ent_path else "") + ")…", flush=True)
    shadow = PackedShadow(pool_path, args.tracks_dir, student, device,
                          stream_ring_layers=ring_layers, fused=not args.no_fused,
                          stream_window_layers=window,
                          ent_path=ent_path, dec_chunks=args.dec_chunks)

    # Every rank tokenizes the same prompt (deterministic lockstep shapes);
    # only the head's embedding is real.
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)

    draft_ks = [int(x) for x in str(args.draft_k).split(",") if x.strip()]
    draft_propose = None
    if any(draft_ks):
        if args.temperature > 0:
            raise SystemExit("[error] draft-verify decode is greedy-only")
        if rank == 0:
            if not args.draft_model:
                raise SystemExit("[error] --draft-k needs --draft-model")
            from transformers import AutoModelForCausalLM

            dm = AutoModelForCausalLM.from_pretrained(
                args.draft_model, dtype=torch.bfloat16).to(device).eval()

            @torch.no_grad()
            def draft_propose(seq, k):
                o = dm.generate(seq, max_new_tokens=k, min_new_tokens=k,
                                do_sample=False, pad_token_id=tok.eos_token_id)
                return o[0, -k:]
    load_peak = torch.cuda.max_memory_allocated(device)
    if rank == 0:
        print(f"[init] pool={shadow.meta.get('config')} prompt={input_ids.shape[1]} tokens "
              f"resident={torch.cuda.memory_allocated(device)/2**30:.2f} GB "
              f"(load peak {load_peak/2**30:.2f} GB)", flush=True)
    torch.cuda.reset_peak_memory_stats(device)

    prof_ctx = None
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        prof_ctx = torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof_ctx.__enter__()
    results = []  # (ki, gen, timing) per sweep entry
    for ki in draft_ks:
        if shadow.ring is not None:
            shadow.ring.copy_log.clear()  # stream stats reflect the last entry
        timing = {"profile": args.profile}
        if ki:
            gen = generate_draft_verify(student, shadow, input_ids, sync,
                                        max_new_tokens=args.max_new_tokens,
                                        draft_propose=draft_propose, k=ki,
                                        latency_s=args.latency_ms / 1e3, timing=timing)
        else:
            gen = generate(student, shadow, input_ids, sync,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature,
                           latency_s=args.latency_ms / 1e3, timing=timing,
                           use_cuda_graphs=not args.no_cuda_graphs)
        torch.cuda.synchronize()
        results.append((ki, gen, timing))
    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)

    # Per-rank HBM ledger (collective — every rank contributes; head prints).
    # "params" = all student weights on this rank (embed + lm_head included on
    # the head); "other" in the table = KV/activations/graph pools/scratch.
    params_b = sum(p.numel() * p.element_size() for p in student.parameters())
    head_b = sum(p.numel() * p.element_size()
                 for m in [student.lm_head] + [tm.embed_tokens for tm in student.text_models]
                 if m is not None for p in m.parameters())
    ledger = {
        "K": len(student.text_models),
        "steady": torch.cuda.memory_allocated(device),
        "peak": torch.cuda.max_memory_allocated(device),
        "load_peak": load_peak,
        "params": params_b,
        "head_extra": head_b,
        "ring": shadow.ring.device_bytes() if shadow.ring is not None else 0,
    }
    ledgers: list = [None] * dist.get_world_size()
    dist.all_gather_object(ledgers, ledger)

    if student.lm_head is not None:
        last_layer = manifest.num_layers - 1
        S = len(set(sync) - {last_layer}) + 1  # boundaries per pass
        for ki, gen, timing in results:
            print("\n" + tok.decode(gen[0]), flush=True)
            if ki:
                pb = timing["per_block_ms"]
                blocks, tokens_n = timing["blocks"], timing["tokens"]
                n_chunks = 1 + blocks
                tau = tokens_n / blocks
                mean = sum(pb) / tokens_n  # ms per emitted token
                rpb = (timing["rounds"] - (S + 1)) / blocks  # rounds per block
                print(f"\n===== draft-verify panel =====")
                print(f"prefill: {timing['prefill_ms']:.1f} ms ({input_ids.shape[1]} tokens)")
                print(f"decode:  k={ki} blocks={blocks} tokens={tokens_n} "
                      f"| τ={tau:.2f} tok/verify")
                print(f"         mean {mean:.1f} ms/tok | {1e3 / mean:.2f} tok/s "
                      f"| block mean {statistics.mean(pb):.1f} ms")
                print(f"syncs:   {S}/pass → {S / tau:.2f}/token | rounds {rpb:.0f}/block "
                      f"→ {rpb / tau:.2f}/token × {args.latency_ms:.0f} ms = "
                      f"{rpb / tau * args.latency_ms:.1f} ms/tok stall")
            else:
                per_tok = timing["per_token_ms"]
                n_chunks = 1 + len(per_tok)
                rounds_per_chunk = timing["rounds"] / n_chunks
                stall = rounds_per_chunk * args.latency_ms
                mean = statistics.mean(per_tok) if per_tok else float("nan")
                print(f"\n===== timing panel =====")
                print(f"prefill: {timing['prefill_ms']:.1f} ms ({input_ids.shape[1]} tokens)")
                print(f"decode:  mean {mean:.1f} ms/tok | p50 {statistics.median(per_tok):.1f} "
                      f"| {1e3/mean:.2f} tok/s")
                print(f"rounds:  {rounds_per_chunk:.0f}/token × {args.latency_ms:.0f} ms "
                      f"= {stall:.0f} ms stall | compute+overhead {mean - stall:.1f} ms")
        print(f"HBM:     run peak {torch.cuda.max_memory_allocated(device)/2**30:.2f} GB "
              f"| steady {torch.cuda.memory_allocated(device)/2**30:.2f} GB")
        gb = 2**30
        print(f"ranks:   {'r':>2} {'K':>2} {'steady':>7} {'peak':>7} {'load':>7} "
              f"{'params':>7} {'ring':>6} {'other':>6}")
        for r, L in enumerate(ledgers):
            other = L["steady"] - L["params"] - L["ring"]
            print(f"         {r:>2} {L['K']:>2} {L['steady']/gb:>6.2f}G {L['peak']/gb:>6.2f}G "
                  f"{L['load_peak']/gb:>6.2f}G {L['params']/gb:>6.2f}G "
                  f"{L['ring']/gb:>5.2f}G {other/gb:>5.2f}G")
        peer = ledgers[-1]
        track_b = (peer["params"] - peer["head_extra"]) / peer["K"]
        proj = peer["steady"] - (peer["K"] - 1) * track_b
        kind = "track" if peer["head_extra"] == 0 else "head"
        print(f"1-track {kind} node (projected from rank {len(ledgers)-1}): steady "
              f"{proj/gb:.2f} GB = {peer['steady']/gb:.2f} − {peer['K']-1} × "
              f"{track_b/gb:.2f} GB/track")
        if shadow.ring is not None:
            host_gb = sum(
                e.mask.nbytes
                + (e.blob.nbytes if e.blob is not None else e.values.nbytes)
                + (e.scale.nbytes if e.scale is not None else 0)
                for e in shadow.entries.values()) / 2**30
            nw = shadow.ring.n_windows
            dec = shadow.ring.copy_stats()[2 * nw:]  # skip prefill + warmup passes
            per_pass = [dec[i:i + nw]
                        for i in range(0, len(dec) - len(dec) % nw, nw)]
            if per_pass:
                ms = [m for _, m in dec]
                bw = sum(b for b, _ in dec) / 2**30 / max(sum(ms) / 1e3, 1e-9)
                # Upper-estimate law: each window's refill budget is >= one
                # stall (window 0 additionally gets the embed round).
                added = statistics.mean(
                    sum(max(0.0, m - args.latency_ms) for _, m in p) for p in per_pass)
                print(f"stream:  {sum(b for b, _ in per_pass[0])/2**30:.2f} GB/"
                      f"{'pass' if results[-1][0] else 'token'} "
                      f"over {nw} windows | copy mean {statistics.mean(ms):.1f} "
                      f"max {max(ms):.1f} ms | H2D {bw:.1f} GB/s | "
                      f"copy beyond one {args.latency_ms:.0f} ms stall each: "
                      f"+{added:.1f} ms/tok | pool pinned {host_gb:.2f} GB host")
        if timing.get("window_ms"):
            decode_wins = timing["window_ms"][1:]  # chunk 0 = prefill
            decode_syncs = timing["sync_ms"][1:]
            win_means = [statistics.mean(col) for col in zip(*decode_wins)]
            sync_means = [statistics.mean(col) for col in zip(*decode_syncs)]
            total = sum(win_means) + sum(sync_means)
            print(f"\n--- profiled serial breakdown (decode, per token; overlap off) ---")
            print(f"windows: " + "  ".join(f"w{j}={m:.0f}ms" for j, m in enumerate(win_means))
                  + f"  | Σ compute {sum(win_means):.0f} ms")
            print(f"syncs:   " + "  ".join(f"s{j}={m:.0f}ms" for j, m in enumerate(sync_means))
                  + f"  | Σ sync {sum(sync_means):.0f} ms (all-reduce wait + {args.latency_ms:.0f} ms sleep)")
            print(f"total:   {total:.0f} ms serial (unprofiled wall {mean:.0f} ms)")
        if prof_ctx is not None:
            def _dev_us(e):
                return getattr(e, "self_device_time_total",
                               getattr(e, "self_cuda_time_total", 0))

            ka = prof_ctx.key_averages()
            cuda_us = sum(_dev_us(e) for e in ka)
            n_kernels = sum(e.count for e in ka if _dev_us(e) > 0)
            print(f"\n--- torch.profiler (whole run: prefill + {n_chunks - 1} decode fwds) ---")
            print(f"GPU busy: {cuda_us / 1e3:.0f} ms total ≈ "
                  f"{cuda_us / 1e3 / n_chunks:.1f} ms per forward pass")
            print(f"kernel executions: {n_kernels} ≈ {n_kernels // n_chunks}/pass")
            top = sorted(ka, key=lambda e: -_dev_us(e))[:6]
            for e in top:
                print(f"  {e.key[:60]:<60} {_dev_us(e)/1e3:7.1f} ms x{e.count}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
