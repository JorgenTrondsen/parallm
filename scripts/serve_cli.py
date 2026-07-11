"""torchrun entry: single-stream lockstep decode through the sparse-replica engine.

One rank per GPU, K = n_tracks/world tracks per rank; rank 0 owns embed +
lm_head (the head). The replica pool streams nothing yet (resident-packed,
P3 adds the host ring); ``--latency-ms`` simulates the inter-node link by
sleeping at every comm round (embed broadcast + each boundary all-reduce).

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
from parallm.engine import PackedShadow, generate
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
                   help="Ring capacity in sync windows (D layers each); 2 = the D*2 "
                        "double buffer, 1 measured equivalent at stall >= 40 ms")
    p.add_argument("--no-fused", action="store_true",
                   help="Disable the packed-GEMV kernel; decode via the dense "
                        "unpack path (the v1 baseline)")
    p.add_argument("--no-cuda-graphs", action="store_true",
                   help="Disable CUDA-graphed decode windows (graphs are on by "
                        "default in resident mode; streamed is always eager)")
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
    student = student.to(device).to(torch.bfloat16).eval()

    pool_path = args.replicas or os.path.join(args.tracks_dir, "replicas.safetensors")
    ring_layers = None
    if args.residency == "streamed":
        num_layers = manifest.num_layers
        window = num_layers // len(sync)  # uniform schedule assumed
        ring_layers = args.ring_windows * window
    if rank == 0:
        print(f"[init] loading packed pool {pool_path} ({args.residency}"
              + (f", ring={ring_layers} layers" if ring_layers else "") + ")…", flush=True)
    shadow = PackedShadow(pool_path, args.tracks_dir, student, device,
                          stream_ring_layers=ring_layers, fused=not args.no_fused)

    # Every rank tokenizes the same prompt (deterministic lockstep shapes);
    # only the head's embedding is real.
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    if rank == 0:
        print(f"[init] pool={shadow.meta.get('config')} prompt={input_ids.shape[1]} tokens "
              f"resident={torch.cuda.memory_allocated(device)/2**30:.2f} GB", flush=True)

    timing: dict = {"profile": args.profile}
    prof_ctx = None
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        prof_ctx = torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof_ctx.__enter__()
    gen = generate(student, shadow, input_ids, sync,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature,
                   latency_s=args.latency_ms / 1e3, timing=timing,
                   use_cuda_graphs=(not args.no_cuda_graphs
                                    and args.residency == "resident"))
    torch.cuda.synchronize()
    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)

    if student.lm_head is not None:
        print("\n" + tok.decode(gen[0]), flush=True)
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
        print(f"HBM:     peak {torch.cuda.max_memory_allocated(device)/2**30:.2f} GB")
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
            n_chunks = 1 + len(timing["per_token_ms"])
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
