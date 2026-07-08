"""Microbenchmark: can layer-slab replica prefetch hide behind the sync stall?

Models one track device of the D=8 phased deployment (Qwen3.5-9B, N=16):
32 layers in 4 windows, a sync stall S at each window boundary (a host-side
wait — the copy engine keeps running, which is the correct model of a
network stall), the replica pool (qwanda:4:0.5 = 15 replicas at 3 bits/w =
2.505 GB) pinned in host DRAM, and a device ring buffer of R bytes streamed
at g-layer slab granularity. Replica weights are static, but cyclic access
with R < pool evicts every slab before its reuse, so each pass re-streams
the full pool. Decode compute per layer is weights-read-bound (~0.1 ms here),
so the stall is essentially the entire hiding budget.

Law under test (per window): added_stall ~= max(0, W/BW - min(S, R/BW))
where W = one window of replica bytes. Per pass the link cannot be beaten:
added_pass >= max(0, POOL/BW - 4*S - compute).

Own-track weights (0.89 GB bf16) stay resident by default; --stream-own
folds them into the streamed slabs (the negative-trade arm, for the record).

No model weights are loaded; slabs are synthetic but byte-exact to the
frontier account. Quality is unaffected by construction — this measures
timing only.
"""

import argparse
import time

import torch

HIDDEN = 4096
LAYERS = 32
WINDOWS = 4
WINDOW_LAYERS = LAYERS // WINDOWS
# 15 replicas x 445.3M params x 3 bits/w, per layer, expressed as bf16 columns
REPLICA_LAYER_COLS = 9554  # 4096*9554*2 B = 78.27 MB/layer -> pool 2.505 GB
OWN_LAYER_COLS = 3399  # 4096*3399*2 B = 27.85 MB/layer -> own blocks 0.891 GB
DECODE_K = 16


def layer_bytes(stream_own: bool) -> int:
    cols = REPLICA_LAYER_COLS + (OWN_LAYER_COLS if stream_own else 0)
    return HIDDEN * cols * 2


def measure_h2d_bandwidth(pool: torch.Tensor, device: str) -> tuple[float, float]:
    """Return (pinned, pageable) H2D bandwidth in GB/s for one-window copies."""
    window = pool[:WINDOW_LAYERS]
    dst = torch.empty_like(window, device=device)
    stream = torch.cuda.Stream()
    results = []
    for src in (window, window.clone()):  # pinned, then pageable
        with torch.cuda.stream(stream):
            for _ in range(5):
                dst.copy_(src, non_blocking=True)
            stream.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(stream)
            for _ in range(20):
                dst.copy_(src, non_blocking=True)
            end.record(stream)
            stream.synchronize()
        gb = 20 * src.numel() * 2 / 1e9
        results.append(gb / (start.elapsed_time(end) / 1e3))
    del dst
    return results[0], results[1]


def run_arm(
    pool: torch.Tensor,
    own_weights: list[torch.Tensor] | None,
    x: torch.Tensor,
    *,
    g: int,
    n_slots: int,
    stall_ms: float,
    warmup: int,
    passes: int,
    device: str,
) -> dict:
    """One streamed configuration; returns per-pass wall time and bytes moved."""
    slabs_per_pass = LAYERS // g
    total_occ = (warmup + passes) * slabs_per_pass
    slots = [
        torch.empty((g, HIDDEN, pool.shape[2]), dtype=torch.bfloat16, device=device)
        for _ in range(n_slots)
    ]
    copy_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()
    ready = {}  # occurrence -> event on copy_stream
    free = [torch.cuda.Event() for _ in range(n_slots)]  # re-recorded per consumption
    state = {"next_copy": 0, "bytes": 0}

    def pump_copies(limit: int) -> None:
        while state["next_copy"] < min(total_occ, limit):
            occ = state["next_copy"]
            slot = occ % n_slots
            src = pool[(occ * g) % LAYERS : (occ * g) % LAYERS + g]
            with torch.cuda.stream(copy_stream):
                if occ >= n_slots:
                    copy_stream.wait_event(free[slot])
                slots[slot].copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(copy_stream)
            ready[occ] = ev
            state["bytes"] += src.numel() * 2
            state["next_copy"] += 1

    pump_copies(n_slots)  # initial fill
    stall_s = stall_ms / 1e3
    occ = 0
    # Timing: host clock at pass boundaries, no device sync — the host is
    # already gated per window by win_done.synchronize(), and the copy
    # stream's in-flight lookahead state is the same (saturated) at both
    # boundaries, so it cancels instead of leaking out of the timed region.
    t0 = None
    for p in range(warmup + passes):
        if p == warmup:
            t0 = time.perf_counter()
        for _ in range(WINDOWS):
            with torch.cuda.stream(comp_stream):
                for _ in range(WINDOW_LAYERS // g):
                    slot = occ % n_slots
                    comp_stream.wait_event(ready.pop(occ))
                    for layer_in_slab in range(g):
                        x @ slots[slot][layer_in_slab]
                        if own_weights is not None:
                            layer = (occ * g + layer_in_slab) % LAYERS
                            x @ own_weights[layer]
                    free[slot].record(comp_stream)
                    occ += 1
                    pump_copies(occ + n_slots)
                win_done = torch.cuda.Event()
                win_done.record(comp_stream)
            win_done.synchronize()
            time.sleep(stall_s)
    wall = (time.perf_counter() - t0) / passes
    assert state["next_copy"] == total_occ  # every slab enqueued exactly once
    torch.cuda.synchronize()
    del slots
    return {"pass_ms": wall * 1e3}


def run_baseline(pool, own_weights, x, *, stall_ms, warmup, passes, device) -> float:
    """All slabs resident, no copies: pass wall = compute + 4 stalls."""
    resident = pool.to(device)
    stall_s = stall_ms / 1e3
    t0 = None
    for p in range(warmup + passes):
        if p == warmup:
            t0 = time.perf_counter()
        for w in range(WINDOWS):
            for layer in range(w * WINDOW_LAYERS, (w + 1) * WINDOW_LAYERS):
                x @ resident[layer]
                if own_weights is not None:
                    x @ own_weights[layer]
            torch.cuda.synchronize()
            time.sleep(stall_s)
    wall = (time.perf_counter() - t0) / passes
    del resident
    return wall * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slab-layers", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--ring-mb", type=float, nargs="+", default=[156, 313, 626, 1252])
    ap.add_argument("--stall-ms", type=float, nargs="+", default=[20, 40, 60, 200])
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--stream-own", action="store_true",
                    help="fold own-track weights into the streamed slabs")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)

    cols = REPLICA_LAYER_COLS + (OWN_LAYER_COLS if args.stream_own else 0)
    pool = torch.randn(LAYERS, HIDDEN, cols, dtype=torch.bfloat16).pin_memory()
    own_weights = None
    if not args.stream_own:
        own_weights = [
            torch.randn(HIDDEN, OWN_LAYER_COLS, dtype=torch.bfloat16, device=args.device)
            for _ in range(LAYERS)
        ]
    x = torch.randn(DECODE_K, HIDDEN, dtype=torch.bfloat16, device=args.device)

    lb = layer_bytes(args.stream_own)
    win_gb = WINDOW_LAYERS * lb / 1e9
    print(f"device={torch.cuda.get_device_name(args.device)}  "
          f"stream_own={args.stream_own}")
    print(f"pool={LAYERS * lb / 1e9:.3f} GB  window={win_gb * 1e3:.0f} MB  "
          f"layer={lb / 1e6:.1f} MB")
    bw_pin, bw_page = measure_h2d_bandwidth(pool, args.device)
    print(f"H2D bandwidth: pinned {bw_pin:.1f} GB/s | pageable {bw_page:.1f} GB/s")
    print()
    header = (f"{'g':>2} {'ring':>8} {'slots':>5} {'stall':>6} {'pass_ms':>8} "
              f"{'base_ms':>8} {'added':>7} {'pred':>6} {'GB/pass':>8} {'resident':>9}")
    print(header)

    for stall in args.stall_ms:
        base_ms = run_baseline(pool, own_weights, x, stall_ms=stall,
                               warmup=args.warmup, passes=args.passes,
                               device=args.device)
        for g in args.slab_layers:
            slab_mb = g * lb / 1e6
            for ring_mb in args.ring_mb:
                n_slots = round(ring_mb * 1e6 / (g * lb))
                if n_slots < 2 or n_slots > LAYERS // g:
                    continue
                ring_gb = n_slots * g * lb / 1e9
                r = run_arm(pool, own_weights, x, g=g, n_slots=n_slots,
                            stall_ms=stall, warmup=args.warmup,
                            passes=args.passes, device=args.device)
                added = r["pass_ms"] - base_ms
                win_ms = win_gb / bw_pin * 1e3
                # max of the buffer-starvation law (copy engine idles during
                # stalls once the ring is full) and the link-saturation bound
                # (the pass can never beat pool/BW).
                pred = max(
                    0.0,
                    WINDOWS * max(0.0, win_ms - min(stall, ring_gb / bw_pin * 1e3)),
                    WINDOWS * win_ms - base_ms,
                )
                resident = ring_gb + (0.0 if args.stream_own
                                      else LAYERS * HIDDEN * OWN_LAYER_COLS * 2 / 1e9)
                print(f"{g:>2} {ring_gb * 1e3:>6.0f}MB {n_slots:>5} {stall:>5.0f}ms "
                      f"{r['pass_ms']:>8.1f} {base_ms:>8.1f} {added:>+6.1f}ms "
                      f"{pred:>5.1f} {win_gb * WINDOWS:>8.2f} {resident:>7.2f}GB")
        print()


if __name__ == "__main__":
    main()
