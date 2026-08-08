"""Device-synced per-phase step timers for the training loop.

`s/step` alone cannot tell you whether a step is compute-bound or launch-bound, and
this trainer had no finer instrumentation at all — which is how an optimization got
chosen off a microbenchmark of a phase worth ~7% of the step. This is the smallest
thing that prevents that: named phases, exclusive wall time, one line per log step.

Same pattern the inference side already uses (`engine._mark`): a
`torch.cuda.synchronize()` on each boundary so the number attributed to a phase is
that phase's, not whatever was still queued from the previous one. **The syncs
serialize overlap the normal path enjoys, so a profiled step reads HIGHER than an
unprofiled one** — this is a serial cost decomposition, not a wall-clock measurement.
Compare phase shares between profiled runs; take totals from unprofiled ones.

Nesting is allowed and reports EXCLUSIVE time: a phase entered inside another has
its cost subtracted from the parent, so `tf_backward` nested inside `tf_block_loop`
does not get counted twice.

Disabled by default and a strict no-op in that state — `PhaseTimer()` costs a
predicate per phase, no syncs, so call sites need no `if`.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class PhaseTimer:
    """Accumulates exclusive wall time per named phase across a step.

    Args:
        enabled: False makes every method a no-op (the default, so the training
                 path pays nothing).
        sync: device-synchronize at phase boundaries. Required for the numbers to
              mean anything on CUDA; leave on unless deliberately measuring
              host-side dispatch time.
        record: emit a `torch.profiler.record_function` range per phase, so phase
                names appear in a kernel trace. Independent of `enabled` on purpose:
                a kernel trace wants the NAMES but not the syncs, which would
                serialize exactly the overlap it is trying to measure. So
                `PhaseTimer(enabled=False, record=True)` is the kernel-trace config.
    """

    def __init__(self, enabled: bool = False, sync: bool = True, record: bool = False):
        self.enabled = enabled
        self.sync = sync
        self.record = record
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._stack: list[float] = []
        self._t0: float | None = None

    def _mark(self) -> float:
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def start_step(self) -> None:
        if self.enabled:
            self._t0 = self._mark()

    @contextmanager
    def phase(self, name: str):
        if not (self.enabled or self.record):
            yield
            return
        if not self.enabled:
            with torch.profiler.record_function(name):
                yield
            return
        rec = torch.profiler.record_function(name) if self.record else None
        if rec is not None:
            rec.__enter__()
        t0 = self._mark()
        self._stack.append(0.0)
        try:
            yield
        finally:
            dt = self._mark() - t0
            child = self._stack.pop()
            # Exclusive: this phase's own time, with nested phases removed.
            self.totals[name] = self.totals.get(name, 0.0) + (dt - child)
            self.counts[name] = self.counts.get(name, 0) + 1
            if self._stack:
                self._stack[-1] += dt
            if rec is not None:
                rec.__exit__(None, None, None)

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self._stack.clear()
        self._t0 = None

    def report(self) -> str:
        """One line: `[prof] total=Xms phase=Yms (Z%) ...`, descending by cost.

        ``other`` is the step wall minus everything attributed — data loading, the
        optimizer under `--optim-in-backward` (which fires inside backward), and
        any host-side work between phases.
        """
        if not self.enabled or self._t0 is None:
            return ""
        wall = (self._mark() - self._t0) * 1000.0
        parts = sorted(self.totals.items(), key=lambda kv: -kv[1])
        accounted = sum(self.totals.values()) * 1000.0
        out = [f"total={wall:.0f}ms"]
        for name, sec in parts:
            ms = sec * 1000.0
            out.append(f"{name}={ms:.0f}({100.0 * ms / max(wall, 1e-9):.0f}%)")
        other = wall - accounted
        out.append(f"other={other:.0f}({100.0 * other / max(wall, 1e-9):.0f}%)")
        return "[prof] " + " ".join(out)


# A kernel at or under this many microseconds cannot fill an A100's SMs at any
# occupancy — it is dominated by launch and teardown. The SHARE of busy time spent
# in these is the statistic that picks the next lever (see `kernel_report`).
TINY_KERNEL_US = 20.0


def kernel_report(prof_ctx, wall_s: float, n_steps: int, top: int = 20) -> str:
    """Kernel-level accounting for a profiled window: is this step launch-bound?

    `s/step` and the phase timers both say WHERE time goes but not WHY, and the two
    remaining step-time levers want opposite answers:

    * **GPU busy ≈ wall** — the device always has work, so the host is keeping up. If
      busy time is also concentrated in kernels too small to fill the SMs, the fix is
      CONCURRENCY: run the K local tracks on separate streams so their tiny kernels
      overlap.
    * **GPU busy ≪ wall** — the device is starved between kernels and the host cannot
      launch fast enough. The fix is fewer launches per sublayer.

    ⚠ **The profiler is not free at this op count.** It charges CPU per op, and at
    321k ops/step that is seconds of observer overhead landing entirely on the wall
    side of this ratio — so a host-bound verdict here is a HINT, not a finding.
    Confirm it by halving `--seq-len` with no profiler attached: launches are
    unchanged and device work roughly halves, so a step time that barely moves is
    host-bound for real (`scratchpad/seqlen_scaling.sh`).

    Reuses the accounting `scripts/serve_cli.py` prints for decode. `wall_s` must be
    the unsynced wall of the profiled window; the ratio is meaningless against a wall
    measured with `PhaseTimer(sync=True)`, which serializes the overlap being counted.
    """
    def _dev_us(e):
        return getattr(e, "self_device_time_total",
                       getattr(e, "self_cuda_time_total", 0))

    # `key_averages()` returns THREE kinds of row and only one of them is a kernel:
    # the CPU-side op (`aten::mm`, carrying the device time it launched), the device
    # kernel itself (`ampere_bf16_...`), and every `record_function` range projected
    # onto the device (`tf_block_loop`). Summing device time over "any row with device
    # time" counts the same work up to three times — it read 143.8% of wall before
    # this filter, which is how the bug announced itself. Kernels are the CUDA-side
    # rows that are not user annotations.
    ka = prof_ctx.key_averages()
    kernels = [e for e in ka
               if str(getattr(e, "device_type", "")) == "DeviceType.CUDA"
               and not getattr(e, "is_user_annotation", False)
               and _dev_us(e) > 0]
    busy_us = sum(_dev_us(e) for e in kernels)
    n_kernels = sum(e.count for e in kernels)
    wall_us = wall_s * 1e6

    # Per-execution mean, not per-key: one key with count=5000 is 5000 launches.
    tiny = [e for e in kernels if _dev_us(e) / max(e.count, 1) <= TINY_KERNEL_US]
    tiny_n = sum(e.count for e in tiny)
    tiny_us = sum(_dev_us(e) for e in tiny)

    busy_frac = busy_us / max(wall_us, 1e-9)
    lines = [
        f"--- kernel profile ({n_steps} steps, {wall_s:.2f}s wall) ---",
        f"GPU busy   : {busy_us / 1e3:.0f} ms = {100 * busy_frac:.1f}% of wall "
        f"({busy_us / 1e3 / n_steps:.0f} ms/step busy vs {1e3 * wall_s / n_steps:.0f} ms/step wall)",
        f"kernels    : {n_kernels} total = {n_kernels // max(n_steps, 1)}/step, "
        f"mean {busy_us / max(n_kernels, 1):.1f} us each",
        f"tiny (<={TINY_KERNEL_US:.0f}us): {tiny_n} launches "
        f"({100 * tiny_n / max(n_kernels, 1):.0f}% of count) "
        f"= {100 * tiny_us / max(busy_us, 1e-9):.0f}% of busy time",
        f"verdict    : {'HOST-BOUND (gaps between kernels) -> fewer ops/sublayer; confirm with a --seq-len halving, the profiler inflates this wall' if busy_frac < 0.6 else 'DEVICE-BUSY on small kernels -> bigger kernels (wider tracks)' if tiny_us / max(busy_us, 1e-9) > 0.3 else 'DEVICE-BOUND on real work -> look at the math'}",
        f"top {top} kernels by self device time:",
    ]
    for e in sorted(kernels, key=lambda e: -_dev_us(e))[:top]:
        us = _dev_us(e)
        lines.append(f"  {e.key[:66]:<66} {us / 1e3:8.1f} ms x{e.count:<6d} "
                     f"{us / max(e.count, 1):7.1f} us")
    return "\n".join(lines)
