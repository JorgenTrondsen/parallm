"""Rails for the step-time instrumentation in `parallm.train.profile`.

`kernel_report`'s verdict line is the branch point for step-time work: it decides
whether the host cannot launch fast enough or the device is busy on kernels too small
to fill the SMs. Those two readings point at opposite fixes, so the classifier gets a
rail rather than a reading of one live log.

`PhaseTimer`'s exclusive-time accounting is railed here too: nested phases must not
be double-counted, or `tf_backward` inside `tf_block_loop` would read as more of the
step than it is.
"""
from __future__ import annotations

from parallm.train.profile import PhaseTimer, kernel_report


class _FakeEntry:
    """Stands in for a `torch.autograd.profiler_util.FunctionEventAvg`.

    `key_averages()` returns THREE kinds of row, and the first version of this fake
    modelled only one — which is how the double-counting bug reached a live run and
    reported GPU busy at 143.8% of wall. The three, all carrying device time:
    `kind="kernel"` (the real thing), `kind="op"` (the CPU-side `aten::` row that
    launched it), and `kind="annotation"` (a `record_function` range projected onto
    the device). Only kernels may be summed.
    """

    def __init__(self, key: str, us: float, count: int, kind: str = "kernel"):
        assert kind in ("kernel", "op", "annotation")
        self.key = key
        self.count = count
        self.self_device_time_total = us
        self.device_type = "DeviceType.CPU" if kind == "op" else "DeviceType.CUDA"
        self.is_user_annotation = kind == "annotation"


class _FakeProf:
    def __init__(self, entries):
        self._entries = entries

    def key_averages(self):
        return self._entries


def _report(entries, wall_s, n_steps=1):
    return kernel_report(_FakeProf(entries), wall_s, n_steps)


def test_host_bound_when_the_device_idles_between_kernels():
    """10 ms of kernels inside a 100 ms step: the GPU is starved, so the fix is fewer
    launches, NOT more concurrency — there is already idle device time to absorb the
    work. The verdict must also demand the profiler-free confirmation, because the
    profiler inflates the wall this ratio is measured against."""
    out = _report([_FakeEntry("triton_poi_fused_add_0", 10_000.0, 1000)], wall_s=0.1)
    assert "HOST-BOUND" in out
    assert "cuda graph" not in out.lower()
    assert "fewer ops" in out and "--seq-len" in out  # the confirmation it must demand
    assert "10.0% of wall" in out


def test_device_busy_on_tiny_kernels_asks_for_concurrency():
    """90 ms of kernels in a 100 ms step, nearly all of it in sub-20us launches:
    the host keeps up and the device is busy, but each kernel is far too small to
    fill an A100 — the fix is wider kernels."""
    out = _report(
        [
            _FakeEntry("triton_poi_fused_mul_3", 80_000.0, 16_000),  # 5.0 us each
            _FakeEntry("ampere_bf16_gemm", 10_000.0, 100),           # 100 us each
        ],
        wall_s=0.1,
    )
    assert "DEVICE-BUSY" in out and "bigger kernels" in out
    assert "89% of busy time" in out, out


def test_device_bound_on_real_work_points_at_the_math():
    """Busy and in big kernels: no scheduling lever left, the math is the cost."""
    out = _report([_FakeEntry("ampere_bf16_gemm", 90_000.0, 100)], wall_s=0.1)
    assert "DEVICE-BOUND" in out and "look at the math" in out


def test_tiny_share_is_per_execution_not_per_key():
    """A key with 20_000 launches at 1us is 20 ms of tiny work, not one 20ms kernel.

    Classifying on a key's TOTAL would call the busiest small-kernel workload
    "large" — exactly backwards, and exactly the config we are trying to diagnose.
    """
    out = _report([_FakeEntry("triton_poi_fused_0", 20_000.0, 20_000)], wall_s=0.021)
    assert "DEVICE-BUSY" in out
    assert "100% of busy time" in out


def test_only_kernel_rows_count_toward_gpu_busy():
    """The regression that produced "GPU busy: 143.8% of wall" on a live run.

    One 50 ms kernel, plus the `aten::` op that launched it and the `record_function`
    range around it — all three rows carry the same 50 ms of device time. Summing the
    rows gives 150 ms in a 100 ms step, which is not merely wrong but impossible, and
    it flipped the verdict from HOST-BOUND to DEVICE-BOUND.
    """
    out = _report(
        [
            _FakeEntry("ampere_bf16_gemm", 50_000.0, 100, kind="kernel"),
            _FakeEntry("aten::mm", 50_000.0, 100, kind="op"),
            _FakeEntry("tf_block_loop", 50_000.0, 1, kind="annotation"),
        ],
        wall_s=0.1,
    )
    assert "50.0% of wall" in out, out
    assert "HOST-BOUND" in out  # busy < 60% of wall
    # And the annotation must not appear as if it were the hottest kernel.
    assert "tf_block_loop" not in out


def test_phase_timer_reports_exclusive_time():
    """A nested phase is charged to itself, not to itself AND its parent."""
    t = PhaseTimer(enabled=True, sync=False)
    t.start_step()
    with t.phase("outer"):
        with t.phase("inner"):
            pass
    assert set(t.totals) == {"outer", "inner"}
    # Exclusive: the parent's total excludes the child, so they cannot sum to more
    # than the wall the report divides by.
    assert t.totals["outer"] >= 0.0
    assert t.totals["outer"] + t.totals["inner"] <= (t._mark() - t._t0) + 1e-6


def test_phase_timer_disabled_is_inert():
    t = PhaseTimer()  # the training default
    with t.phase("teacher_fwd"):
        pass
    assert t.totals == {} and t.report() == ""
