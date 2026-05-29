"""Per-rank CUDA memory attribution for the PT training loop.

The training loop's only built-in memory signal is a single
``max_memory_allocated()`` line in the ``--profile`` path — it reports the peak
but not *what* occupies it. These helpers, driven by the ``--mem-report`` flag
in ``scripts/train_qwen3_5_9b.py``, break the per-GPU footprint down into its
real occupants: the FSDP-sharded dense teacher, the K student tracks (+ embed /
lm_head on the owner rank), gradients, AdamW optimizer state, and — via the
``_phase`` hooks in ``train/distill.py`` — the transient activation peak of each
distill-step phase.

Everything here is measured, not assumed: tensor bytes are summed from the live
modules / optimizer state (so e.g. the actual AdamW state dtype is reported, not
guessed), and ``device_mem`` exposes ``mem_get_info`` so the report also shows
the CUDA-context + NCCL gap that torch's allocator counters never include.

All output is per-rank. Under vocab-parallel the per-rank footprint is nearly
symmetric (every rank holds K tracks + its vocab shard of embed/lm_head); under
the legacy path rank 0 additionally owns the full embed_tokens + lm_head and is
the binding constraint.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn

_GB = 1024 ** 3


def _gb(nbytes: float) -> float:
    return nbytes / _GB


def tensor_bytes(t: torch.Tensor) -> int:
    """Local byte footprint of ``t``.

    FSDP2 (``fully_shard``) replaces parameters with ``DTensor`` whose
    ``numel()`` reports the *global* (unsharded) element count. We want the
    bytes actually resident on this rank, so unwrap to the local shard first.
    """
    local = t.to_local() if hasattr(t, "to_local") else t
    return local.numel() * local.element_size()


def device_mem() -> dict[str, float]:
    """Current-device memory in GB: torch allocator stats + true device usage.

    ``allocated`` / ``reserved`` / ``max_allocated`` are torch's allocator
    counters. ``device_used`` / ``device_total`` come from ``mem_get_info`` and
    include everything the allocator does not — the CUDA context, NCCL buffers,
    cuDNN workspaces — so ``device_used - reserved`` is roughly that overhead.
    """
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": _gb(torch.cuda.memory_allocated()),
        "reserved": _gb(torch.cuda.memory_reserved()),
        "max_allocated": _gb(torch.cuda.max_memory_allocated()),
        "device_used": _gb(total - free),
        "device_total": _gb(total),
    }


def _categorize_student_param(name: str) -> str:
    """Bucket a student parameter name into a report category."""
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name or "vp_embed" in name:
        return "embed_tokens"
    if ".norm" in name or name.endswith("norm.weight") or "layernorm" in name:
        return "norms"
    return "student_layers"


def student_param_bytes(student: nn.Module) -> dict[str, int]:
    """Per-category student parameter + gradient bytes (resident on this rank)."""
    params: dict[str, int] = {}
    grads = 0
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        cat = _categorize_student_param(name)
        params[cat] = params.get(cat, 0) + tensor_bytes(p)
        if p.grad is not None:
            grads += tensor_bytes(p.grad)
    params["grads"] = grads
    return params


def optimizer_state_bytes(optim: torch.optim.Optimizer) -> dict[str, int]:
    """Sum optimizer-state tensor bytes, grouped by ``state_key[dtype]``.

    Reveals the real moment dtype (e.g. ``exp_avg[bf16]`` vs ``[fp32]``) instead
    of assuming it from the param dtype.
    """
    out: dict[str, int] = {}
    for state in optim.state.values():
        for key, val in state.items():
            if torch.is_tensor(val):
                label = f"{key}[{str(val.dtype).replace('torch.', '')}]"
                out[label] = out.get(label, 0) + tensor_bytes(val)
    return out


def teacher_bytes(teacher) -> int:
    """Local (sharded) byte footprint of the frozen teacher on this rank."""
    total = 0
    for mod in (teacher.text_model, teacher.lm_head):
        for p in mod.parameters():
            total += tensor_bytes(p)
        for b in mod.buffers():
            total += tensor_bytes(b)
    return total


def component_breakdown(student: nn.Module, teacher, optim: torch.optim.Optimizer) -> dict[str, float]:
    """Measured per-category resident footprint (GB) on this rank.

    Returns an ordered-ish dict suitable for a report table. Keys: teacher,
    student_layers / embed_tokens / lm_head / norms, grads, and one entry per
    optimizer state key+dtype.
    """
    out: dict[str, float] = {"teacher": _gb(teacher_bytes(teacher))}
    sp = student_param_bytes(student)
    for cat in ("student_layers", "embed_tokens", "lm_head", "norms", "grads"):
        if sp.get(cat):
            out[cat] = _gb(sp[cat])
    for label, nbytes in optimizer_state_bytes(optim).items():
        out[f"optim.{label}"] = _gb(nbytes)
    return out


def print_all_ranks(rank: int, world_size: int, lines: Iterable[str]) -> None:
    """Print each rank's ``lines`` block in rank order (barrier-serialized).

    Every rank must call this with its own ``lines``; the ``dist.barrier()``
    between turns keeps the 8 per-rank blocks from interleaving in the log.
    """
    text = "\n".join(lines)
    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(text, flush=True)
    dist.barrier()


def log_stage(rank: int, world_size: int, label: str) -> None:
    """One barrier-ordered per-rank line: allocator + device memory at ``label``.

    Used at init lifecycle points (baseline → teacher → student → optimizer) so
    the growth attributable to each stage is visible.
    """
    m = device_mem()
    line = (
        f"[mem][rank {rank}] {label:38s} "
        f"alloc={m['allocated']:6.2f} reserved={m['reserved']:6.2f} "
        f"device_used={m['device_used']:6.2f}/{m['device_total']:.0f} GB"
    )
    print_all_ranks(rank, world_size, [line])


def format_report(
    rank: int,
    *,
    label: str,
    components: dict[str, float],
    device: dict[str, float],
    phase_mem: dict[str, dict[str, float]] | None,
    stages: dict[str, dict[str, float]] | None,
) -> list[str]:
    """Build the per-rank report block (returned as lines for ``print_all_ranks``)."""
    lines = [f"[mem][rank {rank}] ===== {label} ====="]

    if stages:
        lines.append(f"[mem][rank {rank}] -- lifecycle (alloc / reserved GB) --")
        prev = 0.0
        for name, m in stages.items():
            delta = m["allocated"] - prev
            lines.append(
                f"[mem][rank {rank}]   {name:38s} "
                f"alloc={m['allocated']:6.2f} (+{delta:5.2f}) reserved={m['reserved']:6.2f}"
            )
            prev = m["allocated"]

    lines.append(f"[mem][rank {rank}] -- resident components (GB) --")
    measured = 0.0
    for name, gb in components.items():
        measured += gb
        lines.append(f"[mem][rank {rank}]   {name:28s} {gb:7.3f}")
    lines.append(f"[mem][rank {rank}]   {'measured total':28s} {measured:7.3f}")

    lines.append(
        f"[mem][rank {rank}] -- allocator vs device --"
    )
    lines.append(
        f"[mem][rank {rank}]   allocated={device['allocated']:.2f}  "
        f"unattributed(alloc-measured)={device['allocated'] - measured:+.2f}  "
        f"reserved={device['reserved']:.2f}  "
        f"device_used={device['device_used']:.2f}/{device['device_total']:.0f} GB  "
        f"(reserved->device gap = CUDA ctx + NCCL ~{device['device_used'] - device['reserved']:.2f})"
    )

    if phase_mem:
        lines.append(f"[mem][rank {rank}] -- per-phase (transient peak / resident-after, GB) --")
        for name, m in phase_mem.items():
            lines.append(
                f"[mem][rank {rank}]   {name:14s} "
                f"peak={m.get('peak_gb', 0.0):6.2f}  resident={m.get('resident_gb', 0.0):6.2f}"
            )

    return lines
