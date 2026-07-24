"""Cross-track gradient sync for replicated parameters.

During PT distillation, parameters that the slicer flagged as held in
identical copies across multiple tracks — RMSNorm scales declared via
``Replicated`` and K/V projection rows sliced via ``KVReplicatedColwise``
when ``n_tracks > num_kv_heads`` — start out bit-identical at conversion.
The trainer must keep them identical: each track receives a *different*
local gradient for its copy, so without intervention the copies drift.

``build_replication_plan`` derives the canonical replication plan for the
current (model, n_tracks, layout); ``sync_replicated_grads`` (called between
``loss.backward()`` and ``optimizer.step()``) averages gradients across all
copies in each group so every member's optimizer step is identical and the
copies remain bit-equal forever (deterministic optimizer assumed).

Ported from the pre-parallm-pivot trainer (imports renamed; semantics
unchanged — the specs' ``replication_groups`` still live in ``slicer.base``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from parallm.adapters import ModelAdapter
from parallm.dist.groups import ProcessGroupLayout
from parallm.slicer.base import SlicerSpec
from parallm.slicer.convert import resolve_param_specs


def _singleton_groups(n_tracks: int) -> list[list[int]]:
    return [[t] for t in range(n_tracks)]


def get_replication_groups(
    spec: SlicerSpec, n_tracks: int, force_sync: bool = False
) -> list[list[int]]:
    """Partition of tracks into replication groups for this spec.

    Tracks in the same group hold identical parameter slices and must share a
    gradient during training. Specs without ``replication_groups`` default to
    singletons (no sync). ``force_sync`` overrides a spec's ``sync=False``
    (diverge) setting back to a single shared group — the legacy
    ``--sync-attention-heads`` behaviour.
    """
    fn = getattr(spec, "replication_groups", None)
    if fn is None:
        return _singleton_groups(n_tracks)
    return fn(n_tracks, force_sync=force_sync)


@dataclass
class ReplicationCoordGroup:
    """One replication group's coordination data, from this rank's perspective.

    ``local_params``: parameter instances on this rank whose ``.grad`` should
    carry the identical (averaged) gradient after sync. ``process_group``:
    group spanning all ranks holding any member (``None`` ⇒ fully local).
    ``group_size``: total tracks in the group (divides the summed grad).
    """

    local_params: list[nn.Parameter]
    process_group: "dist.ProcessGroup | None"
    group_size: int


def _resolve_local_param(
    student: nn.Module,
    text_models_idx: int,
    canonical_param_name: str,
) -> "nn.Parameter | None":
    """Look up a canonical slicer param name on this rank's student module."""
    if canonical_param_name == "lm_head.weight":
        lm_head = getattr(student, "lm_head", None)
        if lm_head is None:
            return None
        return lm_head.weight
    tm = student.text_models[text_models_idx]
    module_path, _, param_name = canonical_param_name.rpartition(".")
    if module_path:
        try:
            submod = tm.get_submodule(module_path)
        except AttributeError:
            return None
    else:
        submod = tm
    return getattr(submod, param_name, None)


def build_replication_plan(
    student: nn.Module,
    *,
    adapter: ModelAdapter,
    text_cfg: Any,
    layout: ProcessGroupLayout,
    force_sync: bool = False,
) -> list[ReplicationCoordGroup]:
    """Construct the gradient-sync plan for all replicated parameters.

    Collective across ranks: every rank must call this in the same order with
    the same arguments (``dist.new_group`` is itself collective). ``force_sync``
    must be passed identically on every rank.
    """
    n_tracks = layout.n_tracks
    world_size = layout.world_size
    rank = layout.rank
    track_to_rank = layout.track_to_rank

    spec_map = resolve_param_specs(adapter, text_cfg)

    # Pass 1: every non-singleton replication group + the distinct rank-sets.
    distinct_rank_sets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    plan_specs: list[tuple[str, list[int], tuple[int, ...]]] = []
    for canonical_name, spec in spec_map.items():
        groups = get_replication_groups(spec, n_tracks, force_sync=force_sync)
        for g in groups:
            if len(g) <= 1:
                continue
            ranks_in_g = tuple(sorted({track_to_rank[t] for t in g}))
            plan_specs.append((canonical_name, list(g), ranks_in_g))
            if ranks_in_g not in seen:
                seen.add(ranks_in_g)
                distinct_rank_sets.append(ranks_in_g)

    # Canonical ordering so every rank constructs subgroups in identical order.
    distinct_rank_sets.sort()
    full_world = tuple(range(world_size))
    rank_set_to_pg: dict[tuple[int, ...], "dist.ProcessGroup | None"] = {}
    for rs in distinct_rank_sets:
        if rs == full_world:
            rank_set_to_pg[rs] = dist.group.WORLD
        else:
            rank_set_to_pg[rs] = dist.new_group(ranks=list(rs))

    # Pass 2: a CoordGroup per replication group with a local member.
    plan: list[ReplicationCoordGroup] = []
    local_track_set = set(layout.local_track_ids)
    for canonical_name, group_tracks, ranks_in_g in plan_specs:
        if rank not in ranks_in_g:
            continue
        local_tracks_in_group = [t for t in group_tracks if t in local_track_set]
        local_params: list[nn.Parameter] = []
        for t in local_tracks_in_group:
            text_models_idx = layout.local_track_ids.index(t)
            p = _resolve_local_param(student, text_models_idx, canonical_name)
            if p is None:
                continue
            local_params.append(p)
        if not local_params:
            continue
        pg = rank_set_to_pg[ranks_in_g]
        if len(ranks_in_g) == 1:
            pg = None  # entire group on this rank — no collective needed
        plan.append(
            ReplicationCoordGroup(
                local_params=local_params,
                process_group=pg,
                group_size=len(group_tracks),
            )
        )

    return plan


def compute_global_grad_norm(
    student: nn.Module,
    plan: list[ReplicationCoordGroup],
) -> torch.Tensor:
    """Global L2 grad norm over the logical model, deduplicated + world-reduced.

    Each replication group's synced ``|g|²`` counts exactly once (each local
    copy contributes ``|g|²/group_size``; the world all-reduce restores
    ``|g|²``). Identical on every rank, so a clip coefficient derived from it
    preserves the bit-equality invariant. Call AFTER ``sync_replicated_grads``.
    """
    replicated_denom: dict[int, float] = {}
    for cg in plan:
        for p in cg.local_params:
            replicated_denom[id(p)] = float(cg.group_size)

    grads_by_denom: dict[float, list[torch.Tensor]] = {}
    for p in student.parameters():
        if p.grad is None:
            continue
        denom = replicated_denom.get(id(p), 1.0)
        grads_by_denom.setdefault(denom, []).append(p.grad.detach())

    sum_sq: "torch.Tensor | None" = None
    # bounds the transient fp32 grad copies to a chunk at a time. Per-tensor
    # _foreach_norm results are independent of grouping, so chunk size changes
    # ONLY peak memory, never the value (bit-equality preserved). Kept small
    # because a single MoE expert grad is ~134 MiB in fp32 (gate_up_proj) — a
    # 256-wide chunk balloons multiple GiB and OOMs N=16 K=2 on a 40 GB card.
    _CHUNK = 16
    for denom, grads in grads_by_denom.items():
        for i in range(0, len(grads), _CHUNK):
            norms = torch._foreach_norm([g.float() for g in grads[i:i + _CHUNK]])
            sq = torch.stack(norms).pow(2).sum()
            if denom != 1.0:
                sq = sq / denom
            sum_sq = sq if sum_sq is None else sum_sq + sq

    if sum_sq is None:
        # No grads on this rank — still participate so peers don't hang.
        device = next(student.parameters()).device
        sum_sq = torch.zeros((), device=device, dtype=torch.float32)

    if dist.is_initialized():
        dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM)

    return sum_sq.sqrt()


def _bucket_groups_by_pg_dtype(
    plan: list[ReplicationCoordGroup],
) -> list[tuple["dist.ProcessGroup | None", torch.dtype, list[ReplicationCoordGroup]]]:
    """Group plan entries that can share a single all_reduce (same pg + dtype).

    Order within a bucket is preserved — required for the flat-concat split on
    the receiving side; guaranteed by the deterministic plan construction.
    """
    buckets: dict[
        tuple[int, torch.dtype],
        tuple["dist.ProcessGroup | None", torch.dtype, list[ReplicationCoordGroup]],
    ] = {}
    for cg in plan:
        if not cg.local_params:
            continue
        dtype = cg.local_params[0].dtype
        key = (id(cg.process_group), dtype)
        if key not in buckets:
            buckets[key] = (cg.process_group, dtype, [])
        buckets[key][2].append(cg)
    return list(buckets.values())


def sync_replicated_grads(plan: list[ReplicationCoordGroup]) -> None:
    """Average gradients across replication groups in-place.

    Call between ``loss.backward()`` and ``optimizer.step()``. Groups sharing a
    (process group, dtype) pair are coalesced into one flat all_reduce. Missing
    gradients are zero-allocated first so every member participates in every
    collective on every step (e.g. the final norm only gets loss gradient on
    the lm_head-owner rank).
    """
    for pg, _dtype, groups in _bucket_groups_by_pg_dtype(plan):
        bufs: list[torch.Tensor] = []
        for cg in groups:
            for p in cg.local_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
            buf = cg.local_params[0].grad.detach().clone()
            for p in cg.local_params[1:]:
                buf.add_(p.grad)
            bufs.append(buf)

        if pg is None:
            for cg, buf in zip(groups, bufs):
                buf.div_(cg.group_size)
                for p in cg.local_params:
                    p.grad.copy_(buf)
            continue

        flat = torch.cat([b.flatten() for b in bufs])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=pg)
        offset = 0
        for cg, buf in zip(groups, bufs):
            n = buf.numel()
            avg = flat[offset:offset + n].view_as(buf).div_(cg.group_size)
            for p in cg.local_params:
                p.grad.copy_(avg)
            offset += n


def assert_replicated_consistent(plan: list[ReplicationCoordGroup]) -> None:
    """Verify every replication group's members hold identical values.

    Local pairs must be ``torch.equal``; cross-rank groups all-reduce MIN and
    MAX and compare. Cheap at startup; raises ``RuntimeError`` on drift.
    """
    for cg in plan:
        if not cg.local_params:
            continue
        head = cg.local_params[0].data
        for p in cg.local_params[1:]:
            if not torch.equal(head, p.data):
                diff = (head - p.data).abs().max().item()
                raise RuntimeError(
                    f"Replicated param drift between local tracks: shape={tuple(head.shape)} "
                    f"max_abs_diff={diff}"
                )
        if cg.process_group is not None:
            mn = head.detach().clone()
            mx = head.detach().clone()
            dist.all_reduce(mn, op=dist.ReduceOp.MIN, group=cg.process_group)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=cg.process_group)
            if not torch.equal(mn, mx):
                diff = (mx - mn).abs().max().item()
                raise RuntimeError(
                    f"Replicated param differs across ranks: shape={tuple(head.shape)} "
                    f"max_abs_diff={diff}"
                )
