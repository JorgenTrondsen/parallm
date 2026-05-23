"""Cross-track gradient sync for replicated parameters.

During PT distillation, parameters that the slicer flagged as held in
identical copies across multiple tracks — RMSNorm scales declared via
``Replicated`` and K/V projection rows sliced via ``KVReplicatedColwise``
when ``n_tracks > num_kv_heads`` — start out bit-identical at conversion.
The trainer must keep them identical: each track receives a *different*
local gradient for its copy (different sliced Q columns drive different
upstream signal into the shared K/V row block; different sliced MLPs feed
different gradient back through the shared norms), so without intervention
the copies will drift.

This module builds the canonical "replication plan" for the current
(model, n_tracks, layout) and provides ``sync_replicated_grads`` to be
called between ``loss.backward()`` and ``optimizer.step()``. The sync
averages gradients across all copies in each replication group, so every
member's optimizer step is identical and the copies remain bit-equal
forever (assuming a deterministic optimizer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from pt_converter.adapters import ModelAdapter
from pt_converter.dist.groups import ProcessGroupLayout
from pt_converter.slicer.base import SlicerSpec
from pt_converter.slicer.convert import resolve_param_specs


def _singleton_groups(n_tracks: int) -> list[list[int]]:
    return [[t] for t in range(n_tracks)]


def get_replication_groups(spec: SlicerSpec, n_tracks: int) -> list[list[int]]:
    """Return the partition of tracks into replication groups for this spec.

    Tracks in the same group hold identical parameter slices and must share
    a gradient during training. Specs that don't implement ``replication_groups``
    default to one singleton group per track (no sync).
    """
    fn = getattr(spec, "replication_groups", None)
    if fn is None:
        return _singleton_groups(n_tracks)
    return fn(n_tracks)


@dataclass
class ReplicationCoordGroup:
    """One replication group's coordination data, from this rank's perspective.

    Attributes:
      local_params: parameter instances on this rank whose ``.grad`` should
        carry an identical (averaged) gradient after sync. Always non-empty
        for any group the trainer should care about.
      process_group: distributed process group spanning all ranks holding any
        member of the replication group. ``None`` ⇒ all members are local
        (no cross-rank collective needed).
      group_size: total number of tracks in the replication group, used to
        divide the summed gradient back to a mean.
    """

    local_params: list[nn.Parameter]
    process_group: "dist.ProcessGroup | None"
    group_size: int


def _resolve_local_param(
    student: nn.Module,
    text_models_idx: int,
    canonical_param_name: str,
) -> "nn.Parameter | None":
    """Look up a canonical slicer param name on this rank's student module.

    Canonical paths live under ``text_models[idx]`` (e.g. ``layers.5.self_attn.k_proj.weight``
    → ``text_models[idx].layers.5.self_attn.k_proj.weight``), except ``lm_head.weight``
    which is on the wrapper itself and only on the owner rank.
    """
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
) -> list[ReplicationCoordGroup]:
    """Construct the gradient-sync plan for all replicated parameters.

    This is collective across all ranks: every rank must call this function in
    the same order with the same arguments. Process subgroups (when a
    replication group spans only a subset of ranks) are created via
    ``dist.new_group``, which itself is collective and must be called by every
    rank in the world in identical order.

    Steps:
      1. Resolve ``canonical_name → SlicerSpec`` for every parameter (same names
         the manifest records).
      2. For each parameter, partition tracks via ``spec.replication_groups``.
      3. Enumerate the distinct rank-sets across all non-singleton replication
         groups, create one ``ProcessGroup`` per distinct rank-set (reusing
         ``dist.group.WORLD`` when the rank-set is everyone).
      4. For each replication group on each rank that holds at least one of its
         tracks, emit a ``ReplicationCoordGroup`` pointing at the local parameter
         instances and the appropriate process group.
    """
    n_tracks = layout.n_tracks
    K = layout.tracks_per_rank
    world_size = layout.world_size
    rank = layout.rank

    spec_map = resolve_param_specs(adapter, text_cfg)

    # First pass: collect (canonical_name, group_tracks, rank_set) for every
    # non-singleton replication group, and the set of distinct rank-sets.
    distinct_rank_sets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    plan_specs: list[tuple[str, list[int], tuple[int, ...]]] = []
    for canonical_name, spec in spec_map.items():
        groups = get_replication_groups(spec, n_tracks)
        for g in groups:
            if len(g) <= 1:
                continue
            ranks_in_g = tuple(sorted({t // K for t in g}))
            plan_specs.append((canonical_name, list(g), ranks_in_g))
            if ranks_in_g not in seen:
                seen.add(ranks_in_g)
                distinct_rank_sets.append(ranks_in_g)

    # Canonical ordering so every rank constructs subgroups in identical order.
    distinct_rank_sets.sort()

    # Create process subgroups. dist.new_group is collective: every rank in the
    # world must call it with the same rank list, in the same order, even when
    # it is not a member. The full-world case reuses dist.group.WORLD to avoid
    # an unnecessary communicator.
    full_world = tuple(range(world_size))
    rank_set_to_pg: dict[tuple[int, ...], "dist.ProcessGroup | None"] = {}
    for rs in distinct_rank_sets:
        if rs == full_world:
            rank_set_to_pg[rs] = dist.group.WORLD
        else:
            rank_set_to_pg[rs] = dist.new_group(ranks=list(rs))

    # Second pass: build a CoordGroup per replication group that includes a
    # local track on this rank.
    plan: list[ReplicationCoordGroup] = []
    local_track_set = set(layout.local_track_ids)
    for canonical_name, group_tracks, ranks_in_g in plan_specs:
        if rank not in ranks_in_g:
            continue  # this rank holds no member of the group; skip.
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
        # If the rank-set is a singleton (entire group is on this rank), no
        # cross-rank collective is needed; setting process_group=None lets the
        # sync skip the NCCL call.
        if len(ranks_in_g) == 1:
            pg = None
        plan.append(
            ReplicationCoordGroup(
                local_params=local_params,
                process_group=pg,
                group_size=len(group_tracks),
            )
        )

    return plan


def sync_replicated_grads(plan: list[ReplicationCoordGroup]) -> None:
    """Average gradients across replication groups in-place.

    Must be called between ``loss.backward()`` and ``optimizer.step()``. For
    each replication group:
      1. Sum the ``.grad`` of all local members on this rank.
      2. All-reduce-SUM across the group's process group (skipped when the
         group is fully local).
      3. Divide by ``group_size`` to get the mean.
      4. Copy the averaged gradient back to every local member's ``.grad`` so
         each member receives an identical optimizer update.

    Some replicated parameters may receive no gradient on some ranks (for
    example the final ``norm.weight``: only the rank that owns track 0
    backprops the LM loss through it). To keep the collective ordering
    symmetric across the replication group, missing gradients are zero-
    allocated before the all-reduce — every member participates in every
    collective on every step.
    """
    for cg in plan:
        if not cg.local_params:
            continue
        # Ensure every local member has a .grad tensor; missing gradients are
        # filled with zero so that this rank's contribution to the collective
        # is a well-defined value rather than a skipped call. Without this,
        # a rank where head.grad is None would skip the all_reduce while peer
        # ranks issue it — that desynchronises the collective queue and
        # eventually hangs NCCL.
        for p in cg.local_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        head = cg.local_params[0]
        if len(cg.local_params) == 1:
            buf = head.grad
        else:
            buf = head.grad.detach().clone()
            for p in cg.local_params[1:]:
                buf.add_(p.grad)
        if cg.process_group is not None:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=cg.process_group)
        buf.div_(cg.group_size)
        for p in cg.local_params:
            p.grad.copy_(buf)


def assert_replicated_consistent(plan: list[ReplicationCoordGroup]) -> None:
    """Verify every replication group's members hold identical parameter values.

    Two checks per group:
      - Local: any two local members must be ``torch.equal``.
      - Cross-rank: all-reduce MIN and MAX of the parameter on the group's
        process group; if min != max element-wise, the copies have drifted.

    Cheap to run at training startup. Raises ``RuntimeError`` on mismatch.
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
