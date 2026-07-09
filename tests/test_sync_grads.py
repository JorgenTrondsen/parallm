"""Tests for cross-track gradient sync of replicated parameters.

Verifies that `sync_replicated_grads` averages gradients in-place across all
members of each replication group and that `assert_replicated_consistent`
catches drift between local copies.

These tests exercise the *local-only* path (no NCCL collective) by building
ReplicationCoordGroup objects with `process_group=None` and multiple local
parameters. The cross-rank path is unit-tested implicitly via the existing
distributed smoke tests; testing it here would require spawning processes.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from parallm.train.sync_grads import (
    ReplicationCoordGroup,
    assert_replicated_consistent,
    sync_replicated_grads,
)


def _make_params(n: int, shape=(4, 4), seed: int = 0) -> list[nn.Parameter]:
    """Create n identical parameters (same value) but distinct nn.Parameter objects."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(shape, generator=g)
    return [nn.Parameter(base.clone()) for _ in range(n)]


def test_sync_replicated_grads_averages_local_members():
    # Three identical params (one replication group of size 3, all local).
    params = _make_params(3)
    # Give each a distinct gradient.
    g0 = torch.full((4, 4), 1.0)
    g1 = torch.full((4, 4), 3.0)
    g2 = torch.full((4, 4), 8.0)
    params[0].grad = g0.clone()
    params[1].grad = g1.clone()
    params[2].grad = g2.clone()
    cg = ReplicationCoordGroup(local_params=params, process_group=None, group_size=3)

    sync_replicated_grads([cg])

    expected = (g0 + g1 + g2) / 3.0
    for p in params:
        assert torch.equal(p.grad, expected), "averaged grad mismatch"
    # All members must now share identical gradient tensors element-wise.
    assert torch.equal(params[0].grad, params[1].grad)
    assert torch.equal(params[1].grad, params[2].grad)


def test_sync_keeps_optimizer_step_in_lockstep():
    # After sync, identical SGD steps on each replica must produce identical updated weights.
    params = _make_params(2)
    params[0].grad = torch.full((4, 4), 1.0)
    params[1].grad = torch.full((4, 4), 5.0)
    cg = ReplicationCoordGroup(local_params=params, process_group=None, group_size=2)
    sync_replicated_grads([cg])
    # Vanilla SGD step.
    lr = 0.1
    for p in params:
        with torch.no_grad():
            p.add_(p.grad, alpha=-lr)
    assert torch.equal(params[0].data, params[1].data), "replicas diverged after sync+step"


def test_sync_zero_allocates_missing_grads_for_symmetric_participation():
    # When .grad is None on some members, sync must still participate in the
    # collective (otherwise ranks would desync); missing grads are zero-allocated
    # and the averaged result is propagated back to every member.
    params = _make_params(2)
    sync_replicated_grads(
        [ReplicationCoordGroup(local_params=params, process_group=None, group_size=2)]
    )
    for p in params:
        assert p.grad is not None
        assert torch.equal(p.grad, torch.zeros_like(p.data))


def test_sync_with_partial_grads_averages_correctly():
    # Only one local member has a real gradient; the other is missing.
    # After sync, both must hold the averaged grad = G / group_size.
    a, b = _make_params(2)
    a.grad = torch.full((4, 4), 4.0)
    # b.grad stays None.
    sync_replicated_grads(
        [ReplicationCoordGroup(local_params=[a, b], process_group=None, group_size=2)]
    )
    expected = torch.full((4, 4), 2.0)  # (4 + 0) / 2
    assert torch.equal(a.grad, expected)
    assert torch.equal(b.grad, expected)


def test_sync_with_singleton_local_group_is_identity_when_divide_by_one():
    # group_size=1, single local param: averaging by 1 leaves grad unchanged.
    p = _make_params(1)[0]
    p.grad = torch.full((4, 4), 7.0)
    cg = ReplicationCoordGroup(local_params=[p], process_group=None, group_size=1)
    sync_replicated_grads([cg])
    assert torch.equal(p.grad, torch.full((4, 4), 7.0))


def test_assert_replicated_consistent_passes_on_identical_locals():
    params = _make_params(3)
    cg = ReplicationCoordGroup(local_params=params, process_group=None, group_size=3)
    assert_replicated_consistent([cg])


def test_assert_replicated_consistent_detects_drift():
    params = _make_params(2)
    with torch.no_grad():
        params[1].add_(1.0)  # introduce drift
    cg = ReplicationCoordGroup(local_params=params, process_group=None, group_size=2)
    with pytest.raises(RuntimeError, match="drift"):
        assert_replicated_consistent([cg])


def test_bucketed_sync_matches_unbucketed():
    """Two replication groups with different group_sizes must end up with
    the same per-group averaged gradient regardless of whether the sync
    flows through one collective per group or one bucketed concat. Tests
    the post-bucket per-slice divide path."""
    # Group A: 2 members, group_size=2 (fully local).
    a0, a1 = _make_params(2, seed=11)
    a0.grad = torch.full((4, 4), 6.0)
    a1.grad = torch.full((4, 4), 2.0)
    # Group B: 4 members, group_size=4 (fully local).
    b0, b1, b2, b3 = _make_params(4, seed=12)
    b0.grad = torch.full((4, 4), 1.0)
    b1.grad = torch.full((4, 4), 2.0)
    b2.grad = torch.full((4, 4), 3.0)
    b3.grad = torch.full((4, 4), 6.0)

    plan = [
        ReplicationCoordGroup(local_params=[a0, a1], process_group=None, group_size=2),
        ReplicationCoordGroup(local_params=[b0, b1, b2, b3], process_group=None, group_size=4),
    ]
    sync_replicated_grads(plan)

    # A: (6+2)/2 = 4.0 across both members.
    assert torch.equal(a0.grad, torch.full((4, 4), 4.0))
    assert torch.equal(a1.grad, torch.full((4, 4), 4.0))
    # B: (1+2+3+6)/4 = 3.0 across all four members.
    assert torch.equal(b0.grad, torch.full((4, 4), 3.0))
    assert torch.equal(b1.grad, torch.full((4, 4), 3.0))
    assert torch.equal(b2.grad, torch.full((4, 4), 3.0))
    assert torch.equal(b3.grad, torch.full((4, 4), 3.0))


def test_sync_handles_multiple_groups_independently():
    # Two separate replication groups should be averaged independently.
    a0, a1 = _make_params(2, seed=1)
    b0, b1 = _make_params(2, seed=2)
    a0.grad = torch.full((4, 4), 2.0)
    a1.grad = torch.full((4, 4), 4.0)
    b0.grad = torch.full((4, 4), 10.0)
    b1.grad = torch.full((4, 4), 20.0)
    plan = [
        ReplicationCoordGroup(local_params=[a0, a1], process_group=None, group_size=2),
        ReplicationCoordGroup(local_params=[b0, b1], process_group=None, group_size=2),
    ]
    sync_replicated_grads(plan)
    assert torch.equal(a0.grad, torch.full((4, 4), 3.0))
    assert torch.equal(a1.grad, torch.full((4, 4), 3.0))
    assert torch.equal(b0.grad, torch.full((4, 4), 15.0))
    assert torch.equal(b1.grad, torch.full((4, 4), 15.0))


# --- Integration: build_replication_plan on a tiny Qwen3.5 student ---


def _tiny_qwen3_5_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def _make_single_rank_layout(n_tracks: int):
    """Construct a ProcessGroupLayout for a single-process test (no dist init)."""
    from parallm.dist.groups import ProcessGroupLayout

    return ProcessGroupLayout(
        world_size=1,
        rank=0,
        n_tracks=n_tracks,
        tracks_per_rank=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        track_to_rank=(0,) * n_tracks,
        track_group=None,
        intra_track_size=1,
        intra_track_rank=0,
        intra_track_group=None,
    )


def test_build_replication_plan_on_tiny_qwen3_5_n4():
    """K=4 tracks on one rank with num_kv_heads=1. By default the full-attention
    head params diverge per track (absent from the plan); the residual-stream
    norms stay synced (fully local ⇒ process_group=None). force_sync re-adds the
    head params, recovering the legacy plan."""
    import torch.nn as nn
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from parallm.adapters import get_adapter_for_config
    from parallm.model.pt_model import PTWrappedModel
    from parallm.slicer.convert import slice_model_to_tracks
    from parallm.train.sync_grads import build_replication_plan

    cfg = _tiny_qwen3_5_config()
    torch.manual_seed(7)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    n_tracks = 4  # num_kv_heads=1 ⇒ all 4 tracks share one kv-group
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )

    student = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=(0, 1, 2, 3),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    student.load_track_state_dicts({i: tracks[i] for i in range(n_tracks)}, strict=False)

    layout = _make_single_rank_layout(n_tracks=n_tracks)
    adapter = get_adapter_for_config(cfg)
    plan = build_replication_plan(student, adapter=adapter, text_cfg=cfg, layout=layout)

    # All replication groups are local (single rank), so process_group is None.
    for cg in plan:
        assert cg.process_group is None
        assert len(cg.local_params) == cg.group_size == n_tracks

    # Default (diverge): only the residual-stream norms are synced.
    #   4 layers × (input_layernorm + post_attention_layernorm) = 8
    #   final norm = 1
    #   3 linear-attention layers × 1 (linear_attn.norm) = 3
    #   1 full-attention layer × (q_norm + k_norm + k_proj + v_proj) = 0 (diverged)
    #   Total = 12
    assert len(plan) == 12

    # force_sync re-adds the 4 diverged full-attention head params → 16.
    plan_synced = build_replication_plan(
        student, adapter=adapter, text_cfg=cfg, layout=layout, force_sync=True
    )
    assert len(plan_synced) == 16
    for cg in plan_synced:
        assert cg.process_group is None
        assert len(cg.local_params) == cg.group_size == n_tracks


def test_asymmetric_layout_track_to_rank_lookup():
    """Non-uniform K layout (rank 0 hosts fewer tracks because it also owns
    embed/lm_head). Verify the pure assignment helper produces the expected
    track_to_rank and local_track_ids, and that a replication group spanning
    multiple ranks resolves to the right rank-set via the new lookup —
    matching what sync_grads.build_replication_plan does at line 143."""
    from parallm.dist.groups import compute_contiguous_assignment

    n_tracks = 16
    world_size = 8
    k_list = [1, 3, 2, 2, 2, 2, 2, 2]  # rank 0 sheds tracks 1 onto rank 1
    assert sum(k_list) == n_tracks
    assert len(k_list) == world_size

    # Expected ownership (contiguous): rank 0 → {0}, rank 1 → {1,2,3},
    # rank 2 → {4,5}, …, rank 7 → {14,15}.
    expected_track_to_rank = (
        0,                           # track 0
        1, 1, 1,                     # tracks 1-3
        2, 2, 3, 3, 4, 4, 5, 5,      # tracks 4-11
        6, 6, 7, 7,                  # tracks 12-15
    )
    expected_local = {
        0: (0,),
        1: (1, 2, 3),
        2: (4, 5),
        3: (6, 7),
        4: (8, 9),
        5: (10, 11),
        6: (12, 13),
        7: (14, 15),
    }
    for r in range(world_size):
        local_ids, track_to_rank = compute_contiguous_assignment(k_list, r)
        assert track_to_rank == expected_track_to_rank, f"rank {r} track_to_rank mismatch"
        assert local_ids == expected_local[r], f"rank {r} local_track_ids mismatch"

    # Simulate the rank-set computation that sync_grads.build_replication_plan
    # performs for each replication group. The old formula `t // K` (assuming
    # uniform K=2) would break under this k_list — verify the new lookup
    # gives the right set on a group that actually diverges.
    track_to_rank = expected_track_to_rank
    # KV-replicated group with n_kv_heads=8: pairs of contiguous tracks. The
    # pair {0,1} straddles rank 0 (track 0) and rank 1 (track 1) under k_list,
    # but `t // 2` would put both on rank 0.
    pair = [0, 1]
    ranks_in_g = tuple(sorted({track_to_rank[t] for t in pair}))
    naive = tuple(sorted({t // 2 for t in pair}))
    assert ranks_in_g == (0, 1)
    assert naive == (0,)  # the formula sync_grads used to use — wrong here
    assert ranks_in_g != naive, "regression check: formulas must actually diverge"

    # A Replicated group spans all 16 tracks → must include every rank.
    full_group = list(range(n_tracks))
    assert tuple(sorted({track_to_rank[t] for t in full_group})) == tuple(range(world_size))


def _build_tiny_student(seed: int):
    """Slice a tiny Qwen3.5 dense model into a single-rank K=4 student."""
    import torch.nn as nn
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from parallm.model.pt_model import PTWrappedModel
    from parallm.slicer.convert import slice_model_to_tracks

    cfg = _tiny_qwen3_5_config()
    torch.manual_seed(seed)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    n_tracks = 4
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
    )
    student = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    student.load_track_state_dicts({i: tracks[i] for i in range(n_tracks)}, strict=False)
    student.train()
    return cfg, student, manifest, n_tracks


def _kproj_full_attention_param(student, track_idx: int):
    """The k_proj.weight of the first full-attention layer in a given track."""
    tm = student.text_models[track_idx]
    for i, lt in enumerate(tm.config.layer_types):
        if lt == "full_attention":
            return tm.get_submodule(f"layers.{i}.self_attn.k_proj").weight
    raise AssertionError("no full_attention layer in tiny config")


def test_force_sync_keeps_replicas_identical():
    """Under force_sync (legacy), one backward+sync+step must keep every
    replicated parameter — including the full-attention head params — bit-
    identical across local tracks."""
    from parallm.adapters import get_adapter_for_config
    from parallm.train.sync_grads import (
        assert_replicated_consistent,
        build_replication_plan,
        sync_replicated_grads,
    )

    cfg, student, _manifest, n_tracks = _build_tiny_student(seed=11)
    layout = _make_single_rank_layout(n_tracks)
    adapter = get_adapter_for_config(cfg)
    plan = build_replication_plan(
        student, adapter=adapter, text_cfg=cfg, layout=layout, force_sync=True
    )
    assert_replicated_consistent(plan)  # bit-identical after slicing

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)

    logits, _ = student(input_ids=input_ids, attention_mask=attention_mask)
    loss = logits.float().pow(2).mean()
    loss.backward()
    sync_replicated_grads(plan)

    optim = torch.optim.SGD(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    optim.step()

    # After the step every replication group's local copies must still be identical.
    assert_replicated_consistent(plan)
    # k_proj copies stay bit-identical across tracks (they were force-synced).
    k0 = _kproj_full_attention_param(student, 0)
    for t in range(1, n_tracks):
        assert torch.equal(k0, _kproj_full_attention_param(student, t))


def test_default_diverges_attention_heads_but_keeps_norms_synced():
    """Default mode: one backward+sync+step must (a) keep the synced norms
    bit-identical and (b) let the full-attention k_proj copies DIVERGE across
    tracks (they are absent from the replication plan)."""
    from parallm.adapters import get_adapter_for_config
    from parallm.train.sync_grads import (
        assert_replicated_consistent,
        build_replication_plan,
        sync_replicated_grads,
    )

    cfg, student, _manifest, n_tracks = _build_tiny_student(seed=11)
    layout = _make_single_rank_layout(n_tracks)
    adapter = get_adapter_for_config(cfg)
    plan = build_replication_plan(student, adapter=adapter, text_cfg=cfg, layout=layout)
    assert_replicated_consistent(plan)  # synced norms identical after slicing

    # k_proj copies are identical at conversion, before any training.
    k0_before = _kproj_full_attention_param(student, 0).detach().clone()
    for t in range(1, n_tracks):
        assert torch.equal(k0_before, _kproj_full_attention_param(student, t))

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    logits, _ = student(input_ids=input_ids, attention_mask=attention_mask)
    loss = logits.float().pow(2).mean()
    loss.backward()
    sync_replicated_grads(plan)
    optim = torch.optim.SGD(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    optim.step()

    # Synced norms still bit-identical (they were in the plan and averaged).
    assert_replicated_consistent(plan)
    # k_proj copies have now diverged across tracks (no sync; distinct grads).
    k0_after = _kproj_full_attention_param(student, 0)
    diverged = any(
        not torch.equal(k0_after, _kproj_full_attention_param(student, t))
        for t in range(1, n_tracks)
    )
    assert diverged, "full-attention k_proj should diverge across tracks by default"
