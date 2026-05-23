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

from pt_converter.train.sync_grads import (
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
    from pt_converter.dist.groups import ProcessGroupLayout

    return ProcessGroupLayout(
        world_size=1,
        rank=0,
        n_tracks=n_tracks,
        tracks_per_rank=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        track_group=None,
        intra_track_size=1,
        intra_track_rank=0,
        intra_track_group=None,
    )


def test_build_replication_plan_on_tiny_qwen3_5_n4():
    """K=4 tracks on one rank with num_kv_heads=1: every Replicated norm and
    every KV-projection has its replication group fully local. The plan must
    cover them all with process_group=None."""
    import torch.nn as nn
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from pt_converter.adapters import get_adapter_for_config
    from pt_converter.model.pt_model import PTWrappedModel
    from pt_converter.slicer.convert import slice_model_to_tracks
    from pt_converter.train.sync_grads import build_replication_plan

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

    # Sanity: at least the final norm + each layer's two norms + the full-
    # attention layer's q_norm/k_norm/k_proj/v_proj should appear.
    # 4 layers × (input_layernorm + post_attention_layernorm) = 8
    # final norm = 1
    # 3 linear-attention layers × 1 (linear_attn.norm) = 3
    # 1 full-attention layer × (q_norm + k_norm + k_proj + v_proj) = 4
    # Total = 16
    assert len(plan) == 16


def test_build_then_sync_then_step_keeps_replicas_identical():
    """Run one backward+sync+step on a tiny K=4 student and confirm every
    replicated parameter still has bit-identical copies across local tracks."""
    import torch.nn as nn
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from pt_converter.adapters import get_adapter_for_config
    from pt_converter.model.pt_model import PTWrappedModel
    from pt_converter.slicer.convert import slice_model_to_tracks
    from pt_converter.train.sync_grads import (
        assert_replicated_consistent,
        build_replication_plan,
        sync_replicated_grads,
    )

    cfg = _tiny_qwen3_5_config()
    torch.manual_seed(11)
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

    layout = _make_single_rank_layout(n_tracks)
    adapter = get_adapter_for_config(cfg)
    plan = build_replication_plan(student, adapter=adapter, text_cfg=cfg, layout=layout)
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
