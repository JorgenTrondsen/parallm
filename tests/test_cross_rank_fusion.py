"""Cross-rank fusion groups: a fusion group that spans GPUs must equal a real
model with that many tracks per group.

Why this exists: `SyncBoundary.fuse` was a pure LOCAL sum, so a fusion group had to
sit whole on one rank and F was capped at ``n_tracks / world_size``. That makes the
quality knob a function of the hardware — adding GPUs LOWERS the reachable F. Cross-
rank fusion adds a subgroup all-reduce at every sublayer no global sync covers, so a
contiguous block of R ranks computes as one track and F is free of the world size.

**This rail is assignment-sensitive, which the exactness rails are not.** Under
`sync_phase="exact"` every sublayer syncs globally, so any partition of the tracks
sums the same deltas and a test would pass with the blocks wired wrong (e.g. ranks
paired {0,2},{1,3} instead of {0,1},{2,3}). So: a SPARSE schedule, and
**N=8 over 4 ranks at F=4 must reproduce a single-process N=2 model** (8 shards in
groups of 4 = 2 effective tracks) — which is true only if the blocks are exactly
{0,1} and {2,3}, since shards 0-3 must pool into one track and 4-7 into the other.

Two directions are checked, because either alone is passable by a bug:
  - cross-rank F=4 == the N=4 convert (right answer), and
  - cross-rank F=4 != the unfused N=8 model (non-vacuous).
Plus the gradient path, which is the piece with a real failure mode: every rank of a
block holds the block's whole state, so only the leader may feed the global boundary
— and if that is done with `zeros_like` rather than `_LeaderOnly`, the non-leaders'
parameters get ZERO gradient and 3/4 of the model silently stops training.
"""
from __future__ import annotations

import os
import socket

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel

from parallm.model.merge import plan_track_layout
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks

# Sparse: layers 1/3/5 sync, the rest own-carry. Fusion only acts on own-carry
# sublayers, so a dense schedule would make this rail vacuous.
SYNC_AFTER = [1, 3, 5]


def _cfg() -> GptOssConfig:
    """Every split dim divides by 8, so the N=8 partition nests inside N=4 and N=2."""
    cfg = GptOssConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
        sliding_window=3,
        layer_types=["sliding_attention", "full_attention"] * 3,
        rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _dense(cfg):
    torch.manual_seed(0)
    dense = GptOssModel(cfg).eval()
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name.endswith(("bias", "sinks")):
                p.normal_(mean=0.0, std=0.05)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    return dense.eval()


def _single_process_logits(cfg, dense, n_tracks, input_ids, fuse_size=1):
    """Reference: all tracks in one process, no collectives (`track_group=None`)."""
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=1, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=SYNC_AFTER,
        track_group=None,
        fuse_size=fuse_size,
    )
    pt.set_sync_phase("post-attn")
    pt.eval()
    pt.load_track_state_dicts({t: tracks[t] for t in range(n_tracks)}, strict=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())
    with torch.no_grad():
        return pt(input_ids=input_ids)[0]


def _worker(rank, world_size, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    from parallm.dist.groups import build_groups

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    cfg = _cfg()
    dense = _dense(cfg)
    n_tracks = 8
    input_ids = torch.randint(
        0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123)
    )

    # F=4 with 2 shards per rank: a fusion group is 2 whole ranks.
    plan = plan_track_layout(
        n_tracks, world_size, 4, supports_merged_tracks=False
    )
    assert plan.fuse_ranks == 2 and plan.fuse_size == 2, plan
    layout = build_groups(n_tracks=n_tracks, fuse_ranks=plan.fuse_ranks)

    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=1, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=SYNC_AFTER,
        track_group=layout.track_group,
        fuse_size=plan.fuse_size,
        fuse_group=layout.fuse_group,
        fuse_ranks=layout.fuse_ranks,
        fuse_rank=layout.fuse_rank,
    )
    pt.set_sync_phase("post-attn")
    pt.load_track_state_dicts(
        {t: tracks[t] for t in layout.local_track_ids}, strict=False
    )
    # Every rank gets an identical lm_head, exactly as the trainer does (a frozen
    # replicated head is what makes the boundary's identity backward the right
    # gradient — see `distill.py`). Without it, peer ranks return logits=None and
    # have no loss to differentiate.
    pt.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())

    pt.eval()
    with torch.no_grad():
        logits, _ = pt(input_ids=input_ids)

    # --- gradient path: EVERY rank's params must receive gradient ------------ #
    # This is what `_LeaderOnly` exists for. Ranks 1 and 3 contribute zero to the
    # global boundary (their state duplicates their block leader's), so a plain
    # `zeros_like` would leave them with no path to the loss at all.
    pt.train()
    pt.use_checkpoint = False
    for p in pt.parameters():
        p.requires_grad_(True)
    grad_logits, _ = pt(input_ids=input_ids)
    (grad_logits.float() ** 2).mean().backward()
    live = {
        n: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for n, p in pt.named_parameters()
        if "lm_head" not in n
    }

    results[rank] = {
        "local_track_ids": layout.local_track_ids,
        "fuse_rank": layout.fuse_rank,
        "logits": logits.detach().clone(),
        "n_params_with_grad": sum(1 for v in live.values() if v > 0),
        "n_params": len(live),
        "dead": sorted(n for n, v in live.items() if v == 0),
        "total_grad_mass": sum(live.values()),
    }
    dist.destroy_process_group()


def _spawn(world_size):
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    results = mgr.dict()
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_worker, args=(world_size, port, results), nprocs=world_size, join=True)
    return dict(results)


def test_cross_rank_fusion_of_two_ranks_equals_a_real_two_track_model():
    world_size = 4
    results = _spawn(world_size)
    assert set(results) == set(range(world_size))

    # Contiguous blocks: ranks 0,1 hold shards 0-3 (one fusion group), ranks 2,3
    # hold 4-7 (the other). Leaders are ranks 0 and 2.
    assert results[0]["local_track_ids"] == (0, 1)
    assert results[3]["local_track_ids"] == (6, 7)
    assert [results[r]["fuse_rank"] for r in range(4)] == [0, 1, 0, 1]

    cfg = _cfg()
    dense = _dense(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))
    # 8 shards in groups of 4 = 2 effective tracks. Shards 0-3 nest inside N=2's
    # track 0 and 4-7 inside its track 1, which is also exactly the rank blocking.
    ref_n2 = _single_process_logits(cfg, dense, 2, ids)
    ref_n8_unfused = _single_process_logits(cfg, dense, 8, ids)

    # Every rank ran the same model; all four must agree, not just the head owner.
    for rank in range(world_size):
        drift = (results[rank]["logits"] - ref_n2).abs().max().item()
        assert drift < 1e-4, (
            f"rank {rank}: cross-rank F=4 != the N=2 convert: max |dlogit| {drift}"
        )

    # Non-vacuous: without the pooling this would be the N=8 model, which differs.
    assert (ref_n8_unfused - ref_n2).abs().max().item() > 1e-3

    # And rank-local fusion ALONE (F=2, no cross-rank tier) is a third, different
    # model — so the agreement above is specifically the cross-rank all-reduce.
    ref_f2 = _single_process_logits(cfg, dense, 8, ids, fuse_size=2)
    assert (ref_f2 - ref_n2).abs().max().item() > 1e-3


def test_every_rank_of_a_fusion_block_receives_gradient():
    """The `_LeaderOnly` contract, and the one bug with a silent failure mode.

    Non-leader ranks contribute ZERO to the global boundary in the forward, because
    their state duplicates their block leader's. Implemented as `torch.zeros_like`
    that would also zero the GRADIENT, cutting ranks 1 and 3 out of the graph — the
    run would look healthy and train half the model. The Function's identity backward
    is what keeps them connected.
    """
    results = _spawn(4)
    # The non-leaders are the ones at risk; assert they are actually present.
    assert [results[r]["fuse_rank"] for r in range(4)].count(1) == 2

    leader = results[0]
    for rank in range(4):
        res = results[rank]
        assert res["total_grad_mass"] > 0, f"rank {rank} got no gradient at all"
        # A non-leader must be as connected as its leader — same live params, and a
        # comparable amount of gradient. `zeros_like` instead of `_LeaderOnly` shows
        # up here as ranks 1 and 3 going dead while 0 and 2 look fine.
        assert res["dead"] == leader["dead"], (
            f"rank {rank} (fuse_rank {res['fuse_rank']}) has a different set of "
            f"zero-gradient params than the block leader: "
            f"{set(res['dead']) ^ set(leader['dead'])}"
        )
        assert res["n_params_with_grad"] >= res["n_params"] - 2, (
            f"rank {rank}: only {res['n_params_with_grad']}/{res['n_params']} "
            f"params got gradient ({res['dead']})"
        )


def test_leader_only_is_zero_forward_identity_backward():
    """The Function in isolation, so the contract is pinned without a 4-rank spawn."""
    from parallm.model.sync import _LeaderOnly

    for is_leader in (True, False):
        x = torch.randn(3, 4, requires_grad=True)
        y = _LeaderOnly.apply(x, is_leader)
        if is_leader:
            assert torch.equal(y, x)
        else:
            assert torch.count_nonzero(y) == 0
        y.backward(torch.ones_like(y))
        # Identity either way — that is the whole point.
        assert torch.equal(x.grad, torch.ones_like(x)), is_leader
