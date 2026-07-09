"""Batch-sharded HookedTeacher forward: parity with the legacy full-batch path.

In training every rank holds the IDENTICAL batch, so the legacy teacher forward
recomputes the same thing world_size times. The sharded path runs each rank on
its ceil(B/world) row slice and all-gathers the captured hiddens (and, when
``need_logits``, the final hidden for the lm_head) back to the full batch.
Checked here on a 2-rank gloo (CPU) group with a tiny Qwen3.5 model:

- captures + logits match the unsharded forward on both ranks; B=3 exercises
  the non-divisible padding path (pad 3 → 4), B=1 the B < world path;
- ``need_logits=False`` skips the lm_head entirely (single-process check).
"""
from __future__ import annotations

import os
import socket

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.train.teacher import HookedTeacher

HOOK_INDICES = [3, 7]


def _tiny_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def _force_torch_gated_delta():
    """Same FLA→pure-torch patch as conftest's autouse fixture, but applied
    inside the spawned gloo workers (fresh processes don't see the fixture)."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    for name in (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "FusedRMSNormGated",
    ):
        setattr(m, name, None)
    m.is_fast_path_available = False


def _build_dense():
    """Tiny dense teacher, deterministic from the fixed seed (identical on every rank)."""
    torch.manual_seed(13)
    cfg = _tiny_config()
    dense = Qwen3_5TextModel(cfg).eval()
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)
    return cfg, dense, lm_head


def _worker(rank, world_size, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    _force_torch_gated_delta()
    cfg, dense, lm_head = _build_dense()

    out = {}
    for B in (3, 1):  # 3: pad 3→4 (non-divisible); 1: B < world (all-padding rank)
        torch.manual_seed(100 + B)
        input_ids = torch.randint(0, cfg.vocab_size, (B, 16))
        attn = torch.ones((B, 16), dtype=torch.long)

        sharded = HookedTeacher(
            dense, lm_head, HOOK_INDICES,
            shard_group=dist.group.WORLD, shard_world_size=world_size, shard_rank=rank,
        )
        s_logits, s_caps = sharded.forward(input_ids, attention_mask=attn)
        sharded.remove_hooks()

        ref = HookedTeacher(dense, lm_head, HOOK_INDICES)
        r_logits, r_caps = ref.forward(input_ids, attention_mask=attn)
        ref.remove_hooks()

        out[B] = {
            "cap_keys": sorted(s_caps),
            "shapes_ok": (
                s_logits.shape == r_logits.shape
                and all(s_caps[i].shape == r_caps[i].shape for i in HOOK_INDICES)
            ),
            "logits_max_err": (s_logits - r_logits).abs().max().item(),
            "caps_max_err": max(
                (s_caps[i] - r_caps[i]).abs().max().item() for i in HOOK_INDICES
            ),
        }
    results[rank] = out
    dist.destroy_process_group()


def test_sharded_teacher_matches_full_batch_two_rank():
    import torch.multiprocessing as mp

    world_size = 2
    mgr = mp.Manager()
    results = mgr.dict()
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_worker, args=(world_size, port, results), nprocs=world_size, join=True)

    assert set(results.keys()) == {0, 1}
    for rank in range(world_size):
        for B, res in results[rank].items():
            assert res["cap_keys"] == HOOK_INDICES, (rank, B)
            assert res["shapes_ok"], (rank, B)
            # Row-independent math: the sharded forward computes the same rows,
            # just batched differently — parity should be near-exact on CPU.
            assert res["logits_max_err"] < 1e-5, (rank, B, res["logits_max_err"])
            assert res["caps_max_err"] < 1e-5, (rank, B, res["caps_max_err"])


def test_need_logits_false_skips_lm_head():
    """need_logits=False must not touch the lm_head and returns logits=None
    with the captures intact (the metrics-off training path)."""
    cfg, dense, _ = _build_dense()

    class _Boom(nn.Module):
        def forward(self, *a, **k):
            raise AssertionError("lm_head must not be called with need_logits=False")

    teacher = HookedTeacher(dense, _Boom(), HOOK_INDICES)
    torch.manual_seed(7)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, caps = teacher.forward(
        input_ids, attention_mask=torch.ones((2, 16), dtype=torch.long), need_logits=False,
    )
    teacher.remove_hooks()
    assert logits is None
    assert sorted(caps) == HOOK_INDICES
    for v in caps.values():
        assert torch.isfinite(v).all()
