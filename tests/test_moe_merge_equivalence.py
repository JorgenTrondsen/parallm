"""Merged tracks on the MoE families: the concatenation rule must equal fusion.

Merging is the step-time lever — F shards' slabs glued into one wide module, so a
rank issues 1/F the kernels over F-times-wider GEMMs. For an MoE that is a *better*
trade than for a dense MLP, because widening the experts leaves the expert COUNT
alone: one `grouped_mm` with E group-dispatches replaces F of them with E each. (The
2026-07-20 refutation measured the opposite transformation — stacking tracks
track-major, which multiplies the group count and so buys nothing on an
expert-group-bound MLP.)

What has to hold for that to be legal is the concatenation rule on the 3-D expert
slabs, and these rails pin it at a SPARSE schedule, where the tracks own-carry
between syncs and a wrong grouping actually shows. Under `exact` every sublayer syncs
globally, so any partition sums the same deltas and the test would pass with the
groups wired wrong — the lesson recorded at the top of
`test_track_fusion_equivalence.py`.

Two layouts, deliberately:
  - gpt-oss: `gate_up_proj [E, H, 2I]` with gate/up INTERLEAVED, split `Colwise(dim=2)`,
    plus `SummedBias` on `down_proj_bias`/`o_proj.bias` and per-head `sinks`.
  - qwen3.5-MoE: `gate_up_proj [E, 2I, H]` with gate/up as SEGMENTS, split
    `FusedSegmentColwise(dim=1)`, whose merge is segment-major rather than a plain cat.
Both must land member-major on each side of the elementwise product, or the gate
lanes multiply the wrong up lanes and the down_proj rows sum the wrong products.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from parallm.adapters import get_adapter_for_config
from parallm.model.merge import merge_track_states, split_track_state
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks


def _gpt_oss_config(inter: int = 32) -> GptOssConfig:
    """Every split dim divisible by 4 so the N=4 slabs nest inside the N=2 ones."""
    cfg = GptOssConfig(
        hidden_size=64,
        intermediate_size=inter,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
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


def _qwen_moe_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        hidden_size=32,
        moe_intermediate_size=16,   # 4 | 16 and 2 | 16
        shared_expert_intermediate_size=16,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention"] * 2 + ["full_attention"]
        + ["linear_attention"] * 2 + ["full_attention"],
        full_attention_interval=3,
        vocab_size=64,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


def _dense_gpt_oss(cfg, seed: int = 0):
    torch.manual_seed(seed)
    dense = GptOssModel(cfg).eval()
    # HF leaves several biases/sinks at zero, which would make the SummedBias half
    # of this rail vacuous.
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name.endswith(("bias", "sinks")):
                p.normal_(mean=0.0, std=0.05)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    return dense.eval()


def _dense_qwen_moe(cfg, seed: int = 0):
    torch.manual_seed(seed)
    dense = Qwen3_5MoeTextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    return dense.eval()


def _run(dense, cfg, n_tracks, sync_after, fuse_size, input_ids, merge_group=1):
    n_shards = n_tracks * merge_group
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_shards, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(n_shards)}
    if merge_group > 1:
        states = merge_track_states(
            get_adapter_for_config(cfg), cfg, n_shards, states, merge_group
        )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=sync_after,
        track_group=None,
        fuse_size=fuse_size,
        merge_group=merge_group,
    )
    pt.set_sync_phase("post-attn")
    pt.eval()
    pt.load_track_state_dicts(states, strict=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())
    with torch.no_grad():
        return pt(input_ids=input_ids)[0]


# Sparse (D=2-shaped): most sublayers own-carry, the only place fusion can act.
SYNC_AFTER = [1, 3, 5]


def _ids(cfg, n=16):
    return torch.randint(0, cfg.vocab_size, (1, n), generator=torch.manual_seed(123))


# --------------------------------------------------------------------------- #
# gpt-oss
# --------------------------------------------------------------------------- #


def test_gpt_oss_merged_equals_summed_fusion():
    cfg = _gpt_oss_config()
    dense = _dense_gpt_oss(cfg)
    ids = _ids(cfg)

    summed = _run(dense, cfg, 4, SYNC_AFTER, 2, ids)
    merged = _run(dense, cfg, 2, SYNC_AFTER, 1, ids, merge_group=2)
    drift = (merged - summed).abs().max().item()
    assert drift < 1e-4, f"gpt-oss merged != summed fusion: max |dlogit| {drift}"

    # Non-vacuous: the unfused 4-track model is a genuinely different function, so
    # the agreement above is not "everything is the same at this schedule".
    unfused = _run(dense, cfg, 4, SYNC_AFTER, 1, ids)
    assert (merged - unfused).abs().max().item() > 1e-3


def test_gpt_oss_merged_pairs_equal_a_real_half_track_model():
    """The stronger claim: 4 shards merged in pairs IS the 2-track convert.

    Needs the N=4 column partition to nest inside the N=2 one, which holds when
    every split dim divides by 4 — hence `_gpt_oss_config`'s sizing.
    """
    cfg = _gpt_oss_config()
    dense = _dense_gpt_oss(cfg)
    ids = _ids(cfg)

    merged = _run(dense, cfg, 2, SYNC_AFTER, 1, ids, merge_group=2)
    real_n2 = _run(dense, cfg, 2, SYNC_AFTER, 1, ids)
    drift = (merged - real_n2).abs().max().item()
    assert drift < 1e-4, f"gpt-oss merged pairs != N=2 convert: max |dlogit| {drift}"


def test_gpt_oss_merged_equals_summed_with_padded_expert_width():
    """Same rail where the per-track expert width needs `EXPERT_WIDTH_ALIGN` padding.

    24/4 = 6 rounds up to 8, so each member carries 2 zero lanes and the merged slab
    is 2*8 wide rather than 2*6. The padded lanes must contribute exactly 0.0 on both
    sides of the merge — `align_chunk(...) * F`, never `align_chunk(... * F)`.
    """
    cfg = _gpt_oss_config(inter=24)
    dense = _dense_gpt_oss(cfg)
    ids = _ids(cfg)

    summed = _run(dense, cfg, 4, SYNC_AFTER, 2, ids)
    merged = _run(dense, cfg, 2, SYNC_AFTER, 1, ids, merge_group=2)
    drift = (merged - summed).abs().max().item()
    assert drift < 1e-4, f"padded merge != summed fusion: max |dlogit| {drift}"


def test_gpt_oss_merge_split_round_trip():
    """`split_track_state` must put a merged track back into ordinary shards — that
    is what keeps merging a training-time representation, invisible to eval/deploy."""
    cfg = _gpt_oss_config()
    dense = _dense_gpt_oss(cfg)
    adapter = get_adapter_for_config(cfg)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=4, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(4)}
    merged = merge_track_states(adapter, cfg, 4, states, 2)
    assert set(merged) == {0, 1}

    for logical, shard_ids in ((0, (0, 1)), (1, (2, 3))):
        # `first_tid` is the first SHARD id of the group, not the logical id.
        back = split_track_state(adapter, cfg, 4, merged[logical], 2, shard_ids[0])
        assert set(back) == set(shard_ids)
        for sid in shard_ids:
            for key, want in states[sid].items():
                got = back[sid][key]
                assert got.shape == want.shape, f"{key}: {got.shape} != {want.shape}"
                assert torch.equal(got, want), f"{key} changed through merge/split"


# --------------------------------------------------------------------------- #
# qwen3.5-MoE — the SEGMENT-major expert slab
# --------------------------------------------------------------------------- #


def test_qwen_moe_merged_equals_summed_fusion():
    cfg = _qwen_moe_config()
    dense = _dense_qwen_moe(cfg)
    ids = _ids(cfg)

    summed = _run(dense, cfg, 4, SYNC_AFTER, 2, ids)
    merged = _run(dense, cfg, 2, SYNC_AFTER, 1, ids, merge_group=2)
    drift = (merged - summed).abs().max().item()
    assert drift < 1e-4, f"qwen-MoE merged != summed fusion: max |dlogit| {drift}"

    unfused = _run(dense, cfg, 4, SYNC_AFTER, 1, ids)
    assert (merged - unfused).abs().max().item() > 1e-3


def test_qwen_moe_merged_pairs_equal_a_real_half_track_model():
    cfg = _qwen_moe_config()
    dense = _dense_qwen_moe(cfg)
    ids = _ids(cfg)

    merged = _run(dense, cfg, 2, SYNC_AFTER, 1, ids, merge_group=2)
    real_n2 = _run(dense, cfg, 2, SYNC_AFTER, 1, ids)
    drift = (merged - real_n2).abs().max().item()
    assert drift < 1e-4, f"qwen-MoE merged pairs != N=2 convert: max |dlogit| {drift}"
