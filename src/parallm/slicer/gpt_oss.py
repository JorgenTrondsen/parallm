"""Per-parameter SlicerSpecs for gpt-oss (openai/gpt-oss-20b, -120b).

Source of truth for how every parameter inside a `GptOssDecoderLayer` is
partitioned across N tracks. Verified against the installed
`transformers/models/gpt_oss/modeling_gpt_oss.py` (5.14.1).

The family is textbook pre-norm MoE self-attention with four wrinkles the specs
have to answer:

1. **Attention sinks.** `GptOssAttention.sinks` is one learned logit PER HEAD,
   concatenated onto that head's attention scores before the softmax and dropped
   again afterwards. It only ever enters its own head's denominator, and heads are
   already disjoint across tracks with `o_proj` summing them, so slicing `sinks`
   with the q heads is EXACT.

2. **Interleaved gate/up experts.** `experts.gate_up_proj` is `[E, H, 2I]` laid out
   `[g0, u0, g1, u1, ...]` (`_apply_gate` reads `[..., ::2]` / `[..., 1::2]`), not
   `[gate | up]`. A plain `Colwise(dim=2)` still works because the chunk is always
   EVEN, so every chunk starts on a gate lane and each track's slab is still a
   well-formed interleaved pair list. No `FusedSegmentColwise` needed — but the
   evenness is load-bearing, hence `intermediate_size` in the adapter's
   `ConstraintSet.divides` and the 2:1 alignment rule below.

3. **The per-track expert width must be a multiple of 8** (`EXPERT_WIDTH_ALIGN`), or
   the `grouped_mm` experts kernel refuses the slab outright. See that constant.

4. **Biases on row-parallel outputs.** `attention_bias: true` gives q/k/v/o biases,
   and the experts carry `gate_up_proj_bias` / `down_proj_bias`. The column-parallel
   ones split with their weight, but `o_proj.bias` and `experts.down_proj_bias` sit
   on outputs that every track produces a PARTIAL of — replicating them would add
   the bias N times at the sync. `SummedBias` puts the whole bias on track 0 and
   zeros on its peers, so the summed partials carry it exactly once.
"""
from __future__ import annotations

from typing import Any

from parallm.slicer.base import (
    Colwise,
    KVReplicatedColwise,
    LayerSpec,
    Replicated,
    Rowwise,
    SummedBias,
    build_decoder_layer_specs,
    standard_top_level_specs,
)


def attention_specs(text_cfg: Any) -> LayerSpec:
    """`GptOssAttention` slicing. Identical for full and sliding layers — they
    differ only in the mask they are handed (see the adapter's `build_masks`).

    k/v (weights and biases) start as a kv-group copy and then DIVERGE per track
    (`sync=False`), the same default the Qwen3.5 full-attention layers use: each
    track gets its own KV head instead of a bit-identical copy of the kv-group's,
    turning GQA into per-track MHA — a capacity superset distillation can exploit,
    free on memory, and leaving each track self-contained. `--sync-attention-heads`
    (force_sync) restores the bit-identical behaviour.
    """
    num_kv = int(text_cfg.num_key_value_heads)
    return {
        # out_features = num_heads * head_dim, head-major, so a plain colwise split
        # hands each track whole heads.
        "q_proj.weight": Colwise(),
        "q_proj.bias": Colwise(),
        # rows are per-kv-head; every track in a kv-group starts from the same copy.
        "k_proj.weight": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "k_proj.bias": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "v_proj.weight": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "v_proj.bias": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "o_proj.weight": Rowwise(),  # cols = num_heads * head_dim -> partial sum
        "o_proj.bias": SummedBias(),  # added once across the tracks' partials
        "sinks": Colwise(),  # [num_heads], one logit per head
    }


#: `torch._grouped_mm` rejects a contracted dim whose row stride is not a multiple of
#: 16 bytes, so a bf16 per-track expert width must be a multiple of 8 — 2880/16 = 180
#: would crash the `grouped_mm` experts path at step 1, and the fallback is 4.7x
#: slower. The zero-padding that fixes it is EXACT here: a padded lane has
#: gate = up = 0 (weight AND bias are zero-filled together), and gpt-oss's gate is
#: `(up + 1) * gate * sigmoid(alpha * gate)` = `(0 + 1) * 0` = 0, so the lane
#: contributes exactly 0.0 to the down_proj sum.
EXPERT_WIDTH_ALIGN = 8


def moe_mlp_specs(text_cfg: Any) -> LayerSpec:
    """One `GptOssMLP` (router + fused expert slabs), keyed under `mlp.`.

    Tracks split by *intra-expert width*: each track holds an `I/N` slab of EVERY
    expert and computes a partial sum of every token's full MoE output, completed by
    the all-reduce at the next sync. The router runs in full on every track, so all
    tracks select the same top-k at a synced boundary.

    The `gate_up` slabs pad to `2 * EXPERT_WIDTH_ALIGN` and `down_proj` to
    `EXPERT_WIDTH_ALIGN`, which stay in an exact 2:1 ratio because N divides I (the
    adapter's `ConstraintSet`) — `align_chunk(2c, 2a) == 2 * align_chunk(c, a)`. That
    ratio is what keeps a track's `gate_up` out-width equal to twice its per-track
    `intermediate_size`, i.e. what makes the shards load.
    """
    inter = int(text_cfg.intermediate_size)
    return {
        "router.weight": Replicated(),
        "router.bias": Replicated(),
        # [E, H, 2I], interleaved gate/up. A contiguous chunk of `2 * width` keeps
        # whole (gate, up) pairs, so `_apply_gate`'s [..., ::2] / [..., 1::2] still
        # picks them apart correctly on the slice.
        "experts.gate_up_proj": Colwise(
            dim=2, pad_full_size=2 * inter, align=2 * EXPERT_WIDTH_ALIGN
        ),
        "experts.gate_up_proj_bias": Colwise(  # [E, 2I], same lanes
            dim=1, pad_full_size=2 * inter, align=2 * EXPERT_WIDTH_ALIGN
        ),
        "experts.down_proj": Rowwise(  # [E, I, H] — the input dim is I
            dim=1, pad_full_size=inter, align=EXPERT_WIDTH_ALIGN
        ),
        "experts.down_proj_bias": SummedBias(),  # [E, H] — row-parallel output
    }


# Both layer types are ordinary self-attention under the same `self_attn` module;
# `sliding_attention` differs only by mask. `valid_layer_types` derives from the keys.
ATTENTION_SPECS = {
    "full_attention": ("self_attn", attention_specs),
    "sliding_attention": ("self_attn", attention_specs),
}
VALID_LAYER_TYPES = tuple(ATTENTION_SPECS)


def decoder_layer_specs(text_cfg: Any, layer_type: str) -> LayerSpec:
    """All sliceable params under one gpt-oss decoder layer (with prefixes)."""
    return build_decoder_layer_specs(
        text_cfg,
        layer_type,
        attention_specs=ATTENTION_SPECS,
        mlp_specs=moe_mlp_specs,
    )


def build_masks(per_track_cfg, inputs_embeds, attention_mask, position_ids) -> dict:
    """``{layer_type: mask}`` — mirrors ``GptOssModel.forward``'s mask mapping."""
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    kwargs = dict(
        config=per_track_cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    return {
        "full_attention": create_causal_mask(**kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**kwargs),
    }


# Embeddings / final norm / lm_head slice identically for every family so far.
top_level_specs = standard_top_level_specs
