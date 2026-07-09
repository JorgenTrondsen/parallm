"""Per-parameter SlicerSpecs for Qwen3.5 (text decoder).

This module is the source of truth for *how* every parameter inside a
Qwen3.5 decoder layer is partitioned across N tracks. The slicer engine in
`convert.py` consumes these dicts; per-track model modules in
`model/tracks/qwen3_5.py` instantiate `nn.Linear`/`nn.Conv1d` modules with
the matching shapes.

References (verified against installed transformers source):
- modeling_qwen3_5.py:645-718 `Qwen3_5Attention`
- modeling_qwen3_5.py:371-559 `Qwen3_5GatedDeltaNet`
- modeling_qwen3_5.py:721-734 `Qwen3_5MLP`
"""
from __future__ import annotations

from typing import Any

from parallm.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    KVReplicatedColwise,
    LayerSpec,
    OwnerOnly,
    PerHead,
    Replicated,
    Rowwise,
    build_decoder_layer_specs,
)


def full_attention_specs(text_cfg: Any) -> LayerSpec:
    num_heads = int(text_cfg.num_attention_heads)
    num_kv = int(text_cfg.num_key_value_heads)
    head_dim = int(text_cfg.head_dim)
    # Full-attention head-local params (k/v projections and their per-head
    # RMSNorms) DIVERGE per track by default (`sync=False`): each track gets its
    # own KV head instead of a bit-identical copy of the kv-group's, turning the
    # full-attention layers from GQA into per-track MHA — a capacity superset
    # distillation can exploit, free on memory (the copies already exist), and
    # leaving each track fully self-contained for per-node inference. The trainer
    # can restore legacy bit-identical sync with `--sync-attention-heads`
    # (force_sync). q_proj/o_proj are already unique per track.
    return {
        # q_proj carries [q | gate] doubled along out_features per head.
        "q_proj.weight": GatedQColwise(num_heads=num_heads, head_dim=head_dim),
        # k/v_proj rows are per-kv-head; start as a kv-group copy, then diverge.
        "k_proj.weight": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "v_proj.weight": KVReplicatedColwise(num_kv_heads=num_kv, sync=False),
        "o_proj.weight": Rowwise(),  # cols = num_heads * head_dim
        "q_norm.weight": Replicated(sync=False),  # RMSNorm on head_dim, per-track
        "k_norm.weight": Replicated(sync=False),
    }


def linear_attention_specs(text_cfg: Any) -> LayerSpec:
    """GatedDeltaNet linear-attention slicing.

    in_proj_qkv fuses [Q | K | V] in its out_features. Slicing must split
    each segment colwise independently so each track's slice still has the
    layout [Q_slice | K_slice | V_slice] that the forward expects.

    conv1d is depthwise (groups=conv_dim), so slicing along the channel dim
    yields a self-consistent sub-conv. We slice along the same fused-segment
    layout because the conv operates on the in_proj_qkv output.
    """
    num_k_heads = int(text_cfg.linear_num_key_heads)
    num_v_heads = int(text_cfg.linear_num_value_heads)
    head_k_dim = int(text_cfg.linear_key_head_dim)
    head_v_dim = int(text_cfg.linear_value_head_dim)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    qkv_segments = (key_dim, key_dim, value_dim)

    return {
        "in_proj_qkv.weight": FusedSegmentColwise(segments=qkv_segments),
        "in_proj_z.weight": Colwise(),  # out = value_dim
        "in_proj_b.weight": Colwise(),  # out = num_v_heads
        "in_proj_a.weight": Colwise(),  # out = num_v_heads
        # conv1d.weight shape: (conv_dim, 1, kernel_size) since groups=conv_dim.
        # We slice along the channel dim (0) using the same Q|K|V segment layout.
        "conv1d.weight": FusedSegmentColwise(segments=qkv_segments, dim=0),
        "A_log": PerHead(num_heads=num_v_heads),
        "dt_bias": PerHead(num_heads=num_v_heads),
        "out_proj.weight": Rowwise(),  # cols = value_dim
        # RMSNormGated weight is on head_v_dim, applied per head: replicated.
        "norm.weight": Replicated(),
    }


def mlp_specs(_text_cfg: Any) -> LayerSpec:
    return {
        "gate_proj.weight": Colwise(),
        "up_proj.weight": Colwise(),
        "down_proj.weight": Rowwise(),
    }


def decoder_layer_specs(text_cfg: Any, layer_type: str) -> LayerSpec:
    """All sliceable params under one dense decoder layer (with prefixes)."""
    return build_decoder_layer_specs(
        text_cfg,
        layer_type,
        full_attention_specs=full_attention_specs,
        linear_attention_specs=linear_attention_specs,
        mlp_specs=mlp_specs,
    )


def top_level_specs(_text_cfg: Any) -> LayerSpec:
    """Params that live outside any decoder layer: embeddings, final norm, lm head.

    `embed_tokens` and `lm_head` live on track 0 only; the input embedding is
    broadcast to peer tracks via the start-of-forward sync (zero-padded
    all-reduce), and lm_head is applied locally on track 0 to the synced
    post-final-norm hidden state. `norm.weight` stays replicated because every
    track applies the final RMSNorm to its (synced) hidden state.
    """
    return {
        "embed_tokens.weight": OwnerOnly(owner_track=0),
        "norm.weight": Replicated(),
        "lm_head.weight": OwnerOnly(owner_track=0),
    }
