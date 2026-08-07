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

import torch

from parallm.slicer.base import (
    Colwise,
    GatedQColwise,
    GDNFusedQKV,
    KVReplicatedColwise,
    LayerSpec,
    PerHead,
    Replicated,
    Rowwise,
    build_decoder_layer_specs,
    standard_top_level_specs,
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

    in_proj_qkv fuses [Q | K | V] in its out_features. `GDNFusedQKV` splits it by
    *value* head (the parallel unit) and hands each track the k-heads its v-heads
    read, so each track's slice keeps the [Q_slice | K_slice | V_slice] layout the
    forward expects — including when the k-heads don't divide across tracks.

    conv1d is depthwise (groups=conv_dim), so slicing along the channel dim
    yields a self-consistent sub-conv. We slice along the same fused-segment
    layout because the conv operates on the in_proj_qkv output.
    """
    qkv = GDNFusedQKV(
        num_k_heads=int(text_cfg.linear_num_key_heads),
        num_v_heads=int(text_cfg.linear_num_value_heads),
        head_k_dim=int(text_cfg.linear_key_head_dim),
        head_v_dim=int(text_cfg.linear_value_head_dim),
    )
    num_v_heads = int(text_cfg.linear_num_value_heads)

    return {
        "in_proj_qkv.weight": qkv,
        "in_proj_z.weight": Colwise(),  # out = value_dim
        "in_proj_b.weight": Colwise(),  # out = num_v_heads
        "in_proj_a.weight": Colwise(),  # out = num_v_heads
        # conv1d.weight shape: (conv_dim, 1, kernel_size) since groups=conv_dim.
        # We slice along the channel dim (0) using the same Q|K|V head layout.
        "conv1d.weight": qkv,
        "A_log": PerHead(num_heads=num_v_heads),
        "dt_bias": PerHead(num_heads=num_v_heads),
        "out_proj.weight": Rowwise(),  # cols = value_dim
        # RMSNormGated weight is on head_v_dim, applied per head: replicated.
        "norm.weight": Replicated(),
    }


def mlp_specs(text_cfg: Any) -> LayerSpec:
    # `pad_full_size` lets intermediate_size not divide n_tracks (27B: 17408 over
    # 24 tracks). Zero-padding a SwiGLU lane is exact: silu(0)*up = 0.
    inter = int(text_cfg.intermediate_size)
    return {
        "gate_proj.weight": Colwise(pad_full_size=inter),
        "up_proj.weight": Colwise(pad_full_size=inter),
        "down_proj.weight": Rowwise(pad_full_size=inter),
    }


# Layer type -> (decoder-layer submodule prefix, spec builder). `valid_layer_types`
# is derived from the keys, so the adapter's declared list cannot drift from what
# this module can actually slice.
ATTENTION_SPECS = {
    "full_attention": ("self_attn", full_attention_specs),
    "linear_attention": ("linear_attn", linear_attention_specs),
}
VALID_LAYER_TYPES = tuple(ATTENTION_SPECS)


def decoder_layer_specs(text_cfg: Any, layer_type: str) -> LayerSpec:
    """All sliceable params under one dense decoder layer (with prefixes)."""
    return build_decoder_layer_specs(
        text_cfg,
        layer_type,
        attention_specs=ATTENTION_SPECS,
        mlp_specs=mlp_specs,
    )


def build_masks(per_track_cfg, inputs_embeds, attention_mask, position_ids) -> dict:
    """``{layer_type: mask}`` — the adapter's mask contract.

    Mirrors ``Qwen3_5TextModel.forward``'s ``causal_mask_mapping``. The linear
    (gated-delta) branch wants the raw padding mask, or None when there is no
    padding to strip.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    return {
        "full_attention": create_causal_mask(
            config=per_track_cfg,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        ),
        "linear_attention": (
            None
            if (attention_mask is not None and torch.all(attention_mask == 1))
            else attention_mask
        ),
    }


# Embeddings / final norm / lm_head slice identically for every family so far.
top_level_specs = standard_top_level_specs
