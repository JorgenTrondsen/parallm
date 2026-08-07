"""Per-parameter SlicerSpecs for Qwen3.5-MoE (text decoder).

The MoE decoder layer is identical to the dense Qwen3.5 layer except the dense
MLP is replaced by a sparse MoE block (`Qwen3_5MoeSparseMoeBlock`):

    mlp.gate.weight              router  [num_experts, hidden]
    mlp.experts.gate_up_proj     fused   [num_experts, 2*moe_inter, hidden]
    mlp.experts.down_proj                [num_experts, hidden, moe_inter]
    mlp.shared_expert.{gate,up,down}_proj.weight   a normal dense MLP
    mlp.shared_expert_gate.weight   sigmoid gate  [1, hidden]

Tracks split by *intra-expert width* (the direct analog of dense MLP slicing):
each track holds a thin `moe_inter // N` slab of **every** expert and computes a
partial sum of every token's full MoE output; the existing all-reduce at the next
sync completes it. The router (top-k selection) and the shared-expert sigmoid gate
run in full on every track, so all tracks select the same experts at a synced
boundary. This needs no new spec class: the fused expert slabs slice with the
`dim`-parameterized `FusedSegmentColwise` / `Rowwise` operating on the leading
`[num_experts, ...]` batch dim untouched.

The attention side (full + linear/GDN) is byte-identical to dense Qwen3.5, so we
import those specs verbatim.
"""
from __future__ import annotations

from typing import Any

from parallm.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    LayerSpec,
    Replicated,
    Rowwise,
    build_decoder_layer_specs,
)
from parallm.slicer.qwen3_5 import (
    ATTENTION_SPECS,  # attention slicing is byte-identical to dense Qwen3.5
    VALID_LAYER_TYPES,
    build_masks,  # re-exported for the adapter (same hybrid mask mapping)
    top_level_specs,  # re-exported for the adapter (embeddings/norm/lm_head)
)

__all__ = [
    "moe_mlp_specs",
    "moe_decoder_layer_specs",
    "top_level_specs",
    "build_masks",
    "VALID_LAYER_TYPES",
]


def moe_mlp_specs(text_cfg: Any) -> LayerSpec:
    """Slicer specs for one `Qwen3_5MoeSparseMoeBlock`, keyed under `mlp.`.

    gate_up_proj is `[E, 2I, H]` with the 2I laid out as `[gate(I) | up(I)]`;
    FusedSegmentColwise on dim 1 slices each segment so the per-track slab is
    `[E, 2*(I/N), H]` still ordered `[gate_slice | up_slice]` (the forward's
    `.chunk(2, -1)` then splits it correctly). down_proj `[E, H, I]` is sliced
    on its input dim (2) = Rowwise partial sum.
    """
    inter = int(text_cfg.moe_intermediate_size)
    return {
        "gate.weight": Replicated(),  # router runs in full on every track
        "experts.gate_up_proj": FusedSegmentColwise(segments=(inter, inter), dim=1),
        "experts.down_proj": Rowwise(dim=2),
        "shared_expert.gate_proj.weight": Colwise(),
        "shared_expert.up_proj.weight": Colwise(),
        "shared_expert.down_proj.weight": Rowwise(),
        "shared_expert_gate.weight": Replicated(),  # [1, H] sigmoid gate, per track
    }


def moe_decoder_layer_specs(text_cfg: Any, layer_type: str) -> LayerSpec:
    """All sliceable params under one MoE decoder layer (with prefixes)."""
    return build_decoder_layer_specs(
        text_cfg,
        layer_type,
        attention_specs=ATTENTION_SPECS,
        mlp_specs=moe_mlp_specs,
    )
