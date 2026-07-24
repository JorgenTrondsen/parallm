"""Decoder-block seam: split a HF Qwen3.5-family decoder layer at the
post-attention point.

The post-attn (lever B) and exact sync schedules place a boundary INSIDE the
layer — after the token mixer's residual add, before the MLP — so the walk
must drive the two halves separately. Canonical home shared by the training
forward (`pt_model._run_post_attn_stack`) and the distill TF block loop.
Ported from the pre-parallm-pivot `cross_head_estimator` module (the seam
split itself was rail-validated there; the estimator machinery was not).
"""
from __future__ import annotations

import torch


def seam_token_mixer(layer, x, position_embeddings, attention_mask, position_ids):
    """First half of a Qwen3.5-family decoder layer: ``input_layernorm`` →
    token mixer → residual add. Returns ``h_attn = x + Y`` where ``Y`` is the
    per-track mixer output (self-attn or gated-delta)."""
    h_ln = layer.input_layernorm(x)
    if layer.block_type == "linear_attention":
        y = layer.linear_attn(
            hidden_states=h_ln, cache_params=None, attention_mask=attention_mask
        )
    else:
        y, _ = layer.self_attn(
            hidden_states=h_ln,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
    return x + y


def seam_mlp(layer, h_attn: torch.Tensor) -> torch.Tensor:
    """Second half: ``post_attention_layernorm`` → ``mlp`` → residual add."""
    return h_attn + layer.mlp(layer.post_attention_layernorm(h_attn))


def checkpointed_halves(use_ckpt: bool, position_embeddings, position_ids):
    """``(run_mixer, run_mlp)`` bound to this forward's no-grad scaffolding.

    ``use_ckpt`` wraps each half in an activation checkpoint, so the backward
    holds one recomputed sublayer instead of a whole own-carry window. The
    scaffolding is captured rather than passed through ``checkpoint``, which
    keeps the residual input the only checkpointed tensor. Shared by the model's
    lever-B walk and the distill TF block loop.
    """
    def _mixer(layer, x, mask):
        return seam_token_mixer(layer, x, position_embeddings, mask, position_ids)

    if not use_ckpt:
        return _mixer, seam_mlp

    from torch.utils.checkpoint import checkpoint

    return (
        lambda layer, x, mask: checkpoint(_mixer, layer, x, mask, use_reentrant=False),
        lambda layer, x: checkpoint(seam_mlp, layer, x, use_reentrant=False),
    )
