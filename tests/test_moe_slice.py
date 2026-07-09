"""MoE slicing correctness: expert-slab spec round-trip, MoE-block partial-sum
parity, and a full tiny-model slice through the adapter.

The decisive check is `test_moe_block_partial_sum_parity`: slicing a
`Qwen3_5MoeSparseMoeBlock` by intra-expert width across N tracks and summing the
N per-track outputs must reproduce the dense block output, because the router is
replicated (same top-k on every track) and each track computes a partial sum over
the expert intermediate dim that the cross-track all-reduce completes.

Runs entirely on CPU in float32.
"""
from __future__ import annotations

import copy

import pytest
import torch

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
)

from parallm.slicer.base import FusedSegmentColwise, Rowwise
from parallm.slicer.convert import slice_model_to_tracks
from parallm.slicer.qwen3_5_moe import moe_decoder_layer_specs, moe_mlp_specs
from parallm.model.tracks.qwen3_5_moe import build_per_track_text_config


def _tiny_config():
    return Qwen3_5MoeTextConfig(
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
        vocab_size=64,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


def test_expert_slab_spec_roundtrip():
    """The two fused expert slabs round-trip through their (existing) specs."""
    E, I, H, N = 8, 16, 32, 4
    gate_up = torch.randn(E, 2 * I, H)  # [E, gate(I)|up(I), H]
    down = torch.randn(E, H, I)

    gu_spec = FusedSegmentColwise(segments=(I, I), dim=1)
    dn_spec = Rowwise(dim=2)

    gu_slices = [gu_spec.slice(gate_up, t, N) for t in range(N)]
    dn_slices = [dn_spec.slice(down, t, N) for t in range(N)]

    # per-track shapes: gate_up [E, 2*(I/N), H], down [E, H, I/N]
    assert tuple(gu_slices[0].shape) == (E, 2 * (I // N), H)
    assert tuple(dn_slices[0].shape) == (E, H, I // N)

    # reassemble reproduces the dense slab
    assert torch.equal(gu_spec.reassemble(gu_slices), gate_up)
    assert torch.equal(dn_spec.reassemble(dn_slices), down)

    # each track's gate slice must be [gate_rows | up_rows] of the SAME sub-range,
    # so .chunk(2) inside the expert forward stays consistent per track.
    per = I // N
    for t in range(N):
        expect = torch.cat(
            [gate_up[:, t * per:(t + 1) * per, :], gate_up[:, I + t * per:I + (t + 1) * per, :]],
            dim=1,
        )
        assert torch.equal(gu_slices[t], expect)


@pytest.mark.parametrize("N", [2, 4])
def test_moe_block_partial_sum_parity(N):
    """Width-sliced MoE tracks summed == dense MoE block (the key correctness gate)."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    dense = Qwen3_5MoeSparseMoeBlock(cfg).eval()
    for p in dense.parameters():
        torch.nn.init.normal_(p, std=0.05)

    x = torch.randn(1, 6, cfg.hidden_size)
    with torch.no_grad():
        dense_out = dense(x)

    specs = moe_mlp_specs(cfg)  # keys match the block's state_dict exactly
    src = dense.state_dict()
    per_track_cfg = build_per_track_text_config(cfg, N)

    acc = torch.zeros_like(dense_out)
    for t in range(N):
        blk = Qwen3_5MoeSparseMoeBlock(per_track_cfg).eval()
        sliced = {k: specs[k].slice(src[k], t, N) for k in specs}
        blk.load_state_dict(sliced, strict=True)  # strict=True asserts key/shape match
        with torch.no_grad():
            acc = acc + blk(x)

    # partial sums over the expert/shared intermediate dim reconstruct the dense op
    assert torch.allclose(acc, dense_out, atol=1e-5, rtol=1e-4), (acc - dense_out).abs().max().item()


def test_full_moe_model_slice_roundtrip():
    """Slice a tiny Qwen3_5MoeTextModel through the adapter; N=1 bit-equal, N=4 reassemble."""
    cfg = _tiny_config()
    model = Qwen3_5MoeTextModel(cfg).eval()
    state = dict(model.state_dict())

    # N=1: every per-track tensor bit-equal to the dense weight (representative MoE keys).
    tracks1, manifest1 = slice_model_to_tracks(model, n_tracks=1, text_config_attr="config")
    t0 = tracks1[0]
    for k in [
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.down_proj",
        "layers.0.mlp.shared_expert.gate_proj.weight",
        "layers.0.mlp.shared_expert.down_proj.weight",
        "layers.0.mlp.shared_expert_gate.weight",
        "layers.3.self_attn.q_proj.weight",  # attention specs reused from dense
    ]:
        assert torch.equal(t0[k], state[k]), k
    assert manifest1.model_type == "qwen3_5_moe_text"

    # N=4: reassemble the sliced expert slabs back to dense.
    N = 4
    tracksN, _ = slice_model_to_tracks(model, n_tracks=N, text_config_attr="config")
    specs = moe_decoder_layer_specs(cfg, "linear_attention")
    for key in ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]:
        full_key = f"layers.0.{key}"
        slices = [tracksN[t][full_key] for t in range(N)]
        assert torch.equal(specs[key].reassemble(slices), state[full_key]), key


@pytest.mark.parametrize("bits", [None, 4])
def test_pack_expert_slab_roundtrip(bits):
    """The 3-D expert-slab pack↔unpack reproduces the per-expert degrade bit-exactly."""
    from parallm.model.replica import wanda_prune_weight, fake_quant_weight
    from parallm.model.replica_pack import pack_expert_slab, unpack_expert_slab

    E, out, inn, frac = 6, 8, 16, 0.5
    w = torch.randn(E, out, inn)
    # Force varying per-expert survivor counts: expert 0 mostly zeros, expert 1 has
    # values that int4-quantize to zero — so nnz differs across experts (the ragged case).
    w[0] *= 0.001
    w[1, :, ::2] = 1e-4
    norms = torch.rand(E, inn) + 0.1  # per-expert input norms

    packed = pack_expert_slab(w, frac, norms, bits=bits)
    got = unpack_expert_slab(packed)

    # reference: degrade each expert independently the same way pack_sparse_weight does
    ref = torch.stack([
        wanda_prune_weight(fake_quant_weight(w[e], bits) if bits else w[e], frac, norms[e])
        for e in range(E)
    ], dim=0)
    assert torch.equal(got, ref.to(got.dtype))
    assert int(packed["num_experts"].item()) == E


if __name__ == "__main__":
    test_expert_slab_spec_roundtrip()
    test_moe_block_partial_sum_parity(2)
    test_moe_block_partial_sum_parity(4)
    test_full_moe_model_slice_roundtrip()
    test_pack_expert_slab_roundtrip(None)
    test_pack_expert_slab_roundtrip(4)
    print("ok")
