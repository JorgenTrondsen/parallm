"""Single-process forward smoke test: simulate an N=8 PT run without NCCL.

The plan calls for an N=16 test on Qwen3.5-9B-shaped configs, but at unit-test
scale we use a tiny model with `num_q_heads=8, num_kv_heads=2` so N=8 exercises
the same code paths (KV replication within groups of 4 tracks) and runs in a
few seconds on CPU.

Pipeline:
  1. Build a dense Qwen3_5TextModel + an lm_head, capture its forward output.
  2. Slice with N=8, sync_block_depth=4.
  3. Build 8 PTWrappedModel instances, load the per-track state into each.
  4. For each block (D=4 layers between syncs), run *every* track's block
     forward locally; compute `delta_t = h_t_post - h_pre_block` per track;
     sum the deltas (proxy for NCCL all_reduce SUM); broadcast
     `h_pre_block + sum(delta_t)` back to every track.
  5. Apply the final norm and lm_head on the synced hidden state.
  6. Compare to dense.

Drift at fresh init is *expected* (this is exactly the gap distillation
closes) — we just bound it loosely and assert finiteness. The strict
correctness signal is the N=1 forward parity test elsewhere.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks


def _smoke_config():
    # num_q_heads=8, num_kv_heads=2 → at N=8, tracks_per_kv_group=4 (real KV replication).
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4,
        vocab_size=64,
        rms_norm_eps=1e-6,
    )


def _run_block(student, h_in, start, end_inclusive, position_embeddings, text_position_ids, causal_mask, linear_attn_mask):
    """Run student.text_models[0].layers[start..end_inclusive] starting from h_in."""
    tm = student.text_models[0]
    h = h_in
    for layer_idx in range(start, end_inclusive + 1):
        layer = tm.layers[layer_idx]
        layer_mask = (
            linear_attn_mask
            if tm.config.layer_types[layer_idx] == "linear_attention"
            else causal_mask
        )
        h = layer(
            h,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
    return h


def test_n8_manual_sync_forward_produces_finite_bounded_output():
    cfg = _smoke_config()
    n_tracks = 8
    sync_block_depth = 4

    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=sync_block_depth, text_config_attr="config"
    )
    assert manifest.sync_layer_indices == [3, 7]

    # Build N students; share weights via the slicer output.
    students: list[PTWrappedModel] = []
    for t in range(n_tracks):
        st = PTWrappedModel(
            text_config=cfg,
            n_tracks=n_tracks,
            local_track_ids=(t,),
            sync_after_layers=manifest.sync_layer_indices,
            track_group=None,
        ).eval()
        st.load_track_state_dicts({t: tracks[t]}, strict=False)
        students.append(st)

    # Shared input.
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    attention_mask = torch.ones((1, 16), dtype=torch.long)

    # Pre-compute the rotary/mask scaffolding *once* on student[0] (it's the
    # same for every track because they all share the per-track text config).
    tm0 = students[0].text_models[0]
    inputs_embeds = tm0.embed_tokens(input_ids)
    position_ids, text_position_ids = tm0._resolve_position_ids(inputs_embeds, None)
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # local import

    causal_mask = create_causal_mask(
        config=tm0.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = None  # attention_mask is all-ones
    position_embeddings = tm0.rotary_emb(inputs_embeds, position_ids)

    # Block ranges between syncs (D=4): (0..3) and (4..7).
    block_ranges = []
    prev = -1
    for s in manifest.sync_layer_indices:
        block_ranges.append((prev + 1, s))
        prev = s

    with torch.no_grad():
        h_pre = inputs_embeds  # same on every track at the start
        for start, end in block_ranges:
            # Each track runs the block locally.
            h_t_list = [
                _run_block(
                    st, h_pre, start, end,
                    position_embeddings, text_position_ids, causal_mask, linear_attn_mask,
                )
                for st in students
            ]
            # Manual sync: h_synced = h_pre + sum_t (h_t - h_pre)
            delta_sum = torch.zeros_like(h_pre)
            for h_t in h_t_list:
                delta_sum = delta_sum + (h_t - h_pre)
            h_pre = h_pre + delta_sum

        # Final norm + lm_head (replicated across tracks; use student[0]).
        h_final = students[0].text_models[0].norm(h_pre)
        pt_logits = students[0].lm_head(h_final)

        # Dense reference.
        dense_out = dense(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        dense_logits = dense.lm_head(dense_out.last_hidden_state)

    assert torch.isfinite(pt_logits).all(), "PT N=8 manual-sync forward produced non-finite values"

    # Drift expected — bound loosely. (Distillation closes this gap.)
    max_abs = (pt_logits - dense_logits).abs().max().item()
    # On a randomly-initialized tiny model the typical drift is O(1). Pin a
    # generous ceiling to catch catastrophic bugs without being noisy.
    assert max_abs < 50.0, f"manual-sync drift {max_abs} unexpectedly large"
    # Also assert non-trivial difference exists — otherwise the test would be
    # passing for the wrong reason (e.g., zero outputs).
    assert max_abs > 1e-6, "manual-sync output matches dense too closely; suspicious"
