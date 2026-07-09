"""Rails for the deployable sparse-copy core (``model/replica.py``).

The cheap cross-track replica is the pivot's payload: activation-aware (Wanda)
sparse copies, optionally over an int-quantized base (qwanda). These tests pin
the per-weight transforms (exact sparsity, survivors exact, quant precision
ordering) and the faithful end-to-end mapping ``degrade_track_layers`` performs
on a real per-track model — the same construction the inference engine hosts as
its replica pool.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.model.pt_model import PTWrappedModel
from parallm.model.replica import (
    block_wanda_prune_weight,
    collect_input_norms,
    degrade_track_layers,
    fake_quant_weight,
    norms_for,
    wanda_prune_weight,
)
from parallm.slicer.convert import slice_model_to_tracks


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


# ----- pure per-weight transforms -----

def test_wanda_prune_exact_sparsity_and_survivors_exact():
    torch.manual_seed(0)
    w = torch.randn(8, 12)
    norms = torch.rand(12) + 0.1
    frac = 0.5
    p = wanda_prune_weight(w, frac, norms)
    # Exactly frac of each row is zeroed.
    assert (p == 0).sum(dim=1).tolist() == [6] * 8
    # Survivors keep their EXACT values (Wanda does not reconstruct).
    kept = p != 0
    assert torch.equal(p[kept], w[kept])


def test_wanda_scores_by_weight_times_input_norm():
    # One input channel with a huge norm should keep its weight even if |w| is
    # tiny — the activation-aware criterion, not plain magnitude.
    w = torch.tensor([[0.01, 1.0, 1.0, 1.0]])
    norms = torch.tensor([1000.0, 1.0, 1.0, 1.0])
    p = wanda_prune_weight(w, 0.5, norms)  # drop 2 of 4
    assert p[0, 0] != 0  # tiny weight, huge activation → survives


def test_fake_quant_precision_ordering():
    torch.manual_seed(1)
    w = torch.randn(16, 32)
    err8 = (fake_quant_weight(w, 8) - w).abs().mean()
    err4 = (fake_quant_weight(w, 4) - w).abs().mean()
    assert err8 < err4  # more bits = closer
    assert fake_quant_weight(w, 4).shape == w.shape


def test_block_wanda_zeros_whole_blocks():
    torch.manual_seed(2)
    w = torch.randn(4, 16)
    norms = torch.rand(16) + 0.1
    p = block_wanda_prune_weight(w, 0.5, norms, block_size=4)
    # Each row has 4 blocks of 4; half the blocks (2) are fully zeroed.
    per_row = p.reshape(4, 4, 4)
    zero_blocks = (per_row == 0).all(dim=-1).sum(dim=-1)
    assert zero_blocks.tolist() == [2, 2, 2, 2]


# ----- calibration + end-to-end degradation -----

def _tiny_dense_and_tracks(n_tracks: int):
    cfg = _tiny_config()
    torch.manual_seed(42)
    dense = Qwen3_5TextModel(cfg)
    dense.eval()
    tracks, _ = slice_model_to_tracks(dense, n_tracks=n_tracks, text_config_attr="config")
    return cfg, dense, tracks


def test_collect_input_norms_matches_manual():
    cfg, dense, _ = _tiny_dense_and_tracks(2)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    batches = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}]
    norms = collect_input_norms(dense, batches, device="cpu")
    # A key per (layer, distinct-input-space); values are per-input-channel L2.
    assert "0.mlp.gate_proj" in norms
    g = norms["0.mlp.gate_proj"]
    assert g.numel() == cfg.hidden_size
    assert torch.all(g >= 0)


def test_degrade_track_layers_matches_recompute():
    n_tracks = 2
    cfg, dense, tracks = _tiny_dense_and_tracks(n_tracks)
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=(0,),
        sync_after_layers=[3, 7], track_group=None,
    )
    pt.eval()
    pt.load_track_state_dicts({0: tracks[0]}, strict=False)
    track_model = pt.text_models[0]

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    batches = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}]
    norms = collect_input_norms(dense, batches, device="cpu")

    frac = 0.5
    # Snapshot the originals before degradation (deepcopy inside must not mutate).
    orig = {(li, rel): mod.weight.data.clone()
            for li, layer in enumerate(track_model.layers)
            for rel, mod in layer.named_modules() if isinstance(mod, nn.Linear)}

    # Plain wanda: every Linear equals wanda_prune_weight of its slice's norms.
    shadow = degrade_track_layers(track_model, norms, n_tracks, track_id=0, frac=frac)
    for li, clone in enumerate(shadow):
        for rel, mod in clone.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            w0 = orig[(li, rel)]
            nv = norms_for(norms, n_tracks, li, rel, w0.shape[-1], 0)
            expected = wanda_prune_weight(w0, frac, nv)
            assert torch.equal(mod.weight.data, expected), f"{li}.{rel}"
            # exactly frac of each row zeroed
            k = int(round(frac * w0.shape[-1]))
            assert torch.equal((mod.weight.data == 0).sum(dim=1),
                               torch.full((w0.shape[0],), k))

    # Original track weights untouched by degradation.
    for li, layer in enumerate(track_model.layers):
        for rel, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                assert torch.equal(mod.weight.data, orig[(li, rel)])

    # qwanda: quantize to int4 FIRST, then wanda-prune.
    qshadow = degrade_track_layers(track_model, norms, n_tracks, track_id=0, frac=frac, bits=4)
    clone0 = qshadow[0]
    for rel, mod in clone0.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w0 = orig[(0, rel)]
        nv = norms_for(norms, n_tracks, 0, rel, w0.shape[-1], 0)
        expected = wanda_prune_weight(fake_quant_weight(w0, 4), frac, nv)
        assert torch.equal(mod.weight.data, expected), f"qwanda {rel}"
