"""Rails for the window-parallel slice forward (Gate 2,
``PTWrappedModel._run_window_parallel_stack``).

* N=1 rail: the 1-track slice must reproduce the DENSE window-parallel target
  function (``eval/dense_parallel.py`` mode ``parallel-attn``) — the slice of a
  healed model computes exactly the function that was healed (at N=1 the seam
  vanishes: own m == Σm).
* Singleton-windows rail: with every layer a boundary the forward must equal
  the deployed lever-B ``post-attn`` D=1 forward (same ops, same syncs).
* Sync-count contract: exactly one ``SyncBoundary`` call per window (+1 embed).
* N=2: the function differs from the phased D=2 forward (sanity that the new
  mode is not silently the old one).
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.eval.dense_parallel import build_windows, dense_window_forward
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import slice_model_to_tracks


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


# The g2-style schedule on 8 layers: window ends [0, 2, 4, 6, 7] ⇒ windows
# [0], [1,2], [3,4], [5,6], [7] — matches build_windows(8, 2).
G2_SCHEDULE = (0, 2, 4, 6, 7)


def _build(n_tracks, sync_after_layers=G2_SCHEDULE):
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=2, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=list(sync_after_layers), track_group=None,
    ).eval()
    pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    return pt, dense, cfg


def _batch(cfg, seq=16):
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    mask = torch.ones((1, seq), dtype=torch.long)
    return ids, mask


def test_n1_matches_dense_parallel_attn_target():
    # THE cross-implementation rail: the 1-track slice in window-parallel mode
    # must equal the dense target function the healing run trained.
    pt, dense, cfg = _build(1)
    pt.set_sync_phase("window-parallel")
    ids, mask = _batch(cfg)
    with torch.no_grad():
        want = dense_window_forward(
            dense, ids, mask, windows=build_windows(8, 2), mode="parallel-attn"
        )
        got, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    assert torch.allclose(want, got, atol=1e-4, rtol=1e-4)


def test_singleton_windows_match_post_attn_d1():
    # Every layer a boundary ⇒ every window is a singleton ⇒ identical to the
    # deployed lever-B post-attn D=1 forward.
    pt, _dense, cfg = _build(2, sync_after_layers=tuple(range(8)))
    ids, mask = _batch(cfg)
    with torch.no_grad():
        pt.set_sync_phase("post-attn")
        want, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        pt.set_sync_phase("window-parallel")
        got, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    assert torch.allclose(want, got, atol=1e-5, rtol=1e-5)


def test_sync_count_is_one_per_window(monkeypatch):
    pt, _dense, cfg = _build(2)
    pt.set_sync_phase("window-parallel")
    ids, mask = _batch(cfg)
    calls = {"n": 0}
    real = pt.sync_module.forward

    def _count(h_list, block_start):
        calls["n"] += 1
        return real(h_list, block_start)

    monkeypatch.setattr(pt.sync_module, "forward", _count)
    with torch.no_grad():
        pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    # 5 windows ([0],[1,2],[3,4],[5,6],[7]) + 1 embed broadcast.
    assert calls["n"] == 6


def test_n2_differs_from_phased_and_runs():
    pt, _dense, cfg = _build(2, sync_after_layers=(1, 3, 5, 7))
    ids, mask = _batch(cfg)
    with torch.no_grad():
        pt.set_sync_phase("post-attn")
        phased, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        pt.set_sync_phase("window-parallel")
        wp, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
    assert wp.shape == phased.shape
    assert torch.isfinite(wp).all()
    assert not torch.allclose(wp, phased, atol=1e-5)


def test_schedule_must_end_at_final_layer():
    pt, _dense, cfg = _build(1, sync_after_layers=(0, 2, 4, 6))  # missing 7
    pt.set_sync_phase("window-parallel")
    ids, mask = _batch(cfg)
    try:
        with torch.no_grad():
            pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        raise AssertionError("expected ValueError for schedule not ending at the final layer")
    except ValueError:
        pass
