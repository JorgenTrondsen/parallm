"""Unit tests for the dense window-parallel forward (``eval/dense_parallel.py``).

Rails on a tiny stock ``Qwen3_5TextModel`` (CPU, no tracks):

* ``serial`` mode == the stock HF forward (any window partition), incl. padding.
* In the parallel modes a size-1 window reduces algebraically to the stock
  serial layer — so all-singleton windows must reproduce ``serial`` exactly.
* With real multi-layer windows the parallel modes move off serial and are
  pairwise distinct (exact seam vs dropped seam vs PaLM-style).
* ``build_windows`` partitions correctly; malformed windows are rejected.
"""
from __future__ import annotations

import pytest
import torch

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from pt_converter.eval.dense_parallel import build_windows, dense_window_forward


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


def _build():
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    return Qwen3_5TextModel(cfg).eval(), cfg


def _batch(cfg, seq=16, pad=0):
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (2, seq))
    mask = torch.ones((2, seq), dtype=torch.long)
    if pad:
        mask[0, :pad] = 0  # left padding, as the lm-eval adapter produces
    return ids, mask


PAIRS = [[0], [1, 2], [3, 4], [5, 6], [7]]


def test_build_windows():
    assert build_windows(8, 2) == PAIRS
    assert build_windows(8, 1) == [[i] for i in range(8)]
    assert build_windows(8, 4) == [[0], [1, 2, 3, 4], [5, 6], [7]]
    assert build_windows(8, 3, first_parallel_layer=2) == [[0], [1], [2, 3, 4], [5, 6], [7]]
    assert build_windows(8, 2, last_solo=False) == [[0], [1, 2], [3, 4], [5, 6], [7]]


def test_windows_validation_and_mode():
    model, cfg = _build()
    ids, mask = _batch(cfg)
    with torch.no_grad():
        for bad in ([[0, 1]], [[1, 0]] + PAIRS[1:], PAIRS[:-1]):
            with pytest.raises(ValueError, match="partition"):
                dense_window_forward(model, ids, mask, windows=bad, mode="serial")
        with pytest.raises(ValueError, match="unknown mode"):
            dense_window_forward(model, ids, mask, windows=PAIRS, mode="bogus")


@pytest.mark.parametrize("pad", [0, 3])
def test_serial_rail_matches_stock_forward(pad):
    model, cfg = _build()
    ids, mask = _batch(cfg, pad=pad)
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=mask).last_hidden_state
        out = dense_window_forward(model, ids, mask, windows=PAIRS, mode="serial")
    assert torch.allclose(stock, out, atol=1e-4, rtol=1e-4)


def test_singleton_windows_reduce_to_serial():
    # A size-1 window in the parallel-attn branch is algebraically the stock
    # layer (attn on W, MLP on W + y) — all-singleton windows == serial. NOT so
    # for parallel-full, whose MLP reads W instead of W + y even at size 1.
    model, cfg = _build()
    ids, mask = _batch(cfg)
    singles = build_windows(8, 1)
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=mask).last_hidden_state
        out = dense_window_forward(model, ids, mask, windows=singles, mode="parallel-attn")
        palm = dense_window_forward(model, ids, mask, windows=singles, mode="parallel-full")
    assert torch.allclose(stock, out, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(stock, palm, atol=1e-5)


def test_parallel_modes_move_and_differ():
    model, cfg = _build()
    ids, mask = _batch(cfg)
    with torch.no_grad():
        serial = dense_window_forward(model, ids, mask, windows=PAIRS, mode="serial")
        outs = {
            mode: dense_window_forward(model, ids, mask, windows=PAIRS, mode=mode)
            for mode in ("parallel-attn", "parallel-attn-dropseam", "parallel-full")
        }
    for mode, out in outs.items():
        assert not torch.allclose(serial, out, atol=1e-5), mode
    assert not torch.allclose(outs["parallel-attn"], outs["parallel-attn-dropseam"], atol=1e-5)
    assert not torch.allclose(outs["parallel-attn"], outs["parallel-full"], atol=1e-5)


def test_exact_seam_beats_dropseam_on_reconstruction():
    # Directional sanity: vs the stock trajectory, the exact-seam arm should be
    # at least as close as the dropped-seam arm (it reads strictly fresher state).
    model, cfg = _build()
    ids, mask = _batch(cfg)
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=mask).last_hidden_state
        exact = dense_window_forward(model, ids, mask, windows=PAIRS, mode="parallel-attn")
        drop = dense_window_forward(
            model, ids, mask, windows=PAIRS, mode="parallel-attn-dropseam"
        )
    def _relmse(a):
        return ((a - stock).pow(2).sum() / stock.pow(2).sum()).item()
    assert _relmse(exact) <= _relmse(drop)
