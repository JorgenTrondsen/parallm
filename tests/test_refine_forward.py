"""Unit tests for the Jacobi / iterative-refinement forward (``eval/refine.py``).

Anchor contracts on a tiny single-process model (``track_group=None`` ⇒ no NCCL):

* N=1 ⇒ every fill is identically zero ⇒ dense parity at any ``iters``.
* Provable exactness: on a freshly sliced model the input to sublayer 0 is the
  embedding, so pass ``k`` makes the first ``k+1`` sublayer sums exact — at
  ``iters = 2L−1`` the forward must equal the dense forward (both carries).
* ``iters=0`` is the comm-free-plus-final-combine floor == the deployed
  boundary forward whose only sync is the mandatory last-layer one.
* ``iters=0`` with a full pass-0 base schedule telescopes to the deployed D=1
  boundary forward (the SyncBoundary-reconstruction identity).
* Exactly ``iters+1`` bulk exchanges and ``len(base_set)`` boundary syncs run.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

import pt_converter.eval.refine as refine_mod
from pt_converter.eval.refine import RefineSpec, refine_forward
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


def _build(n_tracks, sync_after_layers=(7,)):
    """Tiny single-process PT model + the dense model it was sliced from (sdpa
    attention so dense-parity checks hold to fp noise)."""
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(13)
    dense = Qwen3_5TextModel(cfg).eval()
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_tracks, sync_block_depth=4, text_config_attr="config"
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


def _relmse(a, b):
    return ((a.float() - b.float()).pow(2).sum() / b.float().pow(2).sum()).item()


def test_refine_spec_names_and_validation():
    assert RefineSpec(2, "own-fresh").name == "refine:own-fresh:x2"
    assert RefineSpec(0, "shared", (1, 3, 7)).name == "refine:shared:x0+b3"
    for bad in (lambda: RefineSpec(1, "bogus"), lambda: RefineSpec(-1, "shared")):
        try:
            bad()
            raise AssertionError("RefineSpec should have rejected invalid args")
        except ValueError:
            pass


def test_n1_dense_parity_both_carries():
    # N=1 ⇒ the track IS the whole model: every fill S_prev[j] − own[j] is zero,
    # every pass recomputes the same dense trajectory ⇒ dense parity at any iters.
    pt, dense, cfg = _build(1)
    ids, mask = _batch(cfg)
    with torch.no_grad():
        dense_h = dense(input_ids=ids, attention_mask=mask).last_hidden_state
        for carry in ("own-fresh", "shared"):
            for iters in (0, 2):
                out = refine_forward(pt, ids, mask, iters=iters, carry=carry)
                assert torch.allclose(dense_h, out, atol=1e-4, rtol=1e-4), (carry, iters)


def test_convergence_rail_matches_dense():
    # THE provable-exactness rail: pass k makes the first k+1 sublayer sums exact,
    # so iters = 2L−1 = 15 must reconstruct the dense forward on a fresh N=2 slice.
    pt, dense, cfg = _build(2)
    ids, mask = _batch(cfg)
    with torch.no_grad():
        dense_h = dense(input_ids=ids, attention_mask=mask).last_hidden_state
        for carry in ("own-fresh", "shared"):
            floor = refine_forward(pt, ids, mask, iters=0, carry=carry)
            exact = refine_forward(pt, ids, mask, iters=15, carry=carry)
            assert torch.allclose(dense_h, exact, atol=1e-4, rtol=1e-4), carry
            assert _relmse(exact, dense_h) < _relmse(floor, dense_h), carry


def test_iters0_matches_comm_free_final_combine():
    # iters=0 = comm-free per-track run + the mandatory final combine — exactly the
    # deployed boundary forward whose only sync is the last layer.
    pt, _dense, cfg = _build(2, sync_after_layers=(7,))
    ids, mask = _batch(cfg)
    with torch.no_grad():
        ref_h, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        for carry in ("own-fresh", "shared"):
            out = refine_forward(pt, ids, mask, iters=0, carry=carry)
            assert torch.allclose(ref_h, out, atol=1e-4, rtol=1e-4), carry


def test_carries_move_and_differ():
    # At N=2 the deficiency is real: one refinement pass must move the output off
    # the iters=0 floor, and the two carry rules are different reconstructions.
    pt, _dense, cfg = _build(2)
    ids, mask = _batch(cfg)
    with torch.no_grad():
        floor = refine_forward(pt, ids, mask, iters=0, carry="own-fresh")
        own1 = refine_forward(pt, ids, mask, iters=1, carry="own-fresh")
        shared1 = refine_forward(pt, ids, mask, iters=1, carry="shared")
    assert not torch.allclose(floor, own1, atol=1e-5)
    assert not torch.allclose(floor, shared1, atol=1e-5)
    assert not torch.allclose(own1, shared1, atol=1e-5)


def test_exchange_and_boundary_counts(monkeypatch):
    # The comm contract: exactly iters+1 bulk exchanges; base syncs only in pass 0
    # and never at the last layer (subsumed by the final combine).
    pt, _dense, cfg = _build(2)
    ids, mask = _batch(cfg)
    counts = {"exchange": 0, "boundary": 0}
    real_exchange = refine_mod._exchange_all_layers
    real_deltas = refine_mod._sum_track_deltas

    def _count_exchange(stack, group):
        counts["exchange"] += 1
        return real_exchange(stack, group)

    def _count_deltas(deltas, group):
        counts["boundary"] += 1
        return real_deltas(deltas, group)

    monkeypatch.setattr(refine_mod, "_exchange_all_layers", _count_exchange)
    monkeypatch.setattr(refine_mod, "_sum_track_deltas", _count_deltas)
    with torch.no_grad():
        refine_forward(pt, ids, mask, iters=3, carry="own-fresh")
    assert counts == {"exchange": 4, "boundary": 0}
    counts.update(exchange=0, boundary=0)
    with torch.no_grad():
        refine_forward(pt, ids, mask, iters=0, carry="shared", base_sync_indices=(1, 3, 7))
    assert counts == {"exchange": 1, "boundary": 2}


def test_telescoping_identity_with_full_base_schedule():
    # Pass 0 with a real boundary sync after EVERY layer telescopes to exactly the
    # deployed D=1 boundary forward (fp summation order aside) — proves the stack
    # bookkeeping equals SyncBoundary's reconstruction.
    pt, _dense, cfg = _build(2, sync_after_layers=tuple(range(8)))
    ids, mask = _batch(cfg)
    with torch.no_grad():
        ref_h, _ = pt(input_ids=ids, attention_mask=mask, return_hidden_pre_lm_head=True)
        for carry in ("own-fresh", "shared"):
            out = refine_forward(
                pt, ids, mask, iters=0, carry=carry, base_sync_indices=tuple(range(8))
            )
            assert torch.allclose(ref_h, out, atol=1e-4, rtol=1e-4), carry
