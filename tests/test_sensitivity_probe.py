"""Unit tests for the partial-residual probe metric aggregation.

These exercise ``aggregate_layer_stats`` directly on synthetic per-track tensors
(single process, ``group=None`` ⇒ no NCCL), so they run on CPU and don't need a
model. The full ``partial_residual_probe`` forward is covered by the GPU driver.
"""
import math

import torch

from pt_converter.eval.sensitivity import aggregate_layer_stats


def test_aggregate_layer_stats_hand_computed():
    # B=1, T=1, H=2, two tracks, no mask. block_start (layer input) = [1, 1].
    teacher_L = torch.tensor([[[3.0, 4.0]]])          # ‖teacher‖ = 5
    block_start = torch.tensor([[[1.0, 1.0]]])
    p0 = torch.tensor([[[3.0, 4.0]]])                 # == teacher → err 0
    p1 = torch.tensor([[[0.0, 0.0]]])                 # err = 5
    partials = [p0, p1]
    inputs = [block_start, block_start]
    # h_synced = block_start + Σ(partial − block_start) = [1,1] + ([2,3]+[-1,-1]) = [2,3]
    h_synced = torch.tensor([[[2.0, 3.0]]])
    full_delta = h_synced - block_start                # first layer: prev synced = block_start → [1,2]

    out = aggregate_layer_stats(
        partials, inputs, h_synced, full_delta, teacher_L,
        mask=None, n_tracks=2, group=None,
    )

    # rel_err_partial = mean_t ‖p_t − teacher‖ / ‖teacher‖ = (0 + 5) / (2·5) = 0.5
    assert math.isclose(out["rel_err_partial"], 0.5, rel_tol=1e-5)
    # rel_err_synced = ‖[2,3]−[3,4]‖ / 5 = sqrt(2)/5
    assert math.isclose(out["rel_err_synced"], math.sqrt(2) / 5, rel_tol=1e-5)
    # delta norms: ‖[2,3]‖=sqrt13, ‖[-1,-1]‖=sqrt2
    s13, s2 = math.sqrt(13), math.sqrt(2)
    assert math.isclose(out["delta_norm_mean"], (s13 + s2) / 2, rel_tol=1e-5)
    # population std of two values = |a−b|/2 ; imbalance = std/mean
    assert math.isclose(out["delta_imbalance"], (s13 - s2) / (s13 + s2), rel_tol=1e-5)
    # gain_cos = mean_t cos(delta_t, full_delta=[1,2])
    cos0 = (2 + 6) / (s13 * math.sqrt(5))
    cos1 = (-1 - 2) / (s2 * math.sqrt(5))
    assert math.isclose(out["gain_cos"], (cos0 + cos1) / 2, rel_tol=1e-5)


def test_aggregate_layer_stats_parallel_deltas_invariant():
    # All tracks share an identical delta ⇒ each points exactly along the sum,
    # so gain_cos == 1 (scaling recovers direction) and delta_imbalance == 0.
    n = 3
    d = torch.tensor([[[1.0, 2.0, 2.0]]])
    block_start = torch.zeros(1, 1, 3)
    partials = [d.clone() for _ in range(n)]
    inputs = [block_start for _ in range(n)]
    h_synced = d * n                                   # Σ(partial − 0) = n·d
    full_delta = h_synced - block_start                # = n·d, parallel to each d
    teacher_L = torch.tensor([[[5.0, 5.0, 5.0]]])

    out = aggregate_layer_stats(
        partials, inputs, h_synced, full_delta, teacher_L,
        mask=None, n_tracks=n, group=None,
    )
    assert math.isclose(out["gain_cos"], 1.0, rel_tol=1e-5)
    assert out["delta_imbalance"] < 1e-5


def test_aggregate_layer_stats_mask_zeros_out_padding():
    # A fully-masked second position must not change any metric vs the
    # single-valid-position case.
    teacher_L = torch.tensor([[[3.0, 4.0], [9.0, 9.0]]])
    block_start = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    p0 = torch.tensor([[[3.0, 4.0], [7.0, 0.0]]])
    p1 = torch.tensor([[[0.0, 0.0], [0.0, 5.0]]])
    h_synced = torch.tensor([[[2.0, 3.0], [6.0, 4.0]]])
    full_delta = h_synced - block_start
    mask = torch.tensor([[1.0, 0.0]])                  # second position is padding

    out = aggregate_layer_stats(
        [p0, p1], [block_start, block_start], h_synced, full_delta, teacher_L,
        mask=mask, n_tracks=2, group=None,
    )
    # Same numbers as the unmasked single-position case above.
    assert math.isclose(out["rel_err_partial"], 0.5, rel_tol=1e-5)
    assert math.isclose(out["rel_err_synced"], math.sqrt(2) / 5, rel_tol=1e-5)
