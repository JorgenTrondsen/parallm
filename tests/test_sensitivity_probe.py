"""Unit tests for the partial-residual probe metric aggregation.

These exercise ``aggregate_layer_stats`` directly on synthetic per-track tensors
(single process, ``group=None`` ⇒ no NCCL), so they run on CPU and don't need a
model. The full ``partial_residual_probe`` forward is covered by the GPU driver.
"""
import math

import torch

from pt_converter.eval.sensitivity import aggregate_layer_stats, _svd_cumulative_energy


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


def test_aggregate_layer_stats_staleness_ratio():
    # The StagFormer exact-cache gate. other_k = h_synced_prev − inputs[k]
    # (= Σ_{j≠k} accumulated delta entering the layer); staleness_k =
    # ‖other_k[t] − other_k[t−1]‖ / ‖other_k[t]‖ over positions ≥1 (position 0
    # excluded — no previous token). H=1, T=2, two tracks.
    h_synced_prev = torch.tensor([[[10.0], [20.0]]])      # (B=1, T=2, H=1)
    inputs = [torch.tensor([[[3.0], [5.0]]]), torch.tensor([[[4.0], [8.0]]])]
    # other_0 = [[7],[15]] → |15−7|/15 = 8/15 ; other_1 = [[6],[12]] → 6/12 = 0.5
    partials = [x.clone() for x in inputs]                 # deltas don't affect staleness
    teacher_L = torch.tensor([[[1.0], [1.0]]])
    h_synced = torch.tensor([[[1.0], [1.0]]])
    full_delta = torch.tensor([[[1.0], [1.0]]])

    out = aggregate_layer_stats(
        partials, inputs, h_synced, full_delta, teacher_L,
        mask=None, n_tracks=2, group=None, h_synced_prev=h_synced_prev,
    )
    expected = ((8.0 / 15.0) + 0.5) / 2.0
    assert math.isclose(out["delta_staleness_ratio"], expected, rel_tol=1e-5)


def test_aggregate_layer_stats_staleness_window_start_is_zero():
    # At a window start every track reads the synced residual ⇒ other_k = 0 ⇒
    # the ratio is undefined and reported as 0 (not the eps/eps == 1 floor).
    block_start = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])   # (B=1, T=2, H=2)
    inputs = [block_start, block_start]
    partials = [block_start.clone(), block_start.clone()]
    teacher_L = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    out = aggregate_layer_stats(
        partials, inputs, block_start, block_start - block_start, teacher_L,
        mask=None, n_tracks=2, group=None, h_synced_prev=block_start,
    )
    assert out["delta_staleness_ratio"] == 0.0


def test_aggregate_layer_stats_staleness_absent_by_default():
    # Without h_synced_prev (e.g. a window-start caller) the ratio is 0 — back-compat.
    teacher_L = torch.tensor([[[3.0, 4.0]]])
    block_start = torch.tensor([[[1.0, 1.0]]])
    out = aggregate_layer_stats(
        [teacher_L.clone()], [block_start], teacher_L, teacher_L - block_start,
        teacher_L, mask=None, n_tracks=1, group=None,
    )
    assert out["delta_staleness_ratio"] == 0.0


def test_svd_cumulative_energy_rank2():
    # A (B=1,T=3,H=4) matrix of exact rank 2 with singular values 3 and 2:
    # cumulative energy at r=1 is 9/13, at r≥2 is 1.0.
    x = torch.zeros(1, 3, 4)
    x[0, 0, 0] = 3.0
    x[0, 1, 1] = 2.0
    e = _svd_cumulative_energy(x, (1, 2, 3), mask=None)
    assert math.isclose(e[1], 9.0 / 13.0, rel_tol=1e-5)
    assert math.isclose(e[2], 1.0, rel_tol=1e-5)
    assert math.isclose(e[3], 1.0, rel_tol=1e-5)


def test_svd_cumulative_energy_mask_drops_padding():
    # A masked-out third row (huge values) must not change the spectrum.
    x = torch.zeros(1, 3, 4)
    x[0, 0, 0] = 3.0
    x[0, 1, 1] = 2.0
    x[0, 2, :] = 100.0                      # padding — would dominate if counted
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    e = _svd_cumulative_energy(x, (1, 2), mask=mask)
    assert math.isclose(e[1], 9.0 / 13.0, rel_tol=1e-5)
    assert math.isclose(e[2], 1.0, rel_tol=1e-5)
