"""Per-layer partial-residual diagnostics for uniform sync schedules.

D=1 (sync every layer) recovers the dense model; uniform D≥2 stalls because,
within a D-layer window, a track's layer ``L+1`` reads its OWN partial residual
``h_in + Σ_{j<L} delta_t^j`` — missing the other tracks' deltas (~``(N-1)/N`` of
each layer's update). This probe quantifies, **per layer**, how wrong that
partial residual is and *why*, so we can pick the right structural recovery
lever before implementing one.

It runs **teacher-forced per window** (each window starts from the teacher's
hidden at the window boundary), mirroring the distill block loop in
``train/distill.py`` — so the numbers are directly comparable to what block-MSE
distillation sees. For each layer it reports four global (all-track) metrics:

- ``rel_err_partial`` — mean over tracks of ``‖partial_t − teacher_L‖/‖teacher_L‖``,
  the relative error of the *un-synced* residual a track carries into the next
  layer. Large (esp. at the full-attention layers / deeper window positions) ⇒
  the mid-window layer is fed a bad residual ⇒ per-intra-layer MSE supervision
  (lever c) is the target.
- ``rel_err_synced`` — ``‖h_synced_L − teacher_L‖/‖teacher_L‖``, the error if we
  *did* sync at L (the block-MSE reference); the floor the partial path is
  drifting away from.
- ``delta_imbalance`` — ``std_t‖delta_t‖ / mean_t‖delta_t‖`` of the per-track
  per-layer residual deltas. High ⇒ a few tracks dominate the update ⇒ head/
  neuron scattering (lever b). Low ⇒ each track is ~1/N, so a ×N gain is
  well-posed.
- ``gain_cos`` — mean over tracks of ``cos(delta_t, Σ_t' delta_t')``. High ⇒ a
  track's delta points the same way as the full update, so scaling it (×N gain,
  lever a) recovers the residual *direction*. Low ⇒ scaling can't recover the
  missing content; needs (b)/(c).

All metrics are reduced across the track group, so every rank returns identical
scalars. ``group=None`` (single process, all tracks local) is supported for unit
testing — the SyncBoundary degenerates to a local sum and the all-reduce is
skipped.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.distill import _block_ranges
from pt_converter.train.teacher import HookedTeacher


def _masked_sse(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Sum of squares of ``x`` over non-pad positions (fp32)."""
    sq = x.float().pow(2)
    if mask is None:
        return sq.sum()
    m = mask
    while m.ndim < sq.ndim:
        m = m.unsqueeze(-1)
    return (sq * m).sum()


def _masked_dot(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Inner product of ``a`` and ``b`` over non-pad positions (fp32)."""
    p = a.float() * b.float()
    if mask is None:
        return p.sum()
    m = mask
    while m.ndim < p.ndim:
        m = m.unsqueeze(-1)
    return (p * m).sum()


def _svd_cumulative_energy(
    x: torch.Tensor,
    r_grid: "tuple[int, ...]",
    mask: torch.Tensor | None,
    eps: float = 1e-12,
) -> dict[int, float]:
    """Cumulative singular-energy of ``x`` over a rank grid.

    Low-rank-exchange viability gate: a rank-``r`` projection recovers the fraction
    ``Σ_{i≤r} σ_i² / Σ_i σ_i²`` of ``x``'s energy. ``x`` is ``(B, T, H)``; pad
    positions (``mask`` 0) are dropped, then the ``(valid_tokens, H)`` matrix's
    singular values give the spectrum. Returns ``{r: cumulative_energy_fraction}``
    (``r ≥ #singular_values`` → 1.0). Pure tensor math, single-process testable.
    """
    h = x.shape[-1]
    flat = x.reshape(-1, h).float()
    if mask is not None:
        flat = flat[mask.reshape(-1) > 0]
    if flat.shape[0] == 0:
        return {r: 0.0 for r in r_grid}
    sv = torch.linalg.svdvals(flat)
    energy = sv.pow(2)
    cum = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=eps)
    n = cum.shape[0]
    return {r: (cum[r - 1] if r <= n else cum[-1]).item() for r in r_grid}


def aggregate_layer_stats(
    partials: list[torch.Tensor],   # K local tracks' residuals leaving layer L
    inputs: list[torch.Tensor],     # K local tracks' residuals entering layer L
    h_synced: torch.Tensor,         # block_start + Σ_all_tracks(partial − block_start)
    full_delta: torch.Tensor,       # h_synced_L − h_synced_{L-1} (the dense layer update)
    teacher_L: torch.Tensor,        # teacher hidden at layer L
    mask: torch.Tensor | None,
    n_tracks: int,
    group: "dist.ProcessGroup | None",
    h_synced_prev: "torch.Tensor | None" = None,  # synced residual ENTERING layer L
    eps: float = 1e-12,
) -> dict[str, float]:
    """Reduce the per-track layer stats into the global metrics.

    Pure tensor math + a single all-reduce of the packed scalars, so it is
    unit-testable single-process (``group=None``, ``n_tracks == len(partials)``).
    When ``h_synced_prev`` is given, also reports ``delta_staleness_ratio`` (the
    StagFormer exact-cache gate; 0 at window-start layers where it is undefined).
    """
    tnorm = _masked_sse(teacher_L, mask).clamp(min=eps).sqrt()
    fdnorm = _masked_sse(full_delta, mask).clamp(min=eps).sqrt()

    # Staleness diagnostic: at a partial-read layer the other tracks' summed delta
    # entering the layer is  other_k = h_synced_prev − inputs[k]  (= Σ_{j≠k} of the
    # accumulated delta — clean identity, no block_start needed). The ratio
    # ‖other_k[t] − other_k[t−1]‖ / ‖other_k[t]‖ says whether substituting the
    # PREVIOUS token's value (the 1-token-stale cache) beats leaving it out:
    # <1 ⇒ cache helps, ≥1 ⇒ it hurts. Position 0 has no previous token ⇒ excluded.
    want_stale = h_synced_prev is not None
    if want_stale:
        roll_mask = teacher_L.new_ones(teacher_L.shape[:2])  # (B, T)
        roll_mask[:, 0] = 0.0
        if mask is not None:
            roll_mask = roll_mask * mask.to(roll_mask.dtype)

    zero = teacher_L.new_zeros((), dtype=torch.float32)
    s_e, s_dn, s_dn2, s_cos = zero.clone(), zero.clone(), zero.clone(), zero.clone()
    s_stale = zero.clone()
    for p, inp in zip(partials, inputs):
        s_e = s_e + _masked_sse(p - teacher_L, mask).clamp(min=eps).sqrt()
        delta = p - inp
        dn = _masked_sse(delta, mask).clamp(min=eps).sqrt()
        cos = _masked_dot(delta, full_delta, mask) / (dn * fdnorm)
        s_dn = s_dn + dn
        s_dn2 = s_dn2 + dn * dn
        s_cos = s_cos + cos
        if want_stale:
            other = h_synced_prev - inp
            stale_den = _masked_sse(other, roll_mask).sqrt()
            stale_num = _masked_sse(
                other - torch.roll(other, shifts=1, dims=1), roll_mask
            ).sqrt()
            # Window-start layers carry other_k == 0 (the layer reads the synced
            # residual, nothing missing) ⇒ ratio undefined ⇒ contribute 0, not the
            # eps/eps == 1 floor a blind clamp would give.
            s_stale = s_stale + torch.where(
                stale_den > eps,
                stale_num / stale_den.clamp(min=eps),
                torch.zeros_like(stale_den),
            )

    packed = torch.stack([s_e, s_dn, s_dn2, s_cos, s_stale])
    if group is not None and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
    s_e, s_dn, s_dn2, s_cos, s_stale = packed.unbind()

    mean_dn = s_dn / n_tracks
    var_dn = (s_dn2 / n_tracks) - mean_dn * mean_dn
    std_dn = var_dn.clamp(min=0).sqrt()
    return {
        "rel_err_partial": (s_e / (n_tracks * tnorm)).item(),
        # Raw components behind rel_err_partial, exposed for de-biased placement
        # scoring (see `debias_partial_error`): abs_err_partial is the mean over
        # tracks of ‖partial_t − teacher_L‖ (the numerator), teacher_norm is
        # ‖teacher_L‖ (the denominator). Both are post-all-reduce, so identical on
        # every rank.
        "abs_err_partial": (s_e / n_tracks).item(),
        "teacher_norm": tnorm.item(),
        "rel_err_synced": (_masked_sse(h_synced - teacher_L, mask).clamp(min=eps).sqrt() / tnorm).item(),
        "delta_norm_mean": mean_dn.item(),
        "delta_imbalance": (std_dn / mean_dn.clamp(min=eps)).item(),
        "gain_cos": (s_cos / n_tracks).item(),
        "delta_staleness_ratio": (s_stale / n_tracks).item(),
    }


def debias_partial_error(
    abs_err: "dict[int, float]",
    teacher_norm: "dict[int, float]",
    quantile: float = 0.5,
) -> "dict[int, float]":
    """Per-layer sync-need score that suppresses the small-norm early-layer bias.

    The plain ``rel_err_partial = abs_err / teacher_norm`` over-ranks the first few
    layers: the residual-stream norm is tiny there, so a track missing 15/16 of a
    layer's (relatively large) update reads as >100% relative error — a
    small-denominator artifact, not real end-to-end sync-need (the deep residual gap
    is the structural one). This divides the **absolute** partial error by a *floored*
    teacher norm, where ``floor`` is the ``quantile``-th quantile of the per-layer
    teacher norms:

      - ``q = 0`` → floor = min norm → no clamp → **pure relative** (legacy behaviour).
      - ``q = 1`` → floor = max norm → **raw** error / one global scale (deep-biased).
      - ``q = 0.5`` → median floor: only the sub-median-norm (early) layers are clamped
        and demoted; mid/deep layers keep their natural relative ranking.

    ``abs_err`` and ``teacher_norm`` are per-layer dicts (the
    ``abs_err_partial`` / ``teacher_norm`` keys from ``aggregate_layer_stats``). Pure
    arithmetic — single-process unit-testable.
    """
    import numpy as np

    layers = sorted(abs_err)
    norms = [teacher_norm[L] for L in layers]
    floor = float(np.quantile(norms, quantile)) if norms else 0.0
    eps = 1e-12
    return {L: abs_err[L] / max(teacher_norm[L], floor, eps) for L in layers}


@torch.no_grad()
def partial_residual_probe(
    student: PTWrappedModel,
    teacher: HookedTeacher,
    batch: dict[str, torch.Tensor],
    test_sync_indices: tuple[int, ...],
    svd_energy: bool = False,
    svd_r_grid: "tuple[int, ...]" = (8, 16, 32, 64, 128, 256),
) -> dict[int, dict[str, float]]:
    """Per-layer diagnostics under a uniform ``test_sync_indices`` schedule.

    ``teacher`` MUST be hooked at every layer (``sync_layer_indices ==
    range(num_layers)``) so a teacher hidden is available at each depth. Returns
    ``{layer_idx: {metric: float}}`` with global (all-track) metrics — identical
    on every rank.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    _, teacher_hiddens = teacher.forward(input_ids, attention_mask=attention_mask)

    inputs_embeds = student.embed(input_ids)
    tm0 = student.text_models[0]
    position_ids, text_position_ids = tm0._resolve_position_ids(inputs_embeds, None)
    causal_mask = create_causal_mask(
        config=tm0.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        past_key_values=None, position_ids=text_position_ids,
    )
    linear_attn_mask = (
        None if (attention_mask is not None and torch.all(attention_mask == 1)) else attention_mask
    )
    position_embeddings = tm0.rotary_emb(inputs_embeds, position_ids)

    num_layers = len(tm0.layers)
    group = student.sync_module.track_group
    n_tracks = student.n_tracks
    results: dict[int, dict[str, float]] = {}

    for start, end in _block_ranges(num_layers, test_sync_indices):
        # Teacher-forced: each window starts from the teacher's hidden at the
        # boundary (embeddings for the first window), mirroring distill_step.
        block_start = inputs_embeds if start == 0 else teacher_hiddens[start - 1]
        per_track_h = [block_start for _ in student.text_models]
        h_synced_prev = block_start
        for layer_idx in range(start, end + 1):
            inputs_into_layer = per_track_h
            new_h: list[torch.Tensor] = []
            for k, tm in enumerate(student.text_models):
                layer = tm.layers[layer_idx]
                layer_mask = (
                    linear_attn_mask
                    if tm.config.layer_types[layer_idx] == "linear_attention"
                    else causal_mask
                )
                new_h.append(
                    layer(
                        per_track_h[k],
                        position_embeddings=position_embeddings,
                        attention_mask=layer_mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        use_cache=False,
                    )
                )
            h_synced = student.sync_module(new_h, block_start)
            full_delta = h_synced - h_synced_prev
            results[layer_idx] = aggregate_layer_stats(
                new_h, inputs_into_layer, h_synced, full_delta,
                teacher_hiddens[layer_idx], attention_mask, n_tracks, group,
                h_synced_prev=h_synced_prev,
            )
            if svd_energy:
                # A = Σⱼ accumulated delta entering this layer (= h_synced_prev −
                # block_start); its singular spectrum sets how well a rank-r
                # cross-track exchange could reconstruct the missing content.
                # Window-start layers read the synced residual (A=0) ⇒ skip.
                if layer_idx > start:
                    energy = _svd_cumulative_energy(
                        h_synced_prev - block_start, svd_r_grid, attention_mask
                    )
                else:
                    energy = {r: 0.0 for r in svd_r_grid}
                results[layer_idx].update({f"svd_r{r}": e for r, e in energy.items()})
            per_track_h = new_h
            h_synced_prev = h_synced

    return results
