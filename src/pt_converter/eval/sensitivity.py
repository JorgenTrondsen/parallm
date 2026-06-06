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


def aggregate_layer_stats(
    partials: list[torch.Tensor],   # K local tracks' residuals leaving layer L
    inputs: list[torch.Tensor],     # K local tracks' residuals entering layer L
    h_synced: torch.Tensor,         # block_start + Σ_all_tracks(partial − block_start)
    full_delta: torch.Tensor,       # h_synced_L − h_synced_{L-1} (the dense layer update)
    teacher_L: torch.Tensor,        # teacher hidden at layer L
    mask: torch.Tensor | None,
    n_tracks: int,
    group: "dist.ProcessGroup | None",
    eps: float = 1e-12,
) -> dict[str, float]:
    """Reduce the per-track layer stats into the four global metrics.

    Pure tensor math + a single all-reduce of four packed scalars, so it is
    unit-testable single-process (``group=None``, ``n_tracks == len(partials)``).
    """
    tnorm = _masked_sse(teacher_L, mask).clamp(min=eps).sqrt()
    fdnorm = _masked_sse(full_delta, mask).clamp(min=eps).sqrt()

    zero = teacher_L.new_zeros((), dtype=torch.float32)
    s_e, s_dn, s_dn2, s_cos = zero.clone(), zero.clone(), zero.clone(), zero.clone()
    for p, inp in zip(partials, inputs):
        s_e = s_e + _masked_sse(p - teacher_L, mask).clamp(min=eps).sqrt()
        delta = p - inp
        dn = _masked_sse(delta, mask).clamp(min=eps).sqrt()
        cos = _masked_dot(delta, full_delta, mask) / (dn * fdnorm)
        s_dn = s_dn + dn
        s_dn2 = s_dn2 + dn * dn
        s_cos = s_cos + cos

    packed = torch.stack([s_e, s_dn, s_dn2, s_cos])
    if group is not None and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
    s_e, s_dn, s_dn2, s_cos = packed.unbind()

    mean_dn = s_dn / n_tracks
    var_dn = (s_dn2 / n_tracks) - mean_dn * mean_dn
    std_dn = var_dn.clamp(min=0).sqrt()
    return {
        "rel_err_partial": (s_e / (n_tracks * tnorm)).item(),
        "rel_err_synced": (_masked_sse(h_synced - teacher_L, mask).clamp(min=eps).sqrt() / tnorm).item(),
        "delta_norm_mean": mean_dn.item(),
        "delta_imbalance": (std_dn / mean_dn.clamp(min=eps)).item(),
        "gain_cos": (s_cos / n_tracks).item(),
    }


@torch.no_grad()
def partial_residual_probe(
    student: PTWrappedModel,
    teacher: HookedTeacher,
    batch: dict[str, torch.Tensor],
    test_sync_indices: tuple[int, ...],
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
            )
            per_track_h = new_h
            h_synced_prev = h_synced

    return results
