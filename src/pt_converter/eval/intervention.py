"""Cross-track *intervention* harness: measure a candidate comm-free channel
end-to-end, forward-only, anchored between two exact references.

The structural problem (see ``eval/sensitivity.py`` and ``model/sync.py``): with
D≥2, a mid-window layer reads only its track's *partial* residual — missing the
other tracks' deltas. Distillation can fix compounding/exposure-bias but cannot
restore that missing cross-track information; only a real (or cheap stand-in)
cross-track channel can. Every cheap-channel idea so far has been judged by a
*proxy* (residual relMSE, SVD energy, ``delta_staleness_ratio``) measured
teacher-forced — never by running the substitution and reading the metric we
actually care about (downstream / KL).

This module *runs* the substitution. At every partial-read layer it forms the
missing cross-track content ``other_k = full − y_k`` (``full`` = the synced
full-residual reconstruction at that layer, ``y_k`` = track k's partial output)
and replaces each track's carried input with ``y_k + channel(other_k)``. The
final logits are then exactly what inference/downstream would see.

Why it is *self-calibrating* — the same forward, two known anchors:

* ``zero`` channel (``S_k = 0``) → every track keeps its partial → **bit-for-bit
  the deployed D-window forward** (the floor).
* ``oracle`` channel (``S_k = other_k``) → every track reads ``full`` at every
  layer → **the D=1 / sync-every-layer forward** (the ceiling).

So ``oracle`` must reproduce the known D=1 numbers and ``zero`` the current D=2
numbers on every run — a built-in correctness check. Any cheap channel
(``stale`` / ``avg:W`` / ``lowrank:r``) lands strictly between, **measured on the
real end-to-end metric**, and its share of the ``zero → oracle`` headroom is the
honest go/no-go signal. Run it on the trained ``best`` weights and the
compounding confound is already gone (distillation fixed it), so the headroom
isolates exactly the cross-track structural value.

The single full-residual reconstruction used throughout is the running sum of
per-track, per-layer residual updates::

    full_L = full_{L-1} + Σ_k (y_k − x_k)          (full_{start} base = window input)

For the ``zero`` channel ``x_k^{(L+1)} = y_k^{(L)}``, so this telescopes to
``block_start + Σ_k(per_track_h_k − block_start)`` — exactly ``SyncBoundary``'s
boundary reconstruction (equal up to fp summation order). For ``oracle`` the
inputs are common each layer, so it is bit-identical to D=1.

Pure-tensor channel math lives at module scope (no model / NCCL), so it is
CPU-unit-testable; ``intervention_forward`` needs a built ``PTWrappedModel``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pt_converter.eval.sensitivity import _svd_cumulative_energy
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.distill import _block_ranges


# --------------------------------------------------------------------------- #
# Channels: other_k → S_k (the reconstruction added to track k's partial).
# Each operates per local track, independently, on an (B, T, H) tensor.
# --------------------------------------------------------------------------- #
def _causal_avg(o: torch.Tensor, window: int) -> torch.Tensor:
    """Mean of the **previous** ``window`` tokens of ``o`` at each position.

    Causal (uses only positions ``< t``), so it is a valid generation-time cache:
    at token ``t`` only already-decoded tokens are available. ``window == 1`` is
    the naive 1-token-stale cache (``o[t-1]``); larger windows are the multi-token
    generalization. Position 0 (no history) → 0. Vectorized via a prefix sum, no
    Python loop over ``T``.
    """
    B, T, H = o.shape
    of = o.float()
    # prefix[:, t] = Σ_{s < t} o[s]  (exclusive ⇒ causal); prefix[:, 0] = 0.
    prefix = F.pad(of.cumsum(dim=1), (0, 0, 1, 0))[:, :T, :]
    t_idx = torch.arange(T, device=o.device)
    lo = (t_idx - window).clamp(min=0)
    prefix_lo = prefix.index_select(1, lo)
    count = (t_idx - lo).to(of.dtype).view(1, T, 1)  # = min(t, window); 0 at t=0
    avg = (prefix - prefix_lo) / count.clamp(min=1.0)
    return (avg * (count > 0)).to(o.dtype)


def _lowrank_proj(o: torch.Tensor, rank: int) -> torch.Tensor:
    """Project ``o`` onto its own top-``rank`` right-singular subspace over H.

    The best rank-``r`` linear summary of this track's cross-track content — the
    *quality ceiling* of any rank-``r`` exchange/trunk bus, now scored end-to-end
    instead of by recovered energy. Uses ``torch.svd_lowrank`` (cheap for small r).
    ``rank`` ≥ the matrix rank → ~exact reconstruction (≈ oracle), a useful sanity
    point.
    """
    B, T, H = o.shape
    flat = o.reshape(-1, H).float()
    q = min(rank, flat.shape[0], H)
    if q <= 0:
        return torch.zeros_like(o)
    # flat ≈ U diag(S) Vᵀ ; projection onto the top-q right space is flat V Vᵀ.
    # niter power-iterations sharpen the randomized estimate of the top subspace.
    _, _, V = torch.svd_lowrank(flat, q=q, niter=4)
    proj = (flat @ V) @ V.transpose(-1, -2)
    return proj.reshape(B, T, H).to(o.dtype)


class Channel:
    """Map a list of per-track ``other_k`` tensors → per-track ``S_k`` tensors.

    ``layer_idx`` (the depth the substitution happens at) is passed by
    ``intervention_forward`` so a channel can keep per-layer state; channels that
    don't need it ignore it.
    """

    name: str

    def __call__(self, others: list[torch.Tensor], layer_idx: int | None = None) -> list[torch.Tensor]:
        raise NotImplementedError


class ZeroChannel(Channel):
    name = "zero"

    def __call__(self, others, layer_idx=None):
        return [torch.zeros_like(o) for o in others]


class OracleChannel(Channel):
    name = "oracle"

    def __call__(self, others, layer_idx=None):
        return list(others)


class MaskedOracleChannel(Channel):
    """Oracle at a CHOSEN subset of mid-window layers, ``zero`` elsewhere.

    Localizes *where* the ``zero → oracle`` headroom lives, end-to-end and
    compounding-aware (the residual proxy in ``eval/sensitivity.py`` cannot — it is
    teacher-forced and per-layer). With ``invert=False`` and ``layers={L}`` it is the
    *single-layer marginal*: the value of restoring cross-track info at exactly L on
    the otherwise-D2 floor. With ``invert=True`` it is *leave-one-out*: full oracle
    everywhere except L, so the drop from the oracle ceiling is L's own contribution.
    Sweeping L and pairing the two brackets the long-window compounding interaction.

    ``layers=∅, invert=False`` ≡ ``zero``; the full mid-window set ≡ ``oracle`` — the
    anchors fall out as special cases.
    """

    def __init__(self, layers: "set[int] | frozenset[int]", invert: bool = False):
        self.layers = frozenset(layers)
        self.invert = invert
        sep = "~" if invert else "@"
        self.name = f"oracle{sep}{','.join(map(str, sorted(self.layers)))}"

    def __call__(self, others, layer_idx=None):
        apply = (layer_idx in self.layers) ^ self.invert
        return list(others) if apply else [torch.zeros_like(o) for o in others]


class CausalAvgChannel(Channel):
    def __init__(self, window: int):
        self.window = window
        self.name = "stale" if window == 1 else f"avg:{window}"

    def __call__(self, others, layer_idx=None):
        return [_causal_avg(o, self.window) for o in others]


class LowRankChannel(Channel):
    """Per-input, oracle-informed SVD: the BEST rank-r basis for *this* input — an
    adaptive optimal ceiling, not a deployable fixed channel."""

    def __init__(self, rank: int):
        self.rank = rank
        self.name = f"lowrank:{rank}"

    def __call__(self, others, layer_idx=None):
        return [_lowrank_proj(o, self.rank) for o in others]


class FixedLowRankChannel(Channel):
    """Rank-r projection onto a FIXED basis fit once (first batch) per (layer, track).

    Unlike ``LowRankChannel`` (a per-input adaptive SVD), this freezes one basis per
    ``(layer_idx, track)`` from the first batch's ``other_k`` and reuses it for every
    subsequent input — the constraint a TRAINED fixed down/up bus lives under.
    Comparing the two isolates how much of the low-rank headroom needs per-input
    adaptivity vs a single fixed subspace (the deployability question for a trunk bus).
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.name = f"fixed-lowrank:{rank}"
        self._bases: dict = {}

    def __call__(self, others, layer_idx=None):
        out = []
        for i, o in enumerate(others):
            H = o.shape[-1]
            flat = o.reshape(-1, H).float()
            key = (layer_idx, i)
            V = self._bases.get(key)
            if V is None:
                q = min(self.rank, flat.shape[0], H)
                if q <= 0:
                    out.append(torch.zeros_like(o))
                    continue
                _, _, V = torch.svd_lowrank(flat, q=q, niter=4)
                V = V.detach()
                self._bases[key] = V  # frozen for the rest of the run
            proj = (flat @ V) @ V.transpose(-1, -2)
            out.append(proj.reshape(o.shape).to(o.dtype))
        return out


class CalibratedFixedLowRankChannel(Channel):
    """Fixed rank-r basis fit by PCA over a CALIBRATION SET (many batches), not one batch.

    The fair version of ``FixedLowRankChannel``: the deployable trunk fits its
    down/up over the whole training corpus, so a single-batch SVD understates what a
    fixed basis can do. This accumulates the per-``(layer, track)`` covariance
    ``C = Σ otherᵀ other`` over a calibration pass (``observing=True``, returns zeros
    so the forward stays on the deployed D=2 trajectory — the content a real bus would
    compress), then ``finalize()`` takes the top-r eigenvectors of ``C`` as the frozen
    basis. After finalize it projects like the fixed channel. Covariance accumulation
    is fixed-memory (H×H per key) regardless of calibration size.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.name = f"calib-fixed-lowrank:{rank}"
        self.observing = False
        self._cov: dict = {}
        self._bases: dict = {}

    def start_observing(self):
        self.observing = True
        self._cov = {}
        self._bases = {}

    def finalize(self):
        for key, C in self._cov.items():
            # Top-r eigenvectors of the symmetric covariance = the PCA basis
            # (eigh returns ascending eigenvalues, so the last r columns).
            _, evecs = torch.linalg.eigh(C)
            r = min(self.rank, evecs.shape[1])
            self._bases[key] = evecs[:, -r:].contiguous()
        self._cov = {}
        self.observing = False

    def __call__(self, others, layer_idx=None):
        if self.observing:
            for i, o in enumerate(others):
                H = o.shape[-1]
                flat = o.reshape(-1, H).float()
                gram = flat.transpose(0, 1) @ flat  # (H, H)
                key = (layer_idx, i)
                self._cov[key] = gram if key not in self._cov else self._cov[key] + gram
            return [torch.zeros_like(o) for o in others]  # D2 trajectory during calib
        out = []
        for i, o in enumerate(others):
            V = self._bases.get((layer_idx, i))
            if V is None:
                out.append(torch.zeros_like(o))
                continue
            H = o.shape[-1]
            flat = o.reshape(-1, H).float()
            out.append(((flat @ V) @ V.transpose(0, 1)).reshape(o.shape).to(o.dtype))
        return out


def parse_channel(spec: str) -> Channel:
    """``"zero" | "oracle" | "oracle@L1,L2" | "oracle~L1,L2" | "stale" | "avg:W" |
    "lowrank:R"`` → a ``Channel``. ``oracle@..`` = oracle only at those layers;
    ``oracle~..`` = oracle everywhere except those layers."""
    s = spec.strip()
    if s == "zero":
        return ZeroChannel()
    if s == "oracle":
        return OracleChannel()
    if s.startswith("oracle@") or s.startswith("oracle~"):
        invert = s[6] == "~"
        layers = {int(x) for x in s[7:].split(",") if x.strip()}
        return MaskedOracleChannel(layers, invert=invert)
    if s == "stale":
        return CausalAvgChannel(1)
    if s.startswith("avg:"):
        return CausalAvgChannel(int(s.split(":", 1)[1]))
    if s.startswith("lowrank:"):
        return LowRankChannel(int(s.split(":", 1)[1]))
    if s.startswith("fixed-lowrank:"):
        return FixedLowRankChannel(int(s.split(":", 1)[1]))
    if s.startswith("calib-fixed-lowrank:"):
        return CalibratedFixedLowRankChannel(int(s.split(":", 1)[1]))
    raise ValueError(f"unknown channel spec: {spec!r}")


# --------------------------------------------------------------------------- #
# The intervention forward.
# --------------------------------------------------------------------------- #
def _sum_track_deltas(
    deltas: list[torch.Tensor], group: "dist.ProcessGroup | None"
) -> torch.Tensor:
    """Σ over **all** tracks of the per-track residual updates (local sum + all-reduce).

    Mirrors ``SyncBoundary``: each rank sums its K local tracks' deltas, then one
    NCCL all-reduce across ``group`` combines the per-rank sums. ``group=None``
    (single-process multi-track) skips the collective.
    """
    total = deltas[0]
    for d in deltas[1:]:
        total = total + d
    if group is not None and dist.is_initialized():
        total = total.contiguous()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return total


@torch.no_grad()
def intervention_forward(
    student: PTWrappedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sync_indices: Sequence[int],
    channel: Channel,
) -> torch.Tensor:
    """Free-running student forward with a cross-track substitution at every
    partial-read (mid-window) layer. Returns the post-final-norm hidden
    ``(B, T, H)`` (pre-``lm_head``), identical on every rank.

    ``sync_indices`` are the *real* boundaries (a track reset + the running
    reconstruction carry forward). Between them, ``channel`` reconstructs the
    missing cross-track content. ``channel = OracleChannel`` reproduces the D=1
    forward; ``channel = ZeroChannel`` reproduces the ``sync_indices`` D-window
    forward (both up to fp summation order). Every rank must call this in
    lockstep — it all-reduces over the track group at each layer.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

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

    block_start = inputs_embeds
    for start, end in _block_ranges(num_layers, sync_indices):
        base = block_start
        per_track_h = [block_start for _ in student.text_models]
        for layer_idx in range(start, end + 1):
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
            # Running full-residual reconstruction: add this layer's per-track
            # residual updates (y_k − x_k) to the previous reconstruction.
            deltas = [y - x for y, x in zip(new_h, per_track_h)]
            full = base + _sum_track_deltas(deltas, group)
            base = full
            if layer_idx == end:
                block_start = full  # boundary reconstruction carries to the next window
            else:
                others = [full - y for y in new_h]  # missing cross-track content
                subs = channel(others, layer_idx)
                per_track_h = [y + s for y, s in zip(new_h, subs)]

    return tm0.norm(block_start)


# --------------------------------------------------------------------------- #
# Intra-block (post-attention) seam intervention — for the D=1 cross-head study.
#
# The whole-layer ``intervention_forward`` above substitutes at the layer
# *output* (the input to the next layer), so it can model the D≥2 gap (a layer
# reading a partial residual). At D=1 the residual is synced before EVERY layer,
# so that harness has nothing to substitute — D=1's *only* remaining gap is
# INSIDE the block: the post-attention RMSNorm + MLP see ``X + Y_self`` (this
# track's partial token-mixer output) instead of ``X + ΣY``.
#
# This seam version splits the HF ``Qwen3_5DecoderLayer`` at that point and
# injects a reconstruction of the missing ``Σ_{others} Y`` into the MLP input,
# while carrying only the TRUE per-track residual delta (``Y_self + mlp_partial``
# — the substitute is NOT carried, so the boundary SyncBoundary never re-sums it
# K×; see ``project_cross_track_estimator`` trunk note). Anchors:
#   * ``zero``   → MLP reads ``X + Y_self``   = the current D=1 forward.
#   * ``oracle`` → MLP reads ``X + ΣY``       = the dense teacher (exact at D=1).
# The same ``Channel`` classes apply, now on ``other_k = ΣY − Y_self_k`` (the
# missing token-mixer output) instead of the missing residual delta.
# --------------------------------------------------------------------------- #
def _seam_token_mixer(layer, x, position_embeddings, attention_mask, position_ids):
    """First half of ``Qwen3_5DecoderLayer``: ``input_layernorm`` → token mixer →
    residual-add. Returns ``(h_attn = x + Y, Y)`` where ``Y`` is the per-track
    token-mixer (self-attn or gated-delta) output before the residual add."""
    residual = x
    h = layer.input_layernorm(x)
    if layer.layer_type == "linear_attention":
        y = layer.linear_attn(
            hidden_states=h, cache_params=None, attention_mask=attention_mask
        )
    else:
        y, _ = layer.self_attn(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
    return residual + y, y


def _seam_mlp(layer, h_attn, s):
    """Second half: ``post_attention_layernorm(h_attn + s)`` → ``mlp`` → residual-add
    onto ``h_attn``. The substitute ``s`` augments the MLP INPUT only; the carried
    residual is ``h_attn + mlp_partial`` (``s`` excluded)."""
    mlp_in = h_attn if s is None else h_attn + s
    return h_attn + layer.mlp(layer.post_attention_layernorm(mlp_in))


@torch.no_grad()
def seam_intervention_forward(
    student: PTWrappedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sync_indices: Sequence[int],
    channel: Channel,
) -> torch.Tensor:
    """Free-running student forward with a cross-HEAD substitution at the
    post-attention seam of every layer. Returns the post-final-norm hidden
    ``(B, T, H)`` (pre-``lm_head``), identical on every rank.

    Designed for D=1 (``sync_indices`` = every layer): each layer starts from the
    synced residual, the token mixer is exact, and ``channel`` reconstructs the
    missing ``Σ_{others} Y`` fed to the MLP. ``OracleChannel`` reproduces the dense
    teacher; ``ZeroChannel`` reproduces the current D=1 forward. (For D≥2 windows it
    fixes only the post-attn gap, not the token-mixer-on-partial gap.)
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

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

    block_start = inputs_embeds
    for start, end in _block_ranges(num_layers, sync_indices):
        base = block_start
        per_track_h = [block_start for _ in student.text_models]
        for layer_idx in range(start, end + 1):
            # First half: token mixer per track on the carried (partial) residual.
            h_attn: list[torch.Tensor] = []
            y_list: list[torch.Tensor] = []
            for k, tm in enumerate(student.text_models):
                layer = tm.layers[layer_idx]
                layer_mask = (
                    linear_attn_mask
                    if tm.config.layer_types[layer_idx] == "linear_attention"
                    else causal_mask
                )
                ha, y = _seam_token_mixer(
                    layer, per_track_h[k], position_embeddings, layer_mask, text_position_ids
                )
                h_attn.append(ha)
                y_list.append(y)
            # ΣY across all tracks (the would-be attention-output all-reduce), then
            # the missing cross-head content for each track and the channel's stand-in.
            sumY = _sum_track_deltas(y_list, group)
            others = [sumY - y for y in y_list]
            subs = channel(others, layer_idx)
            # Second half: MLP on the augmented input; carry only the true delta.
            new_h = [_seam_mlp(tm.layers[layer_idx], h_attn[k], subs[k])
                     for k, tm in enumerate(student.text_models)]
            deltas = [o - x for o, x in zip(new_h, per_track_h)]
            full = base + _sum_track_deltas(deltas, group)
            base = full
            if layer_idx == end:
                block_start = full
            else:
                per_track_h = new_h  # carry the TRUE partial residual (substitute excluded)

    return tm0.norm(block_start)


# --------------------------------------------------------------------------- #
# Phased (post-attn, D≥1) intervention — the WRITE-SIDE memory-correction gate.
#
# In the phased forward (``PTWrappedModel._run_post_attn_stack``: post-attn sync
# at every boundary except the last layer, post-MLP sync at the last, all
# non-boundary layers fully partial) every token mixer computes on a residual
# missing 1–3 sublayers of Σ_others. That corrupts TWO distinct things:
#   (i)  the current token's READ  — q (and the gated-delta output gate z);
#   (ii) the MEMORY it WRITES     — the KV-cache entry (full-attn k/v) and the
#        gated-delta recurrent state update (k/v slices + write gate b + decay a).
# (ii) is retroactively correctable at decode with ZERO new sync events: the true
# residual at layer i's input is known 1–2 layers later at the EXISTING boundary
# all-reduce (ship per-sublayer sums instead of one lumped delta — volume, not
# frequency), so each track can recompute token T's write-side quantities and
# overwrite its cache entry / redo one recurrence step before token T+1 reads
# them. Past-token memory becomes EXACT — no prediction, no staleness (unlike
# every refuted channel, nothing substitutes the CURRENT token's missing content).
#
# ``phased_intervention_forward`` measures how much of the phased-D2 gap that
# write-side corruption owns. Modes (``PhasedMode``):
#   * ``zero``   — no substitution ⇒ the deployed phased forward (floor anchor).
#   * ``oracle`` — per-track carry replaced by the exact full residual after every
#                  sublayer ⇒ perfect-delivery ceiling for these weights.
#   * ``kv``     — the gate: write-side projection inputs swapped to
#                  ``ln_full = input_layernorm(full residual)``; q/z, the MLP and
#                  the carried residual stay partial. All-positions swap (incl.
#                  the diagonal) ⇒ a slightly optimistic upper bound on the
#                  strictly-past deployable form (~1/T of reads at MC lengths).
#   * ``q``      — the read-side complement (q + z from ``ln_full``, write side
#                  partial). NOT deploy-realizable; interprets the split.
# The exact full residual is the same telescoping running sum the whole-layer
# harness uses (``_sum_track_deltas`` after each sublayer), so ``zero`` matches
# the deployed forward up to fp summation order — the established anchor standard.
# --------------------------------------------------------------------------- #
_PHASED_MODES = ("zero", "oracle", "kv", "q")


class PhasedMode:
    """Mode marker for ``phased_intervention_forward`` (duck-types ``Channel.name``
    so the probe's anchor/%head bookkeeping applies unchanged)."""

    def __init__(self, name: str):
        if name not in _PHASED_MODES:
            raise ValueError(
                f"unknown phased mode {name!r} (choose from {', '.join(_PHASED_MODES)})"
            )
        self.name = name


def _apply_padding_mask(h: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mirror HF's ``apply_mask_to_padding_states`` (GatedDeltaNet zeroes padded
    positions of its input before projecting — the swap source must match)."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        return (h * attention_mask[:, :, None]).to(h.dtype)
    return h


class _MixerWriteSwap:
    """Within ONE token-mixer call, source the WRITE-side projection inputs
    (mode ``kv``: full-attn ``k_proj``/``v_proj``; gated-delta k/v output slices +
    ``in_proj_b`` write gate + ``in_proj_a`` decay) — or the READ-side (mode ``q``:
    full-attn ``q_proj``; gated-delta q slice + ``in_proj_z`` output gate) — from
    ``alt`` (the exact full-residual layernorm) instead of the track's partial
    input. Hook-based so the HF layer forward stays untouched; the fused
    ``in_proj_qkv`` is split at its OUTPUT (the causal conv is depthwise —
    ``groups=conv_dim`` — so a pre-conv slice swap stays clean). Modes ``zero`` /
    ``oracle`` install nothing."""

    def __init__(self, layer, mode: str, alt: torch.Tensor,
                 lin_mask: torch.Tensor | None = None):
        self.layer, self.mode = layer, mode
        self.handles: list = []
        if layer.layer_type == "linear_attention":
            alt = _apply_padding_mask(alt, lin_mask)
        self.alt = alt

    def __enter__(self):
        if self.mode not in ("kv", "q"):
            return self
        alt = self.alt

        def pre(module, args):
            return (alt,) + tuple(args[1:])

        if self.layer.layer_type == "linear_attention":
            gdn = self.layer.linear_attn
            kd = gdn.key_dim  # in_proj_qkv output layout: [q (kd) | k (kd) | v]
            alt_from_kd = self.mode == "kv"

            def qkv_hook(module, args, out):
                alt_out = F.linear(alt, module.weight)
                if alt_from_kd:
                    return torch.cat([out[..., :kd], alt_out[..., kd:]], dim=-1)
                return torch.cat([alt_out[..., :kd], out[..., kd:]], dim=-1)

            self.handles.append(gdn.in_proj_qkv.register_forward_hook(qkv_hook))
            side = (gdn.in_proj_b, gdn.in_proj_a) if self.mode == "kv" else (gdn.in_proj_z,)
        else:
            attn = self.layer.self_attn
            side = (attn.k_proj, attn.v_proj) if self.mode == "kv" else (attn.q_proj,)
        for m in side:
            self.handles.append(m.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


@torch.no_grad()
def phased_intervention_forward(
    student: PTWrappedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sync_indices: Sequence[int],
    channel: "PhasedMode",
) -> torch.Tensor:
    """Free-running PHASED (post-attn) student forward with a per-mixer role-
    targeted substitution (see the section comment above). Returns the
    post-final-norm hidden ``(B, T, H)`` (pre-``lm_head``), identical on every
    rank. Mirrors ``PTWrappedModel._run_post_attn_stack`` semantics: post-attn
    sync at every boundary in ``sync_indices`` except the final layer, post-MLP
    sync at the final layer, non-boundary layers fully partial. Every rank must
    call it in lockstep (per-sublayer all-reduces maintain the exact full
    residual). A checkpoint ``cross_head`` is intentionally ignored — the gate
    targets the plain phased frontier."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    mode = channel.name
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

    last = num_layers - 1
    sync_attn_set = set(sync_indices) - {last}
    base = inputs_embeds  # exact full residual at the current point (telescoping sum)
    per_track_h = [inputs_embeds for _ in student.text_models]
    for i in range(num_layers):
        # Norm weights are synced across tracks, so track 0's module serves all.
        ln_full = tm0.layers[i].input_layernorm(base)
        h_attn: list[torch.Tensor] = []
        y_list: list[torch.Tensor] = []
        for k, tm in enumerate(student.text_models):
            layer = tm.layers[i]
            layer_mask = (
                linear_attn_mask
                if tm.config.layer_types[i] == "linear_attention"
                else causal_mask
            )
            with _MixerWriteSwap(layer, mode, ln_full, linear_attn_mask):
                ha, y = _seam_token_mixer(
                    layer, per_track_h[k], position_embeddings, layer_mask, text_position_ids
                )
            h_attn.append(ha)
            y_list.append(y)
        base = base + _sum_track_deltas(list(y_list), group)  # true post-attn residual
        if mode == "oracle":
            h_attn = [base for _ in h_attn]
        if i in sync_attn_set:
            # Boundary: the post-attn sync delivers the true residual to every
            # track's MLP (== sync_module's reconstruction up to fp order).
            new_h = [_seam_mlp(tm.layers[i], base, None) for tm in student.text_models]
            mlp_deltas = [nh - base for nh in new_h]
        else:
            # Fully-partial layer, or the final layer (whose post-MLP "sync" is the
            # ``base`` accumulation itself — the returned hidden is fully synced).
            new_h = [_seam_mlp(tm.layers[i], h_attn[k], None)
                     for k, tm in enumerate(student.text_models)]
            mlp_deltas = [nh - ha for nh, ha in zip(new_h, h_attn)]
        base = base + _sum_track_deltas(mlp_deltas, group)  # true post-MLP residual
        per_track_h = [base for _ in new_h] if mode == "oracle" else new_h

    return tm0.norm(base)


# --------------------------------------------------------------------------- #
# Predictability decomposition of the missing Σ_others Y (training-free).
#
# stale_corr (the hybrid: 1-tok-stale ΣY skip + comm-free drift correction) ties
# the plain stale/fresh frontier (downstream ~0.67) and never reaches the oracle
# (0.70). To know whether a BETTER comm-free predictor is even possible — vs the
# orthogonality wall — we decompose, per layer, how predictable the current full
# cross-track output ΣY (the chp target; the "fresh correct data") is from the
# comm-free signals: the 1-token-stale ΣY ("stale predicted data"), multi-token
# history, and the synced block input h_ln. All numbers are training-free CEILINGS
# (the best any predictor of that class could reach), so they bound what a co-
# trained predictor could ever do. The decisive question is COMPLEMENTARITY: does
# h_ln explain the part of ΣY that stale misses (⇒ a hybrid can beat both ⇒ build),
# or is the staleness drift orthogonal to every comm-free signal (⇒ wall ⇒ D>1)?
# --------------------------------------------------------------------------- #
def _ridge_relmse(A: torch.Tensor, C: torch.Tensor, tot: torch.Tensor, lam: float) -> float:
    """IN-SAMPLE relMSE of the best ridge map ``F→O`` from the sufficient stats
    ``A=FᵀF``, ``C=FᵀO``, ``tot=‖O‖²``: ``1 − tr(Cᵀ(A+λI)⁻¹C)/tot``. ``λ`` is
    relative to the mean diagonal of ``A`` so it is scale-free across layers.
    Optimistic when ``#tokens`` is not ``≫`` the feature dim — use
    ``_ridge_oos_relmse`` for the honest generalization ceiling."""
    H = A.shape[0]
    ridge = lam * (A.diagonal().mean().clamp_min(1e-12))
    Areg = A + ridge * torch.eye(H, device=A.device, dtype=A.dtype)
    X = torch.linalg.solve(Areg, C)              # (A+λI)⁻¹ C
    explained = (C * X).sum()                     # tr(Cᵀ (A+λI)⁻¹ C)
    return (1.0 - explained / tot.clamp_min(1e-12)).item()


def _ridge_oos_relmse(
    A_fit: torch.Tensor, C_fit: torch.Tensor,
    A_eval: torch.Tensor, C_eval: torch.Tensor, tot_eval: torch.Tensor, lam: float,
) -> float:
    """OUT-OF-SAMPLE relMSE: fit ``W=(A_fit+λI)⁻¹C_fit`` on the FIT tokens, then score
    on HELD-OUT tokens via their sufficient stats:
    ``(‖O_e‖² − 2 tr(WᵀC_e) + tr(Wᵀ A_e W)) / ‖O_e‖²``. This is the honest
    generalization ceiling (a full-rank ``H×H`` linear map overfits badly in-sample
    when ``#tokens`` is not ``≫ H``; the held-out score removes that inflation)."""
    H = A_fit.shape[0]
    ridge = lam * (A_fit.diagonal().mean().clamp_min(1e-12))
    W = torch.linalg.solve(A_fit + ridge * torch.eye(H, device=A_fit.device, dtype=A_fit.dtype), C_fit)
    num = tot_eval - 2.0 * (W * C_eval).sum() + (W * (A_eval @ W)).sum()
    return (num / tot_eval.clamp_min(1e-12)).item()


@torch.no_grad()
def seam_predictability_analysis(
    student: PTWrappedModel,
    batches: "Sequence[dict]",
    sync_indices: Sequence[int],
    *,
    svd_r_grid: "tuple[int, ...]" = (16, 64, 256, 512),
    ridge_lambda: float = 1e-2,
    avg_windows: "tuple[int, ...]" = (4, 16),
    use_trained_seam: bool = True,
    eval_stride: int = 3,
    svd_max_batches: int = 3,
) -> "dict[int, dict]":
    """Per-layer predictability decomposition of the current ΣY at the D=1 seam.

    Runs the D=1 forward (using the student's own trained ``cross_head`` injection
    when present and ``use_trained_seam``, so the analysed residual stream matches
    deployment). At every layer it forms the full cross-track output
    ``ΣY = Σ_k Y_k`` (the chp target), the 1-token-stale ``stale = roll(ΣY,1)`` and
    its multi-token-avg generalizations, and the synced block input
    ``h_ln = input_layernorm(X)`` (shared across tracks at D=1). It accumulates the
    closed-form ridge sufficient stats over a calibration set and reports, per layer
    (all relMSE; ``1.0`` = predicting zero):

    * ``stale``       — relMSE(stale, ΣY) = energy of the 1-token drift / ‖ΣY‖² (the
                        skip's ceiling; ``avg:W`` are the multi-token-cache ceilings).
    * ``hln``         — best linear ``h_ln → ΣY`` (the "fresh" comm-free ceiling),
                        scored OUT-OF-SAMPLE (fit on the fit tokens, measured on the
                        held-out eval tokens; ``eval_stride``-th batches are held out).
    * ``drift_by_hln``— OOS fraction of the drift ``ΣY−stale`` h_ln explains (the
                        COMPLEMENTARITY signal: high ⇒ h_ln recovers what stale misses).
    * ``hybrid``      — ``stale + best-linear-h_ln-correction`` ceiling
                        ``= (1−drift_by_hln)·stale``; the linear ceiling of stale_corr.
                        ``hybrid ≪ min(stale, hln)`` ⇒ complementary ⇒ a better
                        predictor is worth building; ``hybrid ≈ stale`` ⇒ redundant ⇒ wall.
    * ``svd_sumy`` / ``svd_drift`` — cumulative SVD energy of ΣY and of the drift over
                        ``svd_r_grid`` (drift full-rank ⇒ the unpredicted part is
                        high-rank/isotropic ⇒ wall).

    The linear ceilings are scored OUT-OF-SAMPLE because a full-rank ``H×H`` map
    overfits in-sample when the token count is not ``≫ H`` (the held-out score is the
    honest generalization ceiling). All quantities are SHARED across tracks/ranks (ΣY
    via the existing all-reduce, h_ln replicated), so every rank computes identical
    stats; the caller prints on rank 0. Pure measurement — no grad, no training.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    device = student.embed.weight.device if hasattr(student.embed, "weight") else next(student.parameters()).device
    tm0 = student.text_models[0]
    num_layers = len(tm0.layers)
    group = student.sync_module.track_group
    ch = getattr(student, "cross_head", None) if use_trained_seam else None

    # Per-layer fp32 accumulators (built lazily on the first batch once H is known).
    # Ridge stats are split FIT vs EVAL (held-out tokens) so the linear ceilings are
    # scored out-of-sample. ``tot``/``sse`` ceilings are measured on the EVAL tokens
    # (the parameter-free baselines share the eval denominator with the regression).
    A_fit: dict[int, torch.Tensor] = {}
    A_eval: dict[int, torch.Tensor] = {}
    C_sumy_fit: dict[int, torch.Tensor] = {}
    C_sumy_eval: dict[int, torch.Tensor] = {}
    C_drift_fit: dict[int, torch.Tensor] = {}
    C_drift_eval: dict[int, torch.Tensor] = {}
    tot_sumy = torch.zeros(num_layers, device=device, dtype=torch.float64)   # eval tokens
    tot_drift = torch.zeros(num_layers, device=device, dtype=torch.float64)  # eval tokens
    sse_avg = {w: torch.zeros(num_layers, device=device, dtype=torch.float64) for w in avg_windows}
    svd_sumy = {L: {r: 0.0 for r in svd_r_grid} for L in range(num_layers)}
    svd_drift = {L: {r: 0.0 for r in svd_r_grid} for L in range(num_layers)}
    svd_count = 0

    for b_idx, batch in enumerate(batches):
        is_eval = (b_idx % eval_stride) == (eval_stride - 1)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        inputs_embeds = student.embed(input_ids)
        position_ids, text_position_ids = tm0._resolve_position_ids(inputs_embeds, None)
        causal_mask = create_causal_mask(
            config=tm0.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=None, position_ids=text_position_ids,
        )
        linear_attn_mask = (
            None if (attention_mask is not None and torch.all(attention_mask == 1)) else attention_mask
        )
        position_embeddings = tm0.rotary_emb(inputs_embeds, position_ids)

        # (B, T) validity mask, position 0 excluded (no stale history there).
        B, T = input_ids.shape
        vmask = torch.ones((B, T), device=device, dtype=torch.float32)
        vmask[:, 0] = 0.0
        if attention_mask is not None:
            vmask = vmask * attention_mask.to(vmask.dtype)
        flat_keep = vmask.reshape(-1) > 0

        block_start = inputs_embeds
        for start, end in _block_ranges(num_layers, sync_indices):
            base = block_start
            per_track_h = [block_start for _ in student.text_models]
            for layer_idx in range(start, end + 1):
                h_attn, y_list = [], []
                for k, tm in enumerate(student.text_models):
                    layer = tm.layers[layer_idx]
                    layer_mask = (
                        linear_attn_mask
                        if tm.config.layer_types[layer_idx] == "linear_attention"
                        else causal_mask
                    )
                    ha, y = _seam_token_mixer(
                        layer, per_track_h[k], position_embeddings, layer_mask, text_position_ids
                    )
                    h_attn.append(ha)
                    y_list.append(y)
                sumY = _sum_track_deltas(y_list, group)            # ΣY (all tracks), shared
                # h_ln is shared across tracks at D=1 (all read the same synced residual).
                h_ln = tm0.layers[layer_idx].input_layernorm(per_track_h[0])

                # ---- accumulate predictability stats for this layer ----
                sumf = sumY.float()
                stale = _causal_avg(sumY, 1).float()
                driftf = sumf - stale
                hlnf = h_ln.float().reshape(-1, h_ln.shape[-1])[flat_keep]
                of = sumf.reshape(-1, sumf.shape[-1])[flat_keep]
                df = driftf.reshape(-1, driftf.shape[-1])[flat_keep]
                if layer_idx not in A_fit:
                    Hd = hlnf.shape[-1]
                    z2 = lambda a, b: torch.zeros(a, b, device=device, dtype=torch.float32)
                    A_fit[layer_idx], A_eval[layer_idx] = z2(Hd, Hd), z2(Hd, Hd)
                    C_sumy_fit[layer_idx], C_sumy_eval[layer_idx] = z2(Hd, of.shape[-1]), z2(Hd, of.shape[-1])
                    C_drift_fit[layer_idx], C_drift_eval[layer_idx] = z2(Hd, df.shape[-1]), z2(Hd, df.shape[-1])
                A_, Csy, Cdr = (A_eval, C_sumy_eval, C_drift_eval) if is_eval else (A_fit, C_sumy_fit, C_drift_fit)
                A_[layer_idx] += hlnf.T @ hlnf
                Csy[layer_idx] += hlnf.T @ of
                Cdr[layer_idx] += hlnf.T @ df
                if is_eval:
                    tot_sumy[layer_idx] += of.pow(2).sum().double()
                    tot_drift[layer_idx] += df.pow(2).sum().double()
                    for w in avg_windows:
                        aw = _causal_avg(sumY, w).float()
                        sse_avg[w][layer_idx] += (sumf - aw).reshape(-1, sumf.shape[-1])[flat_keep].pow(2).sum().double()
                if b_idx < svd_max_batches:  # SVD is the cost bottleneck; the spectrum
                    # is descriptive + stable, so a few batches suffice (ridge uses all).
                    e_sumy = _svd_cumulative_energy(sumf, svd_r_grid, vmask)
                    e_drift = _svd_cumulative_energy(driftf, svd_r_grid, vmask)
                    for r in svd_r_grid:
                        svd_sumy[layer_idx][r] += e_sumy[r]
                        svd_drift[layer_idx][r] += e_drift[r]

                # ---- advance the residual stream (trained seam injection if present) ----
                if ch is not None:
                    subs, _ghat = ch.subs(layer_idx, y_list, sumY, h_ln)
                else:
                    subs = [None] * len(student.text_models)
                new_h = [_seam_mlp(tm.layers[layer_idx], h_attn[k], subs[k])
                         for k, tm in enumerate(student.text_models)]
                deltas = [o - x for o, x in zip(new_h, per_track_h)]
                full = base + _sum_track_deltas(deltas, group)
                base = full
                if layer_idx == end:
                    block_start = full
                else:
                    per_track_h = new_h
        if b_idx < svd_max_batches:
            svd_count += 1

    # ---- per-layer solve (linear ceilings scored out-of-sample) ----
    out: dict[int, dict] = {}
    for L in range(num_layers):
        if L not in A_fit:
            continue
        stale_c = (tot_drift[L] / tot_sumy[L].clamp_min(1e-12)).item()
        hln_c = _ridge_oos_relmse(
            A_fit[L], C_sumy_fit[L], A_eval[L], C_sumy_eval[L], tot_sumy[L].float(), ridge_lambda
        )
        drift_resid = _ridge_oos_relmse(
            A_fit[L], C_drift_fit[L], A_eval[L], C_drift_eval[L], tot_drift[L].float(), ridge_lambda
        )
        out[L] = {
            "stale": stale_c,
            "hln": hln_c,
            "drift_by_hln": 1.0 - drift_resid,            # OOS fraction of drift h_ln explains
            "hybrid": drift_resid * stale_c,              # stale + best linear h_ln correction
            "avg": {w: (sse_avg[w][L] / tot_sumy[L].clamp_min(1e-12)).item() for w in avg_windows},
            "svd_sumy": {r: svd_sumy[L][r] / max(1, svd_count) for r in svd_r_grid},
            "svd_drift": {r: svd_drift[L][r] / max(1, svd_count) for r in svd_r_grid},
        }
    return out
