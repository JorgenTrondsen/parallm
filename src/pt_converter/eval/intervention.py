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
