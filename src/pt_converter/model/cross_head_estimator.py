"""Cross-head attention-output estimator for the D=1 post-attention seam.

At D=1 (sync after every layer) the ONLY remaining structural gap is intra-block:
each track's post-attention RMSNorm + MLP see ``X + Y_self`` (this track's partial
token-mixer output) instead of ``X + ΣY`` (the full token-mixer output). The
attention-output all-reduce that would supply ``Σ_{others} Y`` is dropped. This
module supplies a co-trained estimate of that missing content at the seam, so each
track's MLP runs on ``≈ X + ΣY``.

Crucially the substitute augments the MLP INPUT only; the carried residual stays
``X + Y_self + mlp_partial`` (the substitute is excluded), so the boundary
``SyncBoundary`` never re-sums it K× — the divergence pitfall documented in
``project_cross_track_estimator`` (trunk). With the per-layer gate at its zero
init the whole module is an exact no-op (bit-identical to plain D=1), so a
warm-start from a converged D=1 checkpoint starts on the same trajectory and the
gate opens during training.

Two backends:

* ``fresh`` (comm-free): a shared per-layer module ``g`` reads the freshly-synced
  block input ``h_ln = input_layernorm(X)`` (identical across tracks at D=1) and
  predicts the full token-mixer output ``ΣY``. ``S_k = gate ⊙ (g(h_ln) − Y_self_k)``
  ⇒ ``mlp_in_k = X + Y_self_k + S_k = X + g(h_ln)`` when the gate is open. No
  staleness, input-adaptive, and **no extra inference communication** (``g`` is
  local and identical on every track). Trained by a direct predict loss
  ``relMSE(g, ΣY.detach())`` (the stable driver — end-to-end-only diverged at D≥2,
  see ``project_staggered_xattn``) plus whatever end-to-end loss flows back.

* ``stale`` (sync-payload-only): no predictor — uses the 1-token-stale cross-head
  content (a seq-shift at train time; a residual-stream cache at inference, sent
  on the existing boundary all-reduce as a bigger payload). ``S_k = gate ⊙
  roll(ΣY − Y_self_k, 1)``.

* ``stale_corr`` (HYBRID, frequency-free): the 1-token-stale real ``ΣY`` (the
  ``stale`` payload — no extra sync, piggybacked on the existing per-layer boundary
  all-reduce) as a SKIP, PLUS a comm-free learned predictor of the 1-token DRIFT
  ``ΣY − roll(ΣY, 1)`` from local features ``[h_ln, roll(ΣY,1)]``. The predictor
  outputs only the correction, so ``ghat = roll(ΣY,1) + g([h_ln, roll(ΣY,1)])``;
  ``S_k = gate ⊙ (ghat − Y_self_k) ⇒ mlp_in_k = X + ghat``. If the drift is
  unpredictable (the delta-staleness wall) the correction collapses to ~0, ``ghat →
  roll(ΣY,1)``, and the MLP reads ``X + stale ΣY`` (stale-quality, the plain-
  ``stale`` floor) — so the hybrid can only help. The skip is the real stale data,
  detached; the predictor reads detached inputs (no self-corruption), trained by
  ``relMSE(ghat, ΣY.detach())`` like ``fresh``.

* ``window_stale`` (post-attn D>1 only, gate-only, frequency-free): pairs with
  ``--sync-phase post-attn`` at D>1. The boundary layers already get the exact ``ΣY``
  (post-attention sync); this fills the fully-PARTIAL non-boundary layers with the
  real ``ΣY`` cached from the previous boundary (1 layer of depth stale, no new sync).
  ``S_k = gate ⊙ (prev_sum_y − Y_self_k)`` via ``subs_window_stale`` (not ``subs`` —
  the staleness source is a cross-layer cache, not a token roll). No predictor.

The module is SHARED across the K local tracks (one copy per rank) and replicated
across ranks: ``broadcast_from_rank0`` gives every rank identical params at init,
and ``all_reduce_grads`` SUM-reduces the per-rank (track-split) gradients so a
deterministic optimizer step keeps the copies identical. The fresh predict loss is
replicated (computed identically on every rank), so the caller scales it by
``1 / world_size`` before backward — after the SUM reduce it then contributes at
the same ``1×`` scale as the track-split end-to-end gradients.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Decoder-block seam: split the HF Qwen3_5DecoderLayer at the post-attention
# point so a caller can inject Σ_others(Y) into the MLP input. Canonical home so
# both pt_model (training/inference) and eval/intervention (the gate) share it.
# --------------------------------------------------------------------------- #
def seam_token_mixer(layer, x, position_embeddings, attention_mask, position_ids):
    """First half of ``Qwen3_5DecoderLayer``: ``input_layernorm`` → token mixer →
    residual-add. Returns ``(h_attn = x + Y, Y, h_ln)`` where ``Y`` is the per-track
    token-mixer output (self-attn or gated-delta) before the residual add and
    ``h_ln`` is the layernorm'd input the mixer read (the fresh predictor's input)."""
    residual = x
    h_ln = layer.input_layernorm(x)
    if layer.layer_type == "linear_attention":
        y = layer.linear_attn(
            hidden_states=h_ln, cache_params=None, attention_mask=attention_mask
        )
    else:
        y, _ = layer.self_attn(
            hidden_states=h_ln,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
    return residual + y, y, h_ln


def seam_mlp(layer, h_attn, s):
    """Second half: ``post_attention_layernorm(h_attn + s)`` → ``mlp`` → residual-add
    onto ``h_attn``. The substitute ``s`` augments the MLP INPUT only; the carried
    residual is ``h_attn + mlp_partial`` (``s`` excluded). ``s=None`` ⇒ the stock layer."""
    mlp_in = h_attn if s is None else h_attn + s
    return h_attn + layer.mlp(layer.post_attention_layernorm(mlp_in))


def _roll1(t: torch.Tensor) -> torch.Tensor:
    """1-token causal shift along seq (dim=1); position 0 → 0 (no history)."""
    r = torch.roll(t, shifts=1, dims=1)
    r[:, 0] = 0.0
    return r


def _rollw(t: torch.Tensor, w: int) -> torch.Tensor:
    """``w``-token causal shift along seq (dim=1); first ``w`` positions → 0."""
    r = torch.roll(t, shifts=w, dims=1)
    r[:, :w] = 0.0
    return r


class CrossHeadEstimator(nn.Module):
    """Per-layer estimator of the missing cross-head attention output ``Σ_others Y``.

    See the module docstring. ``backend`` is ``"fresh"`` or ``"stale"``. The
    per-layer gate is zero-init ⇒ an exact no-op at construction.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        *,
        backend: str = "fresh",
        rank: int = 512,
        depth: int = 1,
        window: int = 1,
    ):
        super().__init__()
        if backend not in (
            "fresh", "fresh_yself", "temporal", "oracle_lowrank",
            "stale", "stale_corr", "oracle", "window_stale"
        ):
            raise ValueError(
                "backend must be 'fresh', 'fresh_yself', 'temporal', 'oracle_lowrank', "
                f"'stale', 'stale_corr', 'oracle', or 'window_stale', got {backend!r}"
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.backend = backend
        self.rank = rank
        self.depth = depth
        self.window = window  # temporal backend: # of cached past-token ΣY frames read
        # Calibration knob (oracle backend only): degrade the delivered ΣY to a
        # target relMSE so a co-trained run measures how downstream depends on the
        # predictor's accuracy (1 − oracle_noise = captured fraction). 0 = exact
        # oracle. ``_noise_base`` is set per (step, microbatch) by the train loop so
        # the additive noise is identical on every rank (else the tracks diverge).
        # Runtime-only (not persisted; rebuilt as 0 on load).
        self.oracle_noise = 0.0
        self._noise_base = 0
        # Gate-warmup multiplier (set per-step by the train loop). Holds the effective
        # gate at ~0 while the predictor learns to predict ΣY accurately, then ramps to
        # 1 — decouples "learn to predict" from "use the prediction", avoiding the
        # gate↔predictor feedback divergence. 1.0 = no warmup (default).
        self._gate_scale = 1.0
        # Per-(layer, channel) gate, zero-init ⇒ S_k = 0 at init ⇒ no-op warm start.
        self.gate = nn.Parameter(torch.zeros(num_layers, hidden_size))
        if backend in (
            "fresh", "fresh_yself", "temporal", "oracle_lowrank", "stale_corr"
        ):
            # Shared per-absolute-layer predictor g → ΣY estimate. ``depth`` rank-space
            # hidden blocks (down → [gelu → mid]*(depth-1) → gelu → up); depth=1 is the
            # plain low-rank map. ``fresh`` reads only the shared h_ln; ``fresh_yself``
            # ALSO reads the track's own attention output Y_self (concatenated, in_dim
            # 2·hidden); ``temporal`` reads the REAL cached recent history of ΣY itself
            # (``window`` causal frames concatenated, in_dim window·hidden) — predict the
            # current cross-track content from its own past, not from local activations.
            # ``stale_corr`` reads [h_ln, roll(ΣY,1)] (in_dim 2·hidden) and predicts the
            # 1-token DRIFT (added to the stale skip in ``subs``).
            if backend in ("fresh_yself", "stale_corr"):
                in_dim = hidden_size * 2
            elif backend == "temporal":
                in_dim = hidden_size * window
            else:
                in_dim = hidden_size
            self.down = nn.ModuleList(
                [nn.Linear(in_dim, rank, bias=False) for _ in range(num_layers)]
            )
            self.mid = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(rank, rank, bias=False) for _ in range(depth - 1)]
                    )
                    for _ in range(num_layers)
                ]
            )
            self.up = nn.ModuleList(
                [nn.Linear(rank, hidden_size, bias=False) for _ in range(num_layers)]
            )

    # ----- forward pieces -----
    def predict(
        self, layer_idx: int, h_ln: torch.Tensor, feat2: "torch.Tensor | None" = None
    ) -> torch.Tensor:
        """Estimate ΣY (or the drift, for ``stale_corr``) from the layernorm'd block
        input. ``fresh_yself`` additionally conditions on the track's own attention
        output ``Y_self``; ``stale_corr`` on the stale frame ``roll(ΣY,1)`` — both
        passed as ``feat2`` and concatenated.

        The inputs are DETACHED: the predictor reads the model's activations as a probe
        and the predict loss trains ONLY the predictor — it must NOT reshape the model
        to be "more predictable" (that gradient, harmless for ``fresh`` which only
        touches the shared h_ln, DESTABILIZES ``fresh_yself`` by corrupting every
        track's attention output). The end-to-end gradient still trains the model
        through the live ``Y_self``/residual in ``subs``; only this predictor-input
        path is cut."""
        h_ln = h_ln.detach()
        x = h_ln if feat2 is None else torch.cat([h_ln, feat2.detach()], dim=-1)
        h = self.down[layer_idx](x)
        for lin in self.mid[layer_idx]:
            h = lin(F.gelu(h))
        return self.up[layer_idx](F.gelu(h))

    def subs(
        self,
        layer_idx: int,
        y_list: list[torch.Tensor],
        sum_y: torch.Tensor,
        h_ln: torch.Tensor,
    ) -> "tuple[list[torch.Tensor], torch.Tensor | None]":
        """Per-track substitutes ``S_k`` (added to the MLP input) and, for ``fresh``,
        the predicted ``ΣY`` (``ghat``) for the caller's predict loss; ``None`` for
        ``stale``. ``sum_y`` is the all-reduced ΣY across all tracks; ``h_ln`` the
        shared layernorm'd input. At the zero-init gate every ``S_k`` is zero."""
        g = self.gate[layer_idx] * self._gate_scale  # _gate_scale=1 unless gate-warmup is active
        if self.backend == "fresh":
            ghat = self.predict(layer_idx, h_ln)
            subs = [g * (ghat - y) for y in y_list]
            return subs, ghat
        if self.backend == "fresh_yself":
            # Per-track prediction of ΣY conditioned on each track's own Y_self.
            # ghat returned STACKED (K, …) so the caller's predict loss / chp averages
            # relMSE(ghat_k, ΣY) over tracks; S_k = g·(ghat_k − y_k) ⇒ mlp_in_k ≈ X + ΣY.
            ghats = [self.predict(layer_idx, h_ln, y) for y in y_list]
            subs = [g * (gh - y) for gh, y in zip(ghats, y_list)]
            return subs, torch.stack(ghats, dim=0)
        if self.backend == "temporal":
            # Predict the current ΣY from its OWN real cached recent history (``window``
            # causal frames of the all-reduced ΣY — comm-free-via-cache, not from local
            # activations). Shared across tracks ⇒ coherent (one ghat), like ``fresh``.
            hist = torch.cat(
                [_rollw(sum_y.detach(), w) for w in range(1, self.window + 1)], dim=-1
            )
            ghat = self.predict(layer_idx, hist)  # predict() re-detaches (no-op here)
            subs = [g * (ghat - y) for y in y_list]
            return subs, ghat
        if self.backend == "oracle_lowrank":
            # CHEAP DELIVERY: reconstruct the REAL ΣY through a rank-``rank`` linear
            # bottleneck. ``down`` (bias-free, linear) applied per track + all-reduced =
            # the rank-``rank`` second all-reduce (rank/hidden the bandwidth of a full
            # ΣY all-reduce); ``up`` reconstructs locally. So ``rank`` IS the extra-comm
            # bandwidth; at rank=hidden this is the exact oracle. Co-trained, shared ghat.
            ghat = self.predict(layer_idx, sum_y)  # predict() detaches sum_y (real data)
            subs = [g * (ghat - y) for y in y_list]
            return subs, ghat
        if self.backend == "oracle":
            # CEILING (comm-FULL, NOT deployable): the EXACT current-token missing
            # cross-head content Σ_others Y_k. The learned zero-init gate opens during
            # training, so this measures whether the co-trained model can USE perfect
            # content; at gate→1 the MLP reads X + ΣY = exact tensor parallelism. The
            # per-layer ΣY all-reduce it needs at inference is the comm the premise
            # forbids — a co-trained ceiling measurement, not a solution.
            subs = [g * (sum_y - y) for y in y_list]
            return subs, None
        if self.backend == "stale_corr":
            # HYBRID: the stale real ΣY (skip, no extra sync) + a learned comm-free
            # correction for the 1-token drift. ghat = roll(ΣY,1) + g([h_ln, roll(ΣY,1)]);
            # predictor reads detached features and outputs only the delta, so it
            # degrades gracefully to plain ``stale`` if the drift is unpredictable.
            # Shared across tracks ⇒ one ghat (coherent), like ``fresh``. The predict
            # loss target is the current ΣY (see distill); the skip means it trains the
            # delta. ghat is detached from the model graph (stale skip + detached
            # predictor inputs) — end-to-end grad flows only through the live y_k and
            # the gate, exactly as ``stale``/``oracle``.
            stale = _roll1(sum_y.detach())
            ghat = stale + self.predict(layer_idx, h_ln, stale)
            subs = [g * (ghat - y) for y in y_list]
            return subs, ghat
        # stale: 1-token-stale missing cross-head content per track.
        subs = [g * _roll1(sum_y - y) for y in y_list]
        return subs, None

    def subs_window_stale(
        self,
        layer_idx: int,
        y_list: list[torch.Tensor],
        prev_sum_y: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Depth-stale cross-head fill for the post-attn D>1 stack (``window_stale``).

        At D>1 the non-boundary layers run fully PARTIAL (no sync ⇒ no ``Σ_others Y``),
        which is the per-window compounding source. This injects the REAL ``ΣY`` cached
        from the most recent boundary post-attention all-reduce (``prev_sum_y``, one
        layer back ⇒ 1-layer-of-depth stale, frequency-free — reuses a sync that already
        happened, no new sync point). ``S_k = gate ⊙ (prev_sum_y − Y_self_k)`` ⇒
        ``mlp_in_k = X_k + prev_sum_y`` at full gate. ``prev_sum_y`` is detached (carried
        real data, not predicted; end-to-end grad flows only through the live ``y_k`` and
        the gate, like ``stale``/``oracle``). Zero-init gate ⇒ exact no-op ⇒ the floor is
        plain D>1 phased, so it can only help. Gate-only: no predictor, no predict loss."""
        g = self.gate[layer_idx] * self._gate_scale
        ps = prev_sum_y.detach()
        return [g * (ps - y) for y in y_list]

    # ----- distributed replication (shared module across ranks) -----
    @torch.no_grad()
    def broadcast_from_rank0(self, group=None) -> None:
        """Make every rank's copy identical to rank 0's (call once after init)."""
        if not dist.is_initialized():
            return
        for p in self.parameters():
            dist.broadcast(p, src=0, group=group)

    @torch.no_grad()
    def all_reduce_grads(self, group=None) -> None:
        """SUM the per-rank (track-split) gradients so a deterministic step keeps
        the replicated copies bit-identical. Call between ``backward`` and
        ``optimizer.step()``. The fresh predict loss must be pre-scaled by
        ``1/world_size`` by the caller so it lands at the same ``1×`` scale."""
        if not dist.is_initialized():
            return
        for p in self.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)

    def config_dict(self) -> dict:
        """Sidecar metadata to rebuild the module shape at load time."""
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "backend": self.backend,
            "rank": self.rank,
            "depth": self.depth,
            "window": self.window,
        }
