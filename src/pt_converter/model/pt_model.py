"""PTWrappedModel: the user-facing per-rank model.

One PTWrappedModel instance per *rank*. Holds the K = tracks_per_rank tracks
this rank owns (as `nn.ModuleList` of per-track text models) plus (on the
rank that owns track 0) the shared `lm_head`. Exposes a forward that runs
all K tracks lockstep through the layers, driving cross-track sync at the
configured sync points, and returns `(logits, sync_hiddens)` for the
distillation loop.

The model-specific decoder layer assembly is provided by a `ModelAdapter`
(see `pt_converter.adapters`). PTWrappedModel itself holds no model-family
knowledge: it looks up the adapter by `text_config.model_type`, calls its
per-track config builder, and instantiates its `track_text_model_cls`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from torch.utils.checkpoint import checkpoint

from pt_converter.model.sync import SyncBoundary


def _window_ranges(num_layers: int, sync_indices: Sequence[int]) -> list[tuple[int, int]]:
    """Sync layer indices → (start, end_inclusive) window ranges covering all layers.

    Mirrors ``train.distill._block_ranges`` (kept local — distill imports this
    module). Every sync index ends a window; a trailing remainder window is
    appended if the last layer is not itself a sync boundary.
    """
    ranges: list[tuple[int, int]] = []
    prev_end = -1
    for idx in sync_indices:
        ranges.append((prev_end + 1, idx))
        prev_end = idx
    if prev_end != num_layers - 1:
        ranges.append((prev_end + 1, num_layers - 1))
    return ranges


@dataclass
class PTTrackTextModelConfig:
    """The engine→adapter contract for instantiating a per-track text model.

    Every adapter's `track_text_model_cls` accepts the constructor signature
    `(per_track_text_config, pt_cfg: PTTrackTextModelConfig, sync_module: SyncBoundary)`.
    Fields here are model-agnostic; per-track-specific dim changes live in
    `per_track_text_config` produced by the adapter.
    """

    n_tracks: int
    sync_after_layers: tuple[int, ...]
    track_id: int = 0
    # When False, track 0 does NOT build its full [V,H] embed_tokens — the
    # vocab-parallel path hosts a sharded embedding on the wrapper instead.
    host_embed_tokens: bool = True


class PTWrappedModel(nn.Module):
    LM_HEAD_OWNER_TRACK = 0

    def __init__(
        self,
        text_config,
        *,
        n_tracks: int,
        local_track_ids: Sequence[int],
        sync_after_layers: list[int],
        track_group: "torch.distributed.ProcessGroup | None" = None,
        activation_checkpoint: bool = False,
        checkpoint_granularity: str = "window",
        compile_layers: bool = False,
        compile_mode: str = "default",
        vocab_parallel: bool = False,
        vp_world_size: int = 1,
        vp_rank: int = 0,
    ):
        super().__init__()
        # Late import: pt_converter.adapters imports its registered adapters,
        # which in turn import model/tracks/<model>.py — those import this
        # module for `PTTrackTextModelConfig`. Importing the registry lazily
        # here breaks the cycle without losing the import-time registration.
        from pt_converter.adapters import get_adapter_for_config

        adapter = get_adapter_for_config(text_config)
        per_track_cfg = adapter.build_per_track_text_config(text_config, n_tracks)
        self.text_config = text_config
        self.per_track_text_config = per_track_cfg
        self.n_tracks = n_tracks
        self.local_track_ids = tuple(local_track_ids)
        self.sync_after_layers = tuple(sync_after_layers)
        self._adapter = adapter  # held for `load_track_state_dicts` remap
        # Per-layer activation checkpointing of the full student forward. Under
        # vocab-parallel (the default) EVERY rank backwards through that forward
        # — `_kl_ce_vocab_parallel` ends in one `hidden.backward(grad_h_accum)`
        # on every rank — so checkpointing trades the held forward graph (the
        # ~25 GB `student_fwd`/`klce` peak) for one recompute pass on every rank,
        # lowering the peak uniformly. (Legacy --no-vocab-parallel: only the
        # lm_head-owner rank backwards; peers run the forward for collective
        # ordering and pay no recompute.) The KL/CE backward is a single
        # `hidden.backward` either way (not per-chunk). This is the lever for
        # fitting larger batch sizes at seq_len=4096 on 40 GB GPUs.
        #
        # Granularity: "window" (default) checkpoints each whole sync window
        # (the D layers between boundaries) per track, so only the SHARED synced
        # window input is saved — one (B,T,H) tensor per window instead of the
        # ~K+1 per window that per-"layer" wrapping pins (window input + each
        # track's mid-window hiddens). Math is bit-identical and each layer is
        # recomputed exactly once either way; the saved-tensor set shrinks (~5 GB
        # at n16/d2 B=5 seq=4096) at the cost of a larger transient while a
        # window's backward re-runs its D·K-layer graph. The SyncBoundary stays
        # OUTSIDE the checkpoint, so no collective ever runs inside a recompute
        # (backward remains collective-free on every rank). At D=1 the two
        # granularities coincide. "layer" keeps the legacy per-layer wrap.
        if checkpoint_granularity not in ("window", "layer"):
            raise ValueError(
                f"checkpoint_granularity must be 'window' or 'layer', got {checkpoint_granularity!r}"
            )
        self._use_checkpoint = activation_checkpoint
        self._ckpt_granularity = checkpoint_granularity

        # ----- Vocab-parallel (tensor-parallel over V) setup -----
        # When enabled, embed_tokens + lm_head + the KL/CE softmax are sharded
        # over the vocab dimension across all `vp_world_size` ranks (rank `r`
        # owns rows [v_lo, v_hi)). This decouples those full-vocab-sized tensors
        # from track-0 ownership, balancing memory and parallelizing the KL/CE
        # phase. See `pt_converter.model.vocab_parallel`.
        self.vocab_parallel = vocab_parallel
        self.vp_world_size = vp_world_size
        self.vp_rank = vp_rank
        self.vp_group = track_group  # spans one rank per GPU (== WORLD here)
        if vocab_parallel:
            from pt_converter.model.vocab_parallel import vocab_range
            self.v_lo, self.v_hi = vocab_range(text_config.vocab_size, vp_world_size, vp_rank)
        else:
            self.v_lo, self.v_hi = 0, text_config.vocab_size

        self.sync_module = SyncBoundary(track_group=track_group, n_tracks=n_tracks)
        # Sync placement within a block (lever B). "boundary" (default) = the single
        # per-layer all-reduce lands AFTER the MLP (bit-identical to legacy). "post-attn"
        # = it lands AFTER attention, so the MLP reads the exact Σ_others Y (real, no
        # predictor) while the NEXT layer's attention reads the partial residual; the
        # sync COUNT is unchanged. See ``_run_post_attn_stack``. D=1 only.
        self.sync_phase = "boundary"
        # Optional cross-head attention-output estimator (D=1 post-attention seam).
        # None ⇒ the stock forward (bit-identical). Set by the train/eval scripts.
        self.cross_head = None
        # Eval-only: when True, the seam forward accumulates per-layer predict relMSE
        # (chp = relMSE(ghat, ΣY)) into ``_chp_sums`` (caller inits + counts batches).
        self._collect_chp = False
        self._chp_sums = None
        # Eval-only seam-fidelity diagnostic: when ``_collect_seam`` is True and
        # ``_seam_teacher`` holds the dense teacher's per-layer ``(X_teacher,
        # Y_teacher)``, the seam forward accumulates the per-layer decomposition of
        # how far the actual injected MLP input lands from the teacher's true input
        # ``X_teacher + Y_teacher`` into ``_seam_sums`` (a {metric: tensor[num_layers]}
        # dict the caller inits + counts batches). Works for BOTH backends.
        self._collect_seam = False
        self._seam_teacher = None
        self._seam_sums = None
        self.text_models = nn.ModuleList(
            [
                adapter.track_text_model_cls(
                    per_track_cfg,
                    PTTrackTextModelConfig(
                        n_tracks=n_tracks,
                        sync_after_layers=tuple(sync_after_layers),
                        track_id=tid,
                        host_embed_tokens=not vocab_parallel,
                    ),
                    sync_module=self.sync_module,
                )
                for tid in self.local_track_ids
            ]
        )
        # Per-track decoder layers are tiny at high n_tracks (e.g. 1 attention head
        # and a 768-wide MLP per track at n_tracks=16) and run in a sequential
        # Python loop over the K local tracks, so each rank is launch-bound on many
        # small kernels. Compile each layer's forward in place so inductor fuses the
        # tiny kernels and cuts launch overhead. We use `layer.compile(...)` (the
        # in-place nn.Module method) — NOT `torch.compile(layer)`, which wraps the
        # module in an OptimizedModule and renames state_dict keys to
        # `...layers.{i}._orig_mod.*`, breaking load_track_state_dicts/save. The
        # in-place form leaves module identity and state_dict() untouched, so both
        # forward call sites and all checkpoint I/O keep working. The SyncBoundary
        # all-reduce lives in the forward loop (outside the layer), so no collective
        # is captured into a graph. dynamic=False: shapes are static (B, T fixed).
        if compile_layers:
            for tm in self.text_models:
                for layer in tm.layers:
                    layer.compile(mode=compile_mode, dynamic=False)

        # Sync windows (start, end_inclusive) — the forward's iteration unit and
        # the "window" checkpointing granule. Every sync index ends a window.
        self._window_ranges = _window_ranges(
            len(self.text_models[0].layers), self.sync_after_layers
        )

        if vocab_parallel:
            # Vocab-parallel: every rank holds the embed + lm_head shard for its
            # vocab range [v_lo, v_hi). The embedding is summed across ranks
            # (one all-reduce in `embed`); lm_head produces this rank's logit
            # slice, consumed by the vocab-parallel KL/CE in `train/distill.py`.
            from pt_converter.model.vocab_parallel import VocabParallelEmbedding
            self.vp_embed = VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                self.v_lo,
                self.v_hi,
                padding_idx=getattr(text_config, "pad_token_id", None),
            )
            self.lm_head = nn.Linear(text_config.hidden_size, self.v_hi - self.v_lo, bias=False)
        else:
            # lm_head lives on the owner track only. The final SyncBoundary
            # broadcasts the post-block hidden state to all tracks, so whichever
            # rank hosts the owner already has the correct synced state to
            # project to logits. Peer ranks return logits=None from forward.
            self.vp_embed = None
            if self.LM_HEAD_OWNER_TRACK in self.local_track_ids:
                self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
            else:
                self.lm_head = None

    def set_sync_schedule(self, sync_after_layers: Sequence[int]) -> None:
        """Re-point the sync boundaries after construction.

        Used by the train script, which builds the student with a placeholder
        schedule, runs the sensitivity probe, then installs the dynamically
        chosen boundaries. All sync decisions live in this wrapper's `forward`
        (the per-track `text_models` never sync on their own), so only the
        wrapper's `sync_after_layers` and the derived `_window_ranges` need to
        change — the weights and module structure are schedule-independent.
        """
        indices = tuple(sorted(set(int(i) for i in sync_after_layers)))
        num_layers = len(self.text_models[0].layers)
        if indices and (indices[0] < 0 or indices[-1] >= num_layers):
            raise ValueError(
                f"sync indices {indices} out of range for {num_layers} layers"
            )
        self.sync_after_layers = indices
        self._window_ranges = _window_ranges(num_layers, indices)
        # Keep the per-track configs' copy consistent for any introspection.
        for tm in self.text_models:
            tm.pt_cfg.sync_after_layers = indices

    def set_sync_phase(self, phase: str) -> None:
        """Select where the single per-block sync lands (lever B). ``"boundary"``
        (default) is bit-identical to the legacy forward; ``"post-attn"`` shifts it
        to the post-attention point (real Σ_others Y at the MLP input, same sync
        count). Incompatible with a cross-head seam (B delivers the real content)."""
        if phase not in ("boundary", "post-attn"):
            raise ValueError(f"sync_phase must be 'boundary' or 'post-attn', got {phase!r}")
        if phase == "post-attn" and self.cross_head is not None \
                and getattr(self.cross_head, "backend", None) != "window_stale":
            raise ValueError(
                "sync_phase='post-attn' (lever B) is incompatible with a cross_head seam "
                "(it delivers the real Σ_others Y); only the 'window_stale' backend — which "
                "fills the partial non-boundary layers with the cached boundary ΣY — is allowed"
            )
        self.sync_phase = phase

    @staticmethod
    def _run_track_window(
        tm,
        start: int,
        end_inclusive: int,
        h: torch.Tensor,
        position_embeddings,
        text_position_ids,
        causal_mask,
        linear_attn_mask,
    ) -> torch.Tensor:
        """Run ONE track's layers [start..end_inclusive] from the synced window input.

        The "window" checkpointing granule: collective-free (the SyncBoundary is
        applied by the caller, outside any checkpoint), so it is safe to recompute
        during backward on every rank independently.
        """
        for layer_idx in range(start, end_inclusive + 1):
            mask = (
                linear_attn_mask
                if tm.config.layer_types[layer_idx] == "linear_attention"
                else causal_mask
            )
            h = tm.layers[layer_idx](
                h,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return h

    def compile_seam_submodules(self, mode: str = "default") -> None:
        """Compile the per-track decoder SUBMODULES in place — for the cross-head
        seam path, which calls submodules directly (``input_layernorm`` /
        ``post_attention_layernorm`` / ``mlp``) and so bypasses the whole-layer
        ``layer.compile()`` done at init. Recovers inductor fusion on the heavy MLP
        matmuls. One-time warmup; call after the estimator is attached.

        The token mixer (``self_attn`` / ``linear_attn``) is DELIBERATELY left eager:
        inductor compiles attention into a ``cutlassF`` template that has "no kernel to
        launch" at the downstream-eval padding-mask shapes (crashes run_downstream;
        the eager ``enable_mem_efficient_sdp(False)`` flag does NOT govern inductor's
        codegen). Eager attention dispatches to flash/math, which have kernels for
        every train + eval config (proven: the un-compiled seam completes the
        4-task@1000 downstream eval). MLP is the dominant FLOPs, so most of the
        speedup is retained."""
        for tm in self.text_models:
            for layer in tm.layers:
                for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
                    sub = getattr(layer, name, None)
                    if sub is not None:
                        sub.compile(mode=mode, dynamic=False)

    def _sum_all_tracks(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """ΣY over ALL tracks: local sum across the K hosted tracks + a detached
        all-reduce across ranks. Used as the cross-head predict target / stale
        cache, i.e. as DATA — the collective is detached (no grad flows through it,
        consistent with SyncBoundary being autograd-invisible)."""
        import torch.distributed as dist
        total = tensors[0]
        for t in tensors[1:]:
            total = total + t
        total = total.detach()
        if self.sync_module.track_group is not None and dist.is_initialized():
            total = total.contiguous()
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.sync_module.track_group)
        return total

    def run_seam_layer(
        self,
        layer_idx: int,
        per_track_h: list[torch.Tensor],
        position_embeddings,
        text_position_ids,
        causal_mask,
        linear_attn_mask,
        collect_predict: bool = False,
    ):
        """Run ONE decoder layer across the K local tracks with the cross-head seam.

        Splits each track's block at the post-attention point and injects the
        estimator's reconstruction of the missing ``Σ_others Y`` into the MLP input
        (the substitute is excluded from the carried residual — see
        ``cross_head_estimator``). Returns the K post-block tensors; with
        ``collect_predict`` also returns ``(sum_y, ghat)`` for the fresh predict loss.

        Assumes the D=1 setting where every track enters the layer from the SAME
        synced residual, so ``h_ln`` (and the fresh predictor's estimate) is shared.
        """
        from pt_converter.model.cross_head_estimator import seam_token_mixer, seam_mlp

        h_attn: list[torch.Tensor] = []
        y_list: list[torch.Tensor] = []
        h_ln_shared = None
        for k, tm in enumerate(self.text_models):
            layer = tm.layers[layer_idx]
            mask = (
                linear_attn_mask
                if tm.config.layer_types[layer_idx] == "linear_attention"
                else causal_mask
            )
            ha, y, h_ln = seam_token_mixer(
                layer, per_track_h[k], position_embeddings, mask, text_position_ids
            )
            h_attn.append(ha)
            y_list.append(y)
            h_ln_shared = h_ln
        sum_y = self._sum_all_tracks(y_list)
        # Calibration (oracle backend): degrade the delivered ΣY to a target relMSE
        # so co-training measures the accuracy→quality curve. The additive noise is
        # seeded by (cross_head._noise_base, layer_idx) — identical on every rank, so
        # the corrupted ΣY stays consistent across tracks.
        if getattr(self.cross_head, "oracle_noise", 0.0) > 0.0:
            r = float(self.cross_head.oracle_noise)
            seed = int(self.cross_head._noise_base) * 64 + layer_idx
            gen = torch.Generator(device=sum_y.device).manual_seed(seed)
            n = torch.randn(sum_y.shape, generator=gen, device=sum_y.device, dtype=torch.float32)
            sy = sum_y.float()
            scale = (r * sy.pow(2).sum() / n.pow(2).sum().clamp_min(1e-12)).sqrt()
            sum_y = (sy + scale * n).to(sum_y.dtype)
        subs, ghat = self.cross_head.subs(layer_idx, y_list, sum_y, h_ln_shared)
        # Eval-only seam-fidelity diagnostic (both backends): compare the actual
        # injected MLP input to the dense teacher's true input. Driven by the probe
        # state, independent of ``collect_predict``.
        if self._collect_seam and self._seam_sums is not None \
                and self._seam_teacher is not None and layer_idx in self._seam_teacher:
            self._accumulate_seam(layer_idx, per_track_h, h_attn, y_list, sum_y, subs)
        new_h = [
            seam_mlp(self.text_models[k].layers[layer_idx], h_attn[k], subs[k])
            for k in range(len(self.text_models))
        ]
        if collect_predict:
            return new_h, sum_y, ghat
        return new_h

    @torch.no_grad()
    def _accumulate_seam(
        self,
        layer_idx: int,
        per_track_h: list[torch.Tensor],
        h_attn: list[torch.Tensor],
        y_list: list[torch.Tensor],
        sum_y: torch.Tensor,
        subs: list[torch.Tensor],
    ) -> None:
        """Accumulate the per-layer seam-fidelity decomposition (eval diagnostic).

        Compares the actual injected per-track MLP input ``mlp_in_k = X + Y_self_k +
        S_k`` (``= h_attn[k] + subs[k]``, the input ``seam_mlp`` feeds the post-attn
        norm) to the dense teacher's true MLP input ``X_teacher + Y_teacher``
        (``self._seam_teacher[layer_idx]``). Five per-layer relMSE metrics into
        ``self._seam_sums``:

          residual_drift = relMSE(X_student, X_teacher)              [global]
          oracle_seam    = relMSE(X_student + ΣY, X_t + Y_t)         [global; perfect-substitute ceiling]
          seam_total     = mean_k relMSE(mlp_in_k, X_t + Y_t)        [per-track; the headline]
          substitute_err = mean_k relMSE(mlp_in_k, X_student + ΣY)   [per-track; staleness/predictor cost]
          stale_ratio    = mean_k ‖Δ_t Σ_others Y_k‖ / ‖Σ_others Y_k‖  [per-track; intrinsic stale error]

        At D=1 ``per_track_h`` entering the layer is the synced residual (identical
        across tracks), so ``X_student = per_track_h[0]``. The global terms use only
        rank-identical tensors (synced X, all-reduced ΣY, teacher) so they need no
        reduction; the per-track terms are summed over this rank's local tracks,
        SUM-reduced over the track group, and divided by ``n_tracks``.
        """
        import torch.distributed as dist

        x_t, y_t = self._seam_teacher[layer_idx]
        x_t = x_t.float()
        y_t = y_t.float()
        t_in = x_t + y_t
        t_in_sq = t_in.pow(2).sum().clamp_min(1e-12)
        x_student = per_track_h[0].float()
        sum_yf = sum_y.float()
        base = x_student + sum_yf                     # X + ΣY (exact-substitute input)
        base_sq = base.pow(2).sum().clamp_min(1e-12)

        self._seam_sums["residual_drift"][layer_idx] += (
            (x_student - x_t).pow(2).sum() / x_t.pow(2).sum().clamp_min(1e-12)
        )
        self._seam_sums["oracle_seam"][layer_idx] += (base - t_in).pow(2).sum() / t_in_sq

        seam_tot = base.new_zeros(())
        sub_err = base.new_zeros(())
        stale = base.new_zeros(())
        for k in range(len(y_list)):
            mlp_in = h_attn[k].float() + subs[k].float()
            seam_tot = seam_tot + (mlp_in - t_in).pow(2).sum() / t_in_sq
            sub_err = sub_err + (mlp_in - base).pow(2).sum() / base_sq
            other = sum_yf - y_list[k].float()        # Σ_others Y_k
            o, op = other[:, 1:], other[:, :-1]       # exclude position 0 (no history)
            den = o.pow(2).sum().clamp_min(1e-12).sqrt()
            num = (o - op).pow(2).sum().sqrt()
            stale = stale + num / den
        packed = torch.stack([seam_tot, sub_err, stale])
        if self.sync_module.track_group is not None and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self.sync_module.track_group)
        packed = packed / self.n_tracks
        self._seam_sums["seam_total"][layer_idx] += packed[0]
        self._seam_sums["substitute_err"][layer_idx] += packed[1]
        self._seam_sums["stale_ratio"][layer_idx] += packed[2]

    def _run_post_attn_stack(
        self,
        h: torch.Tensor,
        position_embeddings,
        text_position_ids,
        causal_mask,
        linear_attn_mask,
        sync_hiddens: "dict[int, torch.Tensor] | None" = None,
        boundary_grad_alpha: float = 1.0,
    ):
        """Phase-shifted per-WINDOW sync (lever B), eval forward — D=1 and D>1.

        The one fresh all-reduce per sync window is PHASE-SHIFTED to land POST-ATTENTION
        instead of at the post-MLP boundary, so the boundary layer's MLP reads the FULL
        synced residual (the exact Σ_others Y delivered for free) while everything after
        it reads the PARTIAL one. Concretely: sync after ATTENTION at every boundary in
        ``self.sync_after_layers`` except the final layer, sync after MLP at the final
        layer (so the output is fully synced for the norm), and run all NON-boundary
        layers fully partial (attn+MLP, no sync). The all-reduce sums every per-track
        delta accrued since the previous sync, so each delta is summed exactly once.
        Sync COUNT == the boundary schedule's (e.g. D=2 → 16). Reduces to the D=1 case
        (sync after every layer's attention) when every layer is a boundary."""
        from pt_converter.model.cross_head_estimator import seam_token_mixer, seam_mlp

        L = len(self.text_models[0].layers)
        last = L - 1
        sync_attn_set = set(self.sync_after_layers) - {last}
        # window_stale: fill the fully-partial non-boundary layers with the real ΣY
        # cached from the previous boundary (depth-stale, frequency-free). gate-only.
        ch = (
            self.cross_head
            if getattr(self, "cross_head", None) is not None
            and getattr(self.cross_head, "backend", None) == "window_stale"
            else None
        )
        # Activation checkpointing for the grad-carrying full forward (an on-policy
        # output objective — λ_kl/λ_ce/λ_logit_mse with sync_phase=post-attn —
        # backwards through this whole stack; without recompute the L-layer ×
        # K-track activation set OOMs at training shapes). Per-sublayer granularity:
        # each track's token mixer / MLP is a closure, collectives stay OUTSIDE.
        # The window_stale seam needs per-layer y_list, so it keeps the plain path
        # (mirrors the boundary forward's cross_head exclusion). Eval (no_grad)
        # always takes the plain path.
        use_ckpt = self._use_checkpoint and torch.is_grad_enabled() and ch is None

        def _mixer_ha(layer, x, mask):
            # position_embeddings / text_position_ids are captured (no-grad
            # scaffolding); only the residual input carries gradient.
            return seam_token_mixer(layer, x, position_embeddings, mask, text_position_ids)[0]

        def _mlp(layer, x, s=None):
            if use_ckpt and s is None:
                return checkpoint(seam_mlp, layer, x, None, use_reentrant=False)
            return seam_mlp(layer, x, s)

        prev_sum_y = None  # most recent boundary's ΣY (the depth-stale fill source)
        block_start = h
        per_track_h = [h for _ in self.text_models]
        for i in range(L):
            h_attn: list[torch.Tensor] = []
            y_list: list[torch.Tensor] = []
            for k, tm in enumerate(self.text_models):
                layer = tm.layers[i]
                mask = (
                    linear_attn_mask
                    if tm.config.layer_types[i] == "linear_attention"
                    else causal_mask
                )
                if use_ckpt:
                    ha = checkpoint(_mixer_ha, layer, per_track_h[k], mask, use_reentrant=False)
                    y = None  # y_list is only consumed on the (non-ckpt) seam path
                else:
                    ha, y, _h_ln = seam_token_mixer(
                        layer, per_track_h[k], position_embeddings, mask, text_position_ids
                    )
                h_attn.append(ha)
                y_list.append(y)
            if i in sync_attn_set:
                # Boundary: post-attention sync delivers the real ΣY to this layer's
                # MLP; the per-track MLP delta is carried un-summed (the next attention's
                # partial seam) until the next sync.
                R = self.sync_module(h_attn, block_start)
                if sync_hiddens is not None:
                    # Tap stored UNDAMPED (same contract as the boundary forward):
                    # only the cross-window continuation below is attenuated.
                    sync_hiddens[i] = R
                if boundary_grad_alpha != 1.0 and torch.is_grad_enabled():
                    # Per-window gradient damping of the free-running unroll
                    # (mirrors the boundary forward): forward value is EXACTLY R,
                    # but gradient flowing back across this sync window shrinks by
                    # alpha per boundary crossed — bounds the unrolled-Jacobian
                    # amplification that makes full-unroll output objectives
                    # diverge. Everything downstream (this boundary's MLP + all
                    # later windows + the final logits) is damped together.
                    R = R.detach() + boundary_grad_alpha * (R - R.detach())
                block_start = R
                if ch is not None:
                    prev_sum_y = self._sum_all_tracks(y_list)  # cache real ΣY for the next partial layer
                per_track_h = [_mlp(tm.layers[i], R) for tm in self.text_models]
            else:
                # Non-boundary (fully partial, no sync) OR the final layer (post-MLP sync
                # that produces the fully-synced output for the norm). window_stale fills
                # the partial MLP with the cached previous-boundary ΣY (no-op at zero gate).
                if ch is not None and i != last and prev_sum_y is not None:
                    subs = ch.subs_window_stale(i, y_list, prev_sum_y)
                else:
                    subs = [None] * len(self.text_models)
                new_h = [
                    _mlp(self.text_models[k].layers[i], h_attn[k], subs[k])
                    for k in range(len(self.text_models))
                ]
                if i == last:
                    h = self.sync_module(new_h, block_start)
                    if sync_hiddens is not None:
                        sync_hiddens[i] = h
                else:
                    per_track_h = new_h
        return h, sync_hiddens

    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Full input embedding (B, S, H), grad-connected, identical on every rank.

        Vocab-parallel: each rank looks up its vocab shard (zeros elsewhere) and
        the partials are summed via one all-reduce. Non-VP (legacy): the owner
        track embeds the full vocab, peers contribute zeros, and the same sum
        broadcasts it to every track. Both go through `sync_module`, so the
        cross-track collective ordering is identical.
        """
        if self.vocab_parallel:
            partial = self.vp_embed(input_ids)
            return self.sync_module([partial], torch.zeros_like(partial))
        embeds_per_track: list[torch.Tensor] = []
        for tm in self.text_models:
            if tm.embed_tokens is not None:
                embeds_per_track.append(tm.embed_tokens(input_ids))
            else:
                B, S = input_ids.shape
                embeds_per_track.append(
                    torch.zeros(
                        B,
                        S,
                        tm.config.hidden_size,
                        device=input_ids.device,
                        dtype=tm.norm.weight.dtype,
                    )
                )
        return self.sync_module(embeds_per_track, torch.zeros_like(embeds_per_track[0]))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_sync_hiddens: bool = False,
        return_hidden_pre_lm_head: bool = False,
        return_intra_window_hiddens: bool = False,
        boundary_grad_alpha: float = 1.0,
    ):
        # Local import: keeps the engine model-family-agnostic at import time.
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

        # 1. Embed (vocab-parallel slice, or legacy owner-only) + cross-track
        #    broadcast via the all-reduce in `embed`.
        h = self.embed(input_ids)

        # 2. Scaffolding (rotary, masks) computed once — every track's per-track
        # config is identical, so reuse the first track's modules.
        tm0 = self.text_models[0]
        position_ids_resolved, text_position_ids = tm0._resolve_position_ids(h, position_ids)
        causal_mask = create_causal_mask(
            config=tm0.config,
            inputs_embeds=h,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = (
            None
            if (attention_mask is not None and torch.all(attention_mask == 1))
            else attention_mask
        )
        position_embeddings = tm0.rotary_emb(h, position_ids_resolved)

        # Lever B: the single per-block sync is phase-shifted to post-attention.
        # Self-contained per-layer split path (the window/checkpoint machinery below
        # assumes the boundary sync), D=1 only.
        if self.sync_phase == "post-attn":
            sh = {} if (return_sync_hiddens or return_intra_window_hiddens) else None
            h, sh = self._run_post_attn_stack(
                h, position_embeddings, text_position_ids, causal_mask, linear_attn_mask, sh,
                boundary_grad_alpha=boundary_grad_alpha,
            )
            h = tm0.norm(h)
            if return_hidden_pre_lm_head:
                return h, sh
            logits = self.lm_head(h) if self.lm_head is not None else None
            return logits, sh

        # 3. Lockstep window iteration with per-block syncs. Iterating windows
        # (not layers) is what lets "window" checkpointing wrap each track's
        # whole window in ONE checkpoint — saving only the shared synced window
        # input instead of every mid-window per-track hidden. Sync positions are
        # identical to the legacy per-layer loop (every sync index ends a window).
        block_start = h
        sync_set = set(self.sync_after_layers)
        sync_hiddens: dict[int, torch.Tensor] = (
            {} if (return_sync_hiddens or return_intra_window_hiddens) else None
        )
        per_track_h = [block_start for _ in self.text_models]
        use_ckpt = self._use_checkpoint and torch.is_grad_enabled()
        # The cross-head seam needs per-layer access (split attn/MLP + a per-layer
        # ΣY all-reduce), so it forces the per-layer path (window checkpointing
        # hides the layers inside one closure).
        window_ckpt = use_ckpt and self._ckpt_granularity == "window" and self.cross_head is None
        # Mid-window taps need per-layer access; the "window" checkpoint granule
        # hides the layers inside one closure. Eval (no_grad ⇒ use_ckpt False)
        # always takes the per-layer path, so this only trips a training caller.
        if return_intra_window_hiddens and window_ckpt:
            raise RuntimeError(
                "return_intra_window_hiddens requires the per-layer forward path "
                "(use checkpoint_granularity='layer' or disable activation checkpointing)"
            )
        for start, end in self._window_ranges:
            if window_ckpt:
                per_track_h = [
                    checkpoint(
                        self._run_track_window,
                        tm, start, end, per_track_h[k],
                        position_embeddings, text_position_ids,
                        causal_mask, linear_attn_mask,
                        use_reentrant=False,
                    )
                    for k, tm in enumerate(self.text_models)
                ]
            else:
                for layer_idx in range(start, end + 1):
                    if self.cross_head is not None:
                        # Cross-head seam: inject the missing Σ_others Y at each
                        # layer's MLP input (the substitute is excluded from the
                        # carried residual). Checkpointing is unused here — the seam
                        # forward is either no_grad (eval / metrics) or backwarded
                        # per block (the D=1 block loop drives training).
                        if self._collect_chp and self._chp_sums is not None:
                            per_track_h, _sum_y, _ghat = self.run_seam_layer(
                                layer_idx, per_track_h, position_embeddings,
                                text_position_ids, causal_mask, linear_attn_mask,
                                collect_predict=True,
                            )
                            if _ghat is not None:  # fresh / fresh_yself backends
                                _tgt = (
                                    _sum_y if _ghat.ndim == _sum_y.ndim
                                    else _sum_y.unsqueeze(0).expand_as(_ghat)
                                )
                                rel = _ghat.float().sub(_tgt.float()).pow(2).sum() / (
                                    _tgt.float().pow(2).sum().clamp_min(1e-12)
                                )
                                self._chp_sums[layer_idx] += rel
                        else:
                            per_track_h = self.run_seam_layer(
                                layer_idx, per_track_h, position_embeddings,
                                text_position_ids, causal_mask, linear_attn_mask,
                            )
                        if return_intra_window_hiddens and layer_idx not in sync_set:
                            sync_hiddens[layer_idx] = self.sync_module(per_track_h, block_start)
                        continue
                    new_h: list[torch.Tensor] = []
                    for k, tm in enumerate(self.text_models):
                        layer = tm.layers[layer_idx]
                        mask = (
                            linear_attn_mask
                            if tm.config.layer_types[layer_idx] == "linear_attention"
                            else causal_mask
                        )
                        if use_ckpt:
                            out = checkpoint(
                                layer,
                                per_track_h[k],
                                position_embeddings=position_embeddings,
                                attention_mask=mask,
                                position_ids=text_position_ids,
                                past_key_values=None,
                                use_cache=False,
                                use_reentrant=False,
                            )
                        else:
                            out = layer(
                                per_track_h[k],
                                position_embeddings=position_embeddings,
                                attention_mask=mask,
                                position_ids=text_position_ids,
                                past_key_values=None,
                                use_cache=False,
                            )
                        new_h.append(out)
                    per_track_h = new_h
                    if return_intra_window_hiddens and layer_idx not in sync_set:
                        # Loss-only synced reconstruction at a mid-window depth
                        # (same semantics as training's --intra-window-mse): the
                        # forward keeps feeding each track its PARTIAL hidden —
                        # this tap is observational and never carries forward.
                        sync_hiddens[layer_idx] = self.sync_module(per_track_h, block_start)
            if end in sync_set:
                h = self.sync_module(per_track_h, block_start)
                if sync_hiddens is not None:
                    # Tap stored UNDAMPED: each tap's gradient into its own
                    # window keeps full strength; only the cross-boundary
                    # continuation below is attenuated.
                    sync_hiddens[end] = h
                if boundary_grad_alpha != 1.0 and torch.is_grad_enabled():
                    # Boundary gradient damping for the free-running MSE term:
                    # forward value is EXACTLY h (h - h.detach() == 0), but the
                    # gradient flowing back across this boundary is scaled by
                    # alpha — a tap j's gradient into window w shrinks by
                    # alpha^(j-w), bounding the unrolled-Jacobian amplification
                    # that makes the full-unroll term diverge. alpha=0 hard-
                    # truncates: each tap trains only its own window on the
                    # true free-running input. The KL/CE backward shares this
                    # graph, so alpha<1 requires lambda_kl == lambda_ce == 0
                    # (enforced by the train script).
                    h = h.detach() + boundary_grad_alpha * (h - h.detach())
                block_start = h
                per_track_h = [h for _ in self.text_models]

        h = tm0.norm(h)
        # The caller may want the post-norm hidden state instead of logits so
        # they can chunk the lm_head application themselves (eliminates the
        # held (B, T, V) bf16 logits tensor — ~2 GB at seq=4096). All ranks
        # return `h` regardless of lm_head ownership; this keeps SyncBoundary
        # collective ordering matched (peers must still run the full forward).
        if return_hidden_pre_lm_head:
            return h, sync_hiddens
        # NOTE: in vocab-parallel mode `self.lm_head(h)` is only this rank's
        # vocab SLICE (B, T, v_hi-v_lo), not full logits — callers needing the
        # logit objective must request the hidden state and use the
        # vocab-parallel KL/CE in `train/distill.py`. The legacy path returns
        # full (B, T, V) logits on the owner and None on peers.
        logits = self.lm_head(h) if self.lm_head is not None else None
        return logits, sync_hiddens

    def load_track_state_dicts(
        self,
        track_states: dict[int, dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Load per-track shards into the K local text_models (and lm_head if owned).

        ``track_states`` keys must be exactly ``self.local_track_ids``. Each
        value is the per-track state_dict emitted by
        ``slicer.convert.slice_model_to_tracks`` (top-level keys like
        ``embed_tokens.weight``, ``norm.weight``, ``lm_head.weight``, and
        ``layers.{i}.<adapter-prefix>.*``). We rewrite into the namespaces
        ``text_models.{k}.embed_tokens.weight`` / ``.norm.weight`` /
        ``.layers.{i}.*`` and the rank-shared ``lm_head.weight``.
        """
        provided = set(track_states.keys())
        expected = set(self.local_track_ids)
        if provided != expected:
            raise ValueError(
                f"load_track_state_dicts expected track ids {sorted(expected)}, "
                f"got {sorted(provided)}"
            )

        remapped: dict[str, torch.Tensor] = {}
        for k, tid in enumerate(self.local_track_ids):
            track_state = track_states[tid]
            prefix = f"text_models.{k}."
            for key, val in track_state.items():
                if key in ("embed_tokens.weight", "lm_head.weight") and self.vocab_parallel:
                    # Vocab-parallel: embed + lm_head are sharded onto the
                    # wrapper's vp_embed / lm_head via load_vocab_parallel_weights,
                    # not into the per-track text models. Skip the full tensors
                    # here (they only appear in the track-0 shard).
                    continue
                if key == "embed_tokens.weight":
                    remapped[prefix + "embed_tokens.weight"] = val
                elif key == "norm.weight":
                    remapped[prefix + "norm.weight"] = val
                elif key == "lm_head.weight":
                    # Owner only; routed to the rank-shared head.
                    remapped["lm_head.weight"] = val
                elif key.startswith("layers."):
                    remapped[prefix + key] = val
                else:
                    remapped[prefix + key] = val

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if strict:
            # rotary buffer keys (e.g. text_models.{k}.rotary_emb.inv_freq) may
            # legitimately be missing — they're computed at init.
            missing_critical = [k for k in missing if "rotary_emb" not in k]
            if self.vocab_parallel:
                # vp_embed.weight / lm_head.weight are loaded separately via
                # load_vocab_parallel_weights, so they're expected-missing here.
                missing_critical = [
                    k for k in missing_critical
                    if not (k.startswith("vp_embed.") or k.startswith("lm_head."))
                ]
            if missing_critical or unexpected:
                raise RuntimeError(
                    f"load_track_state_dicts mismatch:\n  missing={missing_critical}\n  unexpected={unexpected}"
                )

    def load_vocab_parallel_weights(
        self,
        full_embed_weight: torch.Tensor,
        full_lm_head_weight: torch.Tensor,
    ) -> None:
        """Slice the full `[V, H]` embed + lm_head into this rank's vocab shard.

        Every rank calls this with the *full* tensors (read from the track-0
        shard, where the slicer stores them as `OwnerOnly`); each keeps rows
        `[v_lo, v_hi)`. Keeps the on-disk checkpoint format unchanged.
        """
        if not self.vocab_parallel:
            raise RuntimeError("load_vocab_parallel_weights called on a non-vocab-parallel model")
        with torch.no_grad():
            self.vp_embed.weight.copy_(full_embed_weight[self.v_lo:self.v_hi].to(self.vp_embed.weight.dtype))
            self.lm_head.weight.copy_(full_lm_head_weight[self.v_lo:self.v_hi].to(self.lm_head.weight.dtype))

    def gather_vocab_parallel_weights(self) -> "tuple[torch.Tensor, torch.Tensor] | None":
        """All-gather the embed + lm_head vocab shards → full `[V, H]` on rank 0.

        Returns `(embed_weight, lm_head_weight)` on rank 0 (both full vocab) and
        `None` on peers, so `save_checkpoint` can write the unchanged on-disk
        format (full tensors in the track-0 shard). Shards are zero-padded to a
        uniform width for the collective, then trimmed back to `V`.
        """
        if not self.vocab_parallel:
            raise RuntimeError("gather_vocab_parallel_weights called on a non-vocab-parallel model")
        import torch.distributed as dist
        V = self.text_config.vocab_size
        shard = math.ceil(V / self.vp_world_size)
        H = self.text_config.hidden_size

        def _gather(local: torch.Tensor) -> "torch.Tensor | None":
            padded = local.new_zeros((shard, H))
            padded[: local.shape[0]] = local
            if dist.is_initialized() and self.vp_group is not None:
                out = [torch.empty_like(padded) for _ in range(self.vp_world_size)]
                dist.all_gather(out, padded, group=self.vp_group)
            else:
                out = [padded]
            if self.vp_rank != 0:
                return None
            return torch.cat(out, dim=0)[:V].contiguous()

        embed_full = _gather(self.vp_embed.weight.detach())
        lm_head_full = _gather(self.lm_head.weight.detach())
        if self.vp_rank != 0:
            return None
        return embed_full, lm_head_full
