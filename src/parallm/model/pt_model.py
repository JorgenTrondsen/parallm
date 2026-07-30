"""PTWrappedModel: the user-facing per-rank model.

One PTWrappedModel instance per *rank*. Holds the K = tracks_per_rank tracks
this rank owns (as `nn.ModuleList` of per-track text models) plus (on the
rank that owns track 0) the shared `lm_head`. Exposes a forward that runs
all K tracks lockstep through the layers, driving cross-track sync at the
configured sync points, and returns `(logits, sync_hiddens)` for eval's
fidelity / downstream loss computation.

The model-specific decoder layer assembly is provided by a `ModelAdapter`
(see `parallm.adapters`). PTWrappedModel itself holds no model-family
knowledge: it looks up the adapter by `text_config.model_type`, calls its
per-track config builder, and instantiates its `track_text_model_cls`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from parallm.model.sync import SyncBoundary


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
        fuse_size: int = 1,
        merge_group: int = 1,
    ):
        super().__init__()
        # Late import: parallm.adapters imports its registered adapters,
        # which in turn import model/tracks/<model>.py — those import this
        # module for `PTTrackTextModelConfig`. Importing the registry lazily
        # here breaks the cycle without losing the import-time registration.
        from parallm.adapters import get_adapter_for_config

        adapter = get_adapter_for_config(text_config)
        # `merge_group` F: each of this model's tracks is F consecutive shards of an
        # (n_tracks * F)-track convert, glued into one F-wide track. `n_tracks` is
        # then the LOGICAL count the sync sees, so SyncBoundary, the replication
        # plan, the forward walk and the state-dict remap all work unchanged — only
        # the per-track config widens. See `parallm.model.merge`.
        if merge_group < 1:
            raise ValueError(f"merge_group must be >= 1, got {merge_group}")
        per_track_cfg = adapter.build_per_track_text_config(
            text_config, n_tracks * merge_group, fuse_size=merge_group
        )
        self.text_config = text_config
        self.per_track_text_config = per_track_cfg
        self.n_tracks = n_tracks
        self.merge_group = merge_group
        self.local_track_ids = tuple(local_track_ids)
        self.sync_after_layers = tuple(sync_after_layers)
        self._adapter = adapter  # held for `load_track_state_dicts` remap

        # Track fusion groups are pooled with a LOCAL sum, so a group that
        # straddled two ranks would silently drop the off-rank member's delta.
        if fuse_size > 1 and merge_group > 1:
            raise ValueError(
                "fuse_size and merge_group are two implementations of the same "
                "grouping (summed vs concatenated); pass one"
            )
        if fuse_size > 1:
            k = len(self.local_track_ids)
            if k % fuse_size or self.local_track_ids[0] % fuse_size:
                ok = [w for w in range(1, n_tracks + 1)
                      if n_tracks % w == 0 and (n_tracks // w) % fuse_size == 0]
                raise ValueError(
                    f"fuse_size {fuse_size} needs whole groups on each rank: this rank "
                    f"holds {k} tracks starting at {self.local_track_ids[0]}. Launch at a "
                    f"world size that keeps tracks/rank a multiple of {fuse_size} "
                    f"(n_tracks={n_tracks} → {ok}), or pass build_groups a "
                    f"tracks_per_rank_list whose blocks are {fuse_size}-aligned."
                )
        self.sync_module = SyncBoundary(
            track_group=track_group, n_tracks=n_tracks, fuse_size=fuse_size
        )
        self.text_models = nn.ModuleList(
            [
                adapter.track_text_model_cls(
                    per_track_cfg,
                    PTTrackTextModelConfig(
                        n_tracks=n_tracks,
                        sync_after_layers=tuple(sync_after_layers),
                        track_id=tid,
                    ),
                    sync_module=self.sync_module,
                )
                for tid in self.local_track_ids
            ]
        )

        if merge_group > 1:
            self._widen_diverged_norms(merge_group)

        # lm_head lives on the owner track only. The final SyncBoundary
        # broadcasts the post-block hidden state to all tracks, so whichever
        # rank hosts the owner already has the correct synced state to
        # project to logits. Peer ranks return logits=None from forward.
        if self.LM_HEAD_OWNER_TRACK in self.local_track_ids:
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        else:
            self.lm_head = None

        # Sync placement phase (see `set_sync_phase`). "post-mlp" = the legacy
        # after-layer boundary walk, bit-identical to the original forward.
        self.sync_phase = "post-mlp"
        # Optional HostResidentLayers: when set, the walk pages each decoder
        # layer in from pinned host DRAM and drops it again (the streamed
        # teacher). None = every layer stays resident, the normal path.
        self.layer_stream = None
        # Per-sublayer activation checkpointing for grad-carrying full forwards
        # (the trainer's on-policy output objective backwards through the whole
        # stack; without recompute the L-layer activation set OOMs at 27B
        # training shapes). Eval / no_grad paths are unaffected.
        self.use_checkpoint = False

    def _widen_diverged_norms(self, merge_group: int) -> None:
        """Give every ``Replicated(sync=False)`` weight a leading per-member dim.

        Those are the params the F merged members each keep their own copy of —
        `q_norm`/`k_norm`, which diverge under training. They are per-head RMSNorm
        scales applied to a ``(..., heads, head_dim)`` tensor, so a ``[F, head_dim]``
        weight broadcasts per head and needs no forward change (verified exact
        against a per-head loop). A ``sync=False`` param that is NOT applied per-head
        would be silently wrong here, which is why the merged path is validated
        per family by the equivalence rail rather than assumed.
        """
        from parallm.slicer.base import Replicated
        from parallm.slicer.convert import resolve_param_specs

        widened = 0
        for name, spec in resolve_param_specs(self._adapter, self.text_config).items():
            if not (isinstance(spec, Replicated) and not spec.sync):
                continue
            module_path, _, param_name = name.rpartition(".")
            for tm in self.text_models:
                try:
                    submod = tm.get_submodule(module_path)
                except AttributeError:
                    continue
                p = getattr(submod, param_name, None)
                if p is None:
                    continue
                setattr(
                    submod,
                    param_name,
                    nn.Parameter(p.detach().unsqueeze(0).repeat(merge_group, 1)),
                )
                widened += 1
        if widened == 0:
            raise RuntimeError(
                "merge_group > 1 but no Replicated(sync=False) params were found — "
                "the spec map and the built modules disagree"
            )

    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Full input embedding (B, S, H), grad-connected, identical on every rank.

        The owner track embeds the full vocab, peers contribute zeros, and the
        cross-track sum broadcasts it to every track — so the SyncBoundary
        collective ordering is identical on every rank.
        """
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

    def set_sync_phase(self, phase: str) -> None:
        """Sync placement. ``"post-mlp"`` (default): the legacy after-layer
        boundary walk. ``"post-attn"`` (lever B): the one per-window sync lands
        after the boundary layer's token mixer (its MLP reads the TRUE
        residual); the final layer keeps a post-MLP sync for the head; the d1b
        schedule = every layer a boundary. ``"exact"``: sync after attention
        AND after MLP of every layer — bit-identical to the dense forward by
        construction (2 syncs/layer; the frozen-slice teacher's schedule)."""
        if phase not in ("post-mlp", "post-attn", "exact"):
            raise ValueError(f"sync_phase must be 'post-mlp', 'post-attn' or 'exact', got {phase!r}")
        self.sync_phase = phase

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_sync_hiddens: bool = False,
        return_intra_window_hiddens: bool = False,
        return_hidden_pre_lm_head: bool = False,
        capture_post_attn: "set[int] | None" = None,
        capture_post_mlp: "set[int] | None" = None,
    ):
        # Local import: keeps the engine model-family-agnostic at import time.
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

        # 1. Embed (owner-only) + cross-track broadcast via the all-reduce in `embed`.
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

        # 3. Lockstep layer iteration: every track advances one layer, then we
        # sync at each configured boundary (recombining the per-track partial
        # residual updates). Optional mid-window taps reconstruct a synced hidden
        # at non-boundary depths for loss/metric purposes only (never fed forward).
        sync_set = set(self.sync_after_layers)
        sync_hiddens: dict[int, torch.Tensor] | None = (
            {} if (return_sync_hiddens or return_intra_window_hiddens) else None
        )
        if self.sync_phase in ("post-attn", "exact"):
            h = self._run_post_attn_stack(
                h,
                position_embeddings,
                text_position_ids,
                causal_mask,
                linear_attn_mask,
                sync_hiddens=sync_hiddens,
                exact=self.sync_phase == "exact",
                capture_post_attn=capture_post_attn,
                capture_post_mlp=capture_post_mlp,
            )
        else:
            if self.sync_module.fuse_size > 1:
                # Silently ignoring the flag here would report a fused number for
                # an unfused run; nothing trains on post-mlp, so just refuse.
                raise ValueError("fuse_size > 1 is implemented for post-attn/exact only")
            block_start = h
            per_track_h = [h for _ in self.text_models]
            for layer_idx in range(len(tm0.layers)):
                new_h: list[torch.Tensor] = []
                for k, tm in enumerate(self.text_models):
                    mask = (
                        linear_attn_mask
                        if tm.config.layer_types[layer_idx] == "linear_attention"
                        else causal_mask
                    )
                    new_h.append(
                        tm.layers[layer_idx](
                            per_track_h[k],
                            position_embeddings=position_embeddings,
                            attention_mask=mask,
                            position_ids=text_position_ids,
                            past_key_values=None,
                            use_cache=False,
                        )
                    )
                per_track_h = new_h
                if layer_idx in sync_set:
                    h = self.sync_module(per_track_h, block_start)
                    if sync_hiddens is not None:
                        sync_hiddens[layer_idx] = h
                    block_start = h
                    per_track_h = [h for _ in self.text_models]
                elif return_intra_window_hiddens:
                    sync_hiddens[layer_idx] = self.sync_module(per_track_h, block_start)

        h = tm0.norm(h)
        if return_hidden_pre_lm_head:
            # The trainer's chunked KL applies lm_head itself — hand back the
            # post-norm hidden so no (B, T, V) logits tensor materializes here.
            return h, sync_hiddens
        # In-owner rank returns full (B, T, V) logits; peer ranks (lm_head=None)
        # return None — the legacy PTLM contract. Peers still run the full forward
        # so the SyncBoundary collective ordering stays matched across ranks.
        logits = self.lm_head(h) if self.lm_head is not None else None
        return logits, sync_hiddens

    def _run_post_attn_stack(
        self,
        h: torch.Tensor,
        position_embeddings,
        text_position_ids,
        causal_mask,
        linear_attn_mask,
        sync_hiddens: "dict[int, torch.Tensor] | None" = None,
        exact: bool = False,
        capture_post_attn: "set[int] | None" = None,
        capture_post_mlp: "set[int] | None" = None,
    ) -> torch.Tensor:
        """Phase-shifted sync walk — lever B (``exact=False``) and the exact
        schedule (``exact=True``).

        Lever B: the one fresh all-reduce per sync window lands POST-ATTENTION
        at every boundary in ``self.sync_after_layers`` except the final layer
        (that boundary layer's MLP reads the FULL synced residual and its delta
        is then CARRIED un-summed), the final layer syncs post-MLP for the norm,
        and non-boundary layers run fully partial. The all-reduce sums every
        per-track delta accrued since the previous sync, so each delta is summed
        exactly once; the sync COUNT equals the boundary schedule's. Reduces to
        the d1b per-layer loop when every layer is a boundary.

        Exact: sync after attention AND after MLP of every layer — every
        sublayer reads the true residual, so the walk is the dense forward up
        to fp summation order (the frozen-slice teacher path; 2 syncs/layer).

        ``capture_post_attn`` / ``capture_post_mlp``: layer sets whose post-attn
        (resp. post-MLP) synced state is stored into ``sync_hiddens`` — the
        teacher-target contract (a boundary target is post-attn, a final /
        mid-window target is post-MLP; a layer never needs both).
        """
        from parallm.model.seam import checkpointed_halves

        L = len(self.text_models[0].layers)
        last = L - 1
        sync_attn_set = (
            set(range(L)) if exact else set(self.sync_after_layers) - {last}
        )
        if sync_hiddens is None:
            tap_attn = tap_mlp = ()
        else:
            tap_attn, tap_mlp = capture_post_attn or (), capture_post_mlp or ()
        run_mixer, run_mlp = checkpointed_halves(
            self.use_checkpoint and torch.is_grad_enabled(),
            position_embeddings,
            text_position_ids,
        )

        block_start = h
        per_track_h = [h for _ in self.text_models]
        for i in range(L):
            # layer_types[i] is identical on every track, so the mixer mask is
            # per-layer.
            layer_mask = (
                linear_attn_mask
                if self.text_models[0].config.layer_types[i] == "linear_attention"
                else causal_mask
            )
            if self.layer_stream is not None:
                self.layer_stream.acquire(i)
            # Fuse after every sublayer, and feed a global sync one leader per
            # group: at a boundary the group members share their pre-state, so
            # summing all F would count it F times. No-ops at fuse_size=1.
            h_attn = self.sync_module.fuse(
                [run_mixer(tm.layers[i], per_track_h[k], layer_mask)
                 for k, tm in enumerate(self.text_models)],
                per_track_h,
            )
            if i in sync_attn_set:
                # Boundary: post-attention sync delivers the real cross-track
                # content to this layer's MLP.
                R = self.sync_module(self.sync_module.leaders(h_attn), block_start)
                if i in tap_attn:
                    sync_hiddens[i] = R
                block_start = R
                pre = [R] * len(self.text_models)
                new_h = self.sync_module.fuse(
                    [run_mlp(tm.layers[i], R) for tm in self.text_models], pre
                )
                if exact:
                    # Second sync of the exact schedule: post-MLP, every layer.
                    Hm = self.sync_module(self.sync_module.leaders(new_h), R)
                    if i in tap_mlp:
                        sync_hiddens[i] = Hm
                    if i == last:
                        h = Hm
                    block_start = Hm
                    per_track_h = [Hm for _ in self.text_models]
                else:
                    per_track_h = new_h
            else:
                # Own-carry (non-boundary) OR the final layer's post-MLP sync.
                new_h = self.sync_module.fuse(
                    [run_mlp(tm.layers[i], h_attn[k]) for k, tm in enumerate(self.text_models)],
                    h_attn,
                )
                if i == last:
                    h = self.sync_module(self.sync_module.leaders(new_h), block_start)
                    if i in tap_mlp:
                        sync_hiddens[i] = h
                else:
                    per_track_h = new_h
            if self.layer_stream is not None:
                self.layer_stream.release(i)
        return h

    def _pad_narrow_slabs(
        self, remapped: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Zero-pad shards sliced under an older, narrower per-track width.

        `align_chunk` widened the zero-padded MLP lanes to a GEMM-friendly multiple
        (27B/N=24: 726 -> 768), so shards converted before that are short along one
        dim. The added lanes are zeros either way — `silu(0)*up = 0` — so the padded
        slab is the SAME function, and padding here keeps healed checkpoints usable
        instead of forcing a re-convert. Anything mismatched in more than one dim, or
        wider than the model wants, is left alone for `load_state_dict` to report.
        """
        current = self.state_dict()
        out: dict[str, torch.Tensor] = {}
        for key, val in remapped.items():
            want = current.get(key)
            if want is None or want.shape == val.shape or want.dim() != val.dim():
                out[key] = val
                continue
            diff = [d for d in range(val.dim()) if want.shape[d] != val.shape[d]]
            if len(diff) != 1 or want.shape[diff[0]] < val.shape[diff[0]]:
                out[key] = val  # not a pure widening — let the loader complain
                continue
            d = diff[0]
            tail = list(val.shape)
            tail[d] = want.shape[d] - val.shape[d]
            out[key] = torch.cat([val, val.new_zeros(tail)], dim=d)
        return out

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
            prefix = f"text_models.{k}."
            for key, val in track_states[tid].items():
                if key == "lm_head.weight":
                    remapped["lm_head.weight"] = val  # owner only; rank-shared head
                else:
                    remapped[prefix + key] = val

        remapped = self._pad_narrow_slabs(remapped)
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if strict:
            # rotary buffer keys (e.g. text_models.{k}.rotary_emb.inv_freq) may
            # legitimately be missing — they're computed at init.
            missing_critical = [k for k in missing if "rotary_emb" not in k]
            if missing_critical or unexpected:
                raise RuntimeError(
                    f"load_track_state_dicts mismatch:\n  missing={missing_critical}\n  unexpected={unexpected}"
                )


class HostResidentLayers:
    """Frozen decoder layers parked in pinned host DRAM, paged in one at a time.

    The exact-schedule teacher forward is one ``no_grad`` walk over layers 0..L-1,
    so only the current layer need be resident: the teacher's ~30 GB/rank at 122 B
    becomes a 1–2 layer working set, which is what fits it beside the student.
    Assign to ``PTWrappedModel.layer_stream``; the walk drives
    ``acquire``/``release``. Source tensors are pinned, so the H2D is async.

    On CUDA the copies ride a dedicated PREFETCH stream: while the compute stream
    runs layer i, layer i+1's H2D overlaps it, hiding the transfer at the cost of
    one extra resident layer. A per-layer event gates compute so it never reads a
    half-copied layer, and ``record_stream`` stops the allocator recycling a layer
    the compute stream still holds. On CPU (tests) the copies are synchronous —
    one layer resident, no overlap.
    """

    def __init__(self, models, num_layers: int, device):
        # Normalize: torch.cuda.current_device() hands back a bare int.
        self.device = torch.device(device)
        self._n = num_layers
        self._params: list[list[torch.nn.Parameter]] = []
        for i in range(num_layers):
            ps = [p for m in models for p in m.layers[i].parameters()]
            for p in ps:
                # A trainable param would lose its storage before backward could
                # read it — this is a frozen-teacher-only mechanism.
                assert not p.requires_grad, "HostResidentLayers requires frozen params"
                # Pinning is what makes the copies async; it also needs a CUDA
                # context, so the CPU path (tests) just keeps its own copy.
                p._host = (p.data.pin_memory() if self.device.type == "cuda"
                           else p.data.clone())
                p.data = torch.empty(0, dtype=p.dtype)
            self._params.append(ps)
        self._copy = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._inflight: set[int] = set()
        self._done: dict[int, torch.cuda.Event] = {}

    def _load(self, i: int) -> None:
        """Issue layer i's H2D (on the copy stream if prefetching) and mark it."""
        if self._copy is None:
            for p in self._params[i]:
                p.data = p._host.to(self.device, non_blocking=True)
        else:
            with torch.cuda.stream(self._copy):
                for p in self._params[i]:
                    p.data = p._host.to(self.device, non_blocking=True)
                self._done[i] = torch.cuda.Event()
                self._done[i].record(self._copy)
        self._inflight.add(i)

    def acquire(self, i: int) -> None:
        if i not in self._inflight:
            self._load(i)
        if self._copy is not None:
            compute = torch.cuda.current_stream()
            compute.wait_event(self._done[i])  # don't read a half-copied layer
            for p in self._params[i]:
                p.data.record_stream(compute)  # copy-stream alloc, compute-stream use
            if i + 1 < self._n and (i + 1) not in self._inflight:
                self._load(i + 1)  # overlaps this layer's compute

    def release(self, i: int) -> None:
        self._inflight.discard(i)
        self._done.pop(i, None)
        for p in self._params[i]:
            p.data = torch.empty(0, dtype=p.dtype, device=self.device)

    @property
    def host_bytes(self) -> int:
        return sum(p._host.numel() * p._host.element_size()
                   for ps in self._params for p in ps)
