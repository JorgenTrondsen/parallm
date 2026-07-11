"""The parallm inference engine: lockstep decode with sparse-replica replay.

Deployed semantics (fixed by the probe that measured the frontier — git tag
``pre-parallm-pivot``, ``eval/intervention.py::phased_intervention_forward``,
``replica:*`` channel, whole-network copies):

- Between sync boundaries every track carries the same **shadow estimate** of the
  full residual, advanced comm-free by replaying ALL ``N`` tracks' sublayers with
  degraded weight copies (including the rank's own track — the carry must be
  bit-identical on every rank, so the shadow chain never mixes in exact weights).
- Alongside, each track runs its OWN sublayers exactly on that carry and
  accumulates the exact deltas. A boundary is ONE all-reduce of those deltas
  (`SyncBoundary`), which recombines the true residual and resets the carry.
- Sync placement is phased/post-attn (LEVER B): boundaries in ``sync_indices``
  sync after the token mixer (the boundary MLP then reads the TRUE residual);
  the final layer's boundary is post-MLP and produces the fully-synced hidden
  for the lm_head. Total sync events = len(boundaries < last) + 1.

The chunk forward serves both prefill/eval (whole sequence, ``caches=None`` or
fresh) and decode (1-token chunks with caches); incremental ≡ full-sequence by
causality, which is the engine's second correctness rail (the first: undegraded
copies ⇒ bit-equal to the dense model, up to fp summation order).

Comm rounds per decode token = 1 embed broadcast + boundary all-reduces. The
sampled token id itself is never broadcast: peers hold no ``embed_tokens`` and
contribute zeros to the embed collective, so only the head needs real ids.
``latency_s`` sleeps after every round — the single-node simulation of the
multi-node link (a real deployment needs the one-round hierarchical exchange
noted in the plan; stock ring all-reduce would pay (n_nodes-1) rounds).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from parallm.model.pt_model import PTWrappedModel


# --------------------------------------------------------------------------- #
# Shadow providers. The batched replay path never calls the grid modules'
# Linears: all N tracks' projections go through ``blinear`` (one batched op per
# rel), and the diverged small params (q/k norms, conv, dt, A, gated norm) are
# read per-track via ``_stacked``. The grid remains the storage for those
# params and the per-layer module attrs (head dims, eps, ...).
# --------------------------------------------------------------------------- #
_GEMV_MAX_M = 64


def _submodule(mod, dotted: str):
    for p in dotted.split("."):
        mod = getattr(mod, p)
    return mod


def _stacked(shadow, li: int, path: str) -> torch.Tensor:
    """The N tracks' param at ``path`` stacked ``[N, ...]`` (cached). A
    single-track grid gets a zero-copy view — the own-track provider must not
    duplicate the rank's full weights."""
    key = (li, path)
    t = shadow._stacked.get(key)
    if t is None:
        mods = [_submodule(tr[li], path) for tr in shadow.grid]
        t = mods[0].unsqueeze(0) if len(mods) == 1 else torch.stack(mods)
        shadow._stacked[key] = t
    return t


class DenseShadow:
    """Materialized shadow modules, ``grid[track][layer]``. Exact copies can
    share the student's own modules (weights are read-only here); degraded
    copies come from ``model.replica.degrade_track_layers``."""

    def __init__(self, grid: "Sequence[Sequence[nn.Module]]"):
        self.grid = grid
        self.n_tracks = len(grid)
        self._stacked: dict = {}

    def layers(self, i: int) -> "list[nn.Module]":
        return [track[i] for track in self.grid]

    def blinear(self, li: int, rel: str, x: torch.Tensor,
                per_track: bool = False) -> torch.Tensor:
        """All N tracks' Linear at ``rel``: ``[M, in]`` shared x (or ``[N, M,
        in]`` per-track x) → ``[N, M, out]``. matmul broadcasts the shared x."""
        w = _stacked(self, li, f"{rel}.weight")
        return torch.matmul(x, w.transpose(-2, -1))


class _RelEntry:
    """One (layer, rel) of the pool, batched across all N tracks for a single
    unpack kernel chain per rel: packed survivor bitmaps ``[N, bytes]``, the
    tracks' survivor values concatenated flat (codes re-nibble-packed without
    the per-track pad, so residency stays at the packed bits/weight), and
    per-row scales ``[N, out]``. ``masked_scatter_`` consumes the flat values
    in row-major mask order across the stacked ``[N, out, in]`` tensor, which
    is exactly the track-then-row pack order.

    ``pin=False``: tensors live on ``device`` (resident mode). ``pin=True``:
    tensors stay in pinned host DRAM and stream through a `_Ring` slot."""

    def __init__(self, packs: "list[dict]", device, pin: bool = False):
        import numpy as np

        shapes = {tuple(p["shape"].tolist()) for p in packs}
        assert len(shapes) == 1, f"rel shape mismatch across tracks: {shapes}"
        self.shape = (len(packs), *shapes.pop())  # (N, out, in)
        n, o, i = self.shape
        place = (lambda t: t.pin_memory()) if pin else (lambda t: t.to(device))
        self.mask = place(torch.stack([p["mask"] for p in packs]))
        self.scale = None

        # Per-(row, bitmap-block) survivor offsets into the flat value stream
        # (absolute across the track-major concat) — the fused kernel's v2
        # addressing, which frees each block iteration from the previous one's
        # popcount. Derived from the bitmap at load (~2% of pool bytes); small,
        # so always device-resident even when the payload is pinned host-side.
        from parallm.packed_gemv import GEMV_BLOCK_I

        nb = -(-i // GEMV_BLOCK_I)
        bits = np.stack([
            np.unpackbits(p["mask"].numpy())[: o * i].reshape(o, i)
            for p in packs
        ])                                        # [N, o, i]
        pad = np.zeros((n, o, nb * GEMV_BLOCK_I - i), dtype=bits.dtype)
        bpc = np.concatenate([bits, pad], -1).reshape(n, o, nb, -1).sum(-1)
        flat = np.cumsum(bpc.reshape(-1).astype(np.int64))  # inclusive, track-major
        starts = np.concatenate([[0], flat[:-1]]).reshape(n, o, nb)
        self.block_start = torch.from_numpy(starts.astype(np.int32)).to(device)
        track_nnz = bits.sum((1, 2))

        if "vals" in packs[0]:
            self.bits = None
            self.vals = place(torch.cat([p["vals"] for p in packs]))
            self.nnz = self.vals.numel()
        else:
            self.bits = int(packs[0]["bits"].item())
            self.scale = place(torch.stack([p["scale"] for p in packs]).float())
            if self.bits == 4:
                per_track = [
                    _unpack_nibbles_torch(p["codes"], int(track_nnz[k]))
                    for k, p in enumerate(packs)
                ]
                flat = torch.cat(per_track)
                self.nnz = flat.numel()
                a = flat.numpy()
                if a.size % 2:
                    a = np.append(a, np.uint8(0))
                self.codes = place(torch.from_numpy((a[0::2] << 4) | (a[1::2])))
            else:
                self.codes = place(torch.cat([p["codes"] for p in packs]))
                self.nnz = self.codes.numel()

    @property
    def values(self) -> torch.Tensor:
        return self.vals if self.bits is None else self.codes

    def unpack_from(self, mask: torch.Tensor, values: torch.Tensor,
                    out: torch.Tensor, f32: "torch.Tensor | None",
                    scale: "torch.Tensor | None" = None) -> None:
        """Materialize the degraded dense weights into ``out`` ``[N, out, in]``
        from the given (device) packed buffers — bit-exact to per-track
        ``unpack_sparse_weight``."""
        n, o, i = self.shape
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=mask.device)
        mask = (((mask.unsqueeze(-1) >> shifts) & 1)
                .reshape(n, -1)[:, : o * i].to(torch.bool).reshape(n, o, i))
        if self.bits is None:
            out.zero_()
            out.masked_scatter_(mask, values)
            return
        if self.bits == 4:
            values = torch.stack(((values >> 4) & 0xF, values & 0xF), dim=-1)
            values = values.reshape(-1)[: self.nnz]
        signed = values.to(torch.float32) - float(2 ** (self.bits - 1))
        f32.zero_()
        f32.masked_scatter_(mask, signed)
        f32 *= scale[:, :, None]
        out.copy_(f32)


class _Ring:
    """Device ring for the streamed pool: ``n_slots`` per-layer slots whose
    packed buffers are refilled from pinned host DRAM on a dedicated copy
    stream, ``n_slots`` layers ahead of compute — the sync stalls (host sleeps
    + NCCL waits + compute) are the hiding budget, exactly the
    bench_stream_overlap law. Slot ``s`` only ever hosts layers ``≡ s (mod
    n_slots)``, so its preallocated per-rel buffers take those layers' sizes."""

    def __init__(self, shadow: "PackedShadow", n_slots: int, device):
        self.shadow = shadow
        self.device = device
        self.n_slots = n_slots
        self.copy_stream = torch.cuda.Stream(device)
        # Preallocate every slot's buffers for the layers it will host
        # (li ≡ slot mod n_slots), values sized to the max across those layers —
        # no mid-flight growth, so no cross-stream reallocation hazard.
        self.slots: "list[dict]" = []
        for s in range(n_slots):
            slot: dict = {}
            for li in range(s, shadow.num_layers, n_slots):
                for rel in shadow.rels_at[li]:
                    e = shadow.entries[(li, rel)]
                    b = slot.get(rel)
                    if b is None:
                        slot[rel] = {
                            "mask": torch.empty_like(e.mask, device=device),
                            "values": torch.empty_like(e.values, device=device),
                        }
                        if e.scale is not None:
                            slot[rel]["scale"] = torch.empty_like(e.scale, device=device)
                    elif b["values"].numel() < e.values.numel():
                        b["values"] = torch.empty_like(e.values, device=device)
            self.slots.append(slot)
        self.ready: "dict[int, torch.cuda.Event]" = {}
        self.free: "list[torch.cuda.Event | None]" = [None] * n_slots
        self.next_copy = 0
        self.active: dict = {}      # rel -> current layer's slot bufs
        self._pending: "int | None" = None  # occurrence awaiting its free event
        self.pump(n_slots)

    def pump(self, limit: int) -> None:
        """Enqueue host→device slab copies up to occurrence ``limit``."""
        while self.next_copy < limit:
            occ = self.next_copy
            li = occ % self.shadow.num_layers
            slot = self.slots[occ % self.n_slots]
            with torch.cuda.stream(self.copy_stream):
                if self.free[occ % self.n_slots] is not None:
                    self.copy_stream.wait_event(self.free[occ % self.n_slots])
                for rel in self.shadow.rels_at[li]:
                    e = self.shadow.entries[(li, rel)]
                    bufs = slot[rel]
                    bufs["mask"].copy_(e.mask, non_blocking=True)
                    bufs["values"][: e.values.numel()].copy_(e.values, non_blocking=True)
                    if e.scale is not None:
                        bufs["scale"].copy_(e.scale, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
            self.ready[occ] = ev
            self.next_copy += 1

    def consume(self, occ: int, li: int) -> None:
        """Make layer ``li`` (occurrence ``occ``) computable, then release slots
        back to the copy stream: publish the slot's packed buffers (``blinear``
        reads them during the layer's ops, which are enqueued before the NEXT
        consume), so the free event for occurrence ``occ`` records at consume
        ``occ+1`` and the pump lookahead is shortened by one to match."""
        torch.cuda.current_stream().wait_event(self.ready.pop(occ))
        slot = self.slots[occ % self.n_slots]
        self.active = {rel: slot[rel] for rel in self.shadow.rels_at[li]}
        if self._pending is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            self.free[self._pending % self.n_slots] = ev
        self._pending = occ
        self.pump(occ + self.n_slots)


def _unpack_nibbles_torch(packed: torch.Tensor, n: int) -> torch.Tensor:
    u = torch.stack(((packed >> 4) & 0xF, packed & 0xF), dim=-1).reshape(-1)
    return u[:n]


class PackedShadow:
    """Shadow provider backed by a packed replica pool file: HF layer skeletons
    (meta-built, exact non-Linear params from the track shards) hold the small
    diverged params; every Linear is computed by ``blinear`` straight from the
    packed pool (fused kernel at decode M, dense-unpack scratch fallback at
    prefill/eval M). Residency = packed pool bytes + one dense layer of
    scratch — the dense pool (~N× layer bytes × layers) never exists. The
    skeletons' Linear weights stay on meta: calling one fails loudly.
    """

    def __init__(self, pool_path: str, tracks_dir: str, student: PTWrappedModel,
                 device, dtype: torch.dtype = torch.bfloat16,
                 stream_ring_layers: "int | None" = None, fused: bool = False):
        import os

        from safetensors import safe_open

        from parallm.model.replica_pack import load_replica_pool

        pool, self.meta = load_replica_pool(pool_path)
        self.n_tracks = max(k[0] for k in pool) + 1
        num_layers = max(k[1] for k in pool) + 1
        self.num_layers = num_layers
        self.device, self.dtype, self.fused = device, dtype, fused
        rels_at = {}  # layer_idx -> sorted rels
        for (_t, li, rel) in pool:
            rels_at.setdefault(li, set()).add(rel)
        self.rels_at = {li: sorted(rs) for li, rs in rels_at.items()}

        # Batched pool entries + shared scratch (one dense layer, all tracks).
        # Streamed mode keeps the packed entries in pinned host DRAM and pulls
        # them through a device ring; resident mode keeps them on device.
        # Fused mode computes straight from the packed bits (the GEMV kernel),
        # so scratch only materializes lazily for large-M calls (eval/prefill).
        pin = stream_ring_layers is not None
        self.entries = {}  # (li, rel) -> _RelEntry
        self.scratch = {}  # rel -> dense [N, out, in], lazily allocated
        self._f32 = {}     # rel -> fp32 tmp (qwanda rels only)
        self._scratch_mark = {}  # rel -> layer whose weights scratch holds
        self._stacked = {}       # (li, path) -> stacked small params [N, ...]
        for li, rels in self.rels_at.items():
            for rel in rels:
                e = _RelEntry([pool[(t, li, rel)] for t in range(self.n_tracks)],
                              device, pin=pin)
                if fused:
                    assert e.shape[2] % 8 == 0, (rel, e.shape)  # byte-aligned rows
                self.entries[(li, rel)] = e

        # Exact non-Linear params per (track, layer) from the shards.
        smalls: "dict[tuple[int, str], torch.Tensor]" = {}
        for t in range(self.n_tracks):
            with safe_open(os.path.join(tracks_dir, f"track_{t}.safetensors"),
                           framework="pt", device="cpu") as f:
                for key in f.keys():
                    if not key.startswith("layers."):
                        continue
                    parts = key.split(".")
                    li, sub = int(parts[1]), ".".join(parts[2:])
                    rel = sub[: -len(".weight")] if sub.endswith(".weight") else sub
                    if (t, li, rel) in pool:
                        continue  # a degraded Linear — lives in the pool
                    smalls[(t, ".".join(parts[1:]))] = f.get_tensor(key).to(device=device, dtype=dtype)

        # Meta-built skeletons; every non-Linear param must resolve to a shard
        # small (pool Linears are computed via blinear and stay on meta —
        # anything else left on meta fails loudly at first use).
        layer_cls = type(student.text_models[0].layers[0])
        cfg = student.per_track_text_config
        self.grid = []
        for t in range(self.n_tracks):
            track_layers = []
            for li in range(num_layers):
                with torch.device("meta"):
                    layer = layer_cls(cfg, li)
                for name, _p in list(layer.named_parameters()):
                    rel = name[: -len(".weight")] if name.endswith(".weight") else name
                    if (li, rel) not in self.entries:
                        _assign_param(layer, name, smalls[(t, f"{li}.{name}")])
                layer.eval()
                track_layers.append(layer)
            self.grid.append(track_layers)
        self._occ = 0
        self.ring = (_Ring(self, stream_ring_layers, device)
                     if stream_ring_layers is not None else None)

    def layers(self, i: int) -> "list":
        if self.ring is not None:
            # Streamed: strictly sequential layer visits (the replay loop's order).
            assert i == self._occ % self.num_layers, (i, self._occ)
            self.ring.consume(self._occ, i)
            self._occ += 1
        return [track[i] for track in self.grid]

    def blinear(self, li: int, rel: str, x: torch.Tensor,
                per_track: bool = False) -> torch.Tensor:
        """All N tracks' packed Linear at ``rel`` in one op: ``[M, in]`` shared
        x (``x_tstride=0`` — every track reads the same carry-derived input) or
        ``[N, M, in]`` per-track x → ``[N, M, out]``. Decode-M uses the fused
        kernel; larger M (prefill/eval) uses the dense-unpack scratch."""
        e = self.entries[(li, rel)]
        n, o, i = e.shape
        M = x.shape[-2]
        if self.fused and x.is_cuda and M <= _GEMV_MAX_M:
            from parallm.packed_gemv import _launch

            bufs = self.active_bufs(li, rel)
            return _launch(x.contiguous(), bufs["mask"], bufs["values"],
                           bufs.get("scale"), e.block_start, n, M, o, i, e.bits,
                           M * i if per_track else 0)
        return torch.matmul(x, self.dense_weights(li, rel).transpose(-2, -1))

    # ---- packed-buffer plumbing (blinear reads through these) ----
    def active_bufs(self, li: int, rel: str) -> dict:
        """The device-resident packed buffers for (li, rel): the ring's current
        slot when streamed, else the entry's own tensors."""
        if self.ring is not None:
            return self.ring.active[rel]
        e = self.entries[(li, rel)]
        bufs = {"mask": e.mask, "values": e.values}
        if e.scale is not None:
            bufs["scale"] = e.scale
        return bufs

    def _alloc_scratch(self, rel: str, e: _RelEntry) -> None:
        self.scratch[rel] = torch.empty(e.shape, dtype=self.dtype, device=self.device)
        if e.bits is not None:
            self._f32[rel] = torch.empty(e.shape, dtype=torch.float32, device=self.device)

    def dense_weights(self, li: int, rel: str) -> torch.Tensor:
        """The dense degraded weights ``[N, out, in]`` from a lazily allocated
        shared scratch, unpacked once per (layer, rel) visit."""
        e = self.entries[(li, rel)]
        if rel not in self.scratch:
            self._alloc_scratch(rel, e)
        assert e.shape == self.scratch[rel].shape, (li, rel)
        if self._scratch_mark.get(rel) != li:
            bufs = self.active_bufs(li, rel)
            e.unpack_from(bufs["mask"], bufs["values"][: e.values.numel()],
                          self.scratch[rel], self._f32.get(rel), bufs.get("scale"))
            self._scratch_mark[rel] = li
        return self.scratch[rel]


def _assign_param(module, dotted: str, tensor: torch.Tensor) -> None:
    *path, leaf = dotted.split(".")
    for p in path:
        module = getattr(module, p)
    setattr(module, leaf, nn.Parameter(tensor, requires_grad=False))


# --------------------------------------------------------------------------- #
# The track-batched forward: N tracks' sublayers as ONE batched op chain
# (track dim = batch dim) instead of N per-track HF module calls — the
# module-internal tiny-kernel tail was the measured decode floor. It serves
# BOTH streams of the replay: the N shadow tracks (via PackedShadow, summed
# delta advances the carry) and the K own tracks (via a DenseShadow over the
# rank's exact modules, per-track deltas). All inputs are track-shared by
# construction (the carry / the synced base), so only per-track weights ride
# the batch dim. Every rank runs the same shadow chain on the same inputs,
# preserving the rank-bit-identical carry.
# --------------------------------------------------------------------------- #
def _hf():
    """The Qwen3.5 modeling module, resolved at call time so its FLA/CUDA
    kernel bindings (monkeypatched to None on CPU test runs) stay authoritative."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    return m


def _rms_tracks(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-track Qwen3.5 RMSNorm (zero-centered weight): ``x [N, ..., d]``,
    ``w [N, d]``."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + w.float()).view(w.shape[0], *([1] * (x.ndim - 2)), -1)
    return out.type_as(x)


def _batched_attn(shadow, li: int, x, position_embeddings, attention_mask, cache):
    """All N tracks' full-attention mixers on the shared input ``x [B, T, h]``
    → per-track deltas ``[N, B, T, h]``. Mirrors ``Qwen3_5Attention.forward``
    with the track dim folded into batch."""
    m = _hf()
    layer0 = shadow.grid[0][li]
    attn = layer0.self_attn
    N, (B, T, _) = shadow.n_tracks, x.shape
    d = attn.head_dim
    h2 = layer0.input_layernorm(x).reshape(B * T, -1)  # residual norms are track-synced
    q, gate = shadow.blinear(li, "self_attn.q_proj", h2).view(N, B, T, -1, 2 * d).chunk(2, dim=-1)
    q = _rms_tracks(q, _stacked(shadow, li, "self_attn.q_norm.weight"), attn.q_norm.eps)
    k = shadow.blinear(li, "self_attn.k_proj", h2).view(N, B, T, -1, d)
    k = _rms_tracks(k, _stacked(shadow, li, "self_attn.k_norm.weight"), attn.k_norm.eps)
    v = shadow.blinear(li, "self_attn.v_proj", h2)
    q = q.reshape(N * B, T, -1, d).transpose(1, 2)
    k = k.reshape(N * B, T, -1, d).transpose(1, 2)
    v = v.reshape(N * B, T, -1, d).transpose(1, 2)
    cos, sin = position_embeddings
    if B > 1:  # track-major batch: row n*B+b must read position row b
        cos, sin = cos.repeat(N, 1, 1), sin.repeat(N, 1, 1)
    q, k = m.apply_rotary_pos_emb(q, k, cos, sin)
    if cache is not None:
        # Static KV: attention spans the whole s_max buffer; a boolean mask
        # from the device position keeps it causal and hides unwritten slots
        # (shape-static, so CUDA graphs can capture the step).
        k, v = cache.update_kv(li, k, v)
        qpos = cache.pos_t + torch.arange(T, device=k.device)
        mask = (torch.arange(k.shape[2], device=k.device)[None, :]
                <= qpos[:, None]).view(1, 1, T, -1)
    else:
        mask = attention_mask
        if mask is not None and mask.shape[0] > 1:
            mask = mask.repeat(N, 1, 1, 1)
    o = nn.functional.scaled_dot_product_attention(
        q, m.repeat_kv(k, attn.num_key_value_groups),
        m.repeat_kv(v, attn.num_key_value_groups),
        attn_mask=mask, scale=attn.scaling,
        is_causal=cache is None and mask is None and T > 1,
    )
    o = o.transpose(1, 2).reshape(N, B * T, -1) * torch.sigmoid(gate.reshape(N, B * T, -1))
    y = shadow.blinear(li, "self_attn.o_proj", o, per_track=True)
    return y.view(N, B, T, -1)


def _batched_gdn(shadow, li: int, x, attention_mask, cache):
    """All N tracks' gated-delta-net mixers on the shared input → per-track
    deltas. Mirrors ``Qwen3_5GatedDeltaNet.forward``; tracks fold into the conv
    CHANNEL dim (depthwise conv keys weights by channel, not batch) and into
    the BATCH dim of the delta-rule kernels."""
    m = _hf()
    layer0 = shadow.grid[0][li]
    gdn = layer0.linear_attn
    N, (B, T, _) = shadow.n_tracks, x.shape
    Hv, Hk = gdn.num_v_heads, gdn.num_k_heads
    dk, dv = gdn.head_k_dim, gdn.head_v_dim
    C, K = gdn.conv_dim, gdn.conv_kernel_size
    x = m.apply_mask_to_padding_states(x, attention_mask)
    h2 = layer0.input_layernorm(x).reshape(B * T, -1)
    qkv = shadow.blinear(li, "linear_attn.in_proj_qkv", h2)
    z = shadow.blinear(li, "linear_attn.in_proj_z", h2)
    b = shadow.blinear(li, "linear_attn.in_proj_b", h2)
    a = shadow.blinear(li, "linear_attn.in_proj_a", h2)

    conv_w = _stacked(shadow, li, "linear_attn.conv1d.weight").reshape(N * C, K)
    xc = qkv.view(N, B, T, C).permute(1, 0, 3, 2).reshape(B, N * C, T)
    state = cache.conv.get(li) if cache is not None else None
    if state is not None and T == 1:
        conv_update = m.causal_conv1d_update or m.torch_causal_conv1d_update
        xc = conv_update(xc, state, conv_w, None, gdn.activation)
    else:
        if state is not None:  # chunked continuation: prepend the conv context
            xc = torch.cat([state, xc], dim=-1)
        if cache is not None:
            cache.conv[li] = nn.functional.pad(xc, (K - xc.shape[-1], 0))
        L = xc.shape[-1]
        if m.causal_conv1d_fn is not None:
            xc = m.causal_conv1d_fn(x=xc, weight=conv_w, bias=None,
                                    activation=gdn.activation)
        else:
            xc = nn.functional.silu(nn.functional.conv1d(
                xc, conv_w.unsqueeze(1), padding=K - 1, groups=N * C)[..., :L])
        if state is not None:
            xc = xc[..., -T:]
    qkv = xc.view(B, N, C, T).permute(1, 0, 3, 2)  # [N, B, T, C]
    q, k, v = torch.split(qkv, [Hk * dk, Hk * dk, Hv * dv], dim=-1)
    q = q.reshape(N * B, T, Hk, dk)
    k = k.reshape(N * B, T, Hk, dk)
    v = v.reshape(N * B, T, Hv, dv)
    beta = b.view(N * B, T, Hv).sigmoid()
    A_log = _stacked(shadow, li, "linear_attn.A_log")
    dt_bias = _stacked(shadow, li, "linear_attn.dt_bias")
    g = -A_log.float().exp().view(N, 1, 1, Hv) * nn.functional.softplus(
        a.view(N, B, T, Hv).float() + dt_bias.view(N, 1, 1, Hv))
    g = g.reshape(N * B, T, Hv)
    if Hv // Hk > 1:
        q = q.repeat_interleave(Hv // Hk, dim=2)
        k = k.repeat_interleave(Hv // Hk, dim=2)
    rec = cache.rec.get(li) if cache is not None else None
    if rec is not None and T == 1:
        fn = m.fused_recurrent_gated_delta_rule or m.torch_recurrent_gated_delta_rule
    else:
        fn = m.chunk_gated_delta_rule or m.torch_chunk_gated_delta_rule
    core, rec_out = fn(q, k, v, g=g, beta=beta, initial_state=rec,
                       output_final_state=cache is not None,
                       use_qk_l2norm_in_kernel=True)
    if cache is not None:
        cache.update_rec(li, rec_out)
    # Gated RMSNorm (per-track weight, NOT zero-centered), norm before gate —
    # the torch reference math of Qwen3_5RMSNormGated.
    normw = _stacked(shadow, li, "linear_attn.norm.weight")
    cf = core.view(N, B, T, Hv, dv).float()
    cn = cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + gdn.layer_norm_epsilon)
    cn = normw.view(N, 1, 1, 1, dv) * cn.to(core.dtype)
    out = (cn * nn.functional.silu(z.view(N, B, T, Hv, dv).float())).to(core.dtype)
    y = shadow.blinear(li, "linear_attn.out_proj", out.reshape(N, B * T, Hv * dv),
                       per_track=True)
    return y.view(N, B, T, -1)


def _batched_mlp(shadow, li: int, x):
    """All N tracks' MLPs on the shared input ``x`` → per-track deltas."""
    layer0 = shadow.grid[0][li]
    N, (B, T, _) = shadow.n_tracks, x.shape
    h2 = layer0.post_attention_layernorm(x).reshape(B * T, -1)
    g = shadow.blinear(li, "mlp.gate_proj", h2)
    u = shadow.blinear(li, "mlp.up_proj", h2)
    y = shadow.blinear(li, "mlp.down_proj", layer0.mlp.act_fn(g) * u, per_track=True)
    return y.view(N, B, T, -1)


# --------------------------------------------------------------------------- #
# Caches: plain batched tensors (track dim folded into batch/channels,
# matching the batched forward) — one instance for the K own tracks, one for
# the N shadow tracks. No HF cache classes anywhere in decode. Every buffer is
# STATIC (preallocated KV written by position, in-place conv/recurrent states)
# and the write position is a device tensor, so a captured CUDA graph replays
# correctly as ``pos_t`` advances.
# --------------------------------------------------------------------------- #
@dataclass
class BatchedShadowCache:
    pos_t: torch.Tensor          # device scalar: tokens already cached (shared)
    s_max: int                   # KV capacity in tokens
    kv: dict = None              # li -> (k, v) [N*B, kv_heads, s_max, d]
    conv: dict = None            # li -> [B, N*conv_dim, K], updated in place
    rec: dict = None             # li -> [N*B, Hv, dk, dv], updated in place

    def __post_init__(self):
        self.kv, self.conv, self.rec = {}, {}, {}

    def update_kv(self, li: int, k: torch.Tensor, v: torch.Tensor):
        buf = self.kv.get(li)
        if buf is None:
            shape = (k.shape[0], k.shape[1], self.s_max, k.shape[3])
            buf = (k.new_zeros(shape), k.new_zeros(shape))
            self.kv[li] = buf
        idx = self.pos_t + torch.arange(k.shape[2], device=k.device)
        buf[0].index_copy_(2, idx, k)
        buf[1].index_copy_(2, idx, v)
        return buf

    def update_rec(self, li: int, rec_out: torch.Tensor) -> None:
        if li in self.rec:
            self.rec[li].copy_(rec_out)  # keep the static buffer graphs captured
        else:
            self.rec[li] = rec_out


@dataclass
class EngineCaches:
    own: BatchedShadowCache      # the K local tracks, batched
    shadow: BatchedShadowCache   # all N shadow tracks, batched
    pos_t: torch.Tensor          # device mirror of ``pos`` (graph-readable)
    pos: int = 0                 # tokens already cached (authoritative for positions)

    @classmethod
    def build(cls, student: PTWrappedModel, s_max: int = 512) -> "EngineCaches":
        dev = next(student.parameters()).device
        pos_t = torch.zeros((), dtype=torch.long, device=dev)
        return cls(own=BatchedShadowCache(pos_t, s_max),
                   shadow=BatchedShadowCache(pos_t, s_max), pos_t=pos_t)


def _stall(latency_s: float, timing: "dict | None") -> None:
    """One cross-node comm round: simulated wire latency + accounting."""
    if timing is not None:
        timing["rounds"] = timing.get("rounds", 0) + 1
    if latency_s > 0:
        time.sleep(latency_s)


# --------------------------------------------------------------------------- #
# The replay chunk forward.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def replay_chunk(
    student: PTWrappedModel,
    shadow,
    input_ids: torch.LongTensor,
    sync_indices: Sequence[int],
    caches: "EngineCaches | None" = None,
    attention_mask: torch.Tensor | None = None,
    latency_s: float = 0.0,
    timing: "dict | None" = None,
    graphs: "dict | None" = None,
):
    """Advance every track through one chunk of tokens (whole prompt at prefill,
    1 token per decode step). Returns full-chunk logits ``(B, T, V)`` on the
    head rank, ``None`` on peers. All ranks must call in lockstep.

    ``graphs`` (a dict owned by the caller, one per generation) turns on
    CUDA-graph windows for cached 1-token decode: each inter-sync segment is
    captured once (after one eager warmup step) and replayed thereafter, so the
    whole window costs one launch. Comms (embed collective, boundary
    all-reduces), the lm_head, and sampling stay eager between replays; every
    tensor a graph touches is static by construction (see BatchedShadowCache)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

    tm0 = student.text_models[0]
    num_layers = len(tm0.layers)
    last = num_layers - 1
    sync_set = set(int(i) for i in sync_indices) - {last}

    # The K local tracks run through the same batched forward as the shadow,
    # via a provider over their exact modules (stacked-weight views).
    own = getattr(student, "_engine_own_provider", None)
    if own is None:
        own = DenseShadow([list(tm.layers) for tm in student.text_models])
        student._engine_own_provider = own

    # Embed on the owner track; the collective broadcasts h to every rank.
    h = student.embed(input_ids)
    _stall(latency_s, timing)

    # Scaffolding, offset by the cached position (mirrors Qwen3_5TextModel.forward).
    B, T = input_ids.shape
    past = caches.pos if caches is not None else 0
    pos = torch.arange(past, past + T, device=h.device)
    position_ids = pos.view(1, 1, -1).expand(4, B, -1)
    text_position_ids = position_ids[0]
    if past > 0:
        # Cached decode: a single new token attends to everything cached — no
        # mask needed. Chunked continuation (draft-verify) needs an offset
        # mask that nothing builds yet.
        assert T == 1, "chunked decode with past needs an offset causal mask"
        causal_mask = None
        linear_attn_mask = None
    else:
        causal_mask = create_causal_mask(
            config=tm0.config, inputs_embeds=h, attention_mask=attention_mask,
            past_key_values=None, position_ids=text_position_ids,
        )
        linear_attn_mask = (None if attention_mask is None
                            or bool(torch.all(attention_mask == 1))
                            else attention_mask)
    position_embeddings = tm0.rotary_emb(h, position_ids[1:])

    own_cache = caches.own if caches is not None else None
    shadow_cache = caches.shadow if caches is not None else None
    if caches is not None:
        caches.pos_t.fill_(past)  # the device mirror graphs read

    # Optional per-window breakdown: device-synced host clocks at each segment
    # boundary. The synchronize serializes the overlap the normal path enjoys
    # (kernels running through the sleep), so the profiled total reads HIGHER
    # than the unprofiled wall — it is the serial cost decomposition.
    prof = timing is not None and timing.get("profile") and h.is_cuda

    def _mark():
        torch.cuda.synchronize()
        return time.perf_counter()

    if prof:
        win_ms: "list[float]" = []
        sync_ms: "list[float]" = []
        seg_t = _mark()

    def mixer(provider, i, x, mask, cache):
        if tm0.config.layer_types[i] == "linear_attention":
            return _batched_gdn(provider, i, x, mask, cache)
        return _batched_attn(provider, i, x, position_embeddings, mask, cache)

    # The layer walk, segmented at the sync events: segment w runs the
    # PREVIOUS boundary's post-sync MLPs, the full layers up to boundary w, and
    # boundary w's own mixer (post-attn placement; the boundary-layer shadow
    # mixer is skipped — its delta is always discarded by the carry reset).
    # The last boundary is the final layer, synced post-MLP for the head.
    boundaries = sorted(sync_set) + [last]
    K = len(student.text_models)

    def seg_fn(w, carry, base, track_h):
        b = boundaries[w]
        if w > 0:
            pb = boundaries[w - 1]
            # Exact MLP reads the TRUE residual; shadow MLP advances the carry
            # from it with degraded weights.
            y_own = _batched_mlp(own, pb, base)
            track_h = [base + y_own[k] for k in range(K)]
            carry = base + _batched_mlp(shadow, pb, base).sum(0)
        for i in range(boundaries[w - 1] + 1 if w > 0 else 0, b + 1):
            mask = (linear_attn_mask
                    if tm0.config.layer_types[i] == "linear_attention" else causal_mask)
            shadow.layers(i)  # ring prefetch / residency side effects
            # Own exact mixers on the carry (write own KV/GDN state), K-batched.
            y_own = mixer(own, i, carry, mask, own_cache)
            track_h = [th + y_own[k] for k, th in enumerate(track_h)]
            if i < b or b == last:  # full layer (the last layer syncs post-MLP)
                # Shadow mixer: ALL N degraded copies advance the carry comm-free.
                carry_attn = carry + mixer(shadow, i, carry, mask, shadow_cache).sum(0)
                y_own = _batched_mlp(own, i, carry_attn)
                track_h = [th + y_own[k] for k, th in enumerate(track_h)]
                carry = carry_attn + _batched_mlp(shadow, i, carry_attn).sum(0)
        return carry, track_h

    # CUDA-graph windows: cached 1-token decode only, one eager warmup step
    # (allocations, JIT/autotune, cuBLAS workspaces) before capture.
    use_g = (graphs is not None and caches is not None and past > 0 and T == 1
             and h.is_cuda)
    if use_g and not graphs.get("warm"):
        graphs["warm"] = True
        use_g = False
    if use_g:
        st = graphs.get("static")
        if st is None:
            st = graphs["static"] = {
                "carry": torch.empty_like(h), "base": torch.empty_like(h),
                "track_h": [torch.empty_like(h) for _ in range(K)],
                "cos": position_embeddings[0].clone(),
                "sin": position_embeddings[1].clone(),
                "graphs": [None] * len(boundaries),
            }
        else:
            st["cos"].copy_(position_embeddings[0])
            st["sin"].copy_(position_embeddings[1])
        position_embeddings = (st["cos"], st["sin"])
        st["carry"].copy_(h)
        st["base"].copy_(h)
        for tk in st["track_h"]:
            tk.copy_(h)
        carry, base, track_h = st["carry"], st["base"], st["track_h"]
    else:
        carry = h                # the shared shadow estimate
        base = h                 # true residual, valid at boundaries
        track_h = [h] * K        # base + own exact deltas since

    for w in range(len(boundaries)):
        if use_g:
            g = st["graphs"][w]
            if g is None:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):  # capture records, replay executes
                    c2, th2 = seg_fn(w, carry, base, track_h)
                    carry.copy_(c2)
                    for tk, t2 in zip(track_h, th2):
                        tk.copy_(t2)
                st["graphs"][w] = g
            g.replay()
        else:
            carry, track_h = seg_fn(w, carry, base, track_h)
        if prof:
            t = _mark()
            win_ms.append((t - seg_t) * 1e3)
        nb = student.sync_module(track_h, base)  # ONE all-reduce of exact deltas
        if use_g:
            base.copy_(nb)  # keep the static pointer the graphs captured
        else:
            base = nb
        _stall(latency_s, timing)
        if prof:
            seg_t = _mark()
            sync_ms.append((seg_t - t) * 1e3)

    if prof:
        timing.setdefault("window_ms", []).append(win_ms)
        timing.setdefault("sync_ms", []).append(sync_ms)

    if caches is not None:
        caches.pos += T
    if student.lm_head is None:
        return None
    return student.lm_head(tm0.norm(base))


def make_engine_forward_fn(student: PTWrappedModel, shadow, sync_indices):
    """``ForwardFn`` for ``eval/lm_eval_adapter.PTLM``: whole-sequence engine
    replay, so lm-eval scores the exact deployed forward."""
    def _fn(input_ids, attention_mask):
        return replay_chunk(student, shadow, input_ids, sync_indices,
                            attention_mask=attention_mask)
    return _fn


# --------------------------------------------------------------------------- #
# Single-stream generation.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(
    student: PTWrappedModel,
    shadow,
    input_ids: torch.LongTensor,
    sync_indices: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    latency_s: float = 0.0,
    timing: "dict | None" = None,
    use_cuda_graphs: bool = False,
) -> torch.LongTensor:
    """Prefill + token-by-token decode. Returns the generated ids ``(B, new)``
    on the head rank (peers return the placeholder ids they decoded with).

    Peers never see the sampled ids: they hold no ``embed_tokens``, contribute
    zeros to the embed collective, and only need chunk *shapes*, so every rank
    decodes exactly ``max_new_tokens``.
    """
    # ponytail: no EOS early-stop — peers can't observe it without an extra
    # broadcast round; add a stop-flag round if interactive use ever needs it.
    caches = EngineCaches.build(student,
                                s_max=input_ids.shape[1] + max_new_tokens)
    graphs: "dict | None" = {} if use_cuda_graphs else None
    if use_cuda_graphs:
        assert getattr(shadow, "ring", None) is None, \
            "CUDA graphs need --residency resident (the stream ring is eager)"
    per_token_ms: "list[float]" = []

    t0 = time.perf_counter()
    logits = replay_chunk(student, shadow, input_ids, sync_indices, caches,
                          latency_s=latency_s, timing=timing)
    prefill_ms = (time.perf_counter() - t0) * 1e3

    out = []
    tok = torch.zeros((input_ids.shape[0], 1), dtype=torch.long,
                      device=input_ids.device)
    for step_i in range(max_new_tokens):
        if logits is not None:
            step = logits[:, -1, :].float()
            if temperature > 0:
                probs = torch.softmax(step / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            else:
                tok = step.argmax(dim=-1, keepdim=True)
        out.append(tok)
        if step_i == max_new_tokens - 1:
            break  # the last token needs no further forward
        t0 = time.perf_counter()
        logits = replay_chunk(student, shadow, tok, sync_indices, caches,
                              latency_s=latency_s, timing=timing, graphs=graphs)
        per_token_ms.append((time.perf_counter() - t0) * 1e3)

    if timing is not None:
        timing["prefill_ms"] = prefill_ms
        timing["per_token_ms"] = per_token_ms
    return torch.cat(out, dim=1)
