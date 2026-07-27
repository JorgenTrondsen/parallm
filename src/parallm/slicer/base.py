"""Declarative slicing specs for converting dense weights into per-track slices.

Each parameter in a decoder layer is tagged with a `SlicerSpec` that says how
to split it across N tracks. The slicer engine applies the spec to produce N
per-track tensors and (for verification) can reassemble them.

Specs operate on a single tensor at a time. Track index is supplied at slice
time. Specs are deliberately tiny and side-effect free so they can be unit
tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SlicerSpec(Protocol):
    """Protocol for a slicing rule. Each spec knows how to slice and reassemble.

    Specs may optionally implement ``replication_groups(n_tracks, force_sync=False)``
    to declare that some tracks hold *identical* slices and therefore must share a
    gradient during training. Specs that don't implement it are treated as fully
    unique per track (one singleton group per track). Replicated-style specs carry
    a ``sync`` flag: when ``sync`` is False they return singleton groups (the copies
    are free to *diverge* under training), unless ``force_sync=True`` overrides it
    back to a single shared group (the legacy bit-identical behaviour).
    """

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor: ...

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor: ...

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]: ...


def _even_chunk(size: int, n_tracks: int, pad_full_size: int | None, label: str) -> int:
    """Per-track chunk along a split dim. Without `pad_full_size` the dim must
    divide evenly; with it the chunk rounds *up* and short slices are zero-filled."""
    if pad_full_size is None:
        if size % n_tracks != 0:
            raise ValueError(f"{label}: dim size {size} not divisible by n_tracks {n_tracks}")
        return size // n_tracks
    if size != pad_full_size:
        raise ValueError(f"{label}: expected dim size {pad_full_size}, got {size}")
    return -(-size // n_tracks)


def _chunk_slice(weight: torch.Tensor, dim: int, track_idx: int, chunk: int) -> torch.Tensor:
    """Track `track_idx`'s `chunk`-wide slab along `dim`, zero-padded if it runs past the end."""
    size = weight.shape[dim]
    start = min(track_idx * chunk, size)
    got = weight.narrow(dim, start, min(chunk, size - start))
    if got.shape[dim] == chunk:
        return got.contiguous()
    tail = list(got.shape)
    tail[dim] = chunk - got.shape[dim]
    return torch.cat([got, got.new_zeros(tail)], dim=dim).contiguous()


@dataclass(frozen=True)
class Colwise:
    """Split the output dimension (dim 0 of nn.Linear.weight) evenly across tracks.

    Sums back via `torch.cat` along dim 0. The output activations of each
    track represent a *disjoint subset* of output features; the next op
    (typically a row-parallel op) must consume them appropriately.

    ``pad_full_size`` (the dense size along ``dim``) lifts the divisibility
    requirement: slices round up and the overhang is zero. For a SwiGLU MLP that
    is exact — a zero row in gate_proj gives silu(0)*up = 0, so the padded lane
    contributes exactly 0.0 to the down_proj sum.
    """

    dim: int = 0  # output dim of nn.Linear weight is 0
    pad_full_size: int | None = None

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        chunk = _even_chunk(weight.shape[self.dim], n_tracks, self.pad_full_size, "Colwise")
        return _chunk_slice(weight, self.dim, track_idx, chunk)

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        out = torch.cat(slices, dim=self.dim)
        if self.pad_full_size is not None:
            out = out.narrow(self.dim, 0, self.pad_full_size)
        return out.contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = _even_chunk(out[self.dim], n_tracks, self.pad_full_size, "Colwise")
        return tuple(out)


@dataclass(frozen=True)
class Rowwise:
    """Split the input dimension (dim 1 of nn.Linear.weight) evenly across tracks.

    Each track's slice produces a *partial sum* of the output; the all-reduce
    at the next sync point completes the addition. ``pad_full_size`` works as in
    `Colwise` (zero input columns contribute nothing to the partial sum).
    """

    dim: int = 1  # input dim of nn.Linear weight is 1
    pad_full_size: int | None = None

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        chunk = _even_chunk(weight.shape[self.dim], n_tracks, self.pad_full_size, "Rowwise")
        return _chunk_slice(weight, self.dim, track_idx, chunk)

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        out = torch.cat(slices, dim=self.dim)
        if self.pad_full_size is not None:
            out = out.narrow(self.dim, 0, self.pad_full_size)
        return out.contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = _even_chunk(out[self.dim], n_tracks, self.pad_full_size, "Rowwise")
        return tuple(out)


@dataclass(frozen=True)
class PerHead:
    """Slice a 1-D per-head parameter (A_log, dt_bias) across tracks."""

    dim: int = 0
    num_heads: int | None = None  # asserted equal to size along dim if provided

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        size = weight.shape[self.dim]
        if self.num_heads is not None and self.num_heads != size:
            raise ValueError(f"PerHead expected dim {self.dim} == {self.num_heads}, got {size}")
        if size % n_tracks != 0:
            raise ValueError(f"PerHead size {size} not divisible by n_tracks {n_tracks}")
        chunk = size // n_tracks
        return weight.narrow(self.dim, track_idx * chunk, chunk).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = out[self.dim] // n_tracks
        return tuple(out)


@dataclass(frozen=True)
class Replicated:
    """Replicated parameter: every track holds the same full tensor at conversion.

    Used for norms whose statistic is per-head-dim (RMSNorm on head_dim) and
    for final norms that operate on the synced hidden state.

    ``sync`` controls whether the copies are kept bit-identical during training.
    When True (default) they form one replication group and their gradients are
    averaged each step; when False each copy is its own singleton group and is
    free to diverge (e.g. per-track attention head norms).
    """

    sync: bool = True

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        return weight.detach().clone()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return slices[0].detach().clone()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        return tuple(full_shape)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        if self.sync or force_sync:
            # Every track holds the same tensor; one group spanning all tracks.
            return [list(range(n_tracks))]
        # Diverge: each copy trains independently.
        return [[t] for t in range(n_tracks)]


@dataclass(frozen=True)
class OwnerOnly:
    """Parameter that lives on exactly one track. Other tracks omit it entirely.

    Used for `embed_tokens.weight` and `lm_head.weight`: the owner track does
    the embedding lookup and the final logits projection; the embedding output
    is broadcast to peer tracks via the existing all-reduce sync primitive
    (zero-padded delta), and the synced post-final-norm hidden state is local
    to every track so the owner can apply lm_head without further communication.

    `slice` returns `None` on non-owner tracks; the slicer engine interprets
    that as "skip this key" so the per-track state_dict simply omits it.
    """

    owner_track: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor | None:
        if track_idx != self.owner_track:
            return None
        return weight.detach().clone()

    def reassemble(self, slices: list[torch.Tensor | None]) -> torch.Tensor:
        for s in slices:
            if s is not None:
                return s.detach().clone()
        raise ValueError("OwnerOnly.reassemble got no non-None slices")

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        return tuple(full_shape)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        # The param lives on exactly one track. No gradient sync needed.
        # `force_sync` is accepted for signature uniformity and ignored.
        return [[self.owner_track]]


@dataclass(frozen=True)
class FusedSegmentColwise:
    """Slice an nn.Linear whose output is a concatenation of N logical segments.

    For Qwen3.5 `in_proj_qkv`: out_features = [Q (key_dim) | K (key_dim) | V (value_dim)].
    Each segment must be sliced colwise *independently* then re-concatenated in
    segment order. Used wherever the dense layer fuses multiple per-head
    projections into a single weight matrix.

    `segments` is a list of segment sizes along `dim`. They must sum to
    `weight.shape[dim]`. Each segment size must be divisible by n_tracks.
    """

    segments: tuple[int, ...]
    dim: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        total = sum(self.segments)
        if total != weight.shape[self.dim]:
            raise ValueError(
                f"FusedSegmentColwise: segments sum {total} != weight dim {weight.shape[self.dim]}"
            )
        per_track_segs: list[torch.Tensor] = []
        offset = 0
        for seg in self.segments:
            if seg % n_tracks != 0:
                raise ValueError(f"FusedSegmentColwise: segment {seg} not divisible by n_tracks {n_tracks}")
            chunk = seg // n_tracks
            per_track_segs.append(
                weight.narrow(self.dim, offset + track_idx * chunk, chunk).contiguous()
            )
            offset += seg
        return torch.cat(per_track_segs, dim=self.dim).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        # slices is one tensor per track, each already fused [Qslice|Kslice|Vslice].
        # To reassemble we split each track's slice back into segments, then
        # concatenate per-segment across tracks, then re-fuse.
        n_tracks = len(slices)
        track0 = slices[0]
        per_track_chunks = [s // n_tracks for s in self.segments]  # each track holds these per-segment sizes
        # split each track-tensor back into its (per-segment) chunks
        split_per_track = [
            torch.split(s, per_track_chunks, dim=self.dim) for s in slices
        ]
        # gather per segment
        segments_concat = []
        for seg_idx in range(len(self.segments)):
            segments_concat.append(
                torch.cat([split_per_track[t][seg_idx] for t in range(n_tracks)], dim=self.dim)
            )
        return torch.cat(segments_concat, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = out[self.dim] // n_tracks
        return tuple(out)


@dataclass(frozen=True)
class GDNFusedQKV:
    """Slice a GatedDeltaNet `[Q (key_dim) | K (key_dim) | V (value_dim)]` slab by *value* head.

    GDN is GQA-shaped over its value heads: `ratio = num_v_heads // num_k_heads`
    v-heads share one k-head, and the forward repeat-interleaves q/k by that ratio.
    The parallel unit is therefore the v-head — track `t` owns v-heads
    `[t*vpt, (t+1)*vpt)` and needs whichever k-heads those v-heads read
    (`k = v // ratio`).

    Two regimes, both exact:

      - `num_k_heads % n_tracks == 0`: each track's k-heads are one contiguous
        block, emitted **compactly** (`num_k_heads // n_tracks` of them, dense
        ratio preserved). Byte-identical to a plain per-segment colwise split.
      - otherwise: the block straddles k-heads unevenly (e.g. 16 k / 48 v over 24
        tracks gives 1 or 2 k-heads depending on the track), and per-track shapes
        must stay uniform for the batched forward. So every track gets ONE k-head
        copy per v-head — per-track ratio 1, which the forward handles natively
        (it only repeat-interleaves when the ratio exceeds 1). Costs a `ratio`x
        replication of Q/K.

    Deliberately has no ``replication_groups``: a track's slab always carries its
    own unique V segment, so no two tracks hold an identical tensor. Duplicated
    k-heads are free to diverge under training, like full-attention `k_proj`.
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    dim: int = 0

    def _k_head_indices(self, track_idx: int, n_tracks: int) -> list[int]:
        if self.num_v_heads % n_tracks != 0:
            raise ValueError(
                f"GDNFusedQKV: num_v_heads {self.num_v_heads} not divisible by n_tracks {n_tracks}"
            )
        if self.num_k_heads % n_tracks == 0:
            per_track = self.num_k_heads // n_tracks
            return list(range(track_idx * per_track, (track_idx + 1) * per_track))
        vpt = self.num_v_heads // n_tracks
        ratio = self.num_v_heads // self.num_k_heads
        return [(track_idx * vpt + j) // ratio for j in range(vpt)]

    def _dims(self) -> tuple[int, int]:
        return self.num_k_heads * self.head_k_dim, self.num_v_heads * self.head_v_dim

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        key_dim, value_dim = self._dims()
        if weight.shape[self.dim] != 2 * key_dim + value_dim:
            raise ValueError(
                f"GDNFusedQKV: expected dim {self.dim} = {2 * key_dim + value_dim}, "
                f"got {weight.shape[self.dim]}"
            )
        k_idx = self._k_head_indices(track_idx, n_tracks)
        v_chunk = (self.num_v_heads // n_tracks) * self.head_v_dim
        parts = [
            weight.narrow(self.dim, base + i * self.head_k_dim, self.head_k_dim)
            for base in (0, key_dim)
            for i in k_idx
        ]
        parts.append(weight.narrow(self.dim, 2 * key_dim + track_idx * v_chunk, v_chunk))
        return torch.cat(parts, dim=self.dim).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        # Q/K are de-duplicated (a k-head may be held by several tracks); V concatenates.
        n_tracks = len(slices)
        v_chunk = (self.num_v_heads // n_tracks) * self.head_v_dim
        q: dict[int, torch.Tensor] = {}
        k: dict[int, torch.Tensor] = {}
        v: list[torch.Tensor] = []
        for t, s in enumerate(slices):
            k_idx = self._k_head_indices(t, n_tracks)
            n_k = len(k_idx)
            for j, i in enumerate(k_idx):
                q[i] = s.narrow(self.dim, j * self.head_k_dim, self.head_k_dim)
                k[i] = s.narrow(self.dim, (n_k + j) * self.head_k_dim, self.head_k_dim)
            v.append(s.narrow(self.dim, 2 * n_k * self.head_k_dim, v_chunk))
        order = sorted(q)
        return torch.cat([q[i] for i in order] + [k[i] for i in order] + v, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        n_k = len(self._k_head_indices(0, n_tracks))
        out[self.dim] = (
            2 * n_k * self.head_k_dim + (self.num_v_heads // n_tracks) * self.head_v_dim
        )
        return tuple(out)


@dataclass(frozen=True)
class KVReplicatedColwise:
    """Slice an out-dim that is per-kv-head, with replication across tracks in a kv-group.

    Used for `k_proj.weight` and `v_proj.weight` in GQA full-attention when we
    push N above num_kv_heads. Track `t` belongs to kv-group
    `g = t // (n_tracks // num_kv_heads)`; it stores rows
    `[g*chunk : (g+1)*chunk]` where `chunk = weight.shape[dim] // num_kv_heads`.
    Every track in the same kv-group therefore holds an *identical* slice.

    Constraints:
      - n_tracks must be a multiple of num_kv_heads (so the kv-group factor
        `tracks_per_kv_group = n_tracks // num_kv_heads` is integer).
      - weight.shape[dim] must be divisible by num_kv_heads.

    `reassemble` returns the concatenation of the *unique* per-group slices
    (one per kv-group), reconstructing the dense tensor. Caller is expected
    to either (a) drop duplicate tracks within each group before calling, or
    (b) pass exactly one slice per kv-group.

    ``sync`` controls whether the within-group copies are kept bit-identical
    during training. When True (legacy) the copies form one replication group
    per kv-group and their gradients are averaged each step, exactly reproducing
    dense GQA. When False the copies are free to diverge — each track gets its
    own KV head (GQA → per-track MHA), a capacity superset that distillation can
    exploit. Slicing/reassembly are unaffected by this flag; only the
    training-time replication grouping changes.
    """

    num_kv_heads: int
    dim: int = 0
    sync: bool = True

    def _chunk(self, weight_shape: tuple[int, ...]) -> int:
        size = weight_shape[self.dim]
        if size % self.num_kv_heads != 0:
            raise ValueError(
                f"KVReplicatedColwise: dim {self.dim} size {size} not divisible by "
                f"num_kv_heads {self.num_kv_heads}"
            )
        return size // self.num_kv_heads

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        if n_tracks % self.num_kv_heads != 0:
            raise ValueError(
                f"KVReplicatedColwise: n_tracks {n_tracks} must be a multiple of "
                f"num_kv_heads {self.num_kv_heads}"
            )
        chunk = self._chunk(tuple(weight.shape))
        tracks_per_group = n_tracks // self.num_kv_heads
        g = track_idx // tracks_per_group
        return weight.narrow(self.dim, g * chunk, chunk).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        # `slices` is one unique slice per kv-group (length == num_kv_heads).
        if len(slices) != self.num_kv_heads:
            raise ValueError(
                f"KVReplicatedColwise.reassemble expects exactly {self.num_kv_heads} "
                f"unique slices (one per kv-group), got {len(slices)}"
            )
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        # Every track in the same kv-group sees the same shape; n_tracks is
        # accepted for interface uniformity but not used.
        out = list(full_shape)
        out[self.dim] = full_shape[self.dim] // self.num_kv_heads
        return tuple(out)

    def replication_groups(self, n_tracks: int, force_sync: bool = False) -> list[list[int]]:
        if n_tracks % self.num_kv_heads != 0:
            raise ValueError(
                f"KVReplicatedColwise.replication_groups: n_tracks {n_tracks} "
                f"must be a multiple of num_kv_heads {self.num_kv_heads}"
            )
        tpg = n_tracks // self.num_kv_heads
        if self.sync or force_sync:
            # Keep each kv-group's copies bit-identical (dense GQA).
            return [[g * tpg + i for i in range(tpg)] for g in range(self.num_kv_heads)]
        # Diverge: every track's KV head trains independently.
        return [[t] for t in range(n_tracks)]


@dataclass(frozen=True)
class GatedQColwise:
    """Slice Qwen3.5's doubled `q_proj` whose output carries [q_h0, gate_h0, q_h1, gate_h1, ...]
    interleaved per head (size: num_heads * 2 * head_dim along dim 0).

    For each head the q-half and gate-half are interleaved (the view+chunk in
    forward() reshapes to (-1, head_dim*2) then chunks 2 along the last dim).
    So per head the layout is contiguous: head_i occupies a block of
    `2*head_dim` rows starting at `i * 2 * head_dim`.

    To slice per-track we keep complete heads. With n_tracks dividing num_heads,
    track t gets heads [t*heads_per_track : (t+1)*heads_per_track], which is a
    contiguous slab of rows of size `heads_per_track * 2 * head_dim`.
    """

    num_heads: int
    head_dim: int
    dim: int = 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        expected = self.num_heads * 2 * self.head_dim
        if weight.shape[self.dim] != expected:
            raise ValueError(
                f"GatedQColwise: expected dim {self.dim} = {expected}, got {weight.shape[self.dim]}"
            )
        if self.num_heads % n_tracks != 0:
            raise ValueError(f"GatedQColwise: num_heads {self.num_heads} not divisible by n_tracks {n_tracks}")
        heads_per_track = self.num_heads // n_tracks
        block = heads_per_track * 2 * self.head_dim
        return weight.narrow(self.dim, track_idx * block, block).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = (self.num_heads // n_tracks) * 2 * self.head_dim
        return tuple(out)


# Mapping from a fully-qualified parameter sub-path inside a decoder layer to
# its slicer spec. A `LayerSpec` is a dict that callers use to slice every
# parameter in one layer.
LayerSpec = dict[str, "SlicerSpec"]


def build_decoder_layer_specs(
    text_cfg,
    layer_type: str,
    *,
    full_attention_specs,
    linear_attention_specs,
    mlp_specs,
) -> LayerSpec:
    """Assemble one decoder layer's specs: replicated norms + the attention specs
    for `layer_type` (prefixed `self_attn.`/`linear_attn.`) + the MLP specs
    (prefixed `mlp.`). Model families supply the three spec functions; this shape
    is identical across dense and MoE.
    """
    out: LayerSpec = {
        "input_layernorm.weight": Replicated(),
        "post_attention_layernorm.weight": Replicated(),
    }
    if layer_type == "full_attention":
        prefix, specs = "self_attn", full_attention_specs(text_cfg)
    elif layer_type == "linear_attention":
        prefix, specs = "linear_attn", linear_attention_specs(text_cfg)
    else:
        raise ValueError(f"Unknown layer_type {layer_type!r}")
    for k, v in specs.items():
        out[f"{prefix}.{k}"] = v
    for k, v in mlp_specs(text_cfg).items():
        out[f"mlp.{k}"] = v
    return out
