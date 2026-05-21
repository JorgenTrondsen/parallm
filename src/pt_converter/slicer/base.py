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
    """Protocol for a slicing rule. Each spec knows how to slice and reassemble."""

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor: ...

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor: ...

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class Colwise:
    """Split the output dimension (dim 0 of nn.Linear.weight) evenly across tracks.

    Sums back via `torch.cat` along dim 0. The output activations of each
    track represent a *disjoint subset* of output features; the next op
    (typically a row-parallel op) must consume them appropriately.
    """

    dim: int = 0  # output dim of nn.Linear weight is 0

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        size = weight.shape[self.dim]
        if size % n_tracks != 0:
            raise ValueError(f"Colwise: dim {self.dim} size {size} not divisible by n_tracks {n_tracks}")
        chunk = size // n_tracks
        return weight.narrow(self.dim, track_idx * chunk, chunk).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = out[self.dim] // n_tracks
        return tuple(out)


@dataclass(frozen=True)
class Rowwise:
    """Split the input dimension (dim 1 of nn.Linear.weight) evenly across tracks.

    Each track's slice produces a *partial sum* of the output; the all-reduce
    at the next sync point completes the addition.
    """

    dim: int = 1  # input dim of nn.Linear weight is 1

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        size = weight.shape[self.dim]
        if size % n_tracks != 0:
            raise ValueError(f"Rowwise: dim {self.dim} size {size} not divisible by n_tracks {n_tracks}")
        chunk = size // n_tracks
        return weight.narrow(self.dim, track_idx * chunk, chunk).contiguous()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(slices, dim=self.dim).contiguous()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        out = list(full_shape)
        out[self.dim] = out[self.dim] // n_tracks
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
    """Replicated parameter: every track holds the same full tensor.

    Used for norms whose statistic is per-head-dim (RMSNorm on head_dim) and
    for embeddings / final norms outside the sync regime.
    """

    def slice(self, weight: torch.Tensor, track_idx: int, n_tracks: int) -> torch.Tensor:
        return weight.detach().clone()

    def reassemble(self, slices: list[torch.Tensor]) -> torch.Tensor:
        return slices[0].detach().clone()

    def per_track_shape(self, full_shape: tuple[int, ...], n_tracks: int) -> tuple[int, ...]:
        return tuple(full_shape)


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
