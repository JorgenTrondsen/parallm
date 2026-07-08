"""Per-slab low-rank reparameterization of the dense model's sliced Linears.

Gate for the co-trained low-rank replica lever: if every track's weight bands
are *exactly* rank-r (trained U·V factors), then each rank's "degraded copies
of the other 15 tracks" are exact by construction, and the replica forward at
ANY sync count collapses to the full-sync forward — which for slice-partitioned
weights is just the dense forward with per-slab assembled weights. So the
experiment needs no PT/sync machinery: factor each sliced Linear of the DENSE
model into per-slab U·V (slab = one track's band under the slicer specs in
``slicer/qwen3_5.py``), SVD-init from the dense weights, and heal with the
dense trainer (``train/heal_dense.py``, mode ``serial``). Downstream on the
healed model = the deployment quality of exact-copy replica at any D.

Slab geometry mirrors the conversion exactly (``slabs_for_spec`` dispatches on
the SlicerSpec types), so the factor sets ARE the per-track copies:
- Colwise/Rowwise (MLP gate/up/down, o_proj, GDN out/z): n_tracks equal bands.
- GatedQColwise (full-attn q|gate): one contiguous 2*head_dim band per head.
- KVReplicatedColwise (k/v): one band per kv-GROUP (shared by its tracks).
- FusedSegmentColwise (GDN qkv, conv): per segment × per track bands (finer
  than per-track — each band is still a contiguous slab of the dense weight).

A band is factored only when ``rank < min(band.shape)``; bands at or below the
rank (e.g. GDN Q/K bands, 128×4096, at r=128) stay dense parameters — exact
and still trainable. Modules where NO band factors (e.g. in_proj_b/a, 2-row
bands) are left untouched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pt_converter.slicer.base import (
    Colwise,
    FusedSegmentColwise,
    GatedQColwise,
    KVReplicatedColwise,
    Rowwise,
)


def slabs_for_spec(spec, shape: "tuple[int, ...]", n_tracks: int):
    """Contiguous ``(start, size)`` bands along the spec's sliced dim, ordered
    and covering it completely; ``None`` for specs that don't slab-partition
    (Replicated / PerHead / OwnerOnly)."""
    if isinstance(spec, FusedSegmentColwise):
        slabs, offset = [], 0
        for seg in spec.segments:
            chunk = seg // n_tracks
            slabs.extend((offset + t * chunk, chunk) for t in range(n_tracks))
            offset += seg
        return spec.dim, slabs
    if isinstance(spec, GatedQColwise):
        block = (spec.num_heads // n_tracks) * 2 * spec.head_dim
        return spec.dim, [(t * block, block) for t in range(n_tracks)]
    if isinstance(spec, KVReplicatedColwise):
        chunk = shape[spec.dim] // spec.num_kv_heads
        return spec.dim, [(g * chunk, chunk) for g in range(spec.num_kv_heads)]
    if isinstance(spec, (Colwise, Rowwise)):
        chunk = shape[spec.dim] // n_tracks
        return spec.dim, [(t * chunk, chunk) for t in range(n_tracks)]
    return None


def svd_factor(band: torch.Tensor, rank: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Best rank-``rank`` factorization of ``band`` (fp32 SVD), returned as
    balanced factors ``A @ B`` with the singular values split ``sqrt``-wise."""
    w = band.detach().float()
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    sq = S[:rank].clamp(min=0).sqrt()
    A = (U[:, :rank] * sq[None, :]).to(band.dtype)
    B = (sq[:, None] * Vh[:rank, :]).to(band.dtype)
    return A.contiguous(), B.contiguous()


class _FactorBand(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def materialize(self) -> torch.Tensor:
        return self.A @ self.B


class _DenseBand(nn.Module):
    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.W = nn.Parameter(W)

    def materialize(self) -> torch.Tensor:
        return self.W


class LowRankSlabLinear(nn.Module):
    """Drop-in for ``nn.Linear``: the weight is assembled per forward by
    concatenating per-slab bands (factored ``A @ B`` or dense) along the
    sliced dim. Assembly cost is negligible next to the main GEMM."""

    def __init__(self, dim: int, bands: "list[nn.Module]", in_features: int,
                 out_features: int, bias: "torch.Tensor | None" = None):
        super().__init__()
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.bands = nn.ModuleList(bands)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, dim: int,
                    slabs: "list[tuple[int, int]]", rank: int):
        """Factor ``linear`` per slab at ``rank``; returns ``None`` when no
        band is factorable (caller keeps the original module)."""
        W = linear.weight.data
        pos = 0
        for start, size in slabs:
            if start != pos:
                raise ValueError(f"slabs not contiguous at {start} (expected {pos})")
            pos += size
        if pos != W.shape[dim]:
            raise ValueError(f"slabs cover {pos} of dim {dim} size {W.shape[dim]}")
        bands: "list[nn.Module]" = []
        any_factored = False
        for start, size in slabs:
            band = W.narrow(dim, start, size).contiguous()
            if rank < min(band.shape):
                bands.append(_FactorBand(*svd_factor(band, rank)))
                any_factored = True
            else:
                bands.append(_DenseBand(band.clone()))
        if not any_factored:
            return None
        bias = None if linear.bias is None else linear.bias.data.clone()
        return cls(dim, bands, linear.in_features, linear.out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.cat([b.materialize() for b in self.bands], dim=self.dim)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        nf = sum(isinstance(b, _FactorBand) for b in self.bands)
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"dim={self.dim}, bands={len(self.bands)} ({nf} factored)")


def apply_lowrank_slicing(text_model, text_cfg, rank: int, n_tracks: int = 16) -> dict:
    """Replace every sliceable ``nn.Linear`` in ``text_model.layers`` with its
    per-slab factored form at ``rank`` (SVD init from the dense weights).
    Returns replacement stats. Embeddings / norms / lm_head are untouched."""
    from pt_converter.slicer.qwen3_5 import decoder_layer_specs

    stats = {"modules": 0, "bands_factored": 0, "bands_dense": 0,
             "params_before": 0, "params_after": 0}
    for idx, layer in enumerate(text_model.layers):
        specs = decoder_layer_specs(text_cfg, text_cfg.layer_types[idx])
        for path, spec in specs.items():
            if not path.endswith(".weight"):
                continue
            mod_path = path[: -len(".weight")]
            try:
                mod = layer.get_submodule(mod_path)
            except AttributeError:
                continue
            if not isinstance(mod, nn.Linear):
                continue
            res = slabs_for_spec(spec, tuple(mod.weight.shape), n_tracks)
            if res is None:
                continue
            dim, slabs = res
            new = LowRankSlabLinear.from_linear(mod, dim, slabs, rank)
            if new is None:
                continue
            parent_path, _, attr = mod_path.rpartition(".")
            parent = layer.get_submodule(parent_path) if parent_path else layer
            stats["modules"] += 1
            stats["bands_factored"] += sum(isinstance(b, _FactorBand) for b in new.bands)
            stats["bands_dense"] += sum(isinstance(b, _DenseBand) for b in new.bands)
            stats["params_before"] += sum(p.numel() for p in mod.parameters())
            stats["params_after"] += sum(p.numel() for p in new.parameters())
            setattr(parent, attr, new)
    return stats


def assembled_state_dict(model: nn.Module, sd: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    """Fold each ``LowRankSlabLinear``'s band tensors in ``sd`` back into a
    plain ``<module>.weight``, so the result loads into the stock HF model.
    Band products run in fp32 then cast back (matches ``svd_factor``'s framing:
    the assembled weight IS the deployed model)."""
    out = dict(sd)
    for name, mod in model.named_modules():
        if not isinstance(mod, LowRankSlabLinear):
            continue
        bands = []
        for i, band in enumerate(mod.bands):
            if isinstance(band, _FactorBand):
                A = out.pop(f"{name}.bands.{i}.A")
                B = out.pop(f"{name}.bands.{i}.B")
                bands.append((A.float() @ B.float()).to(A.dtype))
            else:
                bands.append(out.pop(f"{name}.bands.{i}.W"))
        out[f"{name}.weight"] = torch.cat(bands, dim=mod.dim).contiguous()
    return out
