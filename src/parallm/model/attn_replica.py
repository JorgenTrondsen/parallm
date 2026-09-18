"""Attention replica: ONE compressed copy of a layer's whole attention, held by every track.

At a layer with no post-attn sync whose predecessor syncs post-MLP, every track holds the
same synced ``R``, so the dense attention output ``A(R)`` is a function of ``R`` and the
weights alone — a node holding a copy computes it with NO collective. Every track's MLP then
reads ``R + Â``, IDENTICAL on every track, so the column-sliced MLP shards sum to exactly
``MLP_dense(LN(R + Â))``: plain SPD's per-track angle defect is gone and one shared
compression error is left. Each track still runs its own attention slice and carries it into
the post-MLP sync, so the residual receives the exact ``A`` and the replica's error never
compounds. `PTWrappedModel.set_attn_replica` installs it.

⚡ **TRUNCATION, NEVER A FIT.** `truncate` projects a projection's TRUE output onto the top
eigenvectors of that output's second moment on real activations — a shrinkage estimate.
On the MLP sum, a fitted map at BETTER relMSE bought 0.2% of the prize where a truncation
bought 82%. The bases come from `scripts/calib_attn_replica.py`.

⚠ **q and k are truncated in a PER-HEAD NORMALIZED metric** (`head_scale`). ``q_norm`` /
``k_norm`` strip each head's scale, so plain L2 would spend the rank on loud heads and hand
a quiet head's normalized query back as noise.

⚡ **Any projection can be PRUNED instead of truncated** (``k:s50``, ``o:s75``): Wanda keeps
each row's top ``|w|·‖x_j‖`` weights exact, with ``‖x_j‖`` the calibration's per-channel input
RMS — ``x_rms`` (the normed ``R`` q, k and v read) or ``o_rms`` (the head outputs o reads). A
rank cut pays for both sides of the matrix; a pruned weight costs one bitmap bit.

⚡ **Rank and density stack** (``q:1024/s50``): cut the rank, THEN prune both factors — ``B`` by the
projection's input RMS, ``A`` by the latent RMS ``z_rms`` (√eigenvalue of each basis column). Never
the other way round: the rank-r factors of a pruned matrix are dense, so its pruning buys nothing.

⚡⚡ **``o`` is FOLDED into each track's own MLP slices** (``o:fold,norm:r32+c``). The replica's
``Â = W_o a`` reaches nothing but this track's ``gate_t``/``up_t`` through one RMSNorm, and RMSNorm
is a per-token scalar times ``γ``: ``gate_t(norm(R + W_o a)) = [gate_t(γ⊙R) +
(gate_t·diag(γ)·W_o)·a] / rms(R + W_o a)``. The folded ``[I/N, q_dim]`` matrices are built from the
track's OWN slices and the teacher's EXACT ``W_o`` (`bind_fold`), so the o stage is exact and the
replica stores no ``o_proj`` — only ``a``, the concatenated heads, leaves the attention. The
compression ratio is ``H/(2·I/N)`` and it grows with N. What is left of ``o`` is the scalar:
``norm:r{RANK}`` keeps a rank-RANK sketch of ``W_o`` (the same truncation) for
``rms(R + Â_sketch)``, ``norm:r0`` uses ``rms(R)``, ``+c`` adds the sizing's per-layer tail-energy
constant under the root, and ``norm:exact`` keeps ``W_o`` whole (the rail). ``o:fold/sPCT``
Wanda-prunes the folded matrices by ``o_rms``. The read is a `seam.FoldRead`.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.nn as nn

from parallm.model.replica import wanda_prune_weight
from parallm.model.replica_pack import _bits_per_weight
from parallm.model.seam import FoldRead

PROJS = ("q", "k", "v", "o")
# Each track also holds its own kv head whole, so an EXACT replica k/v holds that head twice.
KV = ("k", "v")
EPS = 1e-12
# The norm-sketch ranks `scripts/calib_attn_replica.py --norm-c` fits a constant for.
NORM_LADDER = (0, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class FoldSpec:
    """``o:fold[/sPCT]`` + ``norm:X``. ``frac`` prunes the folded gate/up (None = dense).
    ``norm`` names the scalar's estimator: ``exact`` (``W_o`` whole, the rail) or ``r{RANK}``
    (a rank sketch; ``r0`` = ``rms(R)``), either optionally ``+c`` (bias-corrected)."""

    frac: "float | None"
    norm: str
    rank: int = 0
    bias: bool = False

    @property
    def estimator(self) -> str:
        """The sizing's name for the estimator, without ``+c`` — the `norm_c_key` it reads."""
        return self.norm.removesuffix("+c")


def _parse_norm(r: str) -> "tuple[str, int, bool] | None":
    """``(canonical name, rank, bias)`` of a ``norm:`` value, or None when malformed."""
    if r == "exact":
        return ("exact", 0, False)
    name, bias = (r[:-2], True) if r.endswith("+c") else (r, False)
    if name[:1] != "r" or not name[1:].isdigit():
        return None
    return (f"r{int(name[1:])}" + ("+c" if bias else ""), int(name[1:]), bias)


def _parse(spec: str) -> "tuple[dict, dict, FoldSpec | None]":
    """``(ranks, fracs, fold)``: ``q:256,k:s50,o:1024/s60`` → ``({"q": 256, "o": 1024},
    {"k": 0.5, "o": 0.6}, None)``. A projection in both is cut to its rank, then its factors
    pruned. ``o:fold[/sPCT]`` with ``norm:X`` gives a `FoldSpec` instead of an o entry."""
    if spec == "exact":
        return {}, {}, None
    ranks: dict = {}
    fracs: dict = {}
    fold: "tuple | None" = None
    norm: "tuple | None" = None
    err = ValueError(
        f"attn replica spec {spec!r}: expected `exact` or a comma list of q|k|v|o:RANK, "
        f":sPCT (PCT% pruned), :RANK/sPCT (factors pruned), o:fold[/sPCT] with "
        f"norm:exact|r<RANK>[+c], naming each projection at most once")
    for part in spec.split(","):
        name, _, r = part.strip().partition(":")
        if name == "norm":
            if norm is not None or (norm := _parse_norm(r)) is None:
                raise err
            continue
        rank, slash, s = r.partition("/")
        if name == "o" and rank == "fold":
            ok_frac = s[:1] == "s" and s[1:].isdigit() and 0 < int(s[1:]) < 100
            if fold is not None or "o" in ranks or "o" in fracs or (slash and not ok_frac):
                raise err
            fold = (int(s[1:]) / 100 if slash else None,)
            continue
        if not slash and rank[:1] == "s":
            rank, s = "", rank
        ok_rank = rank.isdigit() and int(rank) >= 1
        ok_frac = s[:1] == "s" and s[1:].isdigit() and 0 < int(s[1:]) < 100
        valid = (ok_rank and ok_frac) if slash else (ok_rank if rank else ok_frac)
        if (name not in PROJS or name in ranks or name in fracs or not valid
                or (name == "o" and fold is not None)):
            raise err
        if rank:
            ranks[name] = int(rank)
        if s:
            fracs[name] = int(s[1:]) / 100
    if (fold is None) != (norm is None):
        raise ValueError(f"attn replica spec {spec!r}: o:fold and norm:… go together — the fold "
                         f"keeps no o_proj, so the norm's scalar estimator must be named")
    return ranks, fracs, None if fold is None else FoldSpec(fold[0], *norm)


def fold_spec(spec: str) -> "FoldSpec | None":
    """The `FoldSpec` of an ``o:fold`` spec, else None."""
    return _parse(spec)[2]


def _head_dim(cfg) -> int:
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def _shapes(cfg) -> dict:
    """``name -> (out, in)`` of the DENSE projections."""
    H, hd = cfg.hidden_size, _head_dim(cfg)
    q, kv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    return {"q": (q, H), "k": (kv, H), "v": (kv, H), "o": (H, q)}


# ----- the truncation -----

def head_scale(W: torch.Tensor, C: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Per-row metric scale: every row of head ``h`` gets ``(E‖W_h x‖² / head_dim)^-½``,
    ``C = E[x xᵀ]``. Equalizes the heads the way ``q_norm``/``k_norm`` will."""
    ms = ((W @ C) * W).sum(1)
    return ms.view(-1, head_dim).mean(1).clamp(min=EPS).rsqrt().repeat_interleave(head_dim)


def input_rms(C: torch.Tensor) -> torch.Tensor:
    """``sqrt(diag C)``, ``C = E[x xᵀ]``: the per-channel ``‖x_j‖`` Wanda scores a pruned
    weight by (its scale drops out of a per-row ranking)."""
    return C.diagonal().clamp(min=0).sqrt()


def rms_key(name: str) -> str:
    """The input norm a pruned ``name`` is scored with: q, k and v read the normed ``R``
    (``x_rms``); o reads the concatenated head outputs (``o_rms``)."""
    return "o_rms" if name == "o" else "x_rms"


def output_moment(W: torch.Tensor, C: torch.Tensor, d: "torch.Tensor | None" = None):
    """``D W C Wᵀ D`` — the second moment of the projection's output, in the metric."""
    DW = W if d is None else d[:, None] * W
    return DW @ C @ DW.T


def top_basis(M: torch.Tensor, r_max: int):
    """``(U, evals)``: the top ``r_max`` eigenvectors of a PSD moment as columns, and all
    of its eigenvalues, both descending."""
    evals, evecs = torch.linalg.eigh(M)
    r = min(r_max, M.shape[0])
    return evecs[:, -r:].flip(1), evals.flip(0).clamp(min=0)


def truncate(W: torch.Tensor, U: torch.Tensor, r: int, d: "torch.Tensor | None" = None):
    """``(B, A)`` with ``A @ B = D⁻¹ U_r U_rᵀ D W``: the true output projected onto the
    moment's top ``r`` directions. ``B`` is ``[r, in]``, ``A`` is ``[out, r]``."""
    if r > U.shape[1]:
        raise ValueError(f"rank {r} exceeds the {U.shape[1]} basis vectors available")
    Ur = U[:, :r]
    B = Ur.T @ (W if d is None else d[:, None] * W)
    A = Ur if d is None else Ur / d[:, None]
    return B, A


def norm_c_key(li: int, estimator: str) -> str:
    """The bases key of a layer's bias constant for a scalar estimator, as the sizing
    writes it (``norm_c.safetensors``)."""
    return f"layers.{li}.norm_c.{estimator}"


def replica_memory(cfg, spec: str, n_layers: int, model_bytes: int = 0,
                   n_tracks: "int | None" = None) -> dict:
    """Weights and KV cache (bf16) a replica over ``n_layers`` adds to a node.

    ``kv_per_token`` caches exact k/v. ``kv_per_token_latent`` is what a cache holding
    the rank-r k/v latents would need — priced here, not built. A pruned projection is
    priced packed: a survivor bitmap plus bf16 survivors, ``1 + 16(1−f)`` bits/weight.

    ``gib_net`` is what the node adds beyond its own track: the track already stores its kv
    head bit-exact, so an EXACT replica k/v need not hold that head twice. It stops holding
    the moment the replica's k/v differ from the track's — truncated or pruned.

    A folded o (``o:fold``) is priced per NODE: two ``[I/n_tracks, q_dim]`` matrices (packed
    when pruned) in ``gib_fold``, plus the norm sketch in ``gib_sketch`` — rank·(H + q_dim)
    dense, the whole ``W_o`` for ``norm:exact``, nothing for ``norm:r0``.
    """
    ranks, fracs, fold = _parse(spec)
    bits = own = fold_bits = sketch_bits = 0.0
    for n, (m, i) in _shapes(cfg).items():
        if n == "o" and fold is not None:
            if n_tracks is None:
                raise ValueError("pricing o:fold needs n_tracks: the fold is [I/N, q_dim] per track")
            per = _bits_per_weight(fold.frac, None) if fold.frac is not None else 16
            fold_bits = per * 2 * (cfg.intermediate_size // n_tracks) * i
            sketch_bits = 16 * (m * i if fold.norm == "exact" else fold.rank * (m + i))
        elif n in ranks:
            per = _bits_per_weight(fracs[n], None) if n in fracs else 16
            bits += per * ranks[n] * (m + i)
        elif n in fracs:
            bits += _bits_per_weight(fracs[n], None) * m * i
        else:
            bits += 16 * m * i
            if n in KV:
                own += 16 * (m // cfg.num_key_value_heads) * i
    bits += 16 * (cfg.hidden_size + 2 * _head_dim(cfg))  # input_layernorm, q_norm, k_norm
    bits += fold_bits + sketch_bits
    nbytes = bits / 8 * n_layers
    kv = _shapes(cfg)["k"][0]
    return {
        "gib": nbytes / 2**30,
        "gib_net": (bits - own) / 8 * n_layers / 2**30,
        "gib_fold": fold_bits / 8 * n_layers / 2**30,
        "gib_sketch": sketch_bits / 8 * n_layers / 2**30,
        "pct": 100.0 * nbytes / model_bytes if model_bytes else float("nan"),
        "kv_per_token": 2 * 2 * kv * n_layers,
        "kv_per_token_latent": 2 * (ranks.get("k", kv) + ranks.get("v", kv)) * n_layers,
    }


# ----- the module -----

class AttnReplica(nn.Module):
    """The dense attention of ``layers`` plus the ``input_layernorm`` feeding it.
    ``forward`` maps the shared ``R`` to ``Â`` (to the concatenated heads ``a`` when o is
    folded); ``read`` gives what the MLPs read. Build one with `build_replica`."""

    def __init__(self, text_cfg, layers, fold: "FoldSpec | None" = None):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm

        if getattr(text_cfg, "model_type", None) != "qwen3":
            raise ValueError(
                f"attn replica is built for qwen3, got {getattr(text_cfg, 'model_type', None)!r}")
        if getattr(text_cfg, "attention_bias", False):
            raise ValueError("attn replica assumes bias-free attention projections")
        cfg = copy.deepcopy(text_cfg)
        # The per-track configs' backend (`apply_common_per_track_sizing`), so the masks
        # the walk builds for them mean the same thing here.
        cfg._attn_implementation = "sdpa"
        self.layer_ids = tuple(sorted(layers))
        self.attn = nn.ModuleDict({str(i): Qwen3Attention(cfg, i) for i in self.layer_ids})
        self.norm = nn.ModuleDict({
            str(i): Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps) for i in self.layer_ids})
        self.fold = fold
        # The post-attention norm's eps: the same config value as the input norm's.
        self.eps = float(cfg.rms_norm_eps)
        # Folded o only: the norm sketch per layer, the exact W_o kept for `bind_fold`, the
        # o_rms a pruned fold is scored with, the per-layer bias constant, and the bound
        # [K, I/N, q_dim] gate/up folds.
        self.sketch = nn.ModuleDict()
        self._norm_c: dict = {}
        self._fold_src: dict = {}
        self._fold_norm: dict = {}
        self._fold: dict = {}

    def forward(self, li: int, R: torch.Tensor, position_embeddings, attention_mask):
        return self.attn[str(li)](
            hidden_states=self.norm[str(li)](R),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )[0]

    def read(self, li: int, R: torch.Tensor, position_embeddings, attention_mask):
        """What every MLP reads at ``li``: ``R + Â``, or a `FoldRead` where o is folded —
        ``inv_s = 1/rms(R + Â_sketch)`` in fp32 (``rms(R)`` at ``norm:r0``)."""
        y = self.forward(li, R, position_embeddings, attention_mask)
        if self.fold is None:
            return R + y
        if li not in self._fold:
            raise RuntimeError("a folded replica reads only once `bind_fold` ran — install it "
                               "with PTWrappedModel.set_attn_replica")
        z = R.float()
        if str(li) in self.sketch:
            z = z + self.sketch[str(li)](y).float()
        var = z.pow(2).mean(-1, keepdim=True)
        if li in self._norm_c:  # the sizing's mean tail energy, one constant per layer
            var = (var + self._norm_c[li]).clamp(min=0.0)
        gate, up = self._fold[li]
        return FoldRead(R, y, torch.rsqrt(var + self.eps), gate, up)

    @torch.no_grad()
    def bind_fold(self, student) -> None:
        """Build every layer's folded gate/up from THIS student's own MLP slices and the
        teacher's exact ``W_o`` kept at build: ``G_k = gate_k·diag(γ)·W_o`` (fp32 product,
        stored in the slices' dtype), ``[K, I/N, q_dim]`` for the K tracks this rank walks;
        pruned by ``o_rms`` when the spec says so. Rebinding to another student rebuilds.
        ``W_o`` is parked on the host.

        Rails: (1) PROVENANCE — ``W_o``'s columns for each track's head bit-equal that
        track's own ``o_proj`` slice (the slicer's column layout, `calib.check_o_columns`),
        so another layer's or a misordered ``W_o`` is refused; (2) dense fold: ``a·G_kᵀ``
        reproduces ``gate_k(γ⊙(W_o a))`` to storage rounding on random ``a``.
        """
        if self.fold is None:
            return
        if getattr(student, "exec_groups", 1) > 1 or getattr(student, "merge_group", 1) > 1:
            raise RuntimeError(
                "attn replica fold: the merged/batched walk is not supported — the fold is "
                "built per track off `text_models`. Run with merge_group 1 (eval's default).")
        ids = tuple(student.local_track_ids)
        hd = _head_dim(student.text_config) if hasattr(student, "text_config") else None
        for li in self.layer_ids:
            W = self._fold_src[li]
            layers = [tm.layers[li] for tm in student.text_models]
            gate = torch.stack([l.mlp.gate_proj.weight for l in layers])
            up = torch.stack([l.mlp.up_proj.weight for l in layers])
            gamma = layers[0].post_attention_layernorm.weight
            own_o = torch.stack([l.self_attn.o_proj.weight for l in layers])
            dev, dt = gate.device, gate.dtype
            hd = own_o.shape[-1] if hd is None else hd
            if own_o.shape[0] != len(ids):
                raise RuntimeError(f"attn replica fold: this rank runs {own_o.shape[0]} streams "
                                   f"over {len(ids)} shards at layer {li}")
            # Compared in the coarser of the two dtypes: the teacher's bf16 W_o against a
            # bf16 convert is bit-exact; a wider student is rounded to it first.
            lo = own_o.dtype if own_o.element_size() <= W.element_size() else W.dtype
            for k, tid in enumerate(ids):
                cols = W[:, tid * hd:(tid + 1) * hd].to(dev, lo)
                if not torch.equal(cols, own_o[k].to(dev, lo)):
                    raise RuntimeError(
                        f"attn replica fold at layer {li}: W_o columns {tid * hd}:{(tid + 1) * hd} "
                        f"are not track {tid}'s own o_proj slice — not this layer's o_proj, or "
                        f"the heads are misordered")
            Wf, gf = W.to(dev, torch.float32), gamma.to(dev, torch.float32)
            G = (gate.float() * gf) @ Wf
            U = (up.float() * gf) @ Wf
            if self.fold.frac is not None:
                norm = self._fold_norm[li].to(dev)
                G = torch.stack([wanda_prune_weight(g, self.fold.frac, norm) for g in G])
                U = torch.stack([wanda_prune_weight(u, self.fold.frac, norm) for u in U])
            G, U = G.to(dt).contiguous(), U.to(dt).contiguous()
            if self.fold.frac is None:
                a = torch.randn(4, Wf.shape[1], device=dev, generator=torch.Generator(
                    device=dev).manual_seed(li))
                want = ((a @ Wf.T) * gf) @ gate.float().transpose(-2, -1)
                got = a @ G.float().transpose(-2, -1)
                err = float((got - want).norm() / want.norm().clamp(min=EPS))
                if err > (2e-2 if dt != torch.float32 else 1e-4):
                    raise RuntimeError(f"attn replica fold at layer {li}: a·Gᵀ is {err:.3g} off "
                                       f"gate(γ⊙(W_o a)) — the fold does not reproduce the read")
            self._fold[li] = (G, U)
            self._fold_src[li] = W.cpu()


def _frozen(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t, requires_grad=False)


def _frozen_linear(W: torch.Tensor) -> nn.Linear:
    with torch.device("meta"):
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
    lin.weight = _frozen(W)
    return lin


def _norm(bases: dict, spec: str, name: str, key: str) -> torch.Tensor:
    if key not in bases:
        raise ValueError(f"attn replica {spec!r} prunes {name} and needs {key}")
    return bases[key]


def _fold_needs_bases(fold: "FoldSpec | None") -> bool:
    """A pruned fold reads ``o_rms``; a rank sketch reads ``o.U``; a bias-corrected scalar
    reads its constant."""
    return fold is not None and (fold.frac is not None or fold.rank > 0 or fold.bias)


def build_replica(text_cfg, layers, weights: dict, spec: str = "exact",
                  bases: "dict | None" = None) -> AttnReplica:
    """An `AttnReplica` from dense weights keyed ``layers.{i}.…`` and, for every truncated
    projection, ``bases[f"layers.{i}.{proj}.U"]`` (+ ``.d`` for q/k); a pruned one reads
    ``bases[f"layers.{i}.{rms_key(proj)}"]``, and a pruned factor ``A`` ``.{proj}.z_rms``.

    Built on ``meta`` and filled by assignment, so no multi-GiB random init runs first.
    Truncation runs in fp32 wherever the weights live and casts back to their dtype.
    """
    ranks, fracs, fold = _parse(spec)
    if (ranks or fracs or _fold_needs_bases(fold)) and bases is None:
        raise ValueError(
            f"attn replica {spec!r} compresses {sorted([*ranks, *fracs, *(['o'] if fold else [])])} "
            f"and needs bases")
    with torch.device("meta"):
        rep = AttnReplica(text_cfg, layers, fold)
    for i in rep.layer_ids:
        attn, pre = rep.attn[str(i)], f"layers.{i}."
        for name in PROJS:
            W = weights[f"{pre}self_attn.{name}_proj.weight"]
            if name == "o" and fold is not None:
                # The attention hands out `a`; o lives in the tracks' gate/up (`bind_fold`).
                mod = nn.Identity()
                rep._fold_src[i] = W
                if fold.frac is not None:
                    rep._fold_norm[i] = _norm(bases, spec, "o", f"{pre}o_rms")
                if fold.norm == "exact":
                    rep.sketch[str(i)] = _frozen_linear(W)
                elif fold.rank:
                    B, A = truncate(W.float(), bases[f"{pre}o.U"].to(W.device, torch.float32),
                                    fold.rank)
                    rep.sketch[str(i)] = nn.Sequential(_frozen_linear(B.to(W.dtype)),
                                                       _frozen_linear(A.to(W.dtype)))
                if fold.bias:
                    rep._norm_c[i] = float(_norm(bases, spec, "o's norm bias",
                                                 norm_c_key(i, fold.estimator)))
            elif name in ranks:
                d = bases.get(f"{pre}{name}.d")
                B, A = truncate(W.float(), bases[f"{pre}{name}.U"].to(W.device, torch.float32),
                                ranks[name],
                                None if d is None else d.to(W.device, torch.float32))
                if name in fracs:  # rank first, THEN density: each factor by its own input
                    B = wanda_prune_weight(B, fracs[name],
                                           _norm(bases, spec, name, f"{pre}{rms_key(name)}"))
                    A = wanda_prune_weight(A, fracs[name], _norm(
                        bases, spec, name, f"{pre}{name}.z_rms")[:ranks[name]])
                mod = nn.Sequential(_frozen_linear(B.to(W.dtype)), _frozen_linear(A.to(W.dtype)))
            elif name in fracs:
                mod = _frozen_linear(wanda_prune_weight(
                    W, fracs[name], _norm(bases, spec, name, f"{pre}{rms_key(name)}")))
            else:
                mod = _frozen_linear(W)
            setattr(attn, f"{name}_proj", mod)
        attn.q_norm.weight = _frozen(weights[f"{pre}self_attn.q_norm.weight"])
        attn.k_norm.weight = _frozen(weights[f"{pre}self_attn.k_norm.weight"])
        rep.norm[str(i)].weight = _frozen(weights[f"{pre}input_layernorm.weight"])
    if unfilled := [n for n, t in (*rep.named_parameters(), *rep.named_buffers()) if t.is_meta]:
        raise ValueError(f"attn replica left {unfilled[:3]} unfilled")
    return rep.eval()


def load_replica(hf_model: str, text_cfg, layers, spec: str,
                 bases_path: "str | None", device) -> AttnReplica:
    """Read ONLY ``layers``' attention tensors from the teacher checkpoint and ONLY the
    basis columns the spec needs, then `build_replica` on ``device``.

    ``bases_path`` may list several files (``bases.safetensors,x_rms.safetensors``); each
    key is read from the one file holding it, and a key two files hold is refused."""
    from safetensors import safe_open

    from parallm.slicer.loader import read_hf_tensors

    ranks, fracs, fold = _parse(spec)
    names = [f"layers.{i}.{s}" for i in layers for s in (
        *(f"self_attn.{p}_proj.weight" for p in PROJS),
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight")]
    weights = {k: v.to(device) for k, v in read_hf_tensors(hf_model, names).items()}
    bases = None
    if ranks or fracs or _fold_needs_bases(fold):
        if not bases_path:
            raise ValueError(f"attn replica {spec!r} needs a bases file")
        bases = {}
        # A fold's norm sketch reads o's basis like a rank cut; a pruned fold reads o_rms.
        want_U = dict(ranks)
        if fold is not None and fold.rank:
            want_U["o"] = fold.rank
        with ExitStack() as stack:
            where: dict = {}
            for path in bases_path.split(","):
                f = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                for key in f.keys():
                    if key in where:
                        raise ValueError(f"{key!r} is in more than one of {bases_path}")
                    where[key] = f
            for i in layers:
                for name, r in want_U.items():
                    key = f"layers.{i}.{name}.U"
                    if key not in where:
                        raise ValueError(f"{bases_path} has no {name} basis for layer {i}")
                    bases[key] = where[key].get_slice(key)[:, :r].to(device)
                    if (dkey := f"layers.{i}.{name}.d") in where:
                        bases[dkey] = where[dkey].get_tensor(dkey).to(device)
                need = {f"layers.{i}.{rms_key(n)}" for n in fracs}
                need |= {f"layers.{i}.{n}.z_rms" for n in fracs if n in ranks}
                if fold is not None and fold.frac is not None:
                    need.add(f"layers.{i}.o_rms")
                if fold is not None and fold.bias:
                    need.add(norm_c_key(i, fold.estimator))
                for key in sorted(need):
                    if key not in where:
                        raise ValueError(f"{bases_path} has no {key}, which {spec!r} needs")
                    bases[key] = where[key].get_tensor(key).to(device)
    return build_replica(text_cfg, layers, weights, spec, bases)
