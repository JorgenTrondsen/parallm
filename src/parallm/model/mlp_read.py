"""Local read estimate of the MLP sum a post-attn-only layer leaves unsynced.

At a layer ``i`` that syncs after its token mixer but not after its MLP, every track's MLP
reads the true synced ``R_i``, and track ``k`` carries only ``R_i + Z_k`` into layer ``i+1``:
that attention reads a residual missing ``Σ_{j≠k} Z_j``. ``R_i`` is shared, so any channel of
``MLP_i`` can be recomputed on any track. A base ``S`` of the top-r (track, channel) pairs
(`scripts/build_trunc_estimator.py`) gives track ``k`` the read

    read_k = carry_k + Σ_{c∈S} d_c(x) − Σ_{c∈S∩C_k} d_c(x),    x = post_attention_layernorm(R_i)

— the subtracted sum is the part of ``S`` the carry already holds exactly (track ``k``'s own
channels). Read-only: attention ``i+1`` reads ``read_k`` while its delta joins ``carry_k``, so
the next sync still sums the exact per-track deltas and no collective is added. A full-width
base makes every ``read_k`` the exact residual.

A base compresses without quantization, at load time (`compress_layer`): Wanda or N:M
pruning of gate/up, each channel's row scored by the input RMS ``x_rms``
(`scripts/calib_attn_replica.py --mlp`); `MlpReadBase.bind_fold` then folds ``down`` into the
next layer's q/k/v.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open

from parallm.model.replica import wanda_prune_weight
from parallm.model.replica_pack import _bits_per_weight


class MlpReadBase(nn.Module):
    """Frozen ``gate``/``up`` ``[H, r]`` and ``down`` ``[r, H]`` per layer, plus, per local
    track, the base columns that track owns (from ``layer.{i}.index``)."""

    def __init__(self, tensors: dict, layers, local_track_ids, n_tracks: int,
                 ledger_bytes: "int | None" = None, spec: str = ""):
        super().__init__()
        self.layer_ids = tuple(sorted(layers))
        self.ranks: dict[int, int] = {}
        self.spec = spec
        self.ledger_bytes = ledger_bytes
        for li in self.layer_ids:
            g, u, d = (tensors[f"layer.{li}.{k}"] for k in ("gate", "up", "down"))
            index = tensors[f"layer.{li}.index"].long()
            r = g.shape[1]
            if not (u.shape[1] == r == d.shape[0] == index.shape[0]):
                raise ValueError(f"layer {li}: gate/up/down/index disagree on r "
                                 f"({g.shape[1]}, {u.shape[1]}, {d.shape[0]}, {index.shape[0]})")
            if index.numel() and int(index[:, 0].max()) >= n_tracks:
                raise ValueError(f"layer {li}: base names track {int(index[:, 0].max())} but the "
                                 f"model was sliced into {n_tracks}")
            self.ranks[li] = r
            self.register_buffer(f"g{li}", g, persistent=False)
            self.register_buffer(f"u{li}", u, persistent=False)
            self.register_buffer(f"d{li}", d, persistent=False)
            for k, tid in enumerate(local_track_ids):
                self.register_buffer(f"own{li}_{k}",
                                     (index[:, 0] == tid).nonzero().flatten(), persistent=False)
        self.n_local = len(local_track_ids)
        self.fold_norm: "str | None" = None
        self.source = ""          # the base file's name, for norm-constant provenance
        self._ncal = None         # `start_norm_calibration` state
        self._fold_c: dict = {}   # per-layer `+c` of a folded r<M>+c norm

    @property
    def rank(self) -> str:
        """``r`` of every layer, or ``lo-hi`` for a per-layer allocation (logs)."""
        lo, hi = min(self.ranks.values()), max(self.ranks.values())
        return str(lo) if lo == hi else f"{lo}-{hi}"

    def stored_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for n, b in self.named_buffers()
                   if not n.startswith("own"))

    @torch.no_grad()  # the student's weights require grad; F must not hold a graph
    def bind_fold(self, student, norm: str, stats: "str | None" = None,
                  norm_c: "str | None" = None) -> None:
        """Fold the base's output projection into each local track's NEXT-layer q/k/v (the
        analog of the attention replica's ``o:fold``): the estimate only ever reaches the
        model through layer i+1's ``input_layernorm`` and its q/k/v slices, so track k stores
        ``F_k = D diag(γ) W_qkvᵀ`` ``[r, q+k+v]`` instead of ``D = down`` ``[r, H]``.
        ``norm``: the read's
        RMS scale — ``exact`` forms the estimate in hidden space to get it (keeps ``D``: the
        rail); ``r<M>`` takes it from a rank-M sketch of the estimate, ``z_M = h (D Y) Yᵀ``
        (``r<M>+c`` adds a per-layer constant for the energy the sketch misses, from
        ``norm_c`` — `scripts/calib_fold_norm.py` on this same base and spec)
        with ``Y`` the top-M output directions of the MLP sum (``y_U`` in ``stats``; stores
        ``D Y`` ``[r, M]`` and ``Y`` ``[H, M]``); ``r0`` is the carry's own RMS. ⚠ r0
        COLLAPSES the front (math .26): at layers 0-1 the MLP writes the massive activation
        (one direction holds ~40% of its output energy) and a carry holds only its own share.

        F is a frozen copy of the student's q/k/v at bind time: eval only."""
        from parallm.utils.layers import format_layers

        plus_c = norm.endswith("+c")
        core = norm[:-2] if plus_c else norm
        if not (core == "exact" or (core.startswith("r") and core[1:].isdigit())) or \
                (plus_c and core in ("exact", "r0")):
            raise ValueError(f"fold norm must be exact|r<M>[+c], got {norm!r}")
        M = None if core == "exact" else int(core[1:])
        if plus_c:
            if norm_c is None:
                raise ValueError(f"fold norm {norm} needs the calibrated constants (norm_c)")
            with safe_open(norm_c, "pt") as cf:
                meta = cf.metadata() or {}
                if (meta.get("source"), meta.get("spec", "")) != (self.source, self.spec):
                    raise ValueError(
                        f"{norm_c} was calibrated on {meta.get('source')!r} "
                        f"({meta.get('spec', '')!r}), this base is {self.source!r} ({self.spec!r})")
                self._fold_c = {li: float(cf.get_tensor(f"layer.{li}.c_r{M}"))
                                for li in self.layer_ids}
        if M and stats is None:
            raise ValueError(f"fold norm {norm} needs the stats file's y_U (the MLP sum's output basis)")
        if len(student.text_models) != self.n_local:
            raise ValueError("bind_fold on the looped model the base was loaded for")
        bad = [li for li in self.layer_ids
               if getattr(student.text_models[0].layers[li + 1], "block_type", None)
               == "linear_attention"
               or type(student.text_models[0].layers[li + 1].input_layernorm).__name__
               != "Qwen3RMSNorm"]
        if bad:
            raise ValueError(f"fold needs a Qwen3 self-attention layer (weight·x/rms norm) after "
                             f"each read layer; not after {format_layers(bad)}")
        # Ledger prices are bf16 (`_bytes`) whatever this copy's dtype.
        base = self.ledger_bytes if self.ledger_bytes is not None else sum(
            _bytes(t) for n, t in self.named_buffers() if not n.startswith("own"))
        replaced = folded = 0.0
        for li in self.layer_ids:
            src = getattr(self, f"d{li}")
            for k, tm in enumerate(student.text_models):
                lay = tm.layers[li + 1]
                a = lay.self_attn
                W = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).float()
                F = (src.float() * lay.input_layernorm.weight.float()) @ W.T
                self.register_buffer(f"fold{li}_{k}", F.to(src.dtype).contiguous(), persistent=False)
            self._fold_split = (a.q_proj.out_features, a.k_proj.out_features,
                                a.v_proj.out_features)
            self._fold_eps = lay.input_layernorm.variance_epsilon
            replaced += _bytes(src)
            folded += _bytes(getattr(self, f"fold{li}_0"))  # one track per node
            if M:
                with safe_open(stats, "pt") as sf:
                    Y = sf.get_slice(f"layers.{li}.y_U")[:, :M].to(src.device, torch.float32)
                if Y.shape[1] < M:
                    raise ValueError(f"r{M}: the stats hold only {Y.shape[1]} output directions")
                self.register_buffer(f"ysk{li}", Y.to(src.dtype).contiguous(), persistent=False)
                self.register_buffer(f"dsk{li}", (src.float() @ Y).to(src.dtype).contiguous(),
                                     persistent=False)
                folded += _bytes(getattr(self, f"ysk{li}")) + _bytes(getattr(self, f"dsk{li}"))
            if norm != "exact":
                del self._buffers[f"d{li}"]
        self.ledger_bytes = base - replaced + folded
        self.fold_norm = norm
        self.spec = (self.spec + "," if self.spec else "") + f"fold/{norm}"

    def _folded_reads(self, li, h, carries):
        from parallm.model.seam import FoldedAttnRead

        out = []
        for k, c in enumerate(carries):
            own = getattr(self, f"own{li}_{k}")
            F = getattr(self, f"fold{li}_{k}")
            delta = h @ F - (h.index_select(-1, own) @ F.index_select(0, own)
                             if own.numel() else 0)
            if self.fold_norm == "exact":
                d = getattr(self, f"d{li}")
                z = h @ d - (h.index_select(-1, own) @ d.index_select(0, own) if own.numel() else 0)
            elif self.fold_norm != "r0":
                dsk = getattr(self, f"dsk{li}")
                zk = h @ dsk - (h.index_select(-1, own) @ dsk.index_select(0, own)
                                if own.numel() else 0)
                z = zk @ getattr(self, f"ysk{li}").T
            else:
                z = 0
            ms = (c + z).float().pow(2).mean(-1, keepdim=True)
            if self._fold_c:
                ms = (ms + self._fold_c[li]).clamp(min=0)
            inv_s = torch.rsqrt(ms + self._fold_eps)
            q, kk, v = (delta.float() * inv_s).split(self._fold_split, -1)
            out.append(FoldedAttnRead(None if self.fold_norm == "r0" else inv_s, q, kk, v))
        return out

    def reads(self, li: int, R: torch.Tensor, carries, norm: nn.Module, act):
        """``read_k`` for the ``carries`` (``R + Z_k``) of the local tracks at layer ``li``."""
        x = norm(R)
        h = act(x @ getattr(self, f"g{li}")) * (x @ getattr(self, f"u{li}"))
        if self.fold_norm is not None:
            return self._folded_reads(li, h, carries)
        d = getattr(self, f"d{li}")
        ghat = h @ d
        out = []
        for k, c in enumerate(carries):
            own = getattr(self, f"own{li}_{k}")
            z = ghat - h.index_select(-1, own) @ d.index_select(0, own) if own.numel() else ghat
            if self._ncal is not None:
                self._norm_accumulate(li, c, z)
            out.append(c + z)
        return out

    def start_norm_calibration(self, stats: str, ranks) -> None:
        """Measure the per-layer `+c` a folded ``r<M>+c`` norm adds: the mean over tokens and
        tracks of ``mean_H((carry+z)²) − mean_H((carry+z_M)²)``, ``z_M = z Y_M Y_Mᵀ`` the
        fold's rank-M sketch (``y_U`` in ``stats``) — the energy the sketch misses, which
        otherwise under-states the read's RMS. Runs on the UNFOLDED read, where ``z`` exists."""
        if self.fold_norm is not None:
            raise ValueError("calibrate the norm constant on the unfolded base")
        ranks = sorted(set(int(m) for m in ranks))
        Y = {}
        with safe_open(stats, "pt") as sf:
            for li in self.layer_ids:
                Y[li] = sf.get_slice(f"layers.{li}.y_U")[:, :ranks[-1]].float()
        if any(y.shape[1] < ranks[-1] for y in Y.values()):
            raise ValueError(f"the stats hold fewer than {ranks[-1]} output directions")
        self._ncal = {"ranks": ranks, "Y": Y, "sum": {}, "n": {}}

    def _norm_accumulate(self, li, c, z) -> None:
        cal = self._ncal
        Y = cal["Y"][li].to(z.device)
        cf, zf = c.float(), z.float()
        full = (cf + zf).pow(2).mean(-1)
        zy = zf @ Y
        for m in cal["ranks"]:
            zm = zy[..., :m] @ Y[:, :m].T
            diff = (full - (cf + zm).pow(2).mean(-1)).double().sum()
            cal["sum"][(li, m)] = cal["sum"].get((li, m), 0.0) + diff
        cal["n"][li] = cal["n"].get(li, 0) + full.numel()

    def finish_norm_calibration(self, group=None) -> dict:
        """``{(layer, M): c}``, summed over ``group``'s ranks (every rank must call it)."""
        cal, self._ncal = self._ncal, None
        keys = sorted(cal["sum"])
        dev = next(iter(cal["sum"].values())).device
        sums = torch.stack([cal["sum"][k] for k in keys]).to(dev)
        counts = torch.tensor([float(cal["n"][li]) for li, _ in keys], dtype=torch.float64, device=dev)
        if group is not None or (dist.is_available() and dist.is_initialized()):
            dist.all_reduce(sums, group=group)
            dist.all_reduce(counts, group=group)
        return {k: float(s / n) for k, s, n in zip(keys, sums, counts)}


def load_mlp_read_base(path: str, layers, local_track_ids, n_tracks: int, device,
                       dtype=torch.bfloat16, compress: "str | None" = None,
                       stats: "str | None" = None) -> MlpReadBase:
    """Read only ``layers`` of a base; ``None`` = every layer in it. ``compress`` (a
    `parse_compress_spec` string, with the ``stats`` file it reads) compresses each layer as it
    loads (`compress_layer`, fp32 on ``device``), so no compressed copy is written."""
    spec = parse_compress_spec(compress) if compress else None
    if spec is not None and stats is None:
        raise ValueError("load-time compression needs the calibration stats file")
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        have = sorted({int(k.split(".")[1]) for k in keys if k.startswith("layer.")})
        layers = have if layers is None else sorted(set(layers))
        if missing := sorted(set(layers) - set(have)):
            raise ValueError(f"{path} has no base for layers {missing}")
        if missing := [li for li in layers if f"layer.{li}.index" not in keys]:
            raise ValueError(f"{path} carries no layer.*.index for {missing}: without it a "
                             f"track cannot know which base columns it already holds")
        tensors, ledger = {}, 0.0
        for li in layers:
            index = f.get_tensor(f"layer.{li}.index").to(device)
            tensors[f"layer.{li}.index"] = index
            if spec is None:
                for k in ("gate", "up", "down"):
                    tensors[f"layer.{li}.{k}"] = f.get_tensor(f"layer.{li}.{k}").to(device, dtype)
                continue
            g, u, d = (f.get_tensor(f"layer.{li}.{k}").to(device) for k in ("gate", "up", "down"))
            with safe_open(stats, "pt") as sf:
                st = {k: sf.get_tensor(k) for k in sf.keys() if k.startswith(f"layers.{li}.")}
            out, b = compress_layer(g, u, d, index, st, li, spec)
            del g, u, d
            ledger += b
            for k, v in out.items():
                tensors[f"layer.{li}.{k}"] = v.to(device, dtype).contiguous()
    b = MlpReadBase(tensors, layers, local_track_ids, n_tracks,
                    ledger_bytes=ledger if spec is not None else None, spec=compress or "")
    b.source = path.rsplit("/", 1)[-1]
    return b


def _parse_prune(part: str, val: str):
    n, _, m = val.partition(":")
    if m:
        if not (n.isdigit() and m.isdigit() and 0 < int(n) < int(m)):
            raise ValueError(f"{part!r}: N:M pruning needs 0 < N < M")
        return ("nm", int(n), int(m))
    if val.startswith("s") and val[1:].isdigit() and 0 < int(val[1:]) < 100:
        return ("s", int(val[1:]) / 100)
    raise ValueError(f"{part!r}: pruning is s<1..99> (Wanda) or N:M")


def parse_compress_spec(spec: str) -> dict:
    """``gu:sPCT`` Wanda (→ ``("s", frac)``), ``gu:N:M`` N-of-M structured pruning
    (→ ``("nm", N, M)``), or ``gu:A+sPCT`` (A, then Wanda prunes PCT% of each row's
    survivors, → ``("then", [A, ("s", f)])``) — all on gate/up."""
    name, _, val = spec.strip().partition(":")
    if name != "gu" or not val:
        raise ValueError(f"compression spec is gu:sPCT | gu:N:M | gu:A+sPCT, got {spec!r}")
    if "+" in val:
        terms = [_parse_prune(spec, v) for v in val.split("+")]
        if len(terms) != 2 or terms[1][0] != "s":
            raise ValueError(f"{spec!r}: combine as A+sPCT (A, then Wanda on what is left)")
        return {"gu": ("then", terms)}
    return {"gu": _parse_prune(spec, val)}


def _bits(how) -> float:
    """Stored bits per weight: bf16 dense; Wanda = 1-bit bitmap + bf16 survivors
    (`replica_pack._bits_per_weight`); N:M = bf16 survivors + a ⌈log2 C(M,N)⌉-bit pattern
    per group of M."""
    if how is None:
        return 16.0
    if isinstance(how, float):
        how = ("s", how)
    if how[0] == "s":
        return _bits_per_weight(how[1], None)
    _, n, m = how
    return (16.0 * n + math.ceil(math.log2(math.comb(m, n)))) / m


def _priced(how, w: torch.Tensor):
    """A combined mask fits no single pattern: price it as a bitmap at its realised sparsity."""
    if how is not None and how[0] == "then":
        return ("s", float((w == 0).float().mean()))
    return how


def _bytes(t: torch.Tensor, how=None) -> float:
    return t.numel() * _bits(how) / 8.0


def _prune(w: torch.Tensor, how, norms: torch.Tensor) -> torch.Tensor:
    """Rows of ``w`` ``[rows, in]`` pruned by the activation-aware score ``|w|·norms``: Wanda
    zeroes the lowest ``frac`` per row; N:M keeps the top N of every M consecutive inputs;
    ``then`` applies the first term and lets Wanda prune its fraction of the survivors (zeros
    score lowest, so they go first)."""
    if how[0] == "then":
        first, (_, f) = how[1]
        w1 = _prune(w, first, norms)
        z1 = float((w1 == 0).float().mean())
        return wanda_prune_weight(w1, z1 + f * (1 - z1), norms)
    if how[0] == "s":
        return wanda_prune_weight(w, how[1], norms)
    _, n, m = how
    if w.shape[1] % m:
        raise ValueError(f"N:M pruning needs in_features {w.shape[1]} divisible by {m}")
    score = (w.abs() * norms.to(w.device, w.dtype)[None, :]).view(w.shape[0], -1, m)
    keep = torch.zeros_like(score, dtype=torch.bool).scatter_(
        -1, score.topk(n, dim=-1).indices, True)
    return (w.view_as(score) * keep).view_as(w)


def compress_layer(g, u, d, index, stats: dict, li: int, spec: dict) -> "tuple[dict, float]":
    """One layer's compressed tensors and ledger bytes. ``g``/``u`` ``[H, r]``, ``d`` ``[r, H]``
    (fp32 math); gate/up rows (one per channel) score their inputs with ``layers.{i}.x_rms``."""
    x_norm = stats[f"layers.{li}.x_rms"].float().to(g.device)
    gu = spec["gu"]
    out: dict = {}
    ledger = 0.0
    for name, w in (("gate", g.float()), ("up", u.float())):
        w = _prune(w.T, gu, x_norm).T  # one row per channel, over its inputs
        out[name] = w.contiguous()
        ledger += _bytes(w, _priced(gu, w))
    out["down"] = d.float().contiguous()
    ledger += _bytes(out["down"]) + index.numel() * 4
    return out, ledger
