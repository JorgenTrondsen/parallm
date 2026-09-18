"""Calibrate the attention replica: measure the moments its truncation bases are cut from.

`model.attn_replica` truncates each attention projection by projecting its TRUE output onto
the top eigenvectors of that output's second moment. Two moments per layer determine all
four bases, measured here on teacher-exact states (`--sync-phase exact`):

    C_x = E[x xᵀ],  x = input_layernorm(R)       → q, k, v    (M = D W C_x Wᵀ D)
    S_A = E[A Aᵀ],  A = Σ_k a_k over every head  → o          (M = S_A)

The bases are cut ONCE (fp64 eigh on rank 0) so eval only slices columns, and every rank
builds the same replica from the same file. One pass writes all four artifacts a compressed
arm reads — ``bases.safetensors`` (+ ``.d``, the per-head metric scale for q/k),
``x_rms.safetensors`` (what a pruned q/k/v scores with), ``o_rms.safetensors`` (what a pruned
o or a pruned FOLD scores with) and ``z_rms.safetensors`` (the latent RMS a pruned
``RANK/sPCT`` factor scores its ``A`` with):

    torchrun --standalone --nproc-per-node=8 scripts/calib_attn_replica.py \\
        --hf-model <teacher> --checkpoint-dir convert_out/qwen3/32b_n64_even \\
        --layers 32-63 --out-dir logs/qwen3/attn_replica

⛔ ``A`` is summed EXPLICITLY from every track's attention output in fp32 and all-reduced
after the forward — never ``fl(R + A) − R``. bf16 keeps 7 mantissa bits, so that difference
loses ulps of ``R``, which is the size of the signal in the massive-activation channels.
⚠ Position 0 is KEPT. The sink token carries the massive activations; a basis that never saw
it can drop the direction every head's sink attention reads.

A folded o (``o:fold,norm:r32+c``) stores no ``o_proj``; the post-attention norm's scalar
reads a rank sketch of it and adds a per-layer constant under the root. ``--norm-c`` fits
that constant — a SECOND pass, because it reads the bases the first pass writes:

    torchrun --standalone --nproc-per-node=8 scripts/calib_attn_replica.py --norm-c \\
        --hf-model <teacher> --checkpoint-dir convert_out/qwen3/32b_n64_even \\
        --layers 32-63 --out-dir logs/qwen3/attn_replica

⚡ Its error table SIZES the estimator and never ranks arms — no offline statistic on this
path has ever predicted the macro. The scalar measured FREE at every rung, r0 included.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from parallm.dist.groups import build_groups
from parallm.model.attn_replica import (
    EPS,
    NORM_LADDER,
    PROJS,
    head_scale,
    input_rms,
    norm_c_key,
    output_moment,
    rms_key,
    top_basis,
)
from parallm.model.merge import plan_track_layout
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.loader import read_hf_tensors
from parallm.train.data import (
    DEFAULT_MIXTURE,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_sources,
)
from parallm.utils.checkpoint import load_manifest, load_track
from parallm.utils.layers import format_layers, parse_layers

RANK_LADDER = (64, 128, 256, 512, 1024, 2048)
BASES = "bases.safetensors"
X_RMS = "x_rms.safetensors"
O_RMS = "o_rms.safetensors"
Z_RMS = "z_rms.safetensors"
NORM_C = "norm_c.safetensors"


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


class _Taps:
    """One forward's attention OUTPUTS summed over this rank's tracks per layer (fp32),
    track 0's attention INPUT per layer, and every track's ``o_proj`` input squared and
    summed per channel (fp64) in row ``track id`` — that track's head, so the rows flatten
    into o's DENSE input columns (`check_o_columns`)."""

    def __init__(self, student, layers):
        self.a: dict[int, torch.Tensor] = {}
        self.x: dict[int, torch.Tensor] = {}
        self.o_sq: dict[int, torch.Tensor] = {}
        self.handles = []
        for k, tm in enumerate(student.text_models):
            tid = student.local_track_ids[k]
            for li in layers:
                mod = tm.layers[li].self_attn
                self.handles.append(mod.register_forward_hook(self._out(li)))
                self.handles.append(mod.o_proj.register_forward_pre_hook(
                    self._o_in(li, tid, student.n_tracks)))
                if k == 0:
                    self.handles.append(
                        mod.register_forward_pre_hook(self._inp(li), with_kwargs=True))

    def _out(self, li):
        def hook(_mod, _inp, out):
            y = (out[0] if isinstance(out, tuple) else out).detach().float().flatten(0, 1)
            self.a[li] = y if li not in self.a else self.a[li] + y
        return hook

    def _o_in(self, li, tid, n_tracks):
        def hook(_mod, args):
            sq = args[0].detach().double().pow(2).flatten(0, -2).sum(0)
            if li not in self.o_sq:
                self.o_sq[li] = sq.new_zeros(n_tracks, sq.numel())
            self.o_sq[li][tid] += sq
        return hook

    def _inp(self, li):
        def hook(_mod, args, kwargs):
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self.x[li] = x.detach().flatten(0, 1)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def check_capture(student, probe: dict, x: torch.Tensor, A: torch.Tensor, li: int) -> float:
    """⚠ The capture must BE the walk's states, or every basis is cut from noise.

    ``x`` must equal the MLP-synced ``R`` through this layer's own input norm, bit for bit.
    ``A`` must point where the synced post-attn state minus ``R`` points; bf16 differencing
    is lossy, hence a cosine and a loose bar — a rank-local sum reads far below it.
    """
    R = probe[(li - 1, "mlp")]
    if not torch.equal(x, student.text_models[0].layers[li].input_layernorm(R).flatten(0, 1)):
        raise SystemExit(f"[rail] layer {li}: captured attention input is not "
                         f"input_layernorm(R)")
    diff = (probe[(li, "attn")].float() - R.float()).flatten(0, 1)
    cos = float(F.cosine_similarity(A, diff, dim=-1).mean())
    if cos < 0.9:
        raise SystemExit(f"[rail] layer {li}: summed attention output has cos {cos:.4f} "
                         f"with the synced post-attn delta — not the whole layer's A")
    return cos


def check_o_columns(student, weights: dict, layers) -> None:
    """⚠ `o_rms` is indexed by o's DENSE input column but filled from each track's own
    ``o_proj`` input, so it is right only if track t's o slice IS the teacher's columns
    ``t·hd:(t+1)·hd``. The FOLD stands on the same layout (`AttnReplica.bind_fold`)."""
    for li in layers:
        W = weights[f"layers.{li}.self_attn.o_proj.weight"]
        for k, tm in enumerate(student.text_models):
            t, w = student.local_track_ids[k], tm.layers[li].self_attn.o_proj.weight
            hd = w.shape[1]
            if not torch.equal(w, W[:, t * hd:(t + 1) * hd].to(w.device, w.dtype)):
                raise SystemExit(f"[rail] layer {li}: track {t}'s o_proj is not the teacher's "
                                 f"columns {t * hd}:{(t + 1) * hd} — o_rms would be misindexed")


def compute_bases(weights: dict, C_x: dict, S_A: dict, layers, head_dim: int,
                  r_max: int, device) -> "tuple[dict, dict]":
    """``(bases, energy)``. ``bases`` holds each projection's top-``r_max`` output basis
    (+ the per-head metric scale for q/k), fp32 on the host; ``energy`` the fraction of
    each moment a rank on `RANK_LADDER` captures — SIZING, never a ranking."""
    bases, energy = {}, {}
    for li in layers:
        pre = f"layers.{li}."
        C = C_x[li].to(device, torch.float64)
        evs = {}
        for name in ("q", "k", "v"):
            W = weights[f"{pre}self_attn.{name}_proj.weight"].to(device, torch.float64)
            d = head_scale(W, C, head_dim) if name in ("q", "k") else None
            U, evs[name] = top_basis(output_moment(W, C, d), r_max)
            bases[f"{pre}{name}.U"] = U.float().cpu().contiguous()
            if d is not None:
                bases[f"{pre}{name}.d"] = d.float().cpu().contiguous()
        U, evs["o"] = top_basis(S_A[li].to(device, torch.float64), r_max)
        bases[f"{pre}o.U"] = U.float().cpu().contiguous()
        energy[str(li)] = {
            name: {str(r): round(float(ev[:r].sum() / ev.sum().clamp(min=EPS)), 6)
                   for r in RANK_LADDER if r <= ev.numel()}
            for name, ev in evs.items()
        }
    return bases, energy


def latent_rms(weights: dict, C_x: dict, S_A: dict, bases: dict, layers, device) -> dict:
    """The RMS of every rank latent a truncated projection carries: ``sqrt(u_jᵀ M u_j)`` for
    basis column ``j`` of the moment ``M`` it was cut from — q, k and v from their input
    moment ``D W C_x Wᵀ D`` in the metric `compute_bases` cut them in (the per-head ``d`` for
    q and k, none for v), o from ``S_A`` — √eigenvalue j, whatever the rank. A pruned factor
    ``A`` is scored by it."""
    z_rms = {}
    for li in layers:
        pre = f"layers.{li}."
        C = C_x[li].to(device, torch.float64)
        S = S_A[li].to(device, torch.float64)
        for name in PROJS:
            U = bases[f"{pre}{name}.U"].to(device, torch.float64)
            if name == "o":
                # o is cut from its OUTPUT moment; its latent is a basis column of S_A.
                z2 = ((U.T @ S) * U.T).sum(1)
            else:
                W = weights[f"{pre}self_attn.{name}_proj.weight"].to(device, torch.float64)
                d = bases.get(f"{pre}{name}.d")
                DW = W if d is None else d.to(device, torch.float64)[:, None] * W
                B = U.T @ DW
                z2 = ((B @ C) * B).sum(1)
            if not (torch.isfinite(z2).all() and (z2 > 0).all()):
                raise SystemExit(f"[rail] layer {li} {name}: a rank latent has zero or "
                                 f"non-finite variance")
            if not (z2[1:] <= z2[:-1] + 1e-6 * z2[0]).all():
                raise SystemExit(f"[rail] layer {li} {name}: latent variances leave the basis's "
                                 f"descending order — not this moment's eigenvectors")
            z_rms[f"{pre}{name}.z_rms"] = z2.sqrt().float().cpu().contiguous()
    return z_rms


def o_rms_from_squares(o_sq: dict, n_tok: int) -> dict:
    """``{layers.{i}.o_rms: sqrt(Σ x² / n_tok)}`` over o's dense input columns, from the
    track-summed `_Taps.o_sq` (row t = track t's head = dense head t)."""
    out = {}
    for li, sq in sorted(o_sq.items()):
        rms = (sq.flatten() / n_tok).sqrt().float().cpu()
        if not (torch.isfinite(rms).all() and (rms > 0).all()):
            raise SystemExit(f"[rail] layer {li}: o input has a zero or non-finite channel RMS — "
                             f"a track's head never reached the sum")
        out[f"layers.{li}.{rms_key('o')}"] = rms.contiguous()
    return out


def norm_scalar_errors(R: torch.Tensor, A: torch.Tensor, U: torch.Tensor, eps: float,
                       ranks=NORM_LADDER) -> dict:
    """Per token: ``"true"`` = ``rms(R + A)``, and one estimate per rank —
    ``r{r}`` = ``rms(R + U_r U_rᵀ A)``, ``r0`` = ``rms(R)``. ``U_r U_rᵀ W_o a`` IS the rank-r
    sketch of o that a folded replica keeps for the post-attention norm's scalar."""
    R, A = R.float(), A.float()

    def rms(z):
        return (z.pow(2).mean(-1) + eps).sqrt()

    for r in ranks:
        if r > U.shape[1]:
            raise ValueError(f"norm sketch rank {r} exceeds the {U.shape[1]} basis columns")
    out = {"true": rms(R + A)}
    for r in ranks:
        out[f"r{r}"] = rms(R + ((A @ U[:, :r]) @ U[:, :r].T if r else 0.0))
    return out


def norm_c_report(stats: dict, layers, ranks=NORM_LADDER) -> "tuple[dict, dict]":
    """``(constants, p95)``. The per-layer bias constant ``c = mean(true² − est²)`` a
    ``norm:r{RANK}+c`` fold adds under the root (the systematic tail energy the sketch
    misses, fit in-sample — one number per layer and rank), and the p95 of
    ``|est/true − 1|`` before and after it, printed as SIZING ONLY."""
    consts, p95 = {}, {}
    q = lambda t, p: float(torch.quantile(t, p)) if t.numel() < 2**24 else float(
        torch.quantile(t[torch.randperm(t.numel())[:2**24 - 1]], p))
    for li in layers:
        cat = {k: torch.cat(v) for k, v in stats[li].items()}
        true = cat["true"]
        for r in ranks:
            est = cat[f"r{r}"]
            c = float((true.pow(2) - est.pow(2)).mean())
            consts[norm_c_key(li, f"r{r}")] = torch.tensor(c, dtype=torch.float32)
            p95[(li, r)] = (q((est / true - 1).abs(), 0.95),
                            q(((est.pow(2) + c).clamp(min=0).sqrt() / true - 1).abs(), 0.95))
    print(f"\n===== NORM SCALAR SIZING over {format_layers(layers)} — p95 relative error of "
          f"the post-attention norm's scalar, raw / +c (SIZING ONLY — it picks an estimator, "
          f"it never ranks arms) =====")
    print(f"{'layer':>5} " + " ".join(f"{'r' + str(r):>15}" for r in ranks))
    for li in layers:
        print(f"{li:>5} " + " ".join(f"{p95[(li, r)][0]:7.4f}/{p95[(li, r)][1]:7.4f}"
                                     for r in ranks))
    print("max-over-layers: " + "  ".join(
        f"r{r} {max(p95[(li, r)][0] for li in layers):.4f}/"
        f"{max(p95[(li, r)][1] for li in layers):.4f}" for r in ranks))
    return consts, p95


def _outer(X: torch.Tensor) -> torch.Tensor:
    X = X.double()
    return (X.T @ X).cpu()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-model", required=True, help="Teacher: tokenizer, config, weights")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--norm-c", action="store_true",
                   help="Second pass: fit the per-layer bias constant a `norm:r<RANK>+c` fold "
                        "adds under the root, off the o bases already in --out-dir. Writes "
                        "norm_c.safetensors and leaves every other artifact alone.")
    p.add_argument("--layers", default="32-63")
    p.add_argument("--data-preset", default=DEFAULT_MIXTURE)
    p.add_argument("--data-source", action="append", default=None,
                   metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]",
                   help="Override the preset with explicit sources (repeatable).")
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-rank", type=int, default=2048,
                   help="Basis columns kept per projection. Ask for every rank you might "
                        "want now: re-cutting means re-running the forward pass.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank = dist.get_rank()
    dev = torch.cuda.current_device()

    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    layers = parse_layers(args.layers)
    plan = plan_track_layout(manifest.n_tracks, dist.get_world_size(), 1, allow_merge=False)
    cfg = AutoConfig.from_pretrained(args.hf_model)
    text_cfg = getattr(cfg, "text_config", cfg)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // text_cfg.num_attention_heads)

    student = PTWrappedModel(
        text_config=text_cfg, n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=list(range(manifest.num_layers)),
        track_group=layout.track_group, fuse_size=plan.fuse_size,
    )
    student.set_sync_phase("exact")
    student.load_track_state_dicts(
        {t: load_track(args.checkpoint_dir, t) for t in layout.local_track_ids}, strict=True)
    student = student.to(torch.bfloat16).to(dev).eval()
    _log(rank, f"[init] n_tracks={manifest.n_tracks} layers={format_layers(layers)} "
               f"batches={args.batches}x{args.seq_len} "
               + (f"norm_c only, ranks {NORM_LADDER}" if args.norm_c
                  else f"max_rank={args.max_rank}"))
    U_o: dict = {}
    sketch_stats: dict = {}
    if args.norm_c and rank == 0:
        from safetensors import safe_open

        bases_path = Path(args.out_dir) / BASES
        if not bases_path.is_file():
            raise SystemExit(f"[error] --norm-c reads o's bases from {bases_path}")
        with safe_open(str(bases_path), framework="pt", device="cpu") as f:
            for li in layers:
                U_o[li] = f.get_slice(f"layers.{li}.o.U")[:, :max(NORM_LADDER)].to(dev)
        sketch_stats = {li: {k: [] for k in ("true", *(f"r{r}" for r in NORM_LADDER))}
                        for li in layers}
    ends = sorted({layers[0], layers[-1]})
    check_o_columns(student, read_hf_tensors(
        args.hf_model, [f"layers.{li}.self_attn.o_proj.weight" for li in ends]), ends)
    _log(rank, f"[rail] o columns ok at layers {ends}: each track's o slice is its own "
               f"head's teacher columns")

    tok = AutoTokenizer.from_pretrained(args.hf_model)
    sources = ([parse_source_spec(s) for s in args.data_source] if args.data_source
               else preset_sources(args.data_preset))
    loader = DataLoader(PackedTokenStream(tok, CalibrationDataConfig(
        sources=sources, seq_len=args.seq_len, seed=args.seed,
    )), batch_size=1, num_workers=0)

    C_x: dict[int, torch.Tensor] = {}
    S_A: dict[int, torch.Tensor] = {}
    o_sq: dict[int, torch.Tensor] = {}
    n_tok = 0
    for n, batch in enumerate(loader):
        if n >= args.batches:
            break
        ids = batch["input_ids"].to(dev)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        taps = _Taps(student, layers)
        # The scalar fit needs R (the previous layer's synced post-MLP) for every batch.
        probe = {} if n == 0 or args.norm_c else None
        with torch.no_grad():
            student(input_ids=ids, probe_capture=probe)
        taps.close()
        cos = []
        for li in layers:
            A, sq = taps.a[li], taps.o_sq[li]
            dist.all_reduce(A, group=layout.track_group)  # Σ over EVERY rank's tracks
            dist.all_reduce(sq, group=layout.track_group)  # each row is filled on one rank
            if probe is not None and n == 0:
                cos.append(check_capture(student, probe, taps.x[li], A, li))
            if rank == 0:
                if args.norm_c:
                    R = probe[(li - 1, "mlp")].flatten(0, 1)
                    for k, v in norm_scalar_errors(
                            R, A, U_o[li], text_cfg.rms_norm_eps).items():
                        sketch_stats[li][k].append(v.cpu())
                    continue
                o_sq[li] = sq if li not in o_sq else o_sq[li].add_(sq)
                for acc, X in ((C_x, taps.x[li]), (S_A, A)):
                    G = _outer(X)
                    if li in acc:
                        acc[li].add_(G)
                    else:
                        acc[li] = G
        if cos:
            _log(rank, f"[rail] capture ok: input bit-exact, A vs synced delta cos "
                       f"min {min(cos):.4f} mean {sum(cos) / len(cos):.4f}")
        n_tok += ids.numel()
        del taps, probe
        _log(rank, f"[calib] batch {n + 1}/{args.batches}")

    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return 0

    # ----- Rank 0 alone: the student is done, so its GPU memory goes to eigh. -----
    del student
    torch.cuda.empty_cache()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.norm_c:
        consts, _ = norm_c_report(sketch_stats, layers)
        save_file(consts, str(out / NORM_C), metadata={"n_tokens": str(n_tok)})
        print(f"\n  wrote {out / NORM_C} ({len(consts)} constants)", flush=True)
        return 0

    o_rms = o_rms_from_squares(o_sq, n_tok)
    save_file(o_rms, str(out / O_RMS), metadata={"n_tokens": str(n_tok)})
    print(f"[o_rms] {len(o_rms)} layers ({format_layers(layers)}), channel RMS "
          f"{min(float(v.min()) for v in o_rms.values()):.3g}.."
          f"{max(float(v.max()) for v in o_rms.values()):.3g} → {out / O_RMS}", flush=True)
    C_x = {li: m / n_tok for li, m in C_x.items()}
    S_A = {li: m / n_tok for li, m in S_A.items()}
    x_rms = {}
    for li, m in C_x.items():
        x = input_rms(m).float()
        if not (torch.isfinite(x).all() and (x > 0).all()):
            raise SystemExit(f"[rail] layer {li}: C_x has a non-positive or non-finite diagonal "
                             f"entry — every weight reading that channel would tie")
        x_rms[f"layers.{li}.x_rms"] = x.contiguous()
    save_file(x_rms, str(out / X_RMS))

    names = [f"layers.{li}.self_attn.{p}_proj.weight" for li in layers for p in ("q", "k", "v")]
    weights = read_hf_tensors(args.hf_model, names)
    bases, energy = compute_bases(weights, C_x, S_A, layers, head_dim, args.max_rank, dev)
    save_file(bases, str(out / BASES))
    save_file(latent_rms(weights, C_x, S_A, bases, layers, dev), str(out / Z_RMS))
    (out / "bases.json").write_text(json.dumps({
        "layers": layers, "max_rank": args.max_rank, "n_tokens": n_tok,
        "sync_phase": "exact", "batches": args.batches, "seq_len": args.seq_len,
        "energy": energy,
    }, indent=1))

    print(f"\n===== captured moment energy, mean over {format_layers(layers)} "
          f"(SIZING ONLY — never rank arms with it) =====")
    print(f"{'proj':>5} " + " ".join(f"{'r' + str(r):>8}" for r in RANK_LADDER))
    for name in PROJS:
        row = []
        for r in RANK_LADDER:
            vals = [energy[str(li)][name].get(str(r)) for li in layers]
            vals = [v for v in vals if v is not None]
            row.append(f"{sum(vals) / len(vals):8.4f}" if vals else f"{'-':>8}")
        print(f"{name:>5} " + " ".join(row))
    print(f"\n  wrote {out / BASES}, {out / X_RMS}, {out / Z_RMS}; run --norm-c next for "
          f"a `+c` fold scalar", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
