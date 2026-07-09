"""Step 2 for the NVFP4 27B: calibrate on the dequantized model (multi-GPU),
sweep PER-SUBLAYER quant-aware replica configs, pack the winner into one file.

The base is mixed precision — NVFP4 (4-bit) MLPs, FP8 (8-bit) mixers — so a replica
copy is only cheap if it matches the base per sublayer: MLP copies at int4, mixer
copies at int8. Candidates are (frac, mlp_bits, mixer_bits); ``None`` = bf16.

Selection metric = per-Linear activation-weighted reconstruction relMSE
(``‖(W−Ŵ)·diag‖x‖‖² / ‖W·diag‖x‖‖²``) — it sees the quantization tax. A config is
DEPLOYABLE only if it doesn't exceed the base precision per family (mlp ≤ 4,
mixer ≤ 8); we pick the smallest-byte deployable config under ``--max-proxy``.

    python scripts/build_replicas_27b.py \
        --tracks-dir convert_out/qwen3_6_27b_n8_tracks \
        --hf-model <NVFP4 dir> --out convert_out/qwen3_6_27b_n8_tracks/replicas.safetensors
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open

from parallm.model.replica import (
    _WANDA_KEY,
    collect_input_norms,
    fake_quant_weight,
    norms_for,
    wanda_prune_weight,
)
from parallm.model.replica_pack import pack_sparse_weight, save_replica_pool
from parallm.utils.checkpoint import load_manifest

CANDIDATES = [  # (name, frac, mlp_bits, mixer_bits)   None = bf16 survivors
    ("q4mlp/q8mix:0.5", 0.5, 4, 8),   # match the mixed base exactly
    ("q4mlp/q8mix:0.55", 0.55, 4, 8),
    ("qwanda:4:0.5 (uniform)", 0.5, 4, 4),   # int4 everywhere (mixer under base)
    ("qwanda:8:0.5 (mlp>base)", 0.5, 8, 8),  # int8 everywhere (mlp over base)
    ("wanda:0.5 (bf16 ceiling)", 0.5, None, None),
]


def _is_mlp(rel: str) -> bool:
    return rel.startswith("mlp.")


def _bits_for(rel, mlp_bits, mixer_bits):
    return mlp_bits if _is_mlp(rel) else mixer_bits


def _deployable(mlp_bits, mixer_bits) -> bool:
    # a copy must not exceed the base precision per family (NVFP4 mlp / FP8 mixer)
    return mlp_bits is not None and mlp_bits <= 4 and mixer_bits is not None and mixer_bits <= 8


def _degrade(w, frac, in_norms, bits):
    wq = fake_quant_weight(w, bits) if bits is not None else w
    return wanda_prune_weight(wq, frac, in_norms)


def _fam_bytes(params, frac, bits):
    vb = bits if bits is not None else 16  # bitmap (1) + survivors at value-bits
    return (1.0 + (1.0 - frac) * vb) * params / 8


def _dequant_text_state_dict(hf_model: str) -> "dict[str, torch.Tensor]":
    """bf16 text state_dict keyed for Qwen3_5TextModel; drops lm_head/vision/MTP."""
    from convert_nvfp4_27b import _dequant, _slicer_key
    wm = json.load(open(os.path.join(hf_model, "model.safetensors.index.json")))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, s in wm.items():
        by_shard.setdefault(s, []).append(k)
    sd: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(hf_model, shard), framework="pt") as f:
            present = set(f.keys())
            for k in ks:
                key = _slicer_key(k)
                if key is None or key == "lm_head.weight":
                    continue
                base = k.rsplit(".", 1)[0]
                sib = {t: f.get_tensor(f"{base}.{t}") for t in ("weight_scale", "weight_scale_2")
                       if f"{base}.{t}" in present}
                sd[key] = _dequant(f.get_tensor(k), sib)
    return sd


def _load_dense_multigpu(hf_model, tcfg):
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    with init_empty_weights():
        model = Qwen3_5TextModel(tcfg)
    model.load_state_dict(_dequant_text_state_dict(hf_model), strict=False, assign=True)
    model.eval()
    layer_cls = type(model.layers[0]).__name__
    n = torch.cuda.device_count()
    dmap = infer_auto_device_map(
        model, max_memory={i: "34GiB" for i in range(n)}, no_split_module_classes=[layer_cls],
    )
    return dispatch_model(model, device_map=dmap)


def _linear_items(shard_path):
    with safe_open(shard_path, framework="pt") as f:
        for key in f.keys():
            if not (key.startswith("layers.") and key.endswith(".weight")):
                continue
            parts = key.split(".")
            rel = ".".join(parts[2:-1])
            if rel not in _WANDA_KEY:
                continue
            w = f.get_tensor(key)
            if w.ndim == 2:
                yield int(parts[1]), rel, w


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks-dir", required=True)
    p.add_argument("--hf-model", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--proxy-layers", type=int, default=8, help="# layers sampled for the recon proxy")
    p.add_argument("--max-proxy", type=float, default=0.05, help="max recon relMSE for a deployable pick")
    args = p.parse_args()

    out_path = args.out or os.path.join(args.tracks_dir, "replicas.safetensors")
    manifest = load_manifest(args.tracks_dir)
    n_tracks = manifest.n_tracks
    shard = lambda t: os.path.join(args.tracks_dir, f"track_{t}.safetensors")

    # ----- 1. calibrate on the dequantized dense model (multi-GPU) -----
    from transformers import AutoConfig, AutoTokenizer
    tcfg = AutoConfig.from_pretrained(args.hf_model).text_config
    print("[calib] loading dequantized 27B across GPUs…", flush=True)
    model = _load_dense_multigpu(args.hf_model, tcfg)
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    from build_replicas import get_calib_batches  # wikitext / random fallback (same scripts dir)
    batches = get_calib_batches(tok, args.calib_batches, args.seq_len)
    norms = collect_input_norms(model, batches, device="cuda:0")
    del model
    torch.cuda.empty_cache()
    print(f"[calib] norms for {len(norms)} input spaces", flush=True)

    # ----- 2. sweep: per-Linear recon proxy (sampled) + per-family pool bytes -----
    step = max(1, manifest.num_layers // args.proxy_layers)
    sample_layers = set(range(0, manifest.num_layers, step))
    num = {name: 0.0 for name, *_ in CANDIDATES}
    den = 0.0
    mlp_params = mixer_params = 0
    for t in range(n_tracks):
        for li, rel, w in _linear_items(shard(t)):
            if _is_mlp(rel):
                mlp_params += w.numel()
            else:
                mixer_params += w.numel()
            if li not in sample_layers or t % 3:
                continue
            w = w.cuda()
            nv = norms_for(norms, n_tracks, li, rel, w.shape[-1], t).cuda().float()
            wn = w.float() * nv[None, :]
            den += float(wn.pow(2).sum())
            for name, frac, mlp_bits, mixer_bits in CANDIDATES:
                dw = _degrade(w, frac, nv, _bits_for(rel, mlp_bits, mixer_bits)).float()
                num[name] += float(((w.float() - dw) * nv[None, :]).pow(2).sum())

    print(f"\n[sweep] replicated params: MLP {mlp_params/1e9:.1f}B + mixer {mixer_params/1e9:.1f}B "
          f"(× pooled per node)")
    print(f"{'config':26} {'pool GB':>8} {'recon relMSE':>13} {'deployable':>11}")
    rows = []
    for name, frac, mlp_bits, mixer_bits in CANDIDATES:
        gb = (_fam_bytes(mlp_params, frac, mlp_bits) + _fam_bytes(mixer_params, frac, mixer_bits)) / 2**30
        proxy = num[name] / max(den, 1e-12)
        dep = _deployable(mlp_bits, mixer_bits)
        rows.append((name, frac, mlp_bits, mixer_bits, gb, proxy, dep))
        print(f"{name:26} {gb:8.2f} {proxy:13.4f} {'yes' if dep else 'ceiling':>11}")

    eligible = [r for r in rows if r[6] and r[5] <= args.max_proxy]
    if not eligible:
        best = min((r for r in rows if r[6]), key=lambda r: r[5])
        raise SystemExit(f"[error] no deployable config under recon relMSE {args.max_proxy}; "
                         f"lowest is {best[0]} at {best[5]:.4f}. Raise --max-proxy to accept it.")
    sel = min(eligible, key=lambda r: r[4])  # smallest pool
    name, frac, mlp_bits, mixer_bits, gb, proxy, _ = sel
    print(f"\n[select] {name} — smallest deployable pool ({gb:.2f} GB), recon relMSE {proxy:.4f}")

    # ----- 3. pack the winner (per-Linear bits) → one file -----
    pool = {}
    for t in range(n_tracks):
        for li, rel, w in _linear_items(shard(t)):
            nv = norms_for(norms, n_tracks, li, rel, w.shape[-1], t)
            pool[(t, li, rel)] = pack_sparse_weight(w, frac, nv, bits=_bits_for(rel, mlp_bits, mixer_bits))
        print(f"[pack] track {t}: {sum(1 for k in pool if k[0]==t)} Linears", flush=True)
    meta = {"config": name, "frac": str(frac), "mlp_bits": str(mlp_bits), "mixer_bits": str(mixer_bits),
            "n_tracks": str(n_tracks), "recon_relmse": f"{proxy:.4f}",
            "base": "nvfp4-mlp/fp8-mixer", "source": os.path.abspath(args.tracks_dir)}
    save_replica_pool(out_path, pool, meta)
    print(f"\n[ok] wrote {out_path} ({os.path.getsize(out_path)/2**30:.2f} GB, "
          f"{len(pool)} Linears, config={name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
