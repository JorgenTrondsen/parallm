"""Build the cheap-replica pool for a family's track shards → one packed file.

Step 2 of the parallm flow, family-general (bf16 / NVFP4 / FP8 base, dense or MoE):

  1. Calibrate: instantiate the full (unsliced) text model from a bf16-dequantized
     stream of the checkpoint, dispatch across GPUs, and collect the Wanda
     per-input-channel norms (2-D Linears via ``collect_input_norms``; MoE routed
     experts via ``collect_expert_input_norms``).
  2. Degrade every track's weights to the ``--config`` copy and pack them (survivor
     bitmap + int codes / bf16 survivors; 3-D expert slabs per expert) into one
     ``replicas.safetensors``; verify a sample reloads bit-exactly.

``--config`` grammar (choose to match the base precision per sublayer):
    wanda:0.5            bf16 survivors, 50% sparse       (the quality ceiling)
    qwanda:4:0.5         int4 everywhere                  (int4/bf16 base)
    q4mlp/q8mix:0.5      int4 MLP + int8 mixer, 50%       (NVFP4-mlp / FP8-mixer base)

Config *selection* is exploration — use ``scripts/probe_moe_replica.py`` / the
project's recorded frontier; this builder just packs the chosen config.

    python scripts/build_replicas.py --tracks-dir convert_out/<name> \
        --hf-model <checkpoint dir> --config qwanda:4:0.5
"""
from __future__ import annotations

import argparse
import os
import re

import torch

from parallm.model.replica import (
    collect_expert_input_norms,
    collect_input_norms,
    norms_for,
)
from parallm.model.replica_build import (
    get_calib_batches,
    iter_track_experts,
    iter_track_linears,
    load_calib_text_model,
)
from parallm.model.replica_pack import (
    load_replica_pool,
    pack_expert_slab,
    pack_sparse_weight,
    save_replica_pool,
    unpack_expert_slab,
    unpack_sparse_weight,
)
from parallm.utils.checkpoint import load_manifest


def parse_config(s: str) -> "tuple[float, int | None, int | None]":
    """'wanda:<f>' → (f, None, None); 'qwanda:<b>:<f>' → (f, b, b);
    'q<mb>mlp/q<xb>mix:<f>' → (f, mb, xb) (per-family MLP / mixer bits)."""
    if "/" in s:
        left, frac = s.rsplit(":", 1)
        m = re.fullmatch(r"q(\d+)mlp/q(\d+)mix", left)
        if not m:
            raise ValueError(f"bad per-family --config {s!r}; expected 'q<mlp>mlp/q<mix>mix:<frac>'")
        return float(frac), int(m.group(1)), int(m.group(2))
    parts = s.split(":")
    if parts[0] == "wanda" and len(parts) == 2:
        return float(parts[1]), None, None
    if parts[0] == "qwanda" and len(parts) == 3:
        return float(parts[2]), int(parts[1]), int(parts[1])
    raise ValueError(f"bad --config {s!r}; expected 'wanda:<f>' / 'qwanda:<b>:<f>' / 'q<b>mlp/q<b>mix:<f>'")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks-dir", required=True, help="Output of scripts/convert.py")
    p.add_argument("--hf-model", required=True, help="Checkpoint dir (for Wanda calibration)")
    p.add_argument("--config", required=True, help="Copy config, e.g. wanda:0.5 / qwanda:4:0.5 / q4mlp/q8mix:0.5")
    p.add_argument("--out", default=None, help="Packed file (default <tracks-dir>/replicas.safetensors)")
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    args = p.parse_args()

    frac, mlp_bits, mixer_bits = parse_config(args.config)
    bits_for = lambda rel: mlp_bits if rel.startswith("mlp.") else mixer_bits

    out_path = args.out or os.path.join(args.tracks_dir, "replicas.safetensors")
    manifest = load_manifest(args.tracks_dir)
    n_tracks = manifest.n_tracks
    shard = lambda t: os.path.join(args.tracks_dir, f"track_{t}.safetensors")
    has_experts = any(True for _ in iter_track_experts(shard(0)))

    # ----- 1. calibrate on the full dequantized text model (multi-GPU) -----
    from transformers import AutoConfig, AutoTokenizer

    from parallm.adapters import get_adapter_for_config
    cfg = AutoConfig.from_pretrained(args.hf_model)
    tcfg = getattr(cfg, "text_config", cfg)
    adapter = get_adapter_for_config(tcfg)
    print(f"[calib] loading {args.hf_model} across GPUs…", flush=True)
    model = load_calib_text_model(args.hf_model, adapter, tcfg)
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    batches = get_calib_batches(tok, args.calib_batches, args.seq_len)
    norms = collect_input_norms(model, batches, device="cuda:0")
    expert_norms = collect_expert_input_norms(model, batches, device="cuda:0") if has_experts else {}
    del model
    torch.cuda.empty_cache()
    print(f"[calib] {len(norms)} Linear input spaces"
          + (f" + experts on {len(expert_norms)} MoE layers" if has_experts else ""), flush=True)

    # ----- 2. pack every track (2-D Linears + 3-D expert slabs) → one file -----
    print(f"[config] packing {args.config} (frac={frac}, mlp_bits={mlp_bits}, mixer_bits={mixer_bits})")
    pool: dict = {}
    for t in range(n_tracks):
        for li, rel, w in iter_track_linears(shard(t)):
            nv = norms_for(norms, n_tracks, li, rel, w.shape[-1], t)
            pool[(t, li, rel)] = pack_sparse_weight(w, frac, nv, bits=bits_for(rel))
        for li, name, w in iter_track_experts(shard(t)):
            gu_norm, dn_norm = expert_norms[li]  # [E, H], [E, I]
            if name.endswith("gate_up_proj"):
                en = gu_norm  # input = full hidden, same on every track
            else:  # down_proj input = this track's slice of the intermediate
                s = w.shape[2]
                en = dn_norm[:, t * s:(t + 1) * s]
            pool[(t, li, name)] = pack_expert_slab(w, frac, en, bits=mlp_bits)
        print(f"[pack] track {t}: {sum(1 for k in pool if k[0] == t)} weights", flush=True)

    meta = {"config": args.config, "frac": str(frac), "mlp_bits": str(mlp_bits),
            "mixer_bits": str(mixer_bits), "n_tracks": str(n_tracks), "moe": str(has_experts),
            "source": os.path.abspath(args.tracks_dir)}
    save_replica_pool(out_path, pool, meta)
    print(f"\n[ok] wrote {out_path} ({os.path.getsize(out_path)/2**30:.2f} GB, "
          f"{len(pool)} packed weights, config={args.config})")

    # ----- 3. verify a sample reloads bit-exactly -----
    back, back_meta = load_replica_pool(out_path)
    assert back_meta["config"] == args.config and set(back) == set(pool)
    key = next(iter(pool))
    unpack = unpack_expert_slab if "num_experts" in pool[key] else unpack_sparse_weight
    assert torch.equal(unpack(back[key]), unpack(pool[key]))
    print(f"[verify] reload OK ({len(back)} weights); sample {key} unpacks bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
