"""Probe the cheap-replica sparsity/quant config and save the pool as ONE packed file.

Step 2 of the parallm flow. Given the per-track shards from
``convert_qwen3_5_9b.py``:

  1. Calibrate: one pass over the DENSE model collects the Wanda per-input-channel
     norms (``model.replica.collect_input_norms``).
  2. Sweep candidate configs (activation-aware sparsity ``wanda:<f>`` and int4
     ``qwanda:4:<f>`` — L+S / low-rank was validated but byte-dominated, so it is
     not in the deployable menu). For each: the packed pool size and a norm-only
     pruned-energy proxy (the Wanda importance mass discarded — lower = better),
     annotated with the prior *measured* downstream retention at this sync depth.
  3. Select the smallest-memory config whose measured retention clears the target,
     pack every track's Linears (survivor bitmap + int4 codes / bf16 survivors),
     and write them to one ``replicas.safetensors``.

The quality column is the prior end-to-end downstream probe (recorded in the
project's findings); the local proxy is a same-run sanity check that this slice
ranks the configs as expected — it does not by itself certify a downstream %.

    python scripts/build_replicas.py \
        --tracks-dir convert_out/qwen3_5_9b_n16_tracks \
        --hf-model <dense 9B path> \
        --out convert_out/qwen3_5_9b_n16_tracks/replicas.safetensors
"""
from __future__ import annotations

import argparse
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
from parallm.model.replica_pack import (
    _bits_per_weight,
    load_replica_pool,
    pack_sparse_weight,
    save_replica_pool,
    unpack_sparse_weight,
)
from parallm.utils.checkpoint import load_manifest

# (name, frac, bits) — the deployable menu (wanda sparsity + optional int4).
CANDIDATES = [
    ("qwanda:4:0.55", 0.55, 4),
    ("qwanda:4:0.5", 0.5, 4),
    ("wanda:0.6", 0.6, None),
    ("wanda:0.55", 0.55, None),
    ("wanda:0.5", 0.5, None),
]

# Prior end-to-end downstream retention (% of dense headroom) at D=8 / 4 sync
# events, from the project's intervention-harness probing (see docs/pt_state.md,
# project_cross_track_estimator memory). The quality anchor for selection.
REF_RETENTION = {
    8: {
        "wanda:0.5": 0.994, "wanda:0.55": 0.940, "wanda:0.6": 0.851,
        "qwanda:4:0.5": 0.898, "qwanda:4:0.55": 0.842,
    },
}


def _linear_items(shard_path: str):
    """Yield (layer_idx, rel, weight) for every decoder Linear in a track shard
    whose input space Wanda calibrates (rel in _WANDA_KEY)."""
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not (key.startswith("layers.") and key.endswith(".weight")):
                continue
            parts = key.split(".")
            li = int(parts[1])
            rel = ".".join(parts[2:-1])
            if rel not in _WANDA_KEY:
                continue
            w = f.get_tensor(key)
            if w.ndim != 2:  # skip conv / norm weights that slip through
                continue
            yield li, rel, w


def get_calib_batches(tokenizer, n_batches: int, seq_len: int):
    """Real-text calibration batches from the wikitext preset; random-token
    fallback if the dataset can't be reached (flow still runs, norms weaker)."""
    try:
        from torch.utils.data import DataLoader

        from parallm.train.data import CalibrationDataConfig, PackedTokenStream, preset_sources
        cfg = CalibrationDataConfig(sources=preset_sources("wikitext"), seq_len=seq_len, seed=0)
        loader = DataLoader(PackedTokenStream(tokenizer, cfg), batch_size=1)
        out = []
        for b in loader:
            ids = b["input_ids"]
            out.append({"input_ids": ids if ids.ndim == 2 else ids.unsqueeze(0),
                        "attention_mask": b.get("attention_mask")})
            if len(out) >= n_batches:
                break
        if out:
            print(f"[calib] {len(out)} wikitext batches × {seq_len} tokens", flush=True)
            return out
    except Exception as e:  # noqa: BLE001
        print(f"[calib] wikitext unavailable ({e}); falling back to random tokens", flush=True)
    V = tokenizer.vocab_size if tokenizer is not None else 150000
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, V, (1, seq_len), generator=g),
             "attention_mask": None} for _ in range(n_batches)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tracks-dir", required=True, help="Output of convert_qwen3_5_9b.py")
    p.add_argument("--hf-model", required=True, help="Dense model path (for Wanda calibration)")
    p.add_argument("--out", default=None, help="Packed replica file (default <tracks-dir>/replicas.safetensors)")
    p.add_argument("--sync-depth", type=int, default=8, help="D (layers between syncs); selects the quality anchor")
    p.add_argument("--target-quality", type=float, default=0.95, help="Min downstream retention to clear")
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_path = args.out or os.path.join(args.tracks_dir, "replicas.safetensors")
    manifest = load_manifest(args.tracks_dir)
    n_tracks = manifest.n_tracks
    shard = lambda t: os.path.join(args.tracks_dir, f"track_{t}.safetensors")

    # ----- 1. calibrate on the dense model -----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[calib] loading dense {args.hf_model} on {args.device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device).eval()
    text_model = model.model  # the Qwen3_5TextModel (has .layers and a callable forward)
    batches = get_calib_batches(tok, args.calib_batches, args.seq_len)
    norms = collect_input_norms(text_model, batches, device=args.device)
    del model, text_model
    torch.cuda.empty_cache()
    print(f"[calib] collected norms for {len(norms)} input spaces", flush=True)

    dev = args.device

    # ----- 2. sweep: pruned-energy proxy + packed bytes per config -----
    pruned_sq = {name: 0.0 for name, _, _ in CANDIDATES}
    total_sq = {name: 0.0 for name, _, _ in CANDIDATES}
    total_params = 0
    for t in range(n_tracks):
        for li, rel, w in _linear_items(shard(t)):
            w = w.to(dev)
            nv = norms_for(norms, n_tracks, li, rel, w.shape[-1], t).to(dev)
            total_params += w.numel()
            for name, frac, bits in CANDIDATES:
                wu = fake_quant_weight(w, bits) if bits is not None else w
                score = wu.float().abs() * nv.float()[None, :]
                k = int(round(frac * score.shape[1]))
                thr = score.kthvalue(k, dim=1, keepdim=True).values if k > 0 else None
                pr = (score <= thr) if thr is not None else torch.zeros_like(score, dtype=torch.bool)
                pruned_sq[name] += float((score * pr).pow(2).sum())
                total_sq[name] += float(score.pow(2).sum())
    total_params //= n_tracks  # per-track replicated params (report per track)

    print(f"\n[sweep] per-track replicated params ≈ {total_params/1e6:.0f}M "
          f"(× {n_tracks} tracks pooled per node)")
    print(f"{'config':16} {'pool GB':>8} {'bits/w':>7} {'proxy(lower)':>13} "
          f"{'retention@D'+str(args.sync_depth):>16} {'clears':>7}")
    ref = REF_RETENTION.get(args.sync_depth, {})
    rows = []
    for name, frac, bits in CANDIDATES:
        pool_bytes = _bits_per_weight(frac, bits) * (total_params * n_tracks) / 8
        proxy = pruned_sq[name] / max(total_sq[name], 1e-12)
        ret = ref.get(name)
        clears = ret is not None and ret >= args.target_quality
        rows.append((name, frac, bits, pool_bytes, proxy, ret, clears))
        rr = f"{ret:.3f}" if ret is not None else "  (n/a)"
        print(f"{name:16} {pool_bytes/2**30:8.2f} {_bits_per_weight(frac,bits):7.2f} "
              f"{proxy:13.4f} {rr:>16} {'yes' if clears else 'no':>7}")

    # ----- 3. select smallest-memory config that clears the target -----
    clearing = sorted([r for r in rows if r[6]], key=lambda r: r[3])
    if not clearing:
        raise SystemExit(
            f"[error] no candidate clears retention target {args.target_quality} at D={args.sync_depth}. "
            f"Lower --target-quality, or the D={args.sync_depth} anchor table is missing."
        )
    sel_name, sel_frac, sel_bits, sel_bytes, _, sel_ret, _ = clearing[0]
    print(f"\n[select] {sel_name} — smallest pool ({sel_bytes/2**30:.2f} GB) with "
          f"retention {sel_ret:.3f} ≥ {args.target_quality} at D={args.sync_depth}")

    # ----- 4. pack the selected config for every track → one file -----
    pool: dict = {}
    for t in range(n_tracks):
        for li, rel, w in _linear_items(shard(t)):
            nv = norms_for(norms, n_tracks, li, rel, w.shape[-1], t)
            pool[(t, li, rel)] = pack_sparse_weight(w, sel_frac, nv, bits=sel_bits)
        print(f"[pack] track {t}: {sum(1 for k in pool if k[0]==t)} Linears", flush=True)
    meta = {
        "config": sel_name, "frac": str(sel_frac), "bits": str(sel_bits),
        "sync_depth": str(args.sync_depth), "n_tracks": str(n_tracks),
        "retention": str(sel_ret), "source": os.path.abspath(args.tracks_dir),
    }
    save_replica_pool(out_path, pool, meta)
    size_gb = os.path.getsize(out_path) / 2**30
    print(f"\n[ok] wrote {out_path}  ({size_gb:.2f} GB, {len(pool)} packed Linears, config={sel_name})")

    # ----- 5. verify a sample reloads bit-exactly -----
    back, back_meta = load_replica_pool(out_path)
    assert back_meta["config"] == sel_name and set(back) == set(pool)
    key = next(iter(pool))
    assert torch.equal(unpack_sparse_weight(back[key]), unpack_sparse_weight(pool[key]))
    print(f"[verify] reload OK ({len(back)} Linears); sample {key} unpacks bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
