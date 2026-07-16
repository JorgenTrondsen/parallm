"""Probe the ACTIVE-EXPERTS-ONLY replica lever on an MoE proxy (Qwen3.6-35B-A3B).

The GLM-5.2 deployment target is a 744B / ~40B-active MoE: 256 routed experts,
top-8 per token. A naive replica that degrades *all* experts costs ~272 GB; but a
track only ever needs the experts a token actually routes to, so the pool should
scale with ACTIVE params (~8/256) — IF routing on the degraded (between-sync)
hidden still picks the right experts. We measure both on the same-topology proxy:

  A. Working set — how many distinct experts fire (per token / per sequence), and
     the resulting pool-GB projection (active vs all-256).
  B. Degraded routing — build qwanda:bits:frac expert copies and compare to exact:
       mode 1  exact                      (reference ppl + reference top-8 indices)
       mode 2  degraded experts, FORCED exact routing   (pure weight-degrade tax)
       mode 3  degraded experts, degraded routing        (the real replica; + drift)
     Report top-8 recall (mode3 vs exact) and ppl for modes 2/3. Lever holds if
     mode3 ≈ mode2 (drift benign) and the working set is small.

  C. Expert-level temporal locality (the NVMe→DRAM→HBM tier constants, 2026-07-13):
     churn@h  = fraction of position-t experts NOT routed at position t−h;
     cover@w  = fraction found in the union of the previous w positions' sets
                (per-layer persistence prefetch hit rate);
     LRU(f)   = hit rate of a global (layer, expert)-slab LRU sized to a
                fraction f of the full pool (the DRAM-cache law);
     plus a greedy decode-trace consistency check (teacher-forced positions vs
     true autoregressive generation).

  D. Miss tolerance — degraded experts AND dropped contributions: a "miss"
     zeroes that expert's routed contribution for that token (no renorm — the
     faithful model of a replica slab that didn't arrive; conservative vs the
     real system, where the own-track exact slice still contributes). Arms:
     window-miss (drop exactly the experts a w-token persistence cache lacks)
     and random-miss at fixed rates. Reports ppl vs the no-miss mode-3 base.

Reuses parallm.model.replica.{fake_quant_weight,wanda_prune_weight} verbatim; the
fused-expert calibration lives here (Qwen-specific 3D layout — promote to
replica.py once GLM's expert layout is known).

    .venv/bin/python scripts/probe_moe_replica.py \
        --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<sha> \
        --calib-batches 4 --eval-batches 8 --window 5 --fracs 0.5 0.55 0.65
"""
from __future__ import annotations

import argparse
import sys

import torch

from parallm.model.replica import (
    collect_expert_input_norms,
    fake_quant_weight,
    wanda_prune_weight,
)
from parallm.model.replica_build import get_calib_batches


# ----------------------------------------------------------------------------- #
# model access
# ----------------------------------------------------------------------------- #
def _load(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kw)
    except (ValueError, KeyError):  # not registered as image-text-to-text
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    return model.eval(), tok


def _text_layers(model):
    """The decoder layers, regardless of the CausalLM vs ConditionalGeneration wrap."""
    m = model.model
    lm = getattr(m, "language_model", m)  # ConditionalGeneration nests a language_model
    return lm.layers


def _moe_layers(layers):
    """(layer_idx, decoder_layer, sparse_moe_block) for every MoE layer (has .experts)."""
    out = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            out.append((i, layer, mlp))
    return out


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def _nll(model, batches):
    """Mean per-token next-token NLL over the batches (perplexity = exp(nll))."""
    tot_nll, tot_tok = 0.0, 0
    for ids in batches:
        logits = _logits(model, ids).float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:].to(lp.device)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tot_nll += float(nll.sum())
        tot_tok += tgt.numel()
    return tot_nll / max(tot_tok, 1)


# ----------------------------------------------------------------------------- #
# router capture / forcing (Measurement A + mode 2/3)
# ----------------------------------------------------------------------------- #
class _RouterTap:
    """Forward-hook on each mlp.gate: records the top-8 indices of the latest call;
    (if _forced is set) overrides them with teacher indices for mode 2; (if
    _drop_window / _drop_rand is set) zeroes the routing weight of "missed"
    experts for measurement D; (if trace is set) accumulates per-call indices
    for decode-trace analysis."""

    def __init__(self, moe, num_experts):
        self.last = {}       # layer_idx -> (n_tokens, top_k) indices of last forward
        self._forced = None  # dict layer_idx -> forced indices, or None
        self.E = num_experts
        self._drop_window = None  # int w: drop experts outside the prev-w-position union
        self._drop_rand = None    # float p: drop each routed slot with prob p
        self.drop_count = 0
        self.slot_count = 0
        self.trace = None    # dict layer_idx -> list of per-call indices, or None
        self.handles = [mlp.gate.register_forward_hook(self._mk(i)) for i, _, mlp in moe]

    def _mk(self, li):
        def hook(_mod, _inp, out):
            logits, scores, indices = out
            self.last[li] = indices.detach()
            if self.trace is not None:
                self.trace[li].append(indices.detach().cpu())
            if self._forced is not None and li in self._forced:
                fi = self._forced[li].to(indices.device)
                probs = torch.softmax(logits.float(), dim=-1).gather(-1, fi)
                probs = probs / probs.sum(-1, keepdim=True)
                return logits, probs.to(scores.dtype), fi
            if self._drop_window is not None or self._drop_rand is not None:
                if self._drop_window is not None:
                    miss = _window_miss_mask(indices, self._drop_window, self.E)
                else:
                    miss = torch.rand(scores.shape, device=scores.device) < self._drop_rand
                self.drop_count += int(miss.sum())
                self.slot_count += miss.numel()
                return logits, scores.masked_fill(miss, 0), indices
            return out
        return hook

    def reset_drop_stats(self):
        self.drop_count = self.slot_count = 0

    def snapshot(self):
        return {li: v.clone() for li, v in self.last.items()}

    def remove(self):
        for h in self.handles:
            h.remove()




def degrade_experts(moe, norms, frac, bits):
    """Replace every expert's fused matrices in-place with qwanda copies. Returns a
    CPU stash of the originals for restore. Loops experts (clear > vectorized)."""
    stash = {}
    for li, _, mlp in moe:
        gu_norm, dn_norm = norms[li]
        for name, in_norm in (("gate_up_proj", gu_norm), ("down_proj", dn_norm)):
            p = getattr(mlp.experts, name)
            stash[(li, name)] = p.data.detach().to("cpu", copy=True)
            w = p.data
            for e in range(w.shape[0]):
                we = fake_quant_weight(w[e], bits) if bits is not None else w[e]
                w[e] = wanda_prune_weight(we, frac, in_norm[e].to(w.device))
    return stash


def restore_experts(moe, stash):
    for li, _, mlp in moe:
        for name in ("gate_up_proj", "down_proj"):
            getattr(mlp.experts, name).data.copy_(stash[(li, name)].to(getattr(mlp.experts, name).device))


def _window_miss_mask(indices, w, E):
    """[T, k] bool: True where indices[t, j] is NOT in the union of the previous
    w positions' expert sets at this layer (position-major single sequence).
    The first w positions are treated as warm (prefill fills the cache)."""
    T = indices.shape[0]
    oh = torch.nn.functional.one_hot(indices, E).sum(1).bool()  # [T, E]
    cached = torch.zeros_like(oh)
    for j in range(1, min(w, T - 1) + 1):
        cached[j:] |= oh[:-j]
    hit = cached.gather(1, indices)
    hit[: min(w, T)] = True
    return ~hit


# ----------------------------------------------------------------------------- #
# measurement C: expert-level temporal locality
# ----------------------------------------------------------------------------- #
def _churn_coverage(snapshots, E, horizons=(1, 2, 4, 8), windows=(1, 2, 4, 8, 16)):
    """Position-sequential locality, averaged over layers and traces.
    churn@h = fraction of position-t experts not in position t−h's set;
    cover@w = fraction of position-t experts in the union of the previous w
    positions' sets (== 1 − the miss rate a w-token persistence cache pays)."""
    ch = {h: [] for h in horizons}
    cv = {w: [] for w in windows}
    for snap in snapshots:
        for li in snap:
            oh = torch.nn.functional.one_hot(snap[li], E).sum(1).bool()  # [T, E]
            for h in horizons:
                if h < oh.shape[0]:
                    both = (oh[h:] & oh[:-h]).sum()
                    ch[h].append(1.0 - float(both) / float(oh[h:].sum()))
            acc = torch.zeros_like(oh)
            for j in range(1, max(windows) + 1):
                if j >= oh.shape[0]:
                    break
                acc[j:] |= oh[:-j]
                if j in cv:
                    hit = (acc & oh)[j:].sum()
                    cv[j].append(float(hit) / float(oh[j:].sum()))
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return {h: mean(v) for h, v in ch.items()}, {w: mean(v) for w, v in cv.items()}


def _lru_hit_curve(snapshots, n_moe, E, cap_fracs):
    """Global LRU over (layer, expert) slabs on the position-major access stream
    (all layers of position t before position t+1). Capacity = frac × n_moe × E
    slabs — the DRAM-cache hit law. Cold-start misses included (≤ cap per trace)."""
    from collections import OrderedDict

    out = {}
    for frac in cap_fracs:
        cap = max(1, int(round(frac * n_moe * E)))
        hits = tot = 0
        for snap in snapshots:
            layers = sorted(snap)
            idx = {li: snap[li].cpu().tolist() for li in layers}
            lru: OrderedDict = OrderedDict()
            for t in range(len(idx[layers[0]])):
                for li in layers:
                    for e in idx[li][t]:
                        key = (li, e)
                        if key in lru:
                            hits += 1
                            lru.move_to_end(key)
                        else:
                            lru[key] = None
                            if len(lru) > cap:
                                lru.popitem(last=False)
                        tot += 1
        out[frac] = hits / max(tot, 1)
    return out


# ----------------------------------------------------------------------------- #
# measurements
# ----------------------------------------------------------------------------- #
def _recall_at_k(a, b, num_experts):
    """Mean fraction of a's top-k experts also in b's, per token (a,b: [tokens,k])."""
    oh = lambda t: torch.nn.functional.one_hot(t, num_experts).sum(1)  # [tokens, E] 0/1
    inter = (oh(a) * oh(b)).sum(-1)
    return float((inter.float() / a.shape[-1]).mean())


def _distinct_vs_tokens(snapshots, token_counts):
    """Mean distinct experts/layer among T consecutive tokens of ONE forward, for each
    T, averaged over the snapshots. T=1 is single-stream decode, large T is prefill /
    wide batch. Characterizes where the active-expert win holds."""
    out = {}
    for t in token_counts:
        per_batch = []
        for snap in snapshots:
            d = [snap[li][:t].reshape(-1).unique().numel() for li in snap]
            per_batch.append(sum(d) / len(d))
        out[t] = sum(per_batch) / len(per_batch)
    return out


def per_expert_params(mlp):
    gu = mlp.experts.gate_up_proj[0].numel()
    dn = mlp.experts.down_proj[0].numel()
    return gu + dn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--window", type=int, default=5, help="sync window (# MoE layers) for the stream projection")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.55, 0.65])
    p.add_argument("--bits", type=int, default=4, help="int bits for the expert copy (None-like: pass 0 for bf16)")
    p.add_argument("--sections", default="AB",
                   help="measurements to run: A working-set, B degraded-routing "
                        "(the 2026-07-09 result), C temporal locality, D miss tolerance")
    p.add_argument("--decode-tokens", type=int, default=256,
                   help="C: greedy decode-trace length for the autoregressive "
                        "churn consistency check (0 = skip)")
    p.add_argument("--lru-fracs", type=float, nargs="+",
                   default=[0.025, 0.05, 0.1, 0.2, 0.4, 0.6],
                   help="C: LRU cache sizes as fractions of the full slab pool")
    p.add_argument("--miss-windows", type=int, nargs="+", default=[1, 4, 16],
                   help="D: drop experts outside the prev-w-position union")
    p.add_argument("--miss-rands", type=float, nargs="+", default=[0.05, 0.10],
                   help="D: drop each routed slot at random with prob p")
    args = p.parse_args()
    bits = None if args.bits == 0 else args.bits

    print(f"[load] {args.model}", flush=True)
    model, tok = _load(args.model)
    layers = _text_layers(model)
    moe = _moe_layers(layers)
    E = moe[0][2].experts.num_experts
    top_k = moe[0][2].gate.top_k
    per_exp = per_expert_params(moe[0][2])
    print(f"[model] {len(layers)} layers, {len(moe)} MoE, E={E}, top_k={top_k}, "
          f"per-expert params={per_exp/1e6:.2f}M", flush=True)

    dev = next(model.parameters()).device
    calib = [b["input_ids"].to(dev) for b in get_calib_batches(tok, args.calib_batches, args.seq_len)]
    evalb = [b["input_ids"].to(dev) for b in get_calib_batches(tok, args.eval_batches, args.seq_len)]

    # ---- self-check: every router tapped; indices are a valid top_k set ----
    tap = _RouterTap(moe, E)
    _logits(model, evalb[0])
    assert len(tap.last) == len(moe), f"tapped {len(tap.last)} routers, expected {len(moe)}"
    li0 = moe[0][0]
    idx0 = tap.last[li0]
    assert idx0.shape[-1] == top_k and int(idx0.max()) < E and int(idx0.min()) >= 0
    assert (idx0.sort(-1).values.diff(dim=-1) > 0).all(), "router indices not unique per token"
    print(f"[self-check] all {len(moe)} routers tapped; layer{li0} indices {tuple(idx0.shape)} "
          f"unique & in [0,{E})", flush=True)

    # ---- calibrate expert norms (exact weights; only B/D degrade) ----
    norms = None
    if set("BD") & set(args.sections):
        print("[calib] collecting per-expert Wanda norms…", flush=True)
        norms = collect_expert_input_norms(model, [{"input_ids": ids} for ids in calib], device=dev)

    # ---- Measurement A: working set + exact reference (indices + ppl) ----
    exact_idx = []   # per eval batch: {li: indices}
    for ids in evalb:
        _logits(model, ids)
        exact_idx.append(tap.snapshot())
    # distinct experts per layer over ALL eval tokens (batch/prefill view)
    distinct = []
    for li, _, _ in moe:
        seen = torch.cat([snap[li].reshape(-1) for snap in exact_idx]).unique()
        distinct.append(seen.numel())
    distinct = torch.tensor(distinct, dtype=torch.float)
    exact_nll = _nll(model, evalb)

    tot = len(moe) * E * per_exp
    print("\n===== Measurement A: expert working set =====")
    print(f"exact ppl = {torch.tensor(exact_nll).exp():.3f}   (nll {exact_nll:.4f})")
    print(f"distinct experts/layer over {sum(b.numel() for b in evalb)} eval tokens: "
          f"mean {distinct.mean():.1f}  max {distinct.max():.0f}  of {E}")
    print(f"per-token active = top_k×layers = {top_k*len(moe)} expert-slabs "
          f"({top_k}/{E} = {100*top_k/E:.1f}% per layer)")
    tcounts = [t for t in (1, 2, 4, 8, 16, 64, 256) if t <= evalb[0].shape[1]]
    curve = _distinct_vs_tokens(exact_idx, tcounts)
    print("distinct experts/layer vs tokens-in-flight (the decode-regime curve):")
    for t in tcounts:
        act = curve[t]
        gb = (len(moe) * act * per_exp) * (1.0 + 0.5 * (bits or 16)) / 8 / 2**30
        print(f"    T={t:>4} tokens: {act:6.1f}/{E} experts/layer  → pool {gb:6.2f} GB @frac0.5")
    for frac in args.fracs:
        w = 1.0 + (1.0 - frac) * (bits or 16)
        all_gb = tot * w / 8 / 2**30
        act_gb = (len(moe) * top_k * per_exp) * w / 8 / 2**30
        win_gb = (args.window * top_k * per_exp) * w / 8 / 2**30
        print(f"  frac {frac:.2f} (bpw {w:.2f}): pool all-256 {all_gb:7.2f} GB | "
              f"per-token active {act_gb:6.2f} GB | {args.window}-layer window ring {win_gb:6.3f} GB")

    # ---- Measurement C: expert temporal locality (tier constants) ----
    if "C" in args.sections:
        print("\n===== Measurement C: expert temporal locality (tier constants) =====")
        churn, cover = _churn_coverage(exact_idx, E)
        print("churn@h (fraction of a position's experts new vs position t−h):")
        print("    " + "  ".join(f"h={h}: {churn[h]:.3f}" for h in sorted(churn)))
        print("cover@w (persistence-prefetch hit rate: union of prev w positions):")
        print("    " + "  ".join(f"w={w}: {cover[w]:.3f}" for w in sorted(cover)))
        lru = _lru_hit_curve(exact_idx, len(moe), E, args.lru_fracs)
        print("LRU(f) slab hit rate at DRAM cache = f × full pool "
              f"({len(moe)}×{E} slabs):")
        print("    " + "  ".join(f"f={f:g}: {lru[f]:.3f}" for f in sorted(lru)))
        if args.decode_tokens > 0:
            prompt = evalb[0][:, :64]
            tap.trace = {li: [] for li, _, _ in moe}
            with torch.no_grad():
                model.generate(prompt, max_new_tokens=args.decode_tokens,
                               do_sample=False)
            trace = {li: torch.cat(v)[prompt.shape[1]:]
                     for li, v in tap.trace.items()}
            tap.trace = None
            n_dec = trace[moe[0][0]].shape[0]
            dch, dcv = _churn_coverage([trace], E)
            print(f"decode-trace consistency ({n_dec} greedy tokens): "
                  f"churn@1 {dch[1]:.3f} (teacher-forced {churn[1]:.3f}) | "
                  f"cover@8 {dcv[8]:.3f} (teacher-forced {cover[8]:.3f})", flush=True)

    # ---- Measurement B: degraded routing quality + selection drift ----
    if "B" in args.sections:
        print("\n===== Measurement B: degraded routing (mode2=forced, mode3=drift) =====")
        print(f"{'frac':>5} {'mode2 ppl':>10} {'mode3 ppl':>10} {'recall@8':>9} {'drift Δppl':>11}")
    for frac in args.fracs if "B" in args.sections else []:
        stash = degrade_experts(moe, norms, frac, bits)
        # mode 3: degraded routing (natural forward)
        tap._forced = None
        drift_recall = []
        for k, ids in enumerate(evalb):
            _logits(model, ids)
            for li, _, _ in moe:
                drift_recall.append(_recall_at_k(exact_idx[k][li], tap.last[li], E))
        mode3_nll = _nll(model, evalb)
        # mode 2: forced exact routing per batch
        mode2_tot, mode2_tok = 0.0, 0
        for k, ids in enumerate(evalb):
            tap._forced = exact_idx[k]
            logits = _logits(model, ids).float()
            tap._forced = None
            lp = torch.log_softmax(logits[:, :-1], -1)
            tgt = ids[:, 1:].to(lp.device)
            mode2_tot += float(-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
            mode2_tok += tgt.numel()
        mode2_nll = mode2_tot / mode2_tok
        restore_experts(moe, stash)
        r = sum(drift_recall) / len(drift_recall)
        print(f"{frac:5.2f} {torch.tensor(mode2_nll).exp():10.3f} "
              f"{torch.tensor(mode3_nll).exp():10.3f} {r:9.3f} "
              f"{torch.tensor(mode3_nll).exp()-torch.tensor(mode2_nll).exp():11.3f}", flush=True)

    # ---- Measurement D: miss tolerance under degraded experts ----
    if "D" in args.sections:
        frac0 = args.fracs[0]
        print(f"\n===== Measurement D: miss tolerance (qwanda:{args.bits}:{frac0} "
              f"experts; miss = zeroed contribution, no renorm) =====")
        stash = degrade_experts(moe, norms, frac0, bits)
        base_ppl = float(torch.tensor(_nll(model, evalb)).exp())
        print(f"{'arm':>15} {'miss rate':>9} {'ppl':>8} {'Δppl':>8}")
        print(f"{'no-miss mode3':>15} {0.0:9.3f} {base_ppl:8.3f} {0.0:8.3f}", flush=True)
        for w in args.miss_windows:
            tap._drop_window = w
            tap.reset_drop_stats()
            ppl = float(torch.tensor(_nll(model, evalb)).exp())
            tap._drop_window = None
            rate = tap.drop_count / max(tap.slot_count, 1)
            print(f"{'window w=%d' % w:>15} {rate:9.3f} {ppl:8.3f} "
                  f"{ppl - base_ppl:8.3f}", flush=True)
        for pr in args.miss_rands:
            tap._drop_rand = pr
            tap.reset_drop_stats()
            ppl = float(torch.tensor(_nll(model, evalb)).exp())
            tap._drop_rand = None
            rate = tap.drop_count / max(tap.slot_count, 1)
            print(f"{'random p=%.2f' % pr:>15} {rate:9.3f} {ppl:8.3f} "
                  f"{ppl - base_ppl:8.3f}", flush=True)
        restore_experts(moe, stash)

    tap.remove()
    print("\n[done] lever holds if mode3≈mode2 (benign drift) and working set << 256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
