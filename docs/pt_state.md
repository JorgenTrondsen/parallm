# parallm — state and onboarding

Where the parallel-track (PT) conversion stands, the one result the codebase is
now built around, and where the code lives. Written to onboard a run without
re-deriving the months of dead ends behind it.

**Locked premise (do not violate):** *more tracks, fewer syncs.* Never "fix" a
quality gap by using fewer tracks, a lower `D`, or more syncs — those work but
defeat the point. Valid levers keep `N` high and the sync budget low.

## 1. Architecture

A dense model is sliced across `N` tracks (at `N=16`: one attention head + an MLP
slice per track). The forward runs all tracks lockstep. Between **sync
boundaries**, each track adds only its *own* partial residual update; at a
boundary one all-reduce recombines them — the only cross-track collective
(`SyncBoundary`, [model/sync.py](../src/parallm/model/sync.py)):

```
h_synced = h_pre_block + Σ_t (h_t − h_pre_block)
```

`D` = layers between syncs. Between two syncs a mid-window layer reads only its
track's residual (~`1/N` of the real update), so `D>1` cannot be recovered by any
purely *statistical* estimate of the missing content — that was proven, at
length, refuted (predictors, stale caches, fixed low-rank, geometry, on-policy
training, window-parallel rewiring, cross-head seams). Per-track weights are
schedule-independent; the schedule is chosen at serve/eval time, not baked into
the slice.

## 2. The result the code is built around — sparse-copy recomputation

The between-sync trajectory of any track is a **deterministic function of the last
synced residual and that track's weights**. So instead of *estimating* the missing
cross-track content, each rank **recomputes** the other tracks by replaying their
sublayers through **cheap copies** of their weights — comm-free.

The copies that hold quality are **activation-aware (Wanda) sparse** ones,
optionally over an int-quantized base (**qwanda**): score each weight
`|w_ij|·‖x_j‖` from one dense calibration pass and keep the top `1−frac` per
output row (survivors exact). Measured on the 9B slice, % of the `zero→oracle`
(dense) downstream headroom recovered:

| copy | D=2 (16 syncs) | D=4 (8) | D=8 (4 syncs) | bits/w |
|---|---|---|---|---|
| `wanda:0.5` | 98.0% | 97.7% | **99.4%** | 9 |
| `qwanda:4:0.5` (int4 + 50% sparse) | — | — | **89.8%** | 3 |

**`wanda:0.5` is depth-invariant at ~98–99% of dense quality down to 4 sync
events**, training-free, and its cheapness is *structural* (composes with an
already-quantized base). The frontier is a sparsity↔memory menu; a sub-1 GB
per-track pool is closed in this family (the replica must be a whole-network copy
— attn-only and MLP-only copies each collapse; both together = 99.4%).

The pool is static and accessed layer-sequentially, and every `D=8` pass stalls
≥20 ms at each of 4 sync boundaries, so it **streams from pinned host DRAM behind
the stalls**: a one-window ring (~626 MB) is as good as a full double buffer, cutting
resident HBM ~55% at +0 ms once a real multi-node sync costs ≥40 ms (measured,
[bench_stream_overlap.py](../scripts/bench_stream_overlap.py)).

Streaming is **window-granular and CUDA-graph-compatible** (2026-07-12): ring
slots sit at fixed device addresses the captured windows bake, and all
copy/event work runs in the eager boundary regions — streamed + graphed decode
is bit-identical to resident (`--residency streamed`, rails in
[tests/test_engine.py](../tests/test_engine.py)). On top of it, the codes plane
(the only compressible one: int4 codes carry 2.97/4 bits, the bitmap is AT the
entropy limit) can stream **entropy-coded** (`--pool-codec ent`,
[entropy_codec.py](../src/parallm/entropy_codec.py) +
[scripts/repack_replicas_entropy.py](../scripts/repack_replicas_entropy.py)):
a pool-wide static Huffman table, GPU-decoded into the ring during the stalls
(batched block-parallel triton decoder, ~26 GB/s on int4 pools — int8-heavy
pools drop to ~13 GB/s, the LUT gather scatters), lossless ⇒ tokens stay
bit-identical. Measured on the 9B `qwanda:4:0.5` floor pool (2.50 GB → 2.11 GB):
per-window copies 49.7 → 40.8 ms on a 12.7 GB/s shared link. On a dedicated
PCIe4 node (~26 GB/s) the ent window (~528 MB) ≈ the 20 ms stall budget —
streaming is ~free at S=20 where raw (+4 ms/window) is not, and both are free
at S≥25 or PCIe5.

## 3. Where it lives

- **Convert:** [slicer/](../src/parallm/slicer/) + [scripts/convert.py](../scripts/convert.py) (one streaming converter for bf16 / NVFP4, dense / MoE) → per-track `safetensors` + manifest.
- **Copies (the payload):** [model/replica.py](../src/parallm/model/replica.py) — `collect_input_norms` (dense calibration), `wanda_prune_weight` / `fake_quant_weight` / `block_wanda_prune_weight` (the per-weight transforms), `degrade_track_layers` (build a track's replica pool). Rails: [tests/test_replica.py](../tests/test_replica.py).
- **Forward:** [model/pt_model.py](../src/parallm/model/pt_model.py) `PTWrappedModel` (lockstep window iteration + `SyncBoundary`).
- **Eval:** [eval/fidelity.py](../src/parallm/eval/fidelity.py) (KL/ppl), [eval/downstream.py](../src/parallm/eval/downstream.py) + [eval/lm_eval_adapter.py](../src/parallm/eval/lm_eval_adapter.py). **Judge recovery by downstream retention** (arc_challenge / winogrande / piqa), not KL/ppl — the proxy hid a real failure once (KL ~85% while hard-reasoning was ~22–33%).

## 4. The inference engine (built)

[engine.py](../src/parallm/engine.py) + [scripts/serve_cli.py](../scripts/serve_cli.py):
lockstep decode, track-as-batch forward, CUDA-graphed windows, simulated
inter-node link (`--latency-ms`), streamed/entropy-coded pool residency (§2).
27B decode ≈ 108 ms/tok at S=20 (5 rounds × 20 ms + ~8 ms) resident; streamed
adds `Σ_w max(0, window_bytes/BW − S)` — on the shared-PCIe sim box BW is
~12.7 GB/s/rank (pairs of GPUs share host links; a real one-rank-per-node
deployment keeps the full link). Inference centralizes embed + lm_head on a
head node; per track ≈ own bf16 blocks + a streamed replica ring + KV/state.

**Node-envelope fit (measured 2026-07-13, 9B/N=16 ent-streamed, S=20).** The
serve panel prints a per-rank HBM ledger. Against the deployment envelope of
8 GB VRAM / 16 GB DRAM per node:

| node | VRAM steady | breakdown | host DRAM |
|---|---|---|---|
| track node (1 track, projected from measured peers) | **4.04 GB** | 0.83 own blocks + 1.42 ring(ent) + 1.79 KV/graphs/act | 2.11 GB pinned |
| head node (embed+lm_head add 3.79 GB) | **~7.8 GB** | tight; shed the head's own track via non-uniform `tracks_per_rank_list` if it overflows | 2.11 GB pinned |

Speed on an uncontended link (world=1 arm; the engine realizes ~21–23 GB/s of
the box's 25.9 GB/s PCIe4): ent copies 25.0 ms/window → **+19.8 ms/tok** at
S=20 (raw: 27.3 ms → +29.3); at S=40 measured **+0.0 — fully hidden**; the
crossover is S ≈ 25 ms, and full PCIe4 or PCIe5 puts ent at ≈ the S=20 stall
already. Verdict: the 9B-class config fits the 8/16 node with ~2× VRAM
headroom on track nodes, and stall-hiding follows the law exactly — free from
S≥25 on this link, S=20 on full PCIe4/PCIe5. Anything above the ~2.1–2.5 GB
pool class cannot hide at S=20/PCIe4: 27B (9.14 GB ent wire) adds ~+270 ms/tok
and its own bf16 slice alone breaks 8 GB; GLM-class needs the MoE
active-expert tier account (`docs/glm_sizing.md`).

## 5. The refuted program (recovering `D>1` quality without recomputation)

Everything that tried to *estimate* the missing cross-track content — statistical
predictors, stale/temporal caches, fixed low-rank buses, permutation/rotation
geometry, staggered cross-attention, Jacobi refinement, phased/post-attn and
window-parallel sync placement, on-policy reverse-KL training, cross-head seams,
low-rank co-trained replicas, L+S decompositions — was refuted end-to-end. That
code and its distillation trainer were removed in the parallm pivot; the full
negative record lives in the project memory and in git at tag
`pre-parallm-pivot`. Do not re-run any of it without a genuinely new idea.
