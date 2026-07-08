# PT conversion — state, mechanics, and the cheap-probe playbook

A working map of where the N=16 / D=2 parallel-track (PT) recovery stands, what
the training actually does, **why the remaining gap is structural rather than a
training deficiency**, and the rule for testing new ideas cheaply. Written to
onboard a future run without re-deriving months of dead ends.

The locked premise (do not violate): **more tracks, fewer syncs.** Never "fix"
the gap by using fewer tracks, a lower D, or more syncs — those work but defeat
the entire point. Valid levers keep N high and the sync budget low.

---

## 1. What the architecture is

A dense Qwen3.5-9B is sliced across `N=16` tracks (one attention head / a slice
of each MLP per track). The forward runs all tracks lockstep. Between **sync
boundaries**, each track adds only its *own* partial residual update; at a
boundary one NCCL all-reduce recombines them.

The only cross-track collective is `SyncBoundary`
([model/sync.py](../src/pt_converter/model/sync.py)):

```
h_synced = h_pre_block + Σ_t (h_t − h_pre_block)
```

`D=2` means a sync every 2 layers (17 boundaries, placed **before each
full-attention layer** so the global mixer reads a synced residual). The
per-track weights are **schedule-independent** — boundaries are placed at train
time, not baked into the slice.

**The structural fact that drives everything:** between two syncs, a mid-window
layer reads only its track's residual — roughly `1/N = 1/16` of the real
residual update. For a full-attention layer (the global token mixer) that is
information-destroying by construction. `D=1` (sync every layer) recovers the
dense model (KL ≈ 0.38); `D≥2` cannot, because the mid-window layers literally
never see the other tracks' contributions.

---

## 2. What the training does — and what it provably can't

Distillation ([train/distill.py](../src/pt_converter/train/distill.py)) is a sum
of weighted terms. The dominant one is **block-MSE**:

- For each window, start from the teacher's hidden (teacher-forced), run the
  per-track layers, `SyncBoundary` the partials into a reconstruction, MSE it
  against the teacher's hidden at that depth, backward **immediately**, detach
  before the next window.
- So the block-MSE gradient lives entirely inside the current window's per-track
  layers. **It only pushes the synced reconstruction toward the teacher. It
  never sees cross-track structure** — the tracks don't see each other inside the
  window, so no gradient can teach them to "combine partial residuals correctly."

The other recipe terms each patch a *different* failure mode, none of them the
structural one:

| Term | Flag | What it fixes |
|---|---|---|
| Student-forcing curriculum | `--student-forcing-schedule cosine-full --student-forcing-prob 0.9` | exposure bias: train the deep blocks on the *drifted* inputs they see at inference |
| Intra-window MSE | `--intra-window-mse` | pins mid-window (partial-residual) layers to the teacher trajectory |
| Normalized block-MSE + clamp | `--normalize-block-mse --block-mse-clamp 10` | stops deep high-norm layers dominating / spiking the gradient |
| Adaptive layer weighting | `--adaptive-layer-weight` | steers gradient to the currently-worst band (upper-middle, not the deepest) |
| Free-running MSE + damping | `--free-running-mse --fr-grad-alpha <1` | the one term that sees multi-window compounding; needs boundary-grad damping or it diverges |
| Centered logit-MSE | `--lambda-logit-mse` | output-aware: supervises the residual directions `lm_head` reads (non-saturating) |

**Net result, measured:** distillation took the free-running KL from 7.19 → 0.44
— it fixed *compounding / exposure bias*. But the per-block **teacher-forced**
error is identical untrained vs trained, and downstream hard-reasoning retention
(arc_challenge / winogrande) sits at **~22–33%** of the teacher while the KL/ppl
proxy reads ~85% recovered. **The gap that remains is structural and
training-invariant.** "Is more fine-tuning enough?" → No, and no amount of it can
be: block-MSE is the right tool for compounding, and that job is done.

Judge recovery by **downstream retention**, not val_kl / KL / ppl — the proxy hid
the failure once already.

---

## 3. Why ideas keep costing hours — and the fix

The recurring failure mode in this project's notes is *methodological*: a new
cross-track idea gets judged by a **proxy** measured teacher-forced or untrained —
residual relMSE, SVD energy, `delta_staleness_ratio`
([eval/sensitivity.py](../src/pt_converter/eval/sensitivity.py)) — and then
re-litigated ("inconclusive cheaply", "the proxy hid it"), so it falls back to a
multi-hour training run to actually decide. Proxies on the residual stream do not
reliably predict the downstream metric.

**The cheap-probe playbook (use in this order):**

1. **Sensitivity probe** —
   [scripts/probe_sensitivity.py](../scripts/probe_sensitivity.py). Fast
   per-layer *proxy* triage: `rel_err_partial`, `delta_imbalance`, `gain_cos`,
   `delta_staleness_ratio`, `--svd-energy`. Good for *ruling a lever out* (e.g.
   staleness ≥ 1 ⇒ a naive 1-token cache hurts) and for locating the worst band.
   **Never settle a go-decision on it** — it is a residual-space proxy.

2. **Intervention harness** —
   [scripts/probe_intervention.py](../scripts/probe_intervention.py)
   ([eval/intervention.py](../src/pt_converter/eval/intervention.py)). The
   decisive, forward-only test. It *runs* a candidate cross-track channel inside
   the forward and reads the **real end-to-end metric** (KL + downstream). It is
   self-calibrating: the `oracle` channel reproduces the D=1 ceiling and the
   `zero` channel reproduces the D=2 floor **by construction** (both bit-exact
   anchors, guarded by `tests/test_intervention.py`), so any cheap channel
   (`stale`, `avg:W`, `lowrank:r`) lands between them and its `%head` column is
   its share of the `zero → oracle` headroom. Run it on a **trained** `best/` so
   the compounding confound is already gone — then the headroom isolates the
   cross-track structural value alone.

   Reading it: `oracle ≈ zero` on downstream ⇒ no cross-track headroom, stop.
   `oracle ≫ zero` but every cheap channel ≈ zero ⇒ the content is genuinely
   unreachable cheaply (and now *proven on downstream*, not a proxy). A channel
   recovering a real fraction ⇒ the first quantitative green light for a training
   build, with a measured ceiling.

   *Nuance:* `oracle` is the ceiling **for the checkpoint's fixed weights** — the
   inference-time value of perfect cross-track info holding the trained weights
   constant — not a separately-trained D=1 model. So it need not equal the
   trained-D=1 KL (~0.38); it answers "would *this* model benefit from cross-track
   info," which is exactly the channel question. Use ≥100 batches at the trained
   `--seq-len` for a non-noisy read (a 2-batch smoke is plumbing only).

3. **Only then** a training run.

---

## 4. The lever ledger

**Refuted (don't re-run without a new idea):**

- Cross-track *predictor* (comm-free): val_kl flat.
- Staggered/StagFormer *1-token exact cache*: `delta_staleness_ratio ≥ 1`
  (cross-track delta is volatile token-to-token). *Now testable end-to-end via
  the `stale` / `avg:W` channels — do that before any staggered training build.*
- Low-rank exchange / trained `trunk r=128` bus. The SVD-*energy* proxy was
  ambiguous, but the intervention harness settled it end-to-end: an **adaptive**
  per-input rank-256 projection of the true cross-track content recovers ~96% of
  the downstream headroom, but a **fixed** rank-256 basis (the deployable trunk
  form) lands *below* the D=2 floor — and a *calibrated* fixed basis (PCA over 32
  batches) still fails up to rank 512 (−16%), so it is not a sampling artifact. The
  useful subspace is **input-adaptive**, so no fixed linear bus can capture it on the
  frozen model — confirming and explaining why the trained r=128 trunk was flat. The comm-cheap cross-track *inference* channel is closed in
  every tested form (caches hurt, fixed low-rank hurts, adaptive needs full comm,
  learned predictor flat). A re-implemented **co-trained** trunk (`--trunk-rank`,
  `PTWrappedModel.trunk_augment`, boundary-safe augment-then-subtract) confirmed it:
  warm-start from `best_logit` is stable but mildly *hurts* downstream
  (0.49→0.46); the from-scratch variant oscillated/didn't converge in-constraint
  (the trunk forces no-compile + smaller seq, off the tuned recipe). **The
  cross-track-channel direction is closed** (decision 2026-06-21) and the trunk code
  was removed (kept the negative record in memory); the **intervention harness is
  kept** as the reusable probe. Remaining in-constraint levers: reasoning-targeted
  *data* (a different distribution than qwen-mix's math/code) and co-trained *rotation*.
- Trained back-loaded schedule: per-layer sensitivity ignores long-window
  compounding; downstream stayed at chance.
- Budget-neutral sync **re-placement** (move the fixed 17-sync budget onto the
  highest-value layers). Settled end-to-end by the **partial-oracle headroom map**
  (`MaskedOracleChannel` + `probe_intervention.py --oracle-sweep`, 2026-06-22): the
  real +5–7pt `zero→oracle` downstream headroom is **diffuse** — a 5×3-layer band
  sweep recovered 15–27% per band, an 0.8pt spread that is within the limit-1000
  noise floor (near-additive, Σ≈110%). No hot spot ⇒ no placement beats the existing
  full-attn-aligned uniform schedule. The headroom needs *more comm* or a geometry
  change, not redistribution.
- Permutation basis: hidden-dim coupling is an expander — no reordering makes
  tracks more separable.
- Output objectives (more KL): forward-KL diverges; the stable `kl=0` recipe
  saturates. The objective lever is exhausted.

**Also refuted since (the 2026-06/07 D>1 wave — see memory for each verdict):**

- Reasoning-heavy data, co-trained rotation, staggered cross-attn: all flat or
  divergent end-to-end.
- Every D>1 lever class: stale **delivery** (temporal, depth `window_stale`,
  iteration/Jacobi refine), sync **placement** (phased/post-attn at D=2),
  **training** (on-policy reverse-KL/JSD `a2a`; stable-but-flat continuation),
  write-side memory correction (kv/q gates), and the **architectural**
  window-parallel rewiring (Gate 2: trained slice 0.5167 < the 0.548 D=2
  record). The D=2 residual is structural cross-track content; nothing cheap
  *statistical* delivers it.

**One positive estimator, 2026-07-05 — degraded local RECOMPUTATION (`replica:*`
in the intervention harness).** Every refutation above tried to *estimate* the
missing content (predict / stale-cache / compress a fixed subspace). The
untested family re-derives it: each rank replays all tracks' sublayers from the
last synced residual with **degraded weight copies** (comm-free — a between-sync
trajectory is a deterministic function of the synced residual + weights). On the
untrained slice, %of the zero→oracle (=dense, 0.70) headroom recovered:

| depth | sync events | int8 copies | int4 copies | svd-r128 |
|---|---|---|---|---|
| D=2 | 16 | 100.7% | 96.9% | −5.5% |
| D=4 | 8 | 99.9% | 91.8% | — |
| D=8 | 4 | 100.5% | 88.8% | — |

`replica:exact` reproduces `oracle` to 100.0% (the rail). int8 self-copies
recover **full dense quality at as few as 4 sync events**; int4 erodes with
window depth; the *cheap factorized* form (SVD rank-r) fails (the fixed-rank
ceiling from `fixed-lowrank`/trunk). So the working form is full-rank
low-precision recomputation — a **compute-for-sync-frequency trade** (same shape
as draft–verify): per-rank compute ≈ 0.5× dense (own tracks + int8 shadows of
the rest) buys 4 events at 0.70 quality. Strictly in-premise (comm-free, N=16,
syncs down to 4) but it trades away compute-sharding — replica-int8 ≈ "each rank
runs an int8 copy of the whole model, syncing every D layers to correct drift."
Wins only on a latency/comm-bound multi-node target; pointless single-device.
Likely needs **no co-training** (replica never reads a partial residual ⇒ the raw
slice is already optimal — an inference-time method). Not yet built as a real
deployment. See `project_cross_track_estimator` memory; logs
`train_out/dense_parallel/replica_gate{,_d4,_d8}.log`.

**Requirement update + sparse-copy gate, 2026-07-06.** The premise was clarified:
conversion must work on an **already-quantized base**, so int8/int4 copies are
refuted *by requirement* (their cheapness is precision headroom the base no
longer has; they remain diagnostic anchors above). Copy cheapness must be
**structural**. New channel `replica:prune:<frac>` (per-output-row magnitude
pruning, survivors exact — composes with base quantization), D=2 gate on the
untrained slice (`replica_prune_gate.log`):

| copy | headroom | downstream |
|---|---|---|
| prune:0.5 | **72.0%** | 0.6233 |
| prune:0.75 | 19.0% | 0.4787 |
| prune:0.9 | −0.5% | 0.4253 |

First positive *structural* channel — prune:0.5 nearly doubles svd:256's 38.3%
at similar copy memory, training-free. Plain-magnitude cliff beyond ~50–60%.

**Activation-aware (Wanda-style) copies close the gap (`replica:wanda:<frac>`,
same day):** score `|w|·‖x‖` with one dense calibration pass
(`collect_input_norms`, 128 input spaces, 16×4096 tokens; col-sliced slabs take
the track's slice of the dense norm vector).

| copy | D=2 (16 ev) | D=4 (8 ev) | D=8 (4 ev) |
|---|---|---|---|
| wanda:0.5 | 98.0% (0.6943) | 97.7% (0.6933) | **99.4% (0.6980)** |
| wanda:0.65 | 94.1% (0.6837) | 83.9% (0.6547) | 63.6% (0.5970) |
| wanda:0.75 | 73.0% (0.6260) | — | — |
| int4 (anchor, refuted-by-req) | 96.9% | 91.8% | 88.8% |

**wanda:0.5 is depth-invariant at ~98–99% of dense quality down to FOUR sync
events** (arc_challenge ≥ oracle at D=8), training-free, structural cheapness
that composes with an already-quantized base (copy ≈ half base + sparse index)
— it beats int4 at every depth and matches int8. wanda:0.65 erodes with window
depth (sparser copies drift further before each sync resets them), so ~50%
sparsity is the depth-safe point with plain Wanda. This is the clarified target
architecture at probe level: N=16 tracks, 4 drift-correcting syncs, cheap local
copies between. Caveats: quality-only simulation (real 2:4/sparse kernels
needed for speed claims); per-rank compute between syncs ≈ 0.53× dense
(compute-for-frequency trade — wins on comm/latency-bound multi-node targets);
decode-time shadow replay not yet built.

**Memory-frontier gate (same day, D=8, `replica_sgpt_gate_d8.log`):**

| copy (D=8, 4 events) | bits/w (bf16 base) | headroom | downstream |
|---|---|---|---|
| wanda:0.5 | 9 | 99.4% | 0.6980 |
| **qwanda:4:0.5** (int4 + 50% sparse) | **3** | **89.8%** | **0.6710** |
| sparsegpt:0.65 (reconstructed) | ~6.6 | 66.3% | 0.6047 |
| sparsegpt:0.75 (reconstructed) | ~5 | 15.2% | 0.4607 |
| chanwanda:0.5 (index-free structured) | 8 | 47.4% | 0.5513 |

Findings: (1) **sparsity is ~free on top of int4** — qwanda:4:0.5 ≈ int4's own
88.8% at a fraction of the memory and strictly dominates pure-int4 copies
(3 bits/w ≈ 19% of base ⇒ ~3.6 GB/device total at ≈ D=1-B quality, 4 events).
(2) **SparseGPT reconstruction barely moves >0.5 sparsity at D=8** (+2.7pt over
wanda:0.65) ⇒ the depth erosion is compounding DRIFT, not per-layer
reconstruction error — calibration-only methods are exhausted above 0.5;
the frontier is a sparsity↔depth menu (0.65@D=2 94%, 0.5@D=8 99%).

**2026-07-07 follow-ups (all at D=8, `replica_{blockprof,wanda24}_gate_d8.log`,
co-training logs `…replica_w65_lr{3e5,1e4}.log`): every attempted refinement of
the wanda:0.5 point failed, mapping its optimality.** (a) Sparse-copy
CO-TRAINING (deployed replica forward as the TF chain, frozen lagged copies,
`--replica-copies` in the track trainer, sf-1.0 rail vs
`phased_intervention_forward` passes): wanda:0.65@D=8 dead flat over 1750
steps at lr 3e-5 and no early signal at 1e-4 — drift is not trainable away.
(b) Index-granularity ladder at 0.5: unstructured 99.4% → **2:4 79.9%**
(hardware 2× GEMM costs ~20pt) → block-16 **−1.5%** → channel-structured 47.4%
— per-weight selection freedom is load-bearing; the ~1 bit/weight index is the
price of the content. (c) Non-uniform per-layer budgets (wanda-mass ×
window-depth allocator) HURT: profwanda:0.5 82.6%, profwanda:0.65 18.1% —
uniform per-row wanda is ~optimal. The co-trained **all-low-rank** variant was tried and re-scoped
2026-07-06: three heal arms showed per-slab uniform rank-128 is
capacity-limited at conversion budget, and exact copies make syncs vacuous
(see `project_lowrank_replica_heal`).

**Attn-only replicas (drop the MLP copies) — REFUTED at the content level
2026-07-07 (`replica_attnonly_gate_d8.log`, D=8, anchors zero 0.4177 / oracle
0.6997 reproduce).** Motivation: a <1 GB replica pool/device. MLP is 67.9% of
each track's block params (302M of 445.3M; GDN 101.1M + full-attn 41.9M =
143M), so attn-only qwanda:4:0.5 = 0.80 GB was the only sub-1GB format. New
probe grammar: `replica:<base>:mlp:<sub|none>` (base spec governs mixer
Linears; `mlp:none` drops the MLP copies — the shadow chain advances on
attention deltas only, each track keeps its OWN exact MLP deltas as per-track
offsets reset at boundary syncs; rails: dead-MLP ≡ exact, lossless split).

| channel (D=8, 4 events) | downstream | %headroom |
|---|---|---|
| replica:exact:mlp:none (family CEILING) | 0.4193 | **0.6%** |
| replica:wanda:0.5:mlp:none | 0.4167 | −0.4% |
| replica:qwanda:4:0.5:mlp:none (0.80 GB) | 0.4263 | 3.1% |
| replica:wanda:0.5:mlp:wanda:0.8 | 0.4910 | 26.0% |
| replica:wanda:0.5:mlp:wanda:0.9 | 0.4147 | −1.1% |

Even PERFECT attention replicas with no MLP copies sit at the floor ⇒ ~all
cross-track content rides the MLP deltas (consistent with the kv/q gate: the
residual path is the content, and the residual path is MLP mass). The
fine-tuning arm was not launched per the pre-registered rule (training closes
adaptation gaps, not a missing 15/16 of MLP mass). MLP copies are the most
valuable replica bytes: sparsifying them first (mlp:wanda:0.8/0.9) is strictly
worse than uniform at equal memory; the only open asymmetry is the REVERSE
(sparser attn, MLP at 0.5), ceiling-bounded at ~32% of the pool ⇒ **<1 GB is
unreachable in this family; sub-1GB per track comes from co-location (the pool
is per NODE, shared by K resident tracks) or an untested VQ codebook on the
MLP copies.** Inference memory (centralized embed+lm_head — each track file
duplicates both 1017M-param vocab matrices): track device ≈ 0.89 GB own blocks
(bf16) + 2.5 GB qwanda:4:0.5 pool + shadow KV 32 KB/token (15 replica streams
dedupe to 4 kv-head caches since all replicas read the common shadow input) +
own KV 8 KB/token + O(1) GDN states; head node = 4.07 GB + logit buffers, with
h₀ broadcast / final-hidden gather (8 KB/token) as entry/exit hops, not sync
events. Remaining: the real-deployment build.

**Cheap-replica axes gate 2026-07-07 — user constraints tightened: int4-quantized
base + NO co-location (1 track/node ⇒ full 15-replica pool per device)
(`replica_cheap_gate_d8.log` + `replica_compose_gate_d8.log`, D=8, anchors
reproduce).** Pool bytes = replicated params × [(1−s)·value_bits + index_bits];
the 1-bit/w bitmap at 0.5 density is at its entropy limit, so only three axes
exist: sparsity ↑, value bits ↓, replicated-param share ↓.

| axis | channel | downstream | %headroom |
|---|---|---|---|
| A: sparsity | wanda:0.55 | 0.6827 | **94.0%** (knee) |
| A | wanda:0.6 | 0.6577 | 85.1% |
| B: value bits | qwanda:3:0.5 | 0.4150 | −0.9% (floor) |
| B | qwanda:2:0.5 | 0.4397 | 7.8% (floor) |
| C: attn share | attn 0.8 + MLP 0.5 | 0.5520 | 47.6% |
| C | attn 0.9 + MLP 0.5 | 0.4270 | 3.3% |
| C | attn NONE + MLP 0.5 | 0.4113 | −2.2% (floor) |
| compose | qwanda:4:0.55 (2.34 GB) | 0.6550 | 84.2% |
| compose | qwanda:4:0.6 (2.17 GB) | 0.6253 | 73.6% |

Verdicts: (B) values below 4 bits collapse outright with per-row absmax — the
4→3-bit cliff is total (group-wise scales unrun; floor→90% unprecedented, and
3-bit+g64 ≈ 2.19 GB barely beats wanda:0.55@int4). (C) attention copies are NOT
droppable or strongly compressible even though their content alone is ~nothing:
**the joint-content law** (this gate + the attn-only gate): attn copies alone
0.6%, MLP copies alone −2.2%, both at 0.5 = 99.4% — each sublayer family's
deltas are required to keep the other's inputs on-trajectory; the replica must
be a whole-network copy. (A) the sparsity knee is ~0.55, and the ~10pt int4
value tax is roughly constant across fracs. **Frontier menu under these
constraints: 2.50 GB → 89.8% (qwanda:4:0.5) / 2.34 GB → 84.2% (qwanda:4:0.55) /
2.17 GB → 73.6% (qwanda:4:0.6). A <1 GB pool is definitively closed** — all
three routes to it are dead; second-wave escape hatches (unbuilt, weak priors):
VQ codebooks on the values, low-rank+sparse decomposition. Impl kept
(probe-only): `replica:none:mlp:<sub>` = MLP-only replicas (mixers dropped from
shadow clones — would also eliminate ALL shadow KV/GDN state; own-exact-attn
per-track offsets, reset at boundary syncs; dead-attention ≡ exact rail; 343
tests). qwanda bits 2/3 and `wanda:<f>:mlp:wanda:<f'>` asymmetric specs needed
zero new code.

**Low-rank + sparse (L+S) gate 2026-07-07 (`replica_lsparse_gate_d8.log`, D=8):
format VALIDATED, per-slice economics DOMINATED.** OATS-style alternating
decomposition per copy (`W ≈ L(rank r) + S(frac f sparse)`, activation-scaled,
survivors exact; specs `replica:lsparse:<r>:<f>` / `replica:qlsparse:<b>:<r>:<f>`,
helper `lsparse_decompose_weight`, identity + planted-structure rails, 347
tests):

| channel (D=8) | downstream | %headroom |
|---|---|---|
| wanda:0.7 (pure-sparse control) | 0.5283 | 39.2% |
| lsparse:32:0.7 | 0.6023 | 65.5% |
| lsparse:64:0.7 | 0.6413 | 79.3% |
| lsparse:128:0.7 | 0.6797 | **92.9%** |
| lsparse:64:0.75 | 0.5873 | 60.2% |
| qlsparse:4:64:0.7 (deployable) | 0.6147 | 69.9% |

The complementarity hypothesis is real: pure-L (svd:256 38%) and pure-S (knee
0.55) fail on complementary weight populations, and combining them buys +54pt
over pure sparsity at S=0.7 — the first format to move past wanda's knee. Rank
does not saturate through 128; the int4 tax stays ~10pt. **But per-slice
factors are byte-expensive on thin slices** (r×(m+n) with m+n dominated by the
4096 side): r=128 ⇒ 119M factor params/track ⇒ ×15 int8 = 1.79GB, so the
deployable points (qlsparse:4:64:0.7 = 0.89+1.84 = 2.73 GB @ 69.9%; r=32 =
2.29 GB @ ~56%) are all dominated by qwanda:4:0.5 (2.50 GB @ 89.8%).

**Shared-factor variant measured 2026-07-08 — L+S CLOSED in both forms
(`replica_slsparse_gate_d8.log`, D=8, anchors reproduce).** Built
`compute_shared_lsparse_slices`: each DENSE matrix decomposed once (seeded
randomized SVD; one L per matrix serves all 15 replicas), degraded dense
matrix cut into per-track shadow weights with the converter's own SlicerSpecs;
dense→slice correspondence asserted bit-exact at scale on every slab type
(GatedQ/KVReplicated/FusedSegment/Colwise/Rowwise). Specs
`replica:slsparse:<R>:<f>` / `replica:qslsparse:<b>:<R>:<f>`; identity rail
(rank ≥ full ⇒ ≡ replica:exact); 350 tests.

| channel (D=8) | downstream | %headroom | pool (int8 L + int4 S) |
|---|---|---|---|
| slsparse:256:0.7 (bf16-S science) | 0.6277 | 74.5% | — |
| slsparse:512:0.7 (bf16-S science) | 0.6670 | 88.4% | — |
| qslsparse:4:256:0.7 | 0.5997 | 64.5% | 2.49 GB |
| qslsparse:4:256:0.75 | 0.5560 | 49.1% | 2.26 GB |

Sharing buys byte-efficiency (shared R=512 ≈ per-slice r=128 quality at 1.28 vs
1.79 GB of factors) but never crosses the wanda frontier: at equal bytes the
deployable arms lose by 25pt (2.49 GB: 64.5% vs qwanda:4:0.5's 89.8%) and 35pt
(2.26 GB: 49.1% vs qwanda:4:0.55's 84.2%); the rank slope (~+14/doubling) puts
~93% at R≈1024 ⇒ ~4 GB. **Verdict: L+S is scientifically validated (the only
format past wanda's knee at fixed S-sparsity) but byte-dominated by plain
qwanda in every deployable configuration. The frontier menu stands: 2.50 GB →
89.8% / 2.34 GB → 84.2% / 2.17 GB → 73.6%. The only unbuilt escape hatch left
is VQ codebooks on the values (weakest prior).**

**The surviving result — lever B, D=1 (`--sync-phase post-attn`):** move each
block's single sync to post-attention so the MLP reads the real recombined
residual. 32 events per forward, downstream macro3 **0.665** (lim1000,
arc_challenge/piqa/winogrande) vs teacher 0.70 — the in-premise quality floor.
Checkpoint: `train_out/qwen3_5_9b_n16_d1_postattn/best`.

## 5. Deployment configuration (premise call 2026-07-04)

Sync cost is **frequency per decoded token**, and decode is the only regime
where frequency binds (prefill amortizes its 32 events over the whole
sequence). **Block draft–verify decode** amortizes D=1-B along the token axis:
a Qwen3.5-0.8B draft (replicated per device, zero sync events) proposes k
greedy tokens; ONE D=1-B verify pass over prefix+proposals accepts the
agreeing prefix plus the verifier's own next token. Emitted text is bit-exact
the verifier's greedy decode (rail-tested), so quality is exactly 0.665.

Measured on-generation against the real D=1-B verifier
([probe_draft_verify_decode.py](../scripts/probe_draft_verify_decode.py)):

| k | τ (tok/verify event) | syncs/token | position-compute/token |
|---|---|---|---|
| 8 | 6.35 | **5.0** | ~1.5× |
| 16 | 8.51 | **3.8** | ~2.2× |

Offline teacher-forced τ on data text reads 3.1–3.5 (9–10 syncs/token) — quote
that as the conservative bound for maximally-informative text; on-generation
acceptance is higher because a model's own greedy trajectory is easier to
draft. Both ends dominate D=2 (16 events at 0.548 quality) on both axes.

**Bottom line:** teacher-exact recovery at N=16 / low-comm is refuted in every
tested form; the program's answer is **D=1-B (0.665) + draft–verify decode at
~4–5 sync events per generated token** — a 6–8× sync-frequency win over plain
D=1 decode at zero quality cost. Remaining work is systems engineering
(verifier KV cache, cached draft), not research.

## 6. Streamed replica residency (decode overlap account, measured 2026-07-08)

The qwanda:4:0.5 pool (2.505 GB) does not have to be HBM-resident: replica
weights are static, accessed strictly layer-sequentially, and every D=8 pass
stalls ≥20 ms at each of the 4 sync boundaries — a host-side wait during which
the copy engine keeps running. A device ring buffer of R bytes streamed from a
pinned host-DRAM pool re-streams the full pool every pass (cyclic access
evicts every slab before reuse), and the stalls pay for it. Measured on one
A100 (PCIe4 ×16; sustained overlapped H2D **25.9 GB/s**, probe 21–23;
pageable ~9–13 — pinning is load-bearing) with a byte-exact synthetic
schedule, [bench_stream_overlap.py](../scripts/bench_stream_overlap.py), log
`train_out/dense_parallel/bench_stream_overlap.log`, 2.50 GB moved per pass,
rails green (S=200 ms arms ≈ +0; baseline = 4·S + ~6 ms compute):

| ring R | HBM resident (own 0.89 + R) | added/pass S=20ms | S=40ms | S=60ms |
|---|---|---|---|---|
| none (fully resident) | 3.39 GB | 0 | 0 | 0 |
| 1252 MB (2 windows, the double buffer) | 2.14 GB | +11 ms | 0 | 0 |
| **626 MB (1 window)** | **1.52 GB** | **+12 ms** | **0** | **0** |
| 313 MB | 1.20 GB | +46 ms | +48 ms | +48 ms |
| 157 MB | 1.05 GB | +72 ms | +72 ms | +72 ms |

Laws confirmed: added/pass ≈ max(link saturation: pool/BW − 4·S − compute,
buffer starvation: 4·(W/BW − min(S, R/BW))). Slab granularity is irrelevant
(g = 1/4/8 identical) — only ring bytes matter, so **one window of ring
(626 MB) is exactly as good as the full double buffer**: during one 20 ms
stall the link moves only ~0.5 GB. Streaming the own-track slice too is
dominated everywhere (S=20: +43.5 ms vs +11; S=40: free but 1.70 > 1.52 GB
resident) — keep own weights resident. Disk footnote: this box's NVMe RAID5
reads 12–15 GB/s direct-IO, so even a disk-backed pool only costs ~+120 ms/pass
(slow decode, prefill fine); host pinned DRAM is the recommended backing.

**Deployment account:** track device = 0.89 GB own blocks + **0.63 GB ring**
= 1.52 GB HBM (−55% vs 3.39) + 2.50 GB pinned host DRAM + KV/states, at
89.8%-of-headroom quality (bit-identical weights — no quality dimension), for
+12 ms/pass if sync stalls are exactly 20 ms and **zero added latency once a
real 16-node collective costs ≥40 ms** (one-shot all-gather + jitter typically
does). Zero extra sync events — all traffic is node-local host↔device. Total
bytes don't shrink: streaming moves the pool from HBM to DRAM; if host DRAM
can't hold 2.5 GB either, this lever doesn't apply. Scaling note: at 100B/N=64
the ~37 GB pool sits in host DRAM trivially; per-window slabs (~3.7 GB) hide
behind 40 ms stalls only on ≥PCIe5/C2C-class hosts (PCIe4: ~+90 ms/window).
