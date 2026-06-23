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

**Open, in-constraint, untested:**

- **Reasoning-heavy data** — the one in-recipe lever never tried (keep the
  recipe, change the corpus toward arc/winogrande-like reasoning).
- **Co-trained rotation** — an orthogonal basis learned jointly (the only basis
  hope left; expensive, no positive signal yet).
- **Any cheap cross-track channel** that the §3 intervention harness scores with
  real `zero → oracle` downstream headroom.

**Bottom line:** teacher-exact recovery at N=16 / low-comm looks unlikely. The
honest open question is which in-constraint lever buys the most downstream
retention — and that is now a fast, calibrated measurement, not an hours-long
guess.
