# Draft-verify in track mode — measured gates + the 100B-class ledger

2026-07-16, 8×A100-40GB single-node sim (`--latency-ms 20` = the multi-node
link), Qwen3.6-27B-NVFP4 → N=8 tracks (`convert_out/qwen3_6_27b_n8`), D=16
schedule `15,31,47,63`, pool `q4mlp/q8mix:0.5` = 10.31 GB raw / 9.14 GB ent.
Drafter: Qwen/Qwen3.5-0.8B (same tokenizer, verified id-identical), head rank
only — **zero extra bytes on track nodes**.

## Why this arm exists (the elimination)

For 100B+ targets the replica pool cannot fit the 8 GB VRAM / 16 GB DRAM
node: at N=16 a ≤1×track memory budget gives the copy of the other 15 tracks
`b/15` bits per covered param (~1.07 bf16 / ~0.3 NVFP4) — under the measured
floors (0.5-density bitmap = 1 bit/param AT the entropy limit, value bits <3
collapse, sparsity knee 0.55, minimum viable copy ≈ 2.3 bits/param), **every
weight-space copy (incl. VQ) is closed by arithmetic**. Activation-space
channels are refuted end-to-end (pt_state §5). The surviving class is
**amortized exact input**: a drafter proposes k tokens, ONE lockstep verify
chunk gives every layer REAL synced residuals for k+1 positions, emitted ≡
the verifier's greedy by construction. Syncs/token = S_schedule / τ.

## Measured, 27B

| arm | ms/tok | syncs/tok | rounds/tok | notes |
|---|---|---|---|---|
| plain resident (graphs) | mean 106.7, p50 101.7 | 4 | 5 | reproduces the recorded baseline exactly |
| draft-verify k=16 resident | 449 | **1.17** | 1.76 | τ=3.41 (prose prompt), eager verify |
| draft-verify k=32 resident | 1041 | 1.50 | 2.25 | τ=2.67 (prose) |

**The ≤5 syncs/token goal is met with 3–4× margin at the 27B D=16+pool
verifier — 1.2–1.5 syncs/token on the WORST-τ domain (prose), 4/τ ≈ 0.3 on
code.** Rounds include the one extra accept-count broadcast per block.

### τ is domain-dependent (the finding that reframes the program)

0.8B drafter vs the DENSE 27B (exact-verifier proxy, teacher-forced,
128 tokens/prompt — `serve_out/tau_dense.log`):

| domain | τ@k8 | τ@k16 | τ@k32 | τ@k48 | τ@k128 |
|---|---|---|---|---|---|
| prose (history) | 2.67 | 2.72 | 2.72 (sat) | — | 2.72 |
| science prose | 3.05 | 3.20 (sat) | — | — | 3.20 |
| **code** | 7.11 | 10.67 | 12.80 | 14.22 | **16.00 (k-capped)** |

The engine's on-gen τ vs the degraded D=16+pool verifier (2.6–3.4, prose)
matches τ_TF vs DENSE on the same prompt (2.72) ⇒ **the degraded verifier
does NOT depress τ; the drafter-target agreement does.** The record's
τ 7.05–8.61 (0.8B→9B, k-capped) is code-class. Prose saturates by k=16 —
pushing k buys nothing there; only a better/healed drafter moves prose τ.

### Wall-clock decomposition (why 449 ms/tok, and the fix — BUILT 2026-07-17)

torch.profiler corrected the first attribution: the dominant tax was NOT
launch overhead but the packed-GEMV kernel putting chunk positions on the
launch GRID — each of the k+1 verify positions re-read the ENTIRE pool
(925 of ~1280 ms GPU-busy/pass at M=17; graphs can never remove busy time).
The five fixes, all bit-exact plumbing (rails re-verified):
1. **GEMV M-tiling** (`packed_gemv.py`): weight block decoded once, applied
   to all chunk positions via a tl.dot tensor-core tile (BLOCK_O=16-32,
   M_TILE 16-64, swept). Verify-chunk GEMV: 17×→**1.7× the M=1 cost** at
   M=17, 2.1× at M=33. M=1 keeps the original body bit-identical.
2. **CUDA-graphed verify windows** (T=k+1 statics keyed by width; conv
   commits/rollback switched to `copy_` for address stability; the GDN log
   persists across blocks — replays refresh the same graph-pool tensors).
   Serial window compute 942 → 198 ms/pass (k=16). Graphed ≡ eager emitted
   ids verified bit-identical on the 27B.
3. **Engine-run drafter** (`serve_cli.py EngineDrafter`, `--draft-loop`):
   the 0.8B through the engine's own graphed N=1 decode (SyncBoundary
   no-ops, single-chain fast path in the layer walk since own ≡ shadow at
   N=1): 28 → **5.2 ms/step**; catch-up = the graphed step replayed over
   the ≤8 accepted tokens (longer catch-ups: one eager chunk, ~45 ms flat).
   GDN snapshot/restore (~0.3 ms) replaces re-prefill; KV rolls back by
   pointer. Extra bytes on track nodes: still zero.
4. **Accept broadcast rides the draft time** (peers roll back while the
   head drafts): rounds 6 → 5/block.
5. **Batched rollback**: all logged GDN layers stack on the batch dim of
   ONE chunk-kernel call per cache — 84 → 11 ms/block.

### Measured after the fixes (same assets, standalone runs, 96 new tokens)

| arm | ms/tok mean | ms/tok steady* | τ | syncs/tok | vs before |
|---|---|---|---|---|---|
| plain resident (baseline) | 106.4 | p50 101.6 | — | 4 | unchanged ✓ |
| **resident prose k=8** | **95.9** | **86.5** | 2.46 | 1.62 | 449 (k16) |
| **resident code k=32** | **36.7** | **25.8** | 16.33 | 0.24 | — |
| **streamed+ent code k=32** | — | **70.6** | 16.33 | 0.24 | 213 |
| streamed prose k=8 | — | 413.8 | 2.46 | 1.62 | copy-bound (9.14 GB/2.46 = 3.7 GB wire/tok — the drafter lever, not an engineering item) |

*steady = excluding the two one-time warmup/capture blocks (any long
generation amortizes them to zero). **The ≤100 ms/token target is met on
both domains resident (draft-verify now BEATS the plain 107 ms baseline
everywhere) and on code streamed — the 8/16-node arm.** Prose k=8 is the
operating point (τ saturates by k=8; bigger k only pays draft time).
Per-block decomposition at code k=32 (`--profile`): draft 211 ms
(k×5.2 + catch-up), verify serial 345 + syncs 81, rollback 11.
Sweep caveat: multi-k `--draft-k` lists in one process showed occasional
cross-entry slowdowns; ledger numbers come from one-k-per-process runs.

### Streamed pool ÷ τ (the 27B-reopening arm)

Plain streamed 27B is closed on the record: 9.14 GB ent wire/token ≈
+270 ms/tok. Under draft-verify the pool streams once per PASS and serves τ
tokens (measured this session — `serve_out/dv_streamed_*.log`):

| arm | ms/tok | syncs/tok | pool wire/tok | copy/window |
|---|---|---|---|---|
| plain streamed, k=0 (prose) | 942 | 4 | 9.14 GB | 207 ms @ 11.0 GB/s (8-rank-contended link) |
| dv streamed k=16 (prose, τ=3.41) | 636 | 1.17 | 2.68 GB | 207 ms |
| dv streamed k=32 (code, τ=16.33 on-gen) | 213 → **70.6 steady post-fix** | **0.24** | **0.56 GB** | 195 ms @ 11.7 GB/s |

On-gen τ (16.33) exceeds τ_TF (12.80@k32) — the deployed number is the
better one, as the tag-era module predicted. Code-class decode puts the
streamed 27B pool at 0.56 GB/token — far inside the ~2 GB/token wire
envelope that closed plain-streamed 27B — **27B streamed on 8/16-class
nodes is reopened for code-class decode, now at 70.6 ms/token steady
(14.2 tok/s), i.e. FASTER than the plain resident baseline while holding
the pool in host DRAM** (track-node projection 12.30 GB vs 16.86 resident).
Prose streamed stays copy-bound (pool/τ wire) — the drafter lever.

## The 100B-class ledger (pool-free d1b verify)

At 100B+ no pool fits (23–28 GB > 16 GB DRAM), so the verify schedule is
d1b (post-attn every layer, the lever-B schedule; 95% of teacher downstream
at 9B after healing): S ≈ L ≈ 64–80 boundaries/pass, pool = 0 bytes.

- **syncs/token = S/τ**: code τ 13–16 (k≥48) → **4–6/token ≈ the ≤5 goal**;
  prose τ ~3 → 21–27 ✗. The prose gap is the drafter's, not the schedule's:
  the named lever is a bigger + healed drafter (distill 1–4B on the target's
  greedy outputs, off-node) — 0.8B vs 27B is a 34× size gap; the 9B record
  and today's code numbers say the τ ceiling is real headroom, k-capped.
- **Per-rank VRAM at N=16**: bf16 slices don't fit (2P/N = 12.5 GB @100B).
  NVFP4-resident execution ≈ 3.5 GB + KV ⇒ fits 8 GB — the loader currently
  dequantizes to bf16; quantized-resident track execution is the build item
  100B track mode needs regardless of decode schedule.
- **Extra memory**: drafter on the head node only (track nodes: 0 bytes) or
  ≤1×track if replicated; KV/conv/rec rollback state is transient
  (≤ k+1-token chunk logs).
- **Wall-clock envelope** (both engineering legs now BUILT and measured at
  27B: graphed T=k+1 verify, 5.2 ms/step drafter, accept rides the draft):
  ms/tok ≈ (S+1)·20/τ + verify_compute/τ + k·d/τ. At L=64, τ=14, graphed
  verify (~150 ms/pass est.), k=48: ≈ 93 + 11 + 18 ≈ **122 ms/tok ≈ 8
  tok/s** on code-class decode — rounds-dominated, so the remaining 100B
  levers are τ (drafter) and S (schedule), not compute; prose at τ=3 ≈
  450 ms/tok — again, the drafter lever.

### Pool-free (tracks-only nodes) measured at 27B — 2026-07-17

The 100B d1b arm run on today's assets: `--sync-indices 0..63` (64
boundaries, post-attn placement) with `--replicas none` (serve_cli: shadow =
the rank's OWN tracks, so the carry is per-rank — the trained-d1b semantics;
pool never loaded). Same prompts across arms (τ is prompt-sensitive: the
quicksort prompt reads τ 6.86 on the recorded D=16+pool arm, not 16.33).

| arm (same prompts) | track-node HBM | τ | syncs/tok | ms/tok steady | output |
|---|---|---|---|---|---|
| D=16+pool, code k=32 (control) | 16.86 GB | 6.86 | 0.58 | 73.5 | ✓ |
| d1+pool, code k=32 | 16.86 GB | **18.00** | 3.56 | 132.7 | ✓ (clean code) |
| d1+pool, prose k=8 | 16.86 GB | 4.57 | 14.0 | 278.1 | ✓ (greedy loops) |
| **d1 pool-free, code k=32** | **5.88 GB** | (13.62)* | — | (77.2)* | **✗ collapsed** |
| d1 pool-free, prose k=8 | **5.88 GB** | (6.31)* | — | (196.1)* | **✗ collapsed** |

*Collapsed-output τ is inflated by degeneracy ("1. 1. 1." is trivially
draftable) — mechanics-only numbers, not quality-bearing.

Three findings:
1. **The wall-clock envelope is validated at L=64**: every d1 block costs
   ~1300 ms of rounds (65×20) + only ~100–200 ms compute+draft, ÷ τ — at
   code-class τ 13.6–18 that is 77–133 ms/tok, bracketing the ledger's
   ≈122 est. Rounds-dominated, exactly as modeled; 64-window CUDA-graph
   capture composes fine.
2. **Track-only node = 5.88 GB** (5.71 params + 0.17 runtime) — the 27B/N=8
   track node fits 8 GB VRAM at bf16 once the pool leaves. Head 12.59 GB.
3. **Untrained slices under d1b COLLAPSE** (endoftext/"1 1 1" attractors,
   both domains) while d1+pool on the same schedule emits clean code ⇒ the
   one-sublayer own-carry staleness is fatal without healing — consistent
   with the 9B record (d1b = 95% AFTER healing) and GLM Gate 0b. The 100B
   pool-free ledger therefore **requires the d1b heal of the target's
   slices** (training item), or the exact schedule (2 syncs/layer:
   ~2600 ms/pass ÷ τ_dense ≈ 160+ ms/tok, quality ≡ dense by construction —
   needs post-MLP boundary support in the engine walk).
   Bonus: the truer d1 verifier LIFTS τ (6.86→18.00 code, →4.57 prose) —
   verifier fidelity is a τ lever alongside the drafter.

### Drafter-as-cross-track-anchor — REFUTED (Gate 1, 2026-07-17)

The pool-free collapse's standing fix is a training heal. Tested a memory-cheap
alternative: feed the non-sync layers the DRAFTER's hidden states (free in
draft-verify, head node) as a global anchor for the missing cross-track content
— an EXTERNAL dense model's full forward on the same tokens, genuinely outside
the refuted estimator class (pt_state §5, all of which used the target's OWN
partial/stale internals). Gate 1 = the staggered-xattn kill methodology: can a
linear map of the drafter's hidden PREDICT the missing content a pool-free track
lacks? (27B/N=8 vs 0.8B drafter, both
reference streams from the SAME PT forward — h_full = all-8-tracks synced,
h_own = track-0-alone; missing = h_full − h_own; 10 240 wikitext tokens,
held-out R²).

**Result: KILL.** Best-aligned drafter layer per target layer (max R² over all
24 drafter layers — the mapping-artifact-proof upper bound) = mean **0.377**,
full-attn-layer mean 0.366 — BOTH under the ~0.40 line where staggered-xattn
(a STRONGER, same-model 35–40% signal) already made injection WORSE (remainder
compounds). Proportional single-layer map = 0.318. The mechanism is a clean
**depth decay**: R² 0.46–0.61 in early layers (near-embedding, universal) →
collapses to ~0 by the deep layers (L47 0.16, L59 0.05, L63 0.08) — exactly
where the residual feeds the lm_head and cross-track content matters most. The
0.8B/1024-dim external model cannot linearly carry the target's deep, high-rank,
model-specific cross-track content; the "orthogonal + volatile + high-rank" wall
reasserts against an external anchor. No Gate 2 (predictiveness was decisive, as
planned). Caveat: measured on prose; the depth-collapse is a representational-
capacity wall, not domain-specific, so code is very unlikely to change it. **The
pool-free 100B ledger still requires the d1b heal (or the exact schedule); the
drafter is a τ/verifier lever, not a comms substitute.**

## Verdicts

1. Track mode stands — no pipeline anywhere; the verify chunk is the same
   lockstep track-parallel forward over k+1 positions (engine
   `generate_draft_verify`, +1 tiny accept broadcast per block).
2. Emitted ≡ verifier greedy holds exactly on the CPU rails
   (`test_draft_verify_matches_plain_greedy`); on GPU bf16 the chunked
   verify and the 1-token decode take different kernels (FLA chunk vs
   fused-recurrent, GEMV at M=k+1 vs M=1) and occasionally flip a greedy
   tie — the same ~1e-3 class as the existing incremental≡full rail. The
   verify-pass forward is the SAME full-sequence path all recorded
   downstream numbers were scored on; per-token decode is the deviant.
3. The ≤5 syncs/token goal: **met everywhere at 27B (1.2–1.6 worst-domain)**;
   met on code at 100B-class d1b; open on prose pending the drafter lever.
4. The ≤100 ms/token wall-clock target (2026-07-17): **met on both domains
   resident (prose k=8 95.9/86.5, code k=32 36.7/25.8) and on code streamed
   (70.6)** — draft-verify beats plain decode outright; streamed prose is
   pool-wire physics (pool/τ), waiting on the same drafter lever as 100B
   prose. All fixes bit-exact; plain-decode baseline reproduced unchanged.
