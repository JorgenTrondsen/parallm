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

### Wall-clock decomposition (why 449 ms/tok, and the path down)

Per verify pass (`--profile`): 4 windows × ~235 ms EAGER verify compute
(the ungraphed launch-overhead tax — plain decode pays 6.7 ms/tok compute
only because its windows are CUDA-graphed; the graph machinery gates on
T==1 today), + k×~28 ms drafting (0.8B on the HF loop), + 6×20 ms rounds.
Named engineering items, in order of value:
1. **Graph the verify windows** (fixed T=k+1 ⇒ static shapes; the GDN
   rollback log needs static buffers) — removes ~900 ms/pass.
2. **Drafter decode loop** — 28 ms/tok is launch-overhead-bound for a 0.8B;
   a graphed/compiled drafter is ~3–8 ms.
With both, at code-τ 14: ≈ (120 + ~100)/14 + 48×8/14 ≈ **43 ms/tok ≈ 2.5×
the resident baseline**; at prose-τ 2.7: ~parity with the 107 ms baseline.

### Streamed pool ÷ τ (the 27B-reopening arm)

Plain streamed 27B is closed on the record: 9.14 GB ent wire/token ≈
+270 ms/tok. Under draft-verify the pool streams once per PASS and serves τ
tokens (measured this session — `serve_out/dv_streamed_*.log`):

| arm | ms/tok | syncs/tok | pool wire/tok | copy/window |
|---|---|---|---|---|
| plain streamed, k=0 (prose) | 942 | 4 | 9.14 GB | 207 ms @ 11.0 GB/s (8-rank-contended link) |
| dv streamed k=16 (prose, τ=3.41) | 636 | 1.17 | 2.68 GB | 207 ms |
| **dv streamed k=32 (code, τ=16.33 on-gen)** | **213** | **0.24** | **0.56 GB** | 193 ms @ 11.8 GB/s |

On-gen τ (16.33) exceeds τ_TF (12.80@k32) — the deployed number is the
better one, as the tag-era module predicted. Code-class decode puts the
streamed 27B pool at 0.56 GB/token — far inside the ~2 GB/token wire
envelope that closed plain-streamed 27B — **27B streamed on 8/16-class
nodes is reopened for code-class decode**; prose (2.68 GB/token) rides on
the drafter lever like everything else. Today's ms/tok still carries the
eager-verify and 28 ms-drafter taxes (items 1–2 above); the copies
themselves already hide behind the eager pass.

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
- **Wall-clock envelope**: ms/tok ≈ (S+2)·20/τ + verify_compute/τ + k·d/τ.
  At L=64, τ=14, graphed verify (~150 ms/pass est.), 8 ms drafter, k=48:
  ≈ 94 + 11 + 27 ≈ **132 ms/tok ≈ 7.6 tok/s** on code-class decode; prose at
  τ=3 ≈ 480 ms/tok — again, the drafter lever.

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
3. The ≤5 syncs/token goal: **met everywhere at 27B (1.2–1.5 worst-domain)**;
   met on code at 100B-class d1b; open on prose pending the drafter lever.
