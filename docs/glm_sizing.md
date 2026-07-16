# GLM-5.2 on constrained nodes — replica tier sizing (2026-07-13)

Can the sparse-replica lockstep engine serve GLM-5.2-NVFP4 on decentralized
nodes (8 GB VRAM / 16 GB DRAM / PCIe4-5 / local NVMe ~7 GB/s), keeping decode
stall-dominated (5 comm rounds × S=20 ms ≈ 100-110 ms/tok)? This memo plugs
measured constants into the tier law. Companion probes:
[scripts/probe_moe_replica.py](../scripts/probe_moe_replica.py) sections C/D on
the same-topology proxy (Qwen3.6-35B-A3B, 256 experts, top-8).

## 1. GLM-5.2 parameter account (from the released config)

`GlmMoeDsaForCausalLM`: hidden 6144, 78 layers (3 dense + 75 sparse), MLA/DSA
attention (q_lora 2048, kv_lora 512, 64 heads, qk 192+64, v 256, 32×128
indexer), per layer 256 routed experts (top-8, `moe_intermediate` 2048) + 1
shared expert, vocab 154 880. Derived:

| component | params | notes |
|---|---|---|
| routed experts | **724.8 B** | 75 × 256 × 37.75 M/slab (3 × 6144 × 2048) |
| attention (MLA+DSA) | ~14.9 B | ~191 M/layer, o_proj (100 M) dominates |
| shared experts | 2.83 B | 1 × 37.75 M × 75 |
| dense MLP (3 layers) | 0.68 B | intermediate 12288 |
| embed + lm_head | 1.90 B | centralized on the head node |
| **total** | **~745 B** | matches the marketed 744B; active/tok ≈ 41 B |

## 2. Replica-side sizes (qwanda:4:0.5 — the measured quality frontier, 89.8%)

Storage 3.0 bits/w; ent wire ≈ 2.55 bits/w (int4 codes plane × 0.76).

- **Routed slab** = 37.75 M params → **14.2 MB** storage, 12.0 MB ent wire.
- **Routed pool** = 724.8 B → **271.8 GB** — NVMe-resident (fits ≥1 TB; 17×
  over 16 GB DRAM, so DRAM is a cache tier, never the store).
- **Per-token routed demand** = 75 × 8 = **600 slabs = 8.5 GB raw / 7.2 GB ent**.
- **Trunk replica** (attn + shared + dense; the whole-network joint-content law
  says these copies are not optional) = 18.4 B → **6.9 GB**, touched EVERY
  token — it cannot be tiered by routing locality.
- **Own exact slice** (NVFP4 ≈ 4.5 bits/w) = 419/N GB → N=96: 4.4 GB,
  N=128: 3.3 GB, N=256: 1.6 GB.

## 3. The tier law and the budgets

Per token, with 4 boundary stalls + 1 embed round (min-sync premise — more
syncs is not a lever):

- **PCIe wire budget** (HBM refill hides behind stalls, measured law
  `Σ_w max(0, W/BW − S)`): 4·S·BW_eff. At the engine-realized ~85% of link:
  PCIe4 → **~1.8-2.1 GB/tok**, PCIe5 → **~3.5-4.1 GB/tok** (S=20; scales
  linearly in S).
- **NVMe→DRAM refill** runs the whole ~110 ms token, not just the stalls:
  **0.77 GB/tok** at 7 GB/s (PCIe4 NVMe), 1.5 GB/tok at 14 GB/s.
- **VRAM residency budget** (8 GB): own slice + trunk replica + expert cache
  C_v + ring/KV/CUDA ≈ 1.5 GB.

Wire demand per token = trunk (if streamed) + miss_rate(C_v, C_dram) × 600
slabs. The routed miss rates come from the measured locality constants below.

## 4. Measured locality constants (proxy Qwen3.6-35B-A3B, 40 MoE layers × 256
experts × top-8, wikitext traces; probe sections C/D, 2026-07-13)

- **churn@1 = 0.575** — 57.5% of a token's experts were NOT routed to by the
  previous token. Persistence prefetch barely works: cover@w (hit rate of the
  union of the previous w tokens' sets) = 0.425 / 0.508 / 0.597 / 0.688 / 0.783
  at w = 1/2/4/8/16. *The earlier recall@8 = 0.885 was degraded-vs-exact
  routing agreement on the same token — it says nothing about temporal reuse,
  and temporal reuse is weak.*
- **LRU(f) hit rate** (global slab cache at fraction f of the full pool):
  0.000 / 0.473 / 0.631 / 0.787 / 0.926 / 0.967 at f = 0.025/0.05/0.1/0.2/0.4/0.6.
  The f=0.025 zero is structural: a cache smaller than one token's active set
  (600 slabs at GLM scale) thrashes to 0% under the layer-sequential sweep —
  caches below ~1.2× per-token demand contribute nothing.
- **Decode-trace consistency**: true greedy decode churn@1 = 0.580 vs
  teacher-forced 0.575; cover@8 0.680 vs 0.688 — teacher-forced traces are a
  faithful proxy for autoregressive routing.
- **Miss tolerance (section D, qwanda:4:0.5 experts)**: dropping a missed
  expert's contribution is EXPENSIVE — random 5% drop = +0.45 ppl (+5.2%),
  10% = +1.10 ppl; dropping exactly what a w-token persistence cache lacks:
  w=16 (miss 20.9%) = +2.79 ppl, w=1 (miss 58%) = ppl 8.6 → 29.7 (destroyed).
  Misses must be *fetched*, not skipped; drops are only a ≲5% pressure valve.
- Sanity anchors: exact ppl 7.317; mode-3 degrade tax on these tiny 3.15M-param
  proxy experts +17.9% (matches the known +19% small-expert qwanda tax); the
  T=1→256 working-set curve reproduces the 2026-07-09 numbers.

## 5. Verdict

**The 8 GB / 16 GB node cannot serve GLM-5.2 under the sparse-replica scheme —
closed by two independent measurements, not by engineering shortfall:**

1. **Trunk replica residency.** The MLA+shared+dense replica (6.9 GB at the
   quality frontier) is demanded in full every token: streaming it (5.9 GB/tok
   ent) is ~3× the PCIe4 stall budget, so it must sit in VRAM — impossible
   next to anything else in 8 GB.
2. **Routed-expert locality is too weak for a small cache.** Per-token routed
   demand is 600 slabs = 7.2 GB ent wire. Every cache this envelope can field
   (VRAM ≤ ~6 GB, DRAM ≤ ~12 GB usable → f ≤ 0.045 of the 271.8 GB pool) sits
   at ≤ 0.47 LRU hit → ≥ 3.8 GB/tok from below, over the PCIe4 budget, and
   NVMe (0.77 GB/tok at 7 GB/s) is 5× short of feeding the DRAM tier's misses.

**What the 8/16 node DOES serve: the 35B-A3B class.** When the whole replica
pool fits pinned DRAM and the per-token *active* wire fits the stall budget, no
cache/locality is needed at all: proxy-class MoE (32.3 B routed + 2.7 B trunk)
= 10.3 GB ent pool in DRAM, ~1.05 GB/tok active wire (trunk 0.73 + routed
0.32) < 2.1 GB budget, own slice + ring ≪ 8 GB VRAM. The envelope's MoE
frontier ≈ **~35 B total / ~3-6 B active**; dense frontier ≈ 9-13 B (measured,
`pt_state.md` §4). Engine build needed: demand-driven expert slots (indices
known at window entry; expert→slot indirection updated in the eager boundary
region — composes with CUDA graphs like the ring does).

**GLM-5.2's actual node floor (arithmetic from the measured curves):**
VRAM ≥ ~24 GB (trunk 6.9 + own 3.3 at N=128 + slab cache 12.3 GB → f=0.045,
hit ≈ 0.45) + **PCIe5** (wire 0.55 × 7.2 ≈ 4.0 GB/tok ≈ the 4 GB budget) +
**~28 GB/s of NVMe** (2× PCIe5 drives: the VRAM+DRAM stack at f≈0.09 hits
~0.62, leaving ~2.7 GB/tok for NVMe→DRAM; 1× 14 GB/s covers only 1.5).
Workstream C (trained 0.65-0.7-sparse replicas, ×0.73-0.8 bytes) turns the
PCIe5 knife-edge into margin and is effectively REQUIRED at this tier. Longer
stalls help linearly (S=30 ⇒ +50% on every budget); more syncs stay off the
table by premise.

**Bottom line:** GLM-5.2 on 8 GB/16 GB nodes is refuted by storage and
locality arithmetic under the whole-network-replica premise; the reachable
targets on that envelope are 35B-A3B-class MoE (new engine work: expert-slot
streaming) and the already-shipped 9B dense. GLM-5.2 needs the 24 GB+PCIe5+
dual-NVMe tier plus workstream C. Blocked regardless of tier: the MLA +
streaming convert for `glm_moe_dsa`.
