# pt_converter

`pt_converter` is a model-agnostic conversion and fine-tuning toolkit that turns a pretrained dense transformer into a **parallel-track (PT) transformer**. The dense model's weights are sliced across `N` parallel "tracks". The world layout is one rank per visible GPU, with each rank hosting `K = N / world_size` tracks locally. `embed_tokens`, `lm_head`, and the KL/CE softmax are **vocab-parallel** — sharded over the vocabulary dimension across all ranks — so no rank specially carries them and the layout is uniform. At configurable sync boundaries (every `D` layers), every rank locally sums its `K` partial residual deltas, then a single NCCL all-reduce across the world combines the per-rank sums — the combined forward stays mathematically equivalent to the original dense model. `N=1` short-circuits to a no-op, giving a bit-equal dense-parity gate.

The residual-stream RMSNorm scales (input/post layernorms, the final norm, and the linear-attention gated norm) are held identically across tracks and **kept bit-identical** under training: `sync_replicated_grads` averages each group's gradients between `loss.backward()` and `optimizer.step()`, so a deterministic optimizer step keeps the copies bit-equal forever. The **full-attention head params** — the GQA K/V projections and their per-head `q_norm`/`k_norm` — instead **diverge per track by default**: each track starts from a copy of its kv-group's head and then trains independently, so the 8 full-attention layers go from GQA (`16q/4kv`) to per-track MHA (`16q/16kv`). This is a strict capacity superset (the synced GQA model is the special case where the copies stay equal), it is free on memory (each track already stores its own copy — only the gradient averaging is dropped), it removes the per-step kv-group all-reduces, and it leaves each track fully self-contained in the attention block (aligning with per-node inference). Pass `--sync-attention-heads` to restore the legacy bit-identical / dense-GQA-equivalent path (e.g. for an A/B). New model families are added by registering a `ModelAdapter` — the slicer engine, sync logic, and training loop themselves stay model-agnostic. A short distillation stage (frozen dense teacher → sliced student) using block-wise teacher-forced MSE + end-to-end logit-KL + LM CE recovers any perplexity lost during the static weight conversion.

The first supported family is **Qwen3.5** (mixed full-attention / linear-attention decoder), wired up via the `qwen3_5_text` adapter.

## Install

Requires Python ≥ 3.10, `torch >= 2.4`, `transformers >= 4.57.0.dev0`.

```bash
pip install -e .                # core
pip install -e ".[test]"        # + pytest
pip install -e ".[eval]"        # + lm-eval (for scripts/eval_lm_harness.py)
pip install -e ".[fast]"        # + flash-linear-attention + causal-conv1d (fused gated-delta kernels; GPU)
```

**Strongly recommended for training:** install the `[fast]` extra
(`flash-linear-attention` + `causal-conv1d`). Qwen3.5's 24 linear-attention layers
otherwise fall back to a slow pure-torch chunked recurrence; the fused Triton kernels
are a ~3-4× training step speedup (≈9× combined with `--compile` + SDPA), and
`causal-conv1d` also enables the frozen teacher's linear-attn fast path. GPU + `triton`
only — the test suite still runs on CPU via the pure-torch fallback. If `causal-conv1d`'s
CUDA-extension build fails under pip's build isolation (a torch/nvcc CUDA-version
mismatch), install it with `pip install --no-build-isolation causal-conv1d` (needs
`ninja` and a matching `nvcc` on `PATH`).

## Directory structure

```
pt_converter/
├── pyproject.toml                          # Build metadata, deps; [eval] extra pulls lm-eval.
├── scripts/
│   ├── convert_qwen3_5_9b.py               # CLI: slice a pretrained Qwen3.5-9B into N per-track checkpoints + manifest.
│   ├── train_qwen3_5_9b.py                 # torchrun entry point: distributed distillation.
│   ├── eval_fidelity.py                    # torchrun entry point: KL / top-k / hidden-MSE / ppl-gap, student vs teacher.
│   ├── eval_lm_harness.py                  # torchrun entry point: lm-evaluation-harness over student and/or teacher.
│   └── verify_kv_sync.py                   # torchrun entry point: end-to-end check that replicated params stay bit-identical.
├── src/pt_converter/
│   ├── __init__.py                         # Public API: max_tracks_for_config, slice_model_to_tracks, PTManifest.
│   ├── adapters/
│   │   ├── __init__.py                     # ModelAdapter dataclass + register_model_adapter / get_model_adapter registry.
│   │   └── qwen3_5.py                      # Registers the "qwen3_5_text" adapter (slicer specs + per-track model class).
│   ├── dist/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── groups.py                       # ProcessGroupLayout + build_groups (uniform K; tracks_per_rank_list utility for non-uniform).
│   │   └── fsdp_setup.py                   # No-op student FSDP; teacher FSDP-shard + optional per-layer compile / vocab-sharded lm_head.
│   ├── model/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── pt_model.py                     # PTWrappedModel: per-rank wrapper hosting K local tracks, returns (logits, sync_hiddens).
│   │   ├── sync.py                         # SyncBoundary: local Σ across K tracks, then NCCL all-reduce across ranks.
│   │   ├── vocab_parallel.py               # Vocab-parallel primitives: VocabParallelEmbedding + vocab_range (embed / lm_head / KL-CE sharded over the vocab dim across all ranks).
│   │   └── tracks/
│   │       ├── __init__.py                 # Package marker.
│   │       └── qwen3_5.py                  # Per-track Qwen3.5 decoder with SyncBoundary calls at sync layers.
│   ├── slicer/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── base.py                         # SlicerSpec protocol + Colwise / Rowwise / PerHead / Replicated / Fused / KV / GatedQ / OwnerOnly.
│   │   ├── convert.py                      # slice_model_to_tracks engine: applies adapter specs → N state dicts + PTManifest.
│   │   └── qwen3_5.py                      # Qwen3.5-specific SlicerSpec instances (attention / linear-attention / MLP).
│   ├── train/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── data.py                         # PackedTokenStream IterableDataset (streamed tokenize + pack; seed-interleaved mixture / presets / custom sources).
│   │   ├── distill.py                      # SPD step: per-block MSE (teacher/student-forced, optional normalized+clamped) + memory-chunked, optionally vocab-parallel KL+CE backward.
│   │   ├── losses.py                       # block_mse (raw or normalized/relative), logit_kl, lm_cross_entropy — all with attention-mask support.
│   │   ├── teacher.py                      # HookedTeacher: frozen dense model with hooks capturing hiddens at sync indices.
│   │   └── sync_grads.py                   # Replication plan + sync_replicated_grads (averages grads inside each replication group).
│   ├── eval/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── fidelity.py                     # fidelity_step: KL (fwd+rev), top-k agreement, per-sync hidden MSE, ppl gap.
│   │   └── lm_eval_adapter.py              # lm-evaluation-harness adapter for PTWrappedModel and the FSDP-wrapped teacher.
│   └── utils/
│       ├── __init__.py                     # Package marker.
│       ├── checkpoint.py                   # Save/load per-track safetensors + manifest.json.
│       ├── max_tracks.py                   # Compute maximum valid N under KV-replication and divisibility rules.
│       └── mem_report.py                   # Per-rank CUDA memory attribution (teacher shard / student / grads / AdamW + per-phase activation peak); drives --mem-report.
└── tests/
    ├── __init__.py                         # Package marker.
    ├── conftest.py                         # Autouse fixture forcing the pure-torch gated-delta path so CPU tests skip the GPU-only FLA / Triton kernels.
    ├── test_pt_forward_n1.py               # N=1 PT forward must match dense bit-equal; sync_hiddens captured at right depths.
    ├── test_pt_n8_forward_smoke.py         # N=8 simulated distributed forward on CPU; finite outputs, bounded drift.
    ├── test_pt_k2_local_sync.py            # K=2 single-process: exercises the local-sum SyncBoundary path without NCCL.
    ├── test_kv_replication.py              # KV-replicated slices identical within kv-group, unique across groups, reassemble bit-equal.
    ├── test_replication_groups.py          # SlicerSpec.replication_groups partitions tracks correctly per spec type.
    ├── test_sync_grads.py                  # sync_replicated_grads averages in-place; assert_replicated_consistent catches drift.
    ├── test_slicer_specs.py                # Per-SlicerSpec slice/reassemble round-trip and shape unit tests.
    ├── test_sync_schedule.py               # sync_block_depth + num_layers → correct per-track sync layer indices.
    ├── test_model_adapter.py               # Adapter registry: register / lookup / idempotent re-register; slicer routing via adapter.
    ├── test_slicer_qwen3_5_integration.py  # Tiny Qwen3.5: N=1 bit-equal, N=2 round-trip, per-track shapes match config.
    ├── test_max_tracks.py                  # max_tracks_for_config against synthetic configs under the four-rule constraint set.
    ├── test_vocab_parallel.py              # Vocab-parallel embed + KL/CE parity: single-shard == dense, masked partials sum to nn.Embedding, 2-rank gloo grads stitch back to dense.
    ├── test_data_mixture.py                # Streamed mixture: presets, --data-source spec parsing, packing/labels, seed-interleave determinism across ranks.
    └── test_distill_recipe.py              # Quality recipe: normalized relative block-MSE + deterministic student-forcing scheduled sampling; end-to-end K=2 distill step.
```

## Convert

`scripts/convert_qwen3_5_9b.py` slices a dense Qwen3.5 checkpoint into N per-track safetensors plus a `manifest.json`.

| Flag | Default | Purpose |
|---|---|---|
| `--hf-model` | required | Path or HF id of the dense source model. |
| `--out-dir` | required | Output dir for per-track safetensors + manifest. |
| `--n-tracks` | `max_tracks_for_config(...)` | Number of tracks (defaults to the max valid N for the model). |
| `--sync-block-depth` | `4` | Sync every D layers (the cadence / communication budget). |
| `--sync-schedule` | `full-attn-aligned` | Where sync boundaries fall. `full-attn-aligned` (default): a sync immediately **before every full-attention layer** so it reads a synced (exactly decomposable) residual, plus a final sync before the norm (`D=4` → 2,6,…,30,31). Full attention is the global mixer — feeding it the partial residual the `uniform` schedule leaves it with breaks the per-track decomposition, so aligning is the high-leverage fix at `D≥2` (it drove `val_kl` ~1.4 → 0.77 on n16/d2). `uniform` (legacy): every D layers (`D=4` → after 3,7,…,31). Per-track **weights are identical** either way (only `sync_layer_indices` differ), so a manifest trains/evaluates against existing track shards regardless of schedule. For `D ≥ full_attention_interval` the aligned schedule collapses to one sync per interval. |
| `--device` | `cpu` | Device for slicing. |
| `--dtype` | `bfloat16` | One of `bfloat16` / `float16` / `float32`. |

```bash
python scripts/convert_qwen3_5_9b.py \
    --hf-model Qwen/Qwen3.5-9B \
    --out-dir convert_out/qwen3_5_9b_n16_d4 \
    --n-tracks 16 --sync-block-depth 4
```

This defaults to `--sync-schedule full-attn-aligned` — a sync right *before* every
full-attention layer so the global mixers read a synced (exactly decomposable)
residual instead of the partial one the legacy `uniform` schedule leaves them with.
Pass `--sync-schedule uniform` for the every-D-layers schedule. Per-track weights are
identical either way (only the manifest's `sync_layer_indices` change), so a manifest
trains/evaluates against existing track shards regardless of schedule.

## Train

`scripts/train_qwen3_5_9b.py` runs the distillation under `torchrun`. One rank per GPU; each rank hosts `K = n_tracks / world_size` tracks (uniform). `embed_tokens`, `lm_head`, and the KL/CE softmax are vocab-parallel across all ranks (`--vocab-parallel`, default on), so memory is balanced and the previously rank-0-serial KL/CE phase runs in parallel. The per-track layers are tiny at high `n_tracks` (e.g. one attention head per track at `n_tracks=16`), so by default each layer is `torch.compile`d (`--compile`, ~2× faster steps) and the full-attention layers use SDPA; the frozen teacher's layers are likewise compiled (`--compile-teacher`).

| Flag | Default | Purpose |
|---|---|---|
| `--hf-model` | required | Dense teacher path. |
| `--tracks-dir` | required | Output of the convert script. |
| `--out-dir` | `/train_out` | Checkpoint dir (also writes `best/` when eval improves). |
| `--resume-from` | `None` | Resume model + optimizer state from a prior checkpoint dir. |
| `--vocab-parallel` / `--no-vocab-parallel` | on | Vocab/tensor-parallel `embed_tokens` + `lm_head` + KL/CE across all ranks (balanced memory, parallel KL/CE, uniform layout). `--no-vocab-parallel` selects the legacy track-0-owner path (memory-heavy at `n_tracks=16` on 40 GB). |
| `--sync-attention-heads` / `--no-sync-attention-heads` | off | Keep the full-attention head params (`k_proj`, `v_proj`, `q_norm`, `k_norm`) bit-identical within each kv-group (legacy dense-GQA-equivalent path). Default **off**: those copies **diverge** per track (each track gets its own KV head — GQA → per-track MHA), a capacity superset that is free on memory and drops the per-step kv-group all-reduces. Residual-stream norms stay synced regardless. |
| `--compile-teacher` / `--no-compile-teacher` | on | `torch.compile` each frozen-teacher decoder layer (inference-only); disable alone if it conflicts with FSDP2 / the sync-boundary hooks. |
| `--max-steps` | `1000` | Training steps (counts **optimizer** steps; with `--grad-accum-steps>1` each consumes that many microbatches). |
| `--seq-len` / `--batch-size` | `4096` / `1` | Sequence length and per-rank batch. |
| `--grad-accum-steps` | `1` | Accumulate gradients over this many microbatches before each optimizer step (effective batch = `--batch-size` × this × world). Each microbatch's losses are scaled `1/grad-accum` so the grads **average**, giving a less noisy small-batch block-MSE/KL signal at fixed memory. `--max-steps` / `--eval-every` / the LR schedule all count optimizer steps. `1` = no accumulation (bit-identical to the legacy single-microbatch loop). |
| `--data-preset` | `qwen-mix` | Streamed training mixture. **`qwen-mix`** = `DKYoon/SlimPajama-6B` (0.70) + `open-web-math` (0.15) + `bigcode/the-stack-dedup` (0.15, `content` key), interleaved to approximate Qwen3.5's code/math-heavy distribution. **`the-stack-dedup` is gated** — accept its terms and export `HF_TOKEN`, or use the ungated `slimpajama` / `wikitext` presets. Sources stream, so corpus size is irrelevant (only consumed tokens are fetched); point `HF_HOME` at scratch to save quota. |
| `--data-source` | `None` | Add a custom training source, `NAME[:CONFIG[:TEXT_KEY[:WEIGHT]]]`, repeatable. Empty `CONFIG`/`TEXT_KEY` fields keep defaults (e.g. `DKYoon/SlimPajama-6B::text:0.7`). If **any** `--data-source` is passed it **replaces** `--data-preset`. Sources are normalized to a common `text` column and seed-interleaved by weight. |
| `--val-dataset-name` / `--val-dataset-config` / `--val-split` / `--val-text-key` | `Salesforce/wikitext` / `wikitext-103-raw-v1` / `validation` / `text` | Held-out validation source (single dataset) for the KL eval. Defaults to **WikiText-103 validation on purpose**: a *fixed* comparator keeps `val_kl`/`val_ce` comparable across runs (and to history) for early-stop / `best/` selection, independent of the training mixture. |
| `--lr` / `--warmup-steps` / `--cosine-decay` / `--lr-min-ratio` | `3e-5` / `0` / off / `0.1` | AdamW LR, linear warmup steps, cosine decay (decoupled — warmup/decay/both/neither), and the cosine floor as a fraction of `--lr`. Recommended: a short warmup + `--cosine-decay` (the legacy constant-LR default produced grad-norm spikes / wasted clipped steps). |
| `--max-grad-norm` | `1.0` | Clip gradients before optimizer step. |
| `--activation-checkpoint` | off | Activation-checkpoint the student decoder blocks (memory ↓, compute ↑). Under vocab-parallel (default) every rank backwards through the full student forward, so this lowers the ~25 GB `student_fwd`/`klce` peak on **every** rank — the lever for fitting a larger `--batch-size` at `seq=4096` on 40 GB. |
| `--compile` / `--no-compile` | on | `torch.compile` each per-track decoder layer in place (inductor fusion of the tiny per-track kernels). Default on (~2× faster steps); `--no-compile` to disable. One-time compile warmup on the first steps. |
| `--compile-mode` | `default` | `torch.compile` mode. `default` (inductor fusion). `reduce-overhead` (CUDA graphs) is experimental here. |
| `--profile` / `--profile-trace` | off | `--profile`: per-phase CUDA-synced wall-clock breakdown + per-rank peak mem. `--profile-trace`: adds a torch.profiler kernel trace (Chrome trace + key_averages on rank 0). |
| `--mem-report` / `--mem-report-step` | off / `3` | Per-rank breakdown of *what* occupies GPU memory: lifecycle deltas (baseline → teacher → student → optimizer), a measured resident-component table (teacher shard / student params / grads / AdamW state — real bytes + dtype), the transient per-phase activation peak captured on `--mem-report-step`, and the allocated-vs-reserved-vs-device gap (CUDA ctx + NCCL). Cheap; off = zero overhead. |
| `--kl-ce-chunk-size` | `512` | Seq-chunk size for the KL+CE pass (caps the per-chunk fp32 `(B, chunk, V/world)` expansion). The klce phase is collective/dispatch-bound, so larger = fewer chunks = faster (512 → 8 chunks at `seq=4096` vs 32 at 128) at negligible extra memory; the transient grows ×batch, so keep moderate at large `--batch-size`. |
| `--lambda-block` / `--lambda-kl` / `--lambda-ce` | `1.0` / `1.0` / `0.5` | Loss weights. |
| `--kl-temperature` | `1.0` | KL temperature. |
| `--student-forcing-prob` / `--student-forcing-warmup` | `0.0` / `0` | Scheduled sampling: per block, with this probability feed the block the **student's own** synced hidden (instead of the teacher's) as input, while the MSE target stays the teacher hidden. Closes the exposure-bias gap between teacher-forced training and free-running inference (the cause of the depth-exploding block_mse / chance-level downstream). `--student-forcing-warmup` ramps the prob `0 → prob` over N steps (start teacher-forced). Recommended `0.5` / ~half of `--max-steps`. The per-block draw is deterministic across ranks (seeded by `--seed` + step), so the SyncBoundary all-reduce stays consistent. `0.0` = legacy fully-teacher-forced path. |
| `--student-forcing-schedule` | `hold` | Shape of the student-forcing prob over the run. `hold` (default, legacy): linear ramp `0 → --student-forcing-prob` over `--student-forcing-warmup` then **hold**. `cosine-full`: a free-running **curriculum** — cosine ramp `0 → --student-forcing-prob` across the **whole** run, approaching the high-forcing regime gently and reaching it only near the end. Directly closes the train(teacher-forced)/eval(free-running) gap that drives the depth-exploding block_mse, without the unstable long tail of holding at a high prob. Recommended with `--student-forcing-prob ~0.9` (`--student-forcing-warmup` is ignored in this shape). |
| `--normalize-block-mse` | off | Relative (scale-free) block MSE `Σ(s−t)²/Σt²` per block instead of the raw masked mean. The residual-stream norm grows with depth, so the raw MSE lets deep layers dominate the gradient (and spike the grad norm); normalizing makes every depth contribute comparably. Rescales the block term, so `--lambda-block` becomes a relative weight (1.0 is a fine start). |
| `--block-mse-clamp` | `10.0` | Cap the normalized per-block relative MSE (only active with `--normalize-block-mse`). Under student forcing a block can be fed the student's own drifted hidden, occasionally blowing the ratio up to 100+ on a single batch — a spike that inflates the gradient and trips the `--max-grad-norm` clip, throttling the whole step. Clamping saturates the gradient above the cap (outlier rejection); normal per-block ratios (~0.5–1.5) are untouched. `<=0` disables it. |
| `--free-running-mse` (+ `--lambda-free-running` `1.0` / `--free-running-schedule` `constant` / `--free-running-taps` `all`) | off | **Free-running feature matching.** Relative-MSE the end-to-end **free-running** student forward's synced hiddens (the same full forward the KL/CE pass uses — the student runs on its *own* hiddens throughout) against the teacher hiddens at the sync boundaries, with gradients through the **whole** forward. The block loop detaches at every boundary, so it never trains multi-window error compounding — the deep free-running relMSE plateau is exactly what it cannot see; this term supervises it directly. Reuses the already-paid `student_fwd` pass and shares its single backward with KL/CE, so the marginal compute is ~zero; retaining the boundary teacher hiddens costs ~2.5 GB/rank at `B=5` `D=2` (`--free-running-taps deep-half` halves it, supervising only the deep boundaries where the error concentrates). `--free-running-schedule cosine-full` ramps the weight 0 → 1 across the run (recommended from scratch — early free-running hiddens are garbage); `constant` applies full weight from step 0 (warm-started finishing runs). The term is the mean-over-taps relative MSE clamped by `--block-mse-clamp`, so `--lambda-free-running 1.0` is comparable to a normalized `--lambda-block`. |
| `--adaptive-layer-weight` (+ `--adaptive-layer-weight-ema` `0.9` / `--adaptive-layer-weight-power` `1.0`) | off | Adaptively weight each supervised tap's block-MSE by its **own** running relative error. A per-tap EMA of the relative MSE `Σ(s−t)²/Σt²` is kept; the per-step weight `∝ EMA**power`, mean-1 normalized over the taps, so gradient budget flows to wherever the student is **currently** worst (which the relative metric shows need not be monotone in depth — the worst band is the upper-middle layers, not the deepest) without changing the total block-loss magnitude (so `--lambda-block` keeps its meaning). The relMSE is read off the **synced** hidden (identical on every rank), so the EMA — and the weights — stay in lock-step across ranks. Off = uniform weights. Pure loss-side — no change to the sync schedule / communication budget. |
| `--save-every` / `--save-final` | `0` / off | Checkpoint cadence and final-step save. |
| `--eval-every` / `--val-batches` | `0` / `20` | Held-out KL(teacher ‖ student) eval cadence and size; val_ce also logged. |
| `--early-stop-patience` / `--min-improvement` | `0` / `0.01` | Optional early stopping. |
| `--seed` | `42` | Seeds torch / cuda / python / numpy. |
| `--log-every` | `10` | Log cadence. |

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc-per-node=8 scripts/train_qwen3_5_9b.py \
    --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
    --tracks-dir convert_out/qwen3_5_9b_n16_d4 \
    --out-dir train_out/qwen3_5_9b_n16_d4 \
    --max-steps 4000 --seq-len 4096 --batch-size 5 \
    --activation-checkpoint \
    --eval-every 200 --save-every 500
```

**Larger batch (8× A100-40GB, n16):** `--activation-checkpoint` recomputes the full student
forward instead of holding it (the ~25 GB `student_fwd`/`klce` peak), which is what frees the
headroom to raise `--batch-size` at `seq=4096`. Keep `--kl-ce-chunk-size` moderate (its fp32
transient scales ×batch). Use `--mem-report` to confirm `device_used` stays under ~39 GB before
committing to a long run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc-per-node=8 scripts/train_qwen3_5_9b.py \
    --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
    --tracks-dir convert_out/qwen3_5_9b_n16_d4 \
    --out-dir train_out/qwen3_5_9b_n16_d4 \
    --max-steps 4000 --seq-len 4096 --batch-size 4 \
    --activation-checkpoint --kl-ce-chunk-size 512 \
    --eval-every 200 --save-every 500
```

**Quality-recovery recipe (recommended at high `n_tracks`).** Pure teacher-forced block
distillation trains each block only on the teacher's (correct) input, so at free-running
inference the deep blocks compound error — the symptom is a block_mse that is small during
training but explodes with depth in `eval_fidelity.py`, and chance-level downstream scores.
The combination that closes this gap (and brought `n16/d2` to `val_kl ≈ 0.59`, with the deep
relative error dropping onto the `D=1` floor):

- **A free-running curriculum** — `--student-forcing-schedule cosine-full` with
  `--student-forcing-prob ~0.9`: scheduled sampling that ramps `0 → 0.9` across the **whole**
  run, so the deep blocks are progressively trained on the *drifted* inputs they see at
  inference. This is the dominant lever — the deep-layer gap is mostly free-running exposure
  bias, and it's training-recoverable without extra communication.
- **Normalized block MSE** (`--normalize-block-mse` + `--block-mse-clamp 10`) so deep,
  high-norm layers don't dominate / spike the gradient.
- **Intra-window per-layer MSE** (`--intra-window-mse`) so the mid-window layers — which run
  on partial residuals at `D≥2` — are pinned to the teacher trajectory.
- **Free-running feature matching** (`--free-running-mse` +
  `--free-running-schedule cosine-full`) — the block loop detaches at every boundary, so
  nothing above trains *multi-window* error compounding with gradient flow; this term MSEs the
  end-to-end free-running forward's synced hiddens against the teacher's, backwarding through
  the whole forward, at ~zero extra compute (it shares the already-paid `student_fwd` pass).
- **Adaptive layer weighting** (`--adaptive-layer-weight`) steers gradient budget to whichever
  taps have the highest *running relative* error (the upper-middle band, not strictly the
  deepest), data-driven rather than a fixed depth ramp.
- **Gradient accumulation** (`--grad-accum-steps 2`) averages the noisy small-batch block-MSE
  gradient before each step.
- An **LR schedule** (`--warmup-steps` + `--cosine-decay`, vs the spiky constant-LR default).

At `D≥2` also convert with `--sync-schedule full-attn-aligned` (above): it makes each
full-attention layer's per-track split *structurally* exact, complementary to the curriculum's
exposure-bias fix.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc-per-node=8 scripts/train_qwen3_5_9b.py \
    --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
    --tracks-dir convert_out/qwen3_5_9b_n16_d2 \
    --out-dir train_out/qwen3_5_9b_n16_d2 \
    --max-steps 2001 --seq-len 4096 --batch-size 5 --grad-accum-steps 2 --activation-checkpoint \
    --intra-window-mse --adaptive-layer-weight \
    --student-forcing-prob 0.9 --student-forcing-schedule cosine-full \
    --normalize-block-mse --block-mse-clamp 10 \
    --free-running-mse --free-running-schedule cosine-full \
    --lambda-kl 0 --lambda-ce 0 \
    --lr 1e-4 --warmup-steps 50 --cosine-decay --lr-min-ratio 0.1 \
    --eval-every 200
```

## Evaluate

Three torchrun entry points. They use the uniform per-rank track layout; forward output is layout-independent (the SyncBoundary all-reduce combines all tracks regardless of which rank hosts them). Eval loads checkpoints via the legacy full-logits student path (full `lm_head` on the owner rank), which reads a vocab-parallel-trained checkpoint unchanged.

- **Logit fidelity vs teacher** — `scripts/eval_fidelity.py`. KL (forward + reverse), top-1 / top-5 agreement and top-5 IoU, per-sync-boundary hidden MSE, student / teacher perplexity and gap.
  ```bash
  torchrun --standalone --nproc-per-node=8 scripts/eval_fidelity.py \
      --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
      --checkpoint-dir train_out/qwen3_5_9b_n16_d4/best \
      --num-batches 200
  ```
- **lm-evaluation-harness** — `scripts/eval_lm_harness.py`. Runs the harness on the student and/or teacher (`--target {student,teacher,both}`) with the same seeds, so request streams align across ranks. Default tasks: `hellaswag,arc_easy,arc_challenge,winogrande,piqa`; pass `--include-mmlu` to add MMLU.
  ```bash
  torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \
      --hf-model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \
      --checkpoint-dir train_out/qwen3_5_9b_n16_d4/best \
      --output-json train_out/qwen3_5_9b_n16_d4/lm_eval.json
  ```
