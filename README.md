# pt_converter

`pt_converter` is a model-agnostic conversion and fine-tuning toolkit that turns a pretrained dense transformer into a **parallel-track (PT) transformer**. The dense model's weights are sliced across `N` parallel "tracks". The world layout is one rank per visible GPU, with each rank hosting `K = N / world_size` tracks locally (with `--rank0-tracks` for non-uniform layouts where rank 0 also owns `embed_tokens` and `lm_head`). At configurable sync boundaries (every `D` layers), every rank locally sums its `K` partial residual deltas, then a single NCCL all-reduce across the world combines the per-rank sums — the combined forward stays mathematically equivalent to the original dense model. `N=1` short-circuits to a no-op, giving a bit-equal dense-parity gate.

Parameters that the slicer flagged as identical across multiple tracks (RMSNorm scales, and K/V projection rows when `n_tracks > num_kv_heads`) are kept bit-identical under training: `sync_replicated_grads` averages each group's gradients between `loss.backward()` and `optimizer.step()`, so a deterministic optimizer step keeps the copies bit-equal forever. New model families are added by registering a `ModelAdapter` — the slicer engine, sync logic, and training loop themselves stay model-agnostic. A short distillation stage (frozen dense teacher → sliced student) using block-wise teacher-forced MSE + end-to-end logit-KL + LM CE recovers any perplexity lost during the static weight conversion.

The first supported family is **Qwen3.5** (mixed full-attention / linear-attention decoder), wired up via the `qwen3_5_text` adapter.

## Install

Requires Python ≥ 3.10, `torch >= 2.4`, `transformers >= 4.57.0.dev0`.

```bash
pip install -e .                # core
pip install -e ".[test]"        # + pytest
pip install -e ".[eval]"        # + lm-eval (for scripts/eval_lm_harness.py)
```

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
│   │   ├── groups.py                       # ProcessGroupLayout + build_groups (supports rank0_tracks / tracks_per_rank_list).
│   │   └── fsdp_setup.py                   # Reserved no-op intra-track FSDP wrapping; ready for future layouts.
│   ├── model/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── pt_model.py                     # PTWrappedModel: per-rank wrapper hosting K local tracks, returns (logits, sync_hiddens).
│   │   ├── sync.py                         # SyncBoundary: local Σ across K tracks, then NCCL all-reduce across ranks.
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
│   │   ├── data.py                         # PackedTokenStream IterableDataset (on-the-fly tokenize + pack; WikiText-103 / custom).
│   │   ├── distill.py                      # SPD step: block-wise teacher-forced MSE (backward-per-block) + memory-chunked KL+CE backward.
│   │   ├── losses.py                       # block_mse, logit_kl, lm_cross_entropy — all with attention-mask support.
│   │   ├── teacher.py                      # HookedTeacher: frozen dense model with hooks capturing hiddens at sync indices.
│   │   └── sync_grads.py                   # Replication plan + sync_replicated_grads (averages grads inside each replication group).
│   ├── eval/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── fidelity.py                     # fidelity_step: KL (fwd+rev), top-k agreement, per-sync hidden MSE, ppl gap.
│   │   └── lm_eval_adapter.py              # lm-evaluation-harness adapter for PTWrappedModel and the FSDP-wrapped teacher.
│   └── utils/
│       ├── __init__.py                     # Package marker.
│       ├── checkpoint.py                   # Save/load per-track safetensors + manifest.json.
│       └── max_tracks.py                   # Compute maximum valid N under KV-replication and divisibility rules.
└── tests/
    ├── __init__.py                         # Package marker.
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
    └── test_max_tracks.py                  # max_tracks_for_config against synthetic configs under the four-rule constraint set.
```

## Convert

`scripts/convert_qwen3_5_9b.py` slices a dense Qwen3.5 checkpoint into N per-track safetensors plus a `manifest.json`.

| Flag | Default | Purpose |
|---|---|---|
| `--hf-model` | required | Path or HF id of the dense source model. |
| `--out-dir` | required | Output dir for per-track safetensors + manifest. |
| `--n-tracks` | `max_tracks_for_config(...)` | Number of tracks (defaults to the max valid N for the model). |
| `--sync-block-depth` | `4` | Sync every D layers. |
| `--device` | `cpu` | Device for slicing. |
| `--dtype` | `bfloat16` | One of `bfloat16` / `float16` / `float32`. |

```bash
python scripts/convert_qwen3_5_9b.py \
    --hf-model Qwen/Qwen3.5-9B \
    --out-dir ./pt_tracks/qwen3_5_9b_n16_d4 \
    --n-tracks 16 --sync-block-depth 4
```

## Train

`scripts/train_qwen3_5_9b.py` runs the distillation under `torchrun`. One rank per GPU; each rank hosts `K = n_tracks / world_size` tracks (or fewer on rank 0 if `--rank0-tracks` is set, since rank 0 also owns `embed_tokens` and `lm_head`).

| Flag | Default | Purpose |
|---|---|---|
| `--hf-model` | required | Dense teacher path. |
| `--tracks-dir` | required | Output of the convert script. |
| `--out-dir` | `./pt_train_out` | Checkpoint dir (also writes `best/` when eval improves). |
| `--resume-from` | `None` | Resume model + optimizer state from a prior checkpoint dir. |
| `--rank0-tracks` | `None` | Override K on rank 0 to free memory for embed / lm_head. |
| `--max-steps` | `1000` | Training steps. |
| `--seq-len` / `--batch-size` | `4096` / `1` | Sequence length and per-rank batch. |
| `--lr` / `--warmup-steps` / `--lr-min-ratio` | `3e-5` / `0` / `0.1` | AdamW LR + cosine decay floor. |
| `--max-grad-norm` | `1.0` | Clip gradients before optimizer step. |
| `--activation-checkpoint` | off | Activation-checkpoint the student decoder blocks (memory ↓, compute ↑). |
| `--kl-ce-chunk-size` | `128` | Vocab chunking inside `_kl_ce_chunked` to bound peak KL + CE memory. |
| `--lambda-block` / `--lambda-kl` / `--lambda-ce` | `1.0` / `1.0` / `0.5` | Loss weights. |
| `--kl-temperature` | `1.0` | KL temperature. |
| `--save-every` / `--save-final` | `0` / off | Checkpoint cadence and final-step save. |
| `--eval-every` / `--val-batches` | `0` / `20` | Held-out CE eval cadence and size. |
| `--early-stop-patience` / `--min-improvement` | `0` / `0.01` | Optional early stopping. |
| `--seed` | `42` | Seeds torch / cuda / python / numpy. |
| `--log-every` | `10` | Log cadence. |

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_qwen3_5_9b.py \
    --hf-model Qwen/Qwen3.5-9B \
    --tracks-dir ./pt_tracks/qwen3_5_9b_n16_d4 \
    --out-dir ./pt_train_out \
    --max-steps 4000 --seq-len 4096 --batch-size 1 \
    --activation-checkpoint --rank0-tracks 1 \
    --eval-every 200 --save-every 500
```

## Evaluate

Three torchrun entry points. All reuse the same per-rank track layout as training, so `--rank0-tracks` must match the value used at training time.

- **Logit fidelity vs teacher** — `scripts/eval_fidelity.py`. KL (forward + reverse), top-1 / top-5 agreement and top-5 IoU, per-sync-boundary hidden MSE, student / teacher perplexity and gap.
  ```bash
  torchrun --standalone --nproc-per-node=8 scripts/eval_fidelity.py \
      --hf-model Qwen/Qwen3.5-9B \
      --checkpoint-dir ./pt_train_out/best \
      --num-batches 200
  ```
- **lm-evaluation-harness** — `scripts/eval_lm_harness.py`. Runs the harness on the student and/or teacher (`--target {student,teacher,both}`) with the same seeds, so request streams align across ranks. Default tasks: `hellaswag,arc_easy,arc_challenge,winogrande,piqa`; pass `--include-mmlu` to add MMLU.
  ```bash
  torchrun --standalone --nproc-per-node=8 scripts/eval_lm_harness.py \
      --hf-model Qwen/Qwen3.5-9B \
      --checkpoint-dir ./pt_train_out/best \
      --output-json ./pt_train_out/lm_eval.json
  ```
- **Replicated-param sync sanity check** — `scripts/verify_kv_sync.py`. Runs N distillation steps and asserts that `q_proj` (unique per track) changes, while `k_proj` and `input_layernorm` change *and* remain bit-identical across their replication groups.
  ```bash
  torchrun --standalone --nproc-per-node=8 scripts/verify_kv_sync.py \
      --hf-model Qwen/Qwen3.5-9B \
      --tracks-dir ./pt_tracks/qwen3_5_9b_n16_d4 \
      --steps 2
  ```
