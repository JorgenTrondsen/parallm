# pt_converter

`pt_converter` is a model-agnostic conversion and fine-tuning toolkit that turns a pretrained dense transformer into a **parallel-track (PT) transformer**. The dense model's weights are sliced across `N` parallel "tracks", each of which runs as a smaller decoder on its own rank/GPU. At configurable sync boundaries (every `D` layers) the tracks all-reduce their partial residual updates so the combined forward stays mathematically equivalent to the original dense model. New model families are added by registering a `ModelAdapter` — the slicer engine, sync logic, and training loop themselves stay model-agnostic. A short distillation stage (frozen dense teacher → sliced student) is included to recover any perplexity lost during the static weight conversion.

The first supported family is **Qwen3.5** (mixed full-attention / linear-attention decoder), wired up via the `qwen3_5_text` adapter.

## Directory structure

```
pt_converter/
├── pyproject.toml                          # Build metadata and dependency pins.
├── scripts/
│   ├── convert_qwen3_5_9b.py               # CLI: slice a pretrained Qwen3.5-9B into N per-track checkpoints.
│   └── train_qwen3_5_9b.py                 # torchrun entrypoint for distributed distillation training.
├── src/pt_converter/
│   ├── __init__.py                         # Public API: max_tracks_for_config, slice_model_to_tracks, PTManifest.
│   ├── adapters/
│   │   ├── __init__.py                     # ModelAdapter dataclass + register_model_adapter / get_model_adapter registry.
│   │   └── qwen3_5.py                      # Registers the "qwen3_5_text" adapter (slicer specs + per-track model class).
│   ├── model/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── pt_model.py                     # PTWrappedModel: per-rank wrapper exposing (logits, sync_hiddens) forward.
│   │   ├── sync.py                         # SyncBoundary: cross-track all-reduce of (h_t - h_pre_block) deltas.
│   │   └── tracks/
│   │       ├── __init__.py                 # Package marker.
│   │       └── qwen3_5.py                  # Per-track Qwen3.5 decoder with SyncBoundary calls at sync layers.
│   ├── slicer/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── base.py                         # SlicerSpec protocol + Colwise/Rowwise/PerHead/Replicated/Fused/KV/GatedQ impls.
│   │   ├── convert.py                      # slice_model_to_tracks engine: applies adapter specs → N state dicts + PTManifest.
│   │   └── qwen3_5.py                      # Qwen3.5-specific SlicerSpec instances for attention / linear-attention / MLP.
│   ├── train/
│   │   ├── __init__.py                     # Package marker.
│   │   ├── data.py                         # PackedTokenStream IterableDataset (WikiText-103 / custom, on-the-fly tokenize + pack).
│   │   ├── distill.py                      # SPD-style step: block-wise teacher-forced MSE + end-to-end logit-KL + LM CE.
│   │   ├── losses.py                       # block_mse, logit_kl, lm_cross_entropy — all with attention-mask support.
│   │   └── teacher.py                      # HookedTeacher: frozen dense model with hooks capturing hiddens at sync indices.
│   └── utils/
│       ├── __init__.py                     # Package marker.
│       ├── checkpoint.py                   # Save/load per-track safetensors + manifest.json.
│       └── max_tracks.py                   # Compute maximum valid N under KV-replication and divisibility rules.
└── tests/
    ├── __init__.py                         # Package marker.
    ├── test_pt_forward_n1.py               # N=1 PT forward must match dense bit-equal; sync_hiddens captured at right depths.
    ├── test_pt_n8_forward_smoke.py         # N=8 simulated distributed forward on CPU; finite outputs, bounded drift.
    ├── test_kv_replication.py              # KV-replicated slices identical within kv-group, unique across groups, reassemble bit-equal.
    ├── test_slicer_specs.py                # Per-SlicerSpec slice/reassemble round-trip and shape unit tests.
    ├── test_sync_schedule.py               # sync_block_depth + num_layers → correct per-track sync layer indices.
    ├── test_model_adapter.py               # Adapter registry: register/lookup/idempotent re-register; slicer routing via adapter.
    ├── test_slicer_qwen3_5_integration.py  # Tiny Qwen3.5: N=1 bit-equal, N=2 round-trip, per-track shapes match config.
    └── test_max_tracks.py                  # max_tracks_for_config against synthetic configs under the four-rule constraint set.
```

## File reference

### Top level

#### [pyproject.toml](pyproject.toml)
Standard PEP 621 manifest. Declares the `pt-converter` package, requires Python 3.10+, and pins runtime dependencies `torch>=2.4`, `transformers>=4.57.0.dev0`, `datasets`, `safetensors`, and `accelerate`. Adds `pytest` as the only optional/test dependency and points setuptools at the `src/` layout.

### scripts/

#### [scripts/convert_qwen3_5_9b.py](scripts/convert_qwen3_5_9b.py)
One-shot single-process conversion script. Loads a pretrained Qwen3.5-9B dense model from the HuggingFace hub or a local path, dispatches into the model-agnostic `slice_model_to_tracks` engine, and writes one safetensors file per track alongside a `manifest.json` describing the sync schedule and per-layer metadata.

#### [scripts/train_qwen3_5_9b.py](scripts/train_qwen3_5_9b.py)
`torchrun`-launched entrypoint for distributed distillation. Each rank loads its own sliced student track plus the frozen dense teacher, constructs a process group, and runs the SPD-style step in [src/pt_converter/train/distill.py](src/pt_converter/train/distill.py): block-wise teacher-forced MSE at every sync boundary plus end-to-end logit KL and language-modeling cross-entropy. Optionally saves resumable per-track checkpoints.

### src/pt_converter/

#### [src/pt_converter/__init__.py](src/pt_converter/__init__.py)
The package's public API. Re-exports `max_tracks_for_config`, `slice_model_to_tracks`, and `PTManifest`, and imports the `adapters` subpackage at top level so built-in adapter registrations (`qwen3_5_text`, etc.) run as a side-effect of `import pt_converter`.

### src/pt_converter/adapters/

#### [src/pt_converter/adapters/__init__.py](src/pt_converter/adapters/__init__.py)
Defines the `ModelAdapter` dataclass — the per-family contract that bundles together (a) the slicer-spec factory for each layer kind, (b) the per-track model class, and (c) any state-dict prefix remapping. Exposes `register_model_adapter(model_type, adapter)` and `get_model_adapter(model_type)`; the slicer and runtime route everything through this registry so the engine stays model-agnostic.

#### [src/pt_converter/adapters/qwen3_5.py](src/pt_converter/adapters/qwen3_5.py)
Concrete adapter that wires up Qwen3.5 text models. Imports the per-track decoder from [src/pt_converter/model/tracks/qwen3_5.py](src/pt_converter/model/tracks/qwen3_5.py) and the slicer-spec factory from [src/pt_converter/slicer/qwen3_5.py](src/pt_converter/slicer/qwen3_5.py), assembles them into a `ModelAdapter`, and registers it under `model_type = "qwen3_5_text"` at import time.

### src/pt_converter/model/

#### [src/pt_converter/model/pt_model.py](src/pt_converter/model/pt_model.py)
`PTWrappedModel` is the per-rank user-facing module. It owns this rank's track decoder and `lm_head`, runs a forward that returns `(logits, sync_hiddens)` (the hidden states captured at each sync boundary, used by the distillation step for block-MSE), and remaps the slicer's output state-dict keys onto the per-track module via `load_track_state_dict`. The companion `PTTrackTextModelConfig` is the small engine ↔ adapter contract that the slicer uses to spin up a per-track instance with the right head counts, intermediate dim, and embedding shape.

#### [src/pt_converter/model/sync.py](src/pt_converter/model/sync.py)
Implements the only cross-track collective in the PT forward. At a sync point each track has produced a partial post-block hidden `h_t`, and `SyncBoundary` recombines them via `h_synced = h_pre_block + sum_t (h_t - h_pre_block)` — a single `all_reduce(SUM)` on the delta followed by an add. When `n_tracks <= 1` or no process group is supplied, it degenerates to a pure no-op, which is what makes the `N=1` correctness gate work.

### src/pt_converter/model/tracks/

#### [src/pt_converter/model/tracks/qwen3_5.py](src/pt_converter/model/tracks/qwen3_5.py)
Per-track Qwen3.5 text decoder. Builds standard Qwen3.5 decoder layers (mix of full-attention and linear-attention) but with reduced head counts / intermediate sizes that match the slicer's per-track shapes, and inserts a `SyncBoundary` call after every layer index listed in the sync schedule. Otherwise the forward mirrors the dense Qwen3.5 forward closely so behavior at `N=1` stays bit-equal to dense.

### src/pt_converter/slicer/

#### [src/pt_converter/slicer/base.py](src/pt_converter/slicer/base.py)
Declarative slicing primitives. Defines the `SlicerSpec` protocol (`slice`, `reassemble`, `per_track_shape`) and seven concrete implementations: `Colwise` and `Rowwise` (standard tensor-parallel splits), `PerHead` (1-D per-head params like `A_log` / `dt_bias`), `Replicated` (norms, embeddings), `FusedSegmentColwise` (e.g. fused `[Q | K | V]` in-proj), `KVReplicatedColwise` (lets `N` exceed `num_kv_heads` by replicating k/v slices within each kv-group), and `GatedQColwise` (Qwen's doubled-`q_proj` with interleaved gate columns). All specs are side-effect free and unit-tested in isolation.

#### [src/pt_converter/slicer/convert.py](src/pt_converter/slicer/convert.py)
The model-agnostic conversion engine. `slice_model_to_tracks(model, n_tracks)` looks up the right `ModelAdapter` from the model's `config.model_type`, walks every parameter, applies the adapter-provided `SlicerSpec` to produce `N` per-track tensors, and returns both the list of per-track state dicts and a `PTManifest` describing model metadata (sync layer indices, layer types, per-track shapes).

#### [src/pt_converter/slicer/qwen3_5.py](src/pt_converter/slicer/qwen3_5.py)
The Qwen3.5-specific `SlicerSpec` catalogue. Provides the spec dict for each layer kind: full-attention (`q_proj` / `k_proj` / `v_proj` / `o_proj` plus q/k RMSNorms), linear-attention (fused `in_proj_qkv`, `conv1d`, `A_log`, `dt_bias`, etc.), and the gated MLP. Encodes Qwen-specific quirks like the gated-q output doubling, KV replication within groups, and the fused-segment in-proj layout.

### src/pt_converter/train/

#### [src/pt_converter/train/data.py](src/pt_converter/train/data.py)
`PackedTokenStream`, an `IterableDataset` that streams a text corpus (WikiText-103 by default), tokenizes on the fly with the supplied HF tokenizer, and packs the token stream into fixed-length sequences suitable for calibration or fine-tuning. Avoids materializing the full dataset and works cleanly with multi-worker DataLoader.

#### [src/pt_converter/train/distill.py](src/pt_converter/train/distill.py)
One distillation step in SPD style. Runs the frozen teacher under `no_grad`, recording the pre- and post-block hidden states at each sync boundary. For every block the student is run *teacher-forced* — starting from the teacher's pre-block hidden — then synced, and a per-token block-MSE is taken against the teacher's post-block hidden. A final full student forward produces logits for logit-KL and LM cross-entropy against the labels.

#### [src/pt_converter/train/losses.py](src/pt_converter/train/losses.py)
The three losses used by `distill.py`: `block_mse` (per-token MSE between synced student and teacher hiddens at each sync point), `logit_kl` (temperature-softmaxed forward KL on the final logits), and `lm_cross_entropy` (standard next-token CE on the labels). All three accept an attention mask so padded sequence positions don't pollute the loss.

#### [src/pt_converter/train/teacher.py](src/pt_converter/train/teacher.py)
`HookedTeacher` wraps a frozen dense decoder + `lm_head` and registers forward hooks on the layers immediately preceding each sync boundary. Its `no_grad` forward returns the final logits plus a dict of captured hidden states keyed by sync index, which `distill.py` consumes for block-wise teacher forcing.

### src/pt_converter/utils/

#### [src/pt_converter/utils/checkpoint.py](src/pt_converter/utils/checkpoint.py)
Per-track checkpoint I/O. Saves a list of per-track state dicts as one safetensors file each plus a `manifest.json` (the `PTManifest` content), and loads them back with shape recovery so the trainer can resume from any track count without re-slicing the dense weights.

#### [src/pt_converter/utils/max_tracks.py](src/pt_converter/utils/max_tracks.py)
`max_tracks_for_config` enumerates the divisibility constraints implied by the model config (attention heads, kv heads, MLP intermediate size, linear-attention dims) and reports the largest `N` that satisfies all of them simultaneously. Crucially enforces the KV-replication rule: `N` must be a multiple of `num_key_value_heads` whenever we push past it.

### tests/

#### [tests/test_pt_forward_n1.py](tests/test_pt_forward_n1.py)
The headline correctness gate: with `N=1`, `SyncBoundary` is a no-op, so the wrapped PT model must produce bit-equal logits to the original dense Qwen3.5 model on the same input. Also checks that the `sync_hiddens` dict surfaces the right number of captures at the expected layer depths.

#### [tests/test_pt_n8_forward_smoke.py](tests/test_pt_n8_forward_smoke.py)
Simulates an `N=8` distributed forward on a single CPU process by running each track's forward in turn and stand-in for the all-reduce with an explicit delta-sum / broadcast. Asserts outputs are finite and that the drift from the dense model is bounded (an `O(1)` gap is expected pre-distillation).

#### [tests/test_kv_replication.py](tests/test_kv_replication.py)
End-to-end validation of `KVReplicatedColwise` for `k_proj` and `v_proj` under GQA: slices belonging to the same kv-group must be bit-identical, slices across kv-groups must differ, and the unique per-group slices concatenated must reconstruct the dense tensor exactly.

#### [tests/test_slicer_specs.py](tests/test_slicer_specs.py)
Unit-level round-trip tests for every `SlicerSpec` in [src/pt_converter/slicer/base.py](src/pt_converter/slicer/base.py): for synthetic tensors, `reassemble(slice(...))` recovers the original, `per_track_shape` matches the actual slice shape, and divisibility errors are raised when expected.

#### [tests/test_sync_schedule.py](tests/test_sync_schedule.py)
Validates the small helper that turns `(sync_block_depth=D, num_layers=L)` into the list of layer indices where a `SyncBoundary` is inserted (e.g. `D=4, L=32 → [3, 7, 11, ..., 31]`), including edge cases when `L` is not a clean multiple of `D`.

#### [tests/test_model_adapter.py](tests/test_model_adapter.py)
Exercises the adapter registry: importing the package registers built-in adapters, `get_model_adapter` looks them up by `model_type`, re-registering the same `model_type` is idempotent, and the slicer engine correctly routes a synthetic registered adapter through `slice_model_to_tracks`.

#### [tests/test_slicer_qwen3_5_integration.py](tests/test_slicer_qwen3_5_integration.py)
Integration tests that build a tiny Qwen3.5 config end-to-end. Slice at `N=1` and check bit-equal against dense; slice at `N=2` and verify the per-spec `reassemble` rules recover every parameter; and check that each per-track tensor matches the shape predicted by `PTTrackTextModelConfig`.

#### [tests/test_max_tracks.py](tests/test_max_tracks.py)
Drives `max_tracks_for_config` over a battery of synthetic Qwen-like configs to confirm correct enumeration of valid `N` values under the four divisibility rules (attention heads, kv heads, MLP intermediate, linear-attention dims) and rejection of configs where no satisfying `N > 1` exists.
