"""Engine rails: the shadow-replay forward and cached decode.

Rail 1 — exact copies ≡ dense: with undegraded shadow copies the carry equals
the true residual at every sublayer, so the engine must reproduce the dense
model bit-near-exactly at ANY sync schedule (the ``replica:exact ≡ oracle``
identity the probe established).

Rail 2 — incremental ≡ full-sequence: cached 1-token decode steps must match
the same positions of one whole-sequence replay (causality), with the real
degraded (wanda) copies in the loop. This is the rail that protects decode.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.engine import DenseShadow, EngineCaches, generate, replay_chunk
from parallm.model.pt_model import PTWrappedModel
from parallm.model.replica import collect_input_norms, degrade_track_layers
from parallm.slicer.convert import slice_model_to_tracks

N_TRACKS = 4
SYNC = [3, 7]  # one post-attn boundary (3) + the final post-MLP sync (7)


def _tiny_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def _build():
    cfg = _tiny_config()
    torch.manual_seed(42)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    torch.manual_seed(7)
    nn.init.normal_(dense.lm_head.weight, mean=0.0, std=0.02)
    dense.eval()

    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=4, text_config_attr="config"
    )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=N_TRACKS,
        local_track_ids=tuple(range(N_TRACKS)),
        sync_after_layers=manifest.sync_layer_indices,
        track_group=None,
    )
    pt.eval()
    pt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    return cfg, dense, pt


def test_exact_shadow_matches_dense():
    cfg, dense, pt = _build()
    # Exact copies can share the student's own modules (read-only weights).
    shadow = DenseShadow([list(tm.layers) for tm in pt.text_models])

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        dense_logits = dense.lm_head(dense(input_ids=input_ids, use_cache=False).last_hidden_state)
    engine_logits = replay_chunk(pt, shadow, input_ids, SYNC)

    max_abs = (dense_logits - engine_logits).abs().max().item()
    assert max_abs < 5e-4, f"exact-copy replay drifts from dense by {max_abs}"


def _degraded_shadow(dense, pt):
    calib = [{"input_ids": torch.randint(0, dense.config.vocab_size, (2, 32))}]
    norms = collect_input_norms(dense, calib, device="cpu")
    return DenseShadow([
        degrade_track_layers(tm, norms, N_TRACKS, tid, frac=0.5)
        for tid, tm in enumerate(pt.text_models)
    ])


def test_cached_decode_matches_full_sequence():
    cfg, dense, pt = _build()
    shadow = _degraded_shadow(dense, pt)

    torch.manual_seed(5)
    prompt = torch.randint(0, cfg.vocab_size, (1, 12))
    P, NEW = prompt.shape[1], 6

    gen = generate(pt, shadow, prompt, SYNC, max_new_tokens=NEW)
    full_ids = torch.cat([prompt, gen], dim=1)
    logits_full = replay_chunk(pt, shadow, full_ids, SYNC)

    # Greedy rollout consistency: full-sequence argmax reproduces the decode.
    assert torch.equal(logits_full[:, P - 1 : P + NEW - 1].argmax(-1), gen)

    # Logit-level parity of every cached step against the same full-seq position.
    caches = EngineCaches.build(pt)
    lg = replay_chunk(pt, shadow, prompt, SYNC, caches)
    assert torch.allclose(lg, logits_full[:, :P], atol=1e-3)
    for j in range(NEW):
        lg = replay_chunk(pt, shadow, full_ids[:, P + j : P + j + 1], SYNC, caches)
        max_abs = (lg[:, 0] - logits_full[:, P + j]).abs().max().item()
        assert max_abs < 1e-3, f"decode step {j} drifts from full-seq by {max_abs}"


def test_packed_shadow_matches_dense_shadow(tmp_path):
    """The P2 provider: a packed pool file must reproduce the materialized
    ``degrade_track_layers`` path bit-exactly through the full replay."""
    from parallm.engine import PackedShadow
    from parallm.model.replica import norms_for
    from parallm.model.replica_pack import pack_sparse_weight, save_replica_pool
    from parallm.slicer.convert import slice_model_to_tracks  # noqa: F811
    from parallm.utils.checkpoint import save_tracks

    cfg, dense, pt = _build()
    calib = [{"input_ids": torch.randint(0, cfg.vocab_size, (2, 32))}]
    norms = collect_input_norms(dense, calib, device="cpu")
    dense_sh = DenseShadow([
        degrade_track_layers(tm, norms, N_TRACKS, tid, frac=0.5, bits=4)
        for tid, tm in enumerate(pt.text_models)
    ])

    # Write the track shards + a matching packed pool (the build_replicas path).
    tracks, manifest = slice_model_to_tracks(
        dense, n_tracks=N_TRACKS, sync_block_depth=4, text_config_attr="config"
    )
    save_tracks(tmp_path, tracks, manifest)
    from parallm.model.replica_build import iter_track_linears

    pool = {}
    for t in range(N_TRACKS):
        shard = str(tmp_path / f"track_{t}.safetensors")
        for li, rel, w in iter_track_linears(shard):
            nv = norms_for(norms, N_TRACKS, li, rel, w.shape[-1], t)
            pool[(t, li, rel)] = pack_sparse_weight(w, 0.5, nv, bits=4)
    pool_path = str(tmp_path / "replicas.safetensors")
    save_replica_pool(pool_path, pool, {"config": "qwanda:4:0.5"})

    packed_sh = PackedShadow(pool_path, str(tmp_path), pt, device="cpu",
                             dtype=torch.float32)
    assert packed_sh.n_tracks == N_TRACKS

    torch.manual_seed(11)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    ref = replay_chunk(pt, dense_sh, input_ids, SYNC)
    got = replay_chunk(pt, packed_sh, input_ids, SYNC)
    assert torch.equal(got, ref), (got - ref).abs().max().item()


def test_cuda_graph_decode_matches_eager():
    """Graphed windows must reproduce eager decode token-for-token (same
    kernels, static buffers — capture changes scheduling, not math)."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cfg, dense, pt = _build()
    pt = pt.to("cuda")
    shadow = DenseShadow([list(tm.layers) for tm in pt.text_models])
    torch.manual_seed(5)
    prompt = torch.randint(0, cfg.vocab_size, (1, 12), device="cuda")

    gen_eager = generate(pt, shadow, prompt, SYNC, max_new_tokens=6)
    gen_graph = generate(pt, shadow, prompt, SYNC, max_new_tokens=6,
                         use_cuda_graphs=True)
    assert torch.equal(gen_eager, gen_graph)


def test_generate_reports_comm_rounds():
    cfg, dense, pt = _build()
    shadow = DenseShadow([list(tm.layers) for tm in pt.text_models])
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    timing: dict = {}
    generate(pt, shadow, prompt, SYNC, max_new_tokens=3, timing=timing)
    # Per chunk: 1 embed broadcast + 1 post-attn boundary + 1 final sync;
    # 3 chunks run (prefill + 2 decode steps — the last token skips its forward).
    assert timing["rounds"] == 3 * 3
    assert len(timing["per_token_ms"]) == 2
