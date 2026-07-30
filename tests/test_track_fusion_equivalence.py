"""Track fusion at F must equal a real N/F-track model.

The other fusion rails (`test_indivisible_tracks.py`) are invariant to how tracks
are ASSIGNED to groups: under the `exact` schedule every sublayer syncs globally,
so any partition sums the same deltas, and at F=N there is only one group. So
neither would catch fusion pairing the wrong tracks — yet "F=3 behaves as the
N=8 convert" is the whole claim the lever rests on.

This pins it directly. On a config whose every split dim divides by 4, the N=4
column partition nests exactly inside the N=2 one (chunks [0:8][8:16] vs [0:16]),
so N=4 with fuse_size=2 and N=2 unfused are the same arithmetic on the same
weights — at a SPARSE schedule, where the own-carry path is what differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from parallm.adapters import get_adapter_for_config
from parallm.model.merge import merge_track_states, split_track_state
from parallm.model.pt_model import PTWrappedModel
from parallm.slicer.convert import slice_model_to_tracks


def _nesting_config():
    """Every split dim divisible by 4, so the N=4 slabs nest inside the N=2 ones."""
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=32,        # 4 | 32 and 2 | 32
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,      # divides 4 -> GDN stays in compact mode
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention"] * 2 + ["full_attention"]
        + ["linear_attention"] * 2 + ["full_attention"],
        full_attention_interval=3,
        vocab_size=64,
        rms_norm_eps=1e-6,
    )


def _run(dense, cfg, n_tracks, sync_after, fuse_size, input_ids, merge_group=1):
    """One PT forward. ``merge_group`` runs the MERGED implementation of fusion:
    the model holds n_tracks LOGICAL tracks, each built from `merge_group`
    consecutive shards of an `n_tracks * merge_group`-track convert."""
    n_shards = n_tracks * merge_group
    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_shards, sync_block_depth=1, text_config_attr="config"
    )
    states = {t: tracks[t] for t in range(n_shards)}
    if merge_group > 1:
        states = merge_track_states(
            get_adapter_for_config(cfg), cfg, n_shards, states, merge_group
        )
    pt = PTWrappedModel(
        text_config=cfg,
        n_tracks=n_tracks,
        local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=sync_after,
        track_group=None,
        fuse_size=fuse_size,
        merge_group=merge_group,
    )
    pt.set_sync_phase("post-attn")
    pt.eval()
    pt.load_track_state_dicts(states, strict=False)
    pt.lm_head.load_state_dict(dense.lm_head.state_dict())
    with torch.no_grad():
        return pt(input_ids=input_ids)[0]


def test_n4_fused_pairs_equal_n2():
    cfg = _nesting_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()

    # Sparse (D=2-shaped) schedule: most sublayers own-carry, which is the only
    # place fusion can act — and the only place a wrong grouping would show.
    sync_after = [1, 3, cfg.num_hidden_layers - 1]
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))

    fused = _run(dense, cfg, 4, sync_after, 2, input_ids)
    ref = _run(dense, cfg, 2, sync_after, 1, input_ids)
    drift = (fused - ref).abs().max().item()
    assert drift < 1e-4, f"N=4 fused pairs != N=2: max |dlogit| {drift}"

    # And the claim is non-trivial: unfused N=4 is a genuinely different model.
    unfused = _run(dense, cfg, 4, sync_after, 1, input_ids)
    assert (unfused - ref).abs().max().item() > 1e-3


def test_merged_tracks_equal_summed_fusion():
    """The step-time lever: when a fusion group is a rank's WHOLE track set, running
    it as ONE wide track (concatenated slabs) must equal summing F narrow tracks'
    deltas. Same function, 1/F the kernels and one full-width residual stream
    instead of F — on the 27B at N=24 that is 6.4-7.7 -> ~3 s/step.

    Pinned at a SPARSE schedule, where own-carry is what the merge has to reproduce,
    on the indivisible config's harder cousin: `_nesting_config` keeps the GDN in
    compact mode, so this rail is run BOTH here and (via the exact-schedule rail in
    test_indivisible_tracks) where k-heads replicate and the MLP zero-pads.
    """
    cfg = _nesting_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()

    sync_after = [1, 3, cfg.num_hidden_layers - 1]
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))

    summed = _run(dense, cfg, 4, sync_after, 2, input_ids)
    merged = _run(dense, cfg, 2, sync_after, 1, input_ids, merge_group=2)
    drift = (merged - summed).abs().max().item()
    assert drift < 1e-4, f"merged tracks != summed fusion: max |dlogit| {drift}"

    # Non-vacuous: both differ from the unfused 4-track model.
    unfused = _run(dense, cfg, 4, sync_after, 1, input_ids)
    assert (merged - unfused).abs().max().item() > 1e-3


def test_a_trained_merged_model_saves_back_into_the_summed_model():
    """The save path end to end: TRAIN a merged model, split its weights back to N
    shards, and load those into the summed model — the two forwards must agree.

    `test_merge_then_split_returns_the_original_shards` only checks that splitting
    an UNTRAINED merge is the identity. That misses the case that matters: after
    training, `q_norm`/`k_norm` have diverged per member and the merged tensor's
    leading dim is the only thing that says which member owns which row. If
    `split` unstacked those in the wrong order, or dropped a diverged copy, every
    checkpoint this run writes would be silently mis-sharded and only a later eval
    would notice.
    """
    cfg = _nesting_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    sync_after = [1, 3, cfg.num_hidden_layers - 1]
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))
    adapter = get_adapter_for_config(cfg)
    n_shards, fuse = 4, 2

    tracks, _ = slice_model_to_tracks(
        dense, n_tracks=n_shards, sync_block_depth=1, text_config_attr="config"
    )
    merged_states = merge_track_states(
        adapter, cfg, n_shards, {t: tracks[t] for t in range(n_shards)}, fuse
    )

    merged = PTWrappedModel(
        text_config=cfg, n_tracks=n_shards // fuse,
        local_track_ids=tuple(range(n_shards // fuse)), sync_after_layers=sync_after,
        track_group=None, merge_group=fuse,
    )
    merged.set_sync_phase("post-attn")
    merged.load_track_state_dicts(merged_states, strict=False)
    merged.lm_head.load_state_dict(dense.lm_head.state_dict())

    # One real optimizer step, so the diverged per-member params actually move
    # apart (a fresh slice has them identical — the case that hides mis-ordering).
    merged.train()
    opt = torch.optim.SGD([p for p in merged.parameters() if p.requires_grad], lr=0.05)
    logits, _ = merged(input_ids=input_ids)
    logits.square().mean().backward()
    opt.step()
    merged.eval()

    qn = merged.text_models[0].layers[2].self_attn.q_norm.weight
    assert qn.shape[0] == fuse
    assert not torch.allclose(qn[0], qn[1]), "members must have diverged for this rail"

    with torch.no_grad():
        want, _ = merged(input_ids=input_ids)

    # Split the trained merged weights back into N shards and load them summed.
    shards: dict[int, dict] = {}
    for k, tid in enumerate(merged.local_track_ids):
        one = {
            key[len(f"text_models.{k}."):]: val
            for key, val in merged.state_dict().items()
            if key.startswith(f"text_models.{k}.")
        }
        if merged.lm_head is not None and k == 0:
            one["lm_head.weight"] = merged.state_dict()["lm_head.weight"]
        shards.update(
            split_track_state(adapter, cfg, n_shards, one, fuse, tid * fuse)
        )
    assert sorted(shards) == list(range(n_shards))

    summed = PTWrappedModel(
        text_config=cfg, n_tracks=n_shards, local_track_ids=tuple(range(n_shards)),
        sync_after_layers=sync_after, track_group=None, fuse_size=fuse,
    )
    summed.set_sync_phase("post-attn")
    summed.eval()
    summed.load_track_state_dicts(shards, strict=False)
    with torch.no_grad():
        got, _ = summed(input_ids=input_ids)

    drift = (want - got).abs().max().item()
    assert drift < 1e-4, f"trained merged weights do not round-trip: {drift}"


def test_merge_group_1_is_untouched():
    """Guards the rail above from passing vacuously: merge_group=1 must be the
    bit-identical no-op path, not a differently-shaped model that happens to agree."""
    cfg = _nesting_config()
    torch.manual_seed(0)
    dense = Qwen3_5TextModel(cfg)
    dense.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    dense.eval()
    sync_after = [1, 3, cfg.num_hidden_layers - 1]
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.manual_seed(123))

    assert torch.equal(
        _run(dense, cfg, 4, sync_after, 1, input_ids),
        _run(dense, cfg, 4, sync_after, 1, input_ids, merge_group=1),
    )
