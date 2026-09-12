"""`parallm.utils.layers`: the one definition of "which layers" and "which phase".

Five CLI flags parse a layer list and `--sync-phase` parses a per-layer schedule on
top of it, so the inclusive-range rule and the coverage rule are pinned here rather
than re-argued at each call site.
"""
from __future__ import annotations

import pytest

from parallm.utils.layers import PHASES, format_layers, parse_layers, resolve_phases


def test_ranges_are_inclusive_of_both_ends():
    # L32-62 is 31 layers everywhere in this program — a half-open copy would drop
    # L62 from one arm and nothing would raise.
    assert parse_layers("32-62") == list(range(32, 63))
    assert len(parse_layers("32-62")) == 31
    assert parse_layers("32,40,43") == [32, 40, 43]
    assert parse_layers("8,42-45,63") == [8, 42, 43, 44, 45, 63]
    assert parse_layers("") == []
    assert parse_layers(" 3 , 7 ") == [3, 7]


def test_format_layers_is_the_inverse():
    assert format_layers({0, 1, 2, 32}) == "0-2,32"
    assert format_layers([]) == "-"
    assert format_layers([5]) == "5"
    for spec in ("0-63", "8,42-45,63", "3", "0-2,32"):
        assert format_layers(parse_layers(spec)) == spec


@pytest.mark.parametrize("phase", PHASES)
def test_a_bare_phase_covers_every_layer(phase):
    assert resolve_phases(phase, 8) == [phase] * 8
    # ...and spelling it out explicitly is the same schedule.
    assert resolve_phases(f"0-7:{phase}", 8) == [phase] * 8


def test_segments_apply_in_order_so_a_leading_bare_phase_is_the_default():
    assert resolve_phases("0-3:post-attn,4-7:post-mlp", 8) == (
        ["post-attn"] * 4 + ["post-mlp"] * 4
    )
    # The isolation cell: exact everywhere, one band carved out.
    got = resolve_phases("exact,4-5:post-attn", 8)
    assert got == ["exact"] * 4 + ["post-attn"] * 2 + ["exact"] * 2
    # Last write wins.
    assert resolve_phases("post-attn,0-7:post-mlp", 8) == ["post-mlp"] * 8
    # A segment's layers may themselves be a comma list.
    assert resolve_phases("post-mlp,1,3:post-attn", 8) == [
        "post-mlp", "post-attn", "post-mlp", "post-attn",
        "post-mlp", "post-mlp", "post-mlp", "post-mlp",
    ]


def test_every_layer_must_be_named():
    """No implicit default. A spec that covers only part of the stack would score a
    different network than the flag describes, silently — the failure `train_meta_arg`
    exists to prevent."""
    with pytest.raises(ValueError, match=r"names no phase for layers 4-7"):
        resolve_phases("0-3:post-attn", 8)


def test_a_typo_names_itself():
    with pytest.raises(ValueError, match="unknown sync phase 'post-atn'"):
        resolve_phases("post-atn", 8)
    with pytest.raises(ValueError, match="unknown sync phase 'postmlp'"):
        resolve_phases("0-7:postmlp", 8)
    with pytest.raises(ValueError, match=r"layer 8 is outside 0\.\.7"):
        resolve_phases("0-3:post-attn,4-8:exact", 8)
    # Layers stranded before a bare phase would be silently overwritten by it.
    with pytest.raises(ValueError, match="would drop them"):
        resolve_phases("3,7,exact", 8)
    # A trailing layer list with no phase is a layer list, not a phase name.
    with pytest.raises(ValueError, match="name no phase"):
        resolve_phases("exact,4-5", 8)


def test_a_spec_round_trips_through_train_meta(tmp_path):
    """The reason the per-layer schedule rides the EXISTING `--sync-phase` string.

    `args.sync_phase` is already written into train_meta.json by the trainer and read
    back by the eval scripts, so a spec needs no new plumbing to survive a checkpoint.
    A separate flag would need its own round-trip, or eval scores a different network
    — the recorded 0.553-vs-0.700 failure.
    """
    import json

    from parallm.utils.checkpoint import train_meta_arg

    spec = "0-31:post-attn,32-63:post-mlp"
    (tmp_path / "train_meta.json").write_text(
        json.dumps({"args": {"sync_phase": spec, "fuse_tracks": 1}})
    )
    got, source = train_meta_arg(tmp_path, "sync_phase", "post-attn")
    assert (got, source) == (spec, "train_meta.json")
    assert resolve_phases(got, 64) == resolve_phases(spec, 64)

    # No metadata beside the weights: the documented default, not the model's ctor one.
    assert train_meta_arg(tmp_path / "nope", "sync_phase", "post-attn") == (
        "post-attn", "default")
