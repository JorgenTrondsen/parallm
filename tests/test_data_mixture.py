"""Unit tests for the streaming data mixture (train/data.py).

Network-free: the packing tests inject plain in-memory iterables via the
``_streams`` test seam; only the interleave-determinism test needs the optional
``datasets`` package (skipped if absent).
"""
from __future__ import annotations

import pytest
import torch

from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    DataSourceSpec,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)


class _CharTok:
    """Deterministic stand-in tokenizer: one id per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


# ----- presets -----

def test_preset_names_and_default():
    names = preset_names()
    assert {"wikitext", "slimpajama", "qwen-mix"} <= set(names)
    assert DEFAULT_PRESET == "qwen-mix"


def test_wikitext_preset_is_legacy_single_source():
    (src,) = preset_sources("wikitext")
    assert src.dataset_name == "Salesforce/wikitext"
    assert src.dataset_config == "wikitext-103-raw-v1"
    assert src.text_key == "text"


def test_qwen_mix_weights_and_keys():
    srcs = preset_sources("qwen-mix")
    assert len(srcs) == 3
    names = [s.dataset_name for s in srcs]
    assert "DKYoon/SlimPajama-6B" in names  # broad parquet base
    # code source carries a non-default text key.
    code = next(s for s in srcs if "the-stack" in s.dataset_name)
    assert code.text_key == "content"
    # weights are tilted toward broad web but include code+math.
    assert pytest.approx(sum(s.weight for s in srcs), rel=1e-6) == 1.0


def test_preset_sources_returns_independent_copies():
    a = preset_sources("qwen-mix")
    a[0].weight = 999.0
    b = preset_sources("qwen-mix")
    assert b[0].weight != 999.0


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        preset_sources("does-not-exist")


# ----- --data-source spec parsing -----

@pytest.mark.parametrize(
    "spec, expected",
    [
        ("ds", DataSourceSpec("ds", None, text_key="text", weight=1.0)),
        ("ds:cfg", DataSourceSpec("ds", "cfg", text_key="text", weight=1.0)),
        ("ds:cfg:body", DataSourceSpec("ds", "cfg", text_key="body", weight=1.0)),
        ("ds:cfg:body:0.3", DataSourceSpec("ds", "cfg", text_key="body", weight=0.3)),
        # empty config / text_key fields fall back to defaults
        ("ds::code:0.2", DataSourceSpec("ds", None, text_key="code", weight=0.2)),
    ],
)
def test_parse_source_spec(spec, expected):
    got = parse_source_spec(spec)
    assert got.dataset_name == expected.dataset_name
    assert got.dataset_config == expected.dataset_config
    assert got.text_key == expected.text_key
    assert got.weight == pytest.approx(expected.weight)


def test_parse_source_spec_empty_name_raises():
    with pytest.raises(ValueError):
        parse_source_spec(":cfg")


# ----- packing -----

def test_pack_shapes_labels_and_contiguous_tiling():
    seq_len = 4
    # 3 docs → 30 chars → 30 tokens → 7 full chunks of seq_len (28 tokens used).
    docs = [{"text": "abcdefghij"}, {"text": "klmnopqrst"}, {"text": "uvwxyz0123"}]
    cfg = CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len)
    stream = PackedTokenStream(_CharTok(), cfg, _streams=[docs])

    all_tokens = [ord(c) for d in docs for c in d["text"]]
    out = list(stream)
    assert len(out) == len(all_tokens) // seq_len  # contiguous tiling, stride == seq_len

    flat = []
    for item in out:
        assert item["input_ids"].shape == (seq_len,)
        assert item["input_ids"].dtype == torch.long
        # labels are NOT pre-shifted (consumers shift internally).
        assert torch.equal(item["labels"], item["input_ids"])
        assert torch.equal(item["attention_mask"], torch.ones(seq_len, dtype=torch.long))
        flat.extend(item["input_ids"].tolist())
    # yielded chunks reconstruct the token stream in order, no gaps/overlap.
    assert flat == all_tokens[: len(out) * seq_len]


def test_empty_documents_are_skipped():
    seq_len = 4
    docs = [{"text": ""}, {"text": "abcdefgh"}, {"text": ""}, {"text": "ijkl"}]
    cfg = CalibrationDataConfig(sources=[DataSourceSpec("x")], seq_len=seq_len)
    out = list(PackedTokenStream(_CharTok(), cfg, _streams=[docs]))
    expected = [ord(c) for c in "abcdefghijkl"]
    flat = [t for item in out for t in item["input_ids"].tolist()]
    assert flat == expected[: len(out) * seq_len]


# ----- interleave determinism (needs `datasets`) -----

def _build_two_source_stream(seed):
    from datasets import Dataset

    a = Dataset.from_dict({"text": [chr(ord("a") + i) * 12 for i in range(20)]}).to_iterable_dataset()
    # second source uses a DIFFERENT text key to exercise rename → "text".
    b_raw = Dataset.from_dict({"body": [chr(ord("A") + i) * 12 for i in range(20)]}).to_iterable_dataset()
    b = b_raw.rename_column("body", "text").select_columns(["text"])
    cfg = CalibrationDataConfig(
        sources=[DataSourceSpec("a", weight=0.5), DataSourceSpec("b", text_key="body", weight=0.5)],
        seq_len=4,
        seed=seed,
    )
    return PackedTokenStream(_CharTok(), cfg, _streams=[a, b])


def test_interleave_is_deterministic_across_ranks():
    pytest.importorskip("datasets")
    # Two independent builds with the same seed == two ranks: identical batches.
    s1, s2 = _build_two_source_stream(123), _build_two_source_stream(123)
    out1 = [item["input_ids"] for item, _ in zip(s1, range(8))]
    out2 = [item["input_ids"] for item, _ in zip(s2, range(8))]
    assert len(out1) == 8
    for a, b in zip(out1, out2):
        assert torch.equal(a, b)


def test_interleave_seed_changes_order():
    pytest.importorskip("datasets")
    s1 = _build_two_source_stream(1)
    s2 = _build_two_source_stream(2)
    out1 = [item["input_ids"].tolist() for item, _ in zip(s1, range(8))]
    out2 = [item["input_ids"].tolist() for item, _ in zip(s2, range(8))]
    # Different seeds should give a different interleave (mixed-source content differs).
    assert out1 != out2
