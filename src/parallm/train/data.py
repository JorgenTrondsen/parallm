"""Calibration / fine-tune data loader.

Perplexity-recovery distillation matches a frozen teacher via KL / CE / block-MSE,
so the student only learns to match the teacher on inputs it *actually sees*. To
recover quality across the teacher's (Qwen3.5) code/math-heavy distribution — not
just encyclopedic English — the default is a streamable **mixture** (broad web +
up-weighted code + math), weighted-interleaved on the fly. Swap mixtures with
``CalibrationDataConfig.from_preset(...)``, a custom list of ``DataSourceSpec``, or
``CalibrationDataConfig.single(...)`` for a single dataset.

All sources stream (``streaming=True``): nominal dataset size is irrelevant — only
the tokens actually consumed are fetched and then discarded, nothing is downloaded
in full (a 4k-step run at seq=4096 touches ~tens of millions of tokens regardless
of whether the source nominally holds 600M or 600B).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


@dataclass
class DataSourceSpec:
    """One streaming text source in a (possibly interleaved) mixture.

    ``weight`` is a relative interleave probability (normalized across the mixture).
    Sources must be parquet-native (standard-format) datasets — modern ``datasets``
    no longer supports script-based loaders (e.g. ``codeparrot/github-code-clean``);
    use parquet mirrors instead.
    """

    dataset_name: str
    dataset_config: str | None = None
    split: str = "train"
    text_key: str = "text"
    weight: float = 1.0


# Named mixtures. `qwen-mix` (default) approximates Qwen3.5's code/math tilt with
# parquet-native sources. Its code source `bigcode/the-stack-dedup` is multi-language
# (matching Qwen better than Python-only codeparrot) but **gated**: the running HF
# account must accept its terms at huggingface.co/datasets/bigcode/the-stack-dedup
# AND have HF_TOKEN set, else streaming 404s. `slimpajama` (DKYoon's 6B parquet mirror,
# already blending web/books/wiki/arxiv/github/stackexchange) and `wikitext` are both
# ungated — use them for token-less runs.
_PRESETS: dict[str, list[DataSourceSpec]] = {
    "wikitext": [
        DataSourceSpec("Salesforce/wikitext", "wikitext-103-raw-v1", text_key="text"),
    ],
    "slimpajama": [
        DataSourceSpec("DKYoon/SlimPajama-6B", text_key="text"),
    ],
    "qwen-mix": [
        DataSourceSpec("DKYoon/SlimPajama-6B", text_key="text", weight=0.70),
        DataSourceSpec("open-web-math/open-web-math", text_key="text", weight=0.15),
        DataSourceSpec("bigcode/the-stack-dedup", text_key="content", weight=0.15),  # gated; needs HF_TOKEN
    ],
}

DEFAULT_PRESET = "qwen-mix"


def preset_names() -> list[str]:
    """Names of the built-in mixtures (for argparse `choices`)."""
    return list(_PRESETS)


def preset_sources(name: str) -> list[DataSourceSpec]:
    """Return a fresh copy of the named preset's source list."""
    if name not in _PRESETS:
        raise KeyError(f"unknown data preset {name!r}; choose from {preset_names()}")
    return [replace(s) for s in _PRESETS[name]]


def parse_source_spec(spec: str) -> DataSourceSpec:
    """Parse a CLI ``NAME[:CONFIG[:TEXT_KEY[:WEIGHT]]]`` source string.

    Empty CONFIG / TEXT_KEY fields fall back to defaults (None / "text"), so
    ``name::code:0.2`` sets text_key + weight while leaving config unset.
    """
    parts = spec.split(":")
    if not parts[0]:
        raise ValueError(f"empty dataset name in --data-source spec: {spec!r}")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    text_key = parts[2] if len(parts) > 2 and parts[2] else "text"
    weight = float(parts[3]) if len(parts) > 3 and parts[3] else 1.0
    return DataSourceSpec(name, config, text_key=text_key, weight=weight)


@dataclass
class CalibrationDataConfig:
    sources: list[DataSourceSpec] = field(
        default_factory=lambda: preset_sources(DEFAULT_PRESET)
    )
    seq_len: int = 4096
    # Fixed interleave seed: the loader is consumed with num_workers=0 and NO
    # DistributedSampler, so every rank must read the identical stream (under
    # vocab-parallel every rank backwards the same batch and the SyncBoundary
    # all-reduce assumes identical inputs). Keep this equal across ranks.
    seed: int = 42
    # Held-out boundary: discard the first ``skip_docs`` raw documents of the
    # (interleaved) stream before packing. Used to carve a disjoint val set out of
    # the SAME mixture: the train stream sets ``skip_docs=N`` while a mirror val set
    # reads the front (``skip_docs=0``) of the identical seeded sequence, so the two
    # cover non-overlapping document ranges. 0 = read from the start (legacy).
    skip_docs: int = 0

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "CalibrationDataConfig":
        return cls(sources=preset_sources(name), **kwargs)

    @classmethod
    def single(
        cls,
        dataset_name: str = "Salesforce/wikitext",
        dataset_config: str | None = "wikitext-103-raw-v1",
        split: str = "train",
        text_key: str = "text",
        **kwargs,
    ) -> "CalibrationDataConfig":
        """One-source config (e.g. the held-out validation set)."""
        return cls(
            sources=[DataSourceSpec(dataset_name, dataset_config, split, text_key)],
            **kwargs,
        )


def _interleave(streams: list, weights: list[float], seed: int):
    """Single stream → itself; multiple → seeded weighted interleave.

    ``stopping_strategy="all_exhausted"`` keeps the combined stream alive until
    every source is exhausted (the huge streaming sources never exhaust within a
    run); the seed makes the source-choice sequence deterministic across ranks.
    """
    if len(streams) == 1:
        return streams[0]
    from datasets import interleave_datasets  # local: avoid hard dep at import

    total = sum(weights)
    probs = [w / total for w in weights]
    return interleave_datasets(
        streams, probabilities=probs, seed=seed, stopping_strategy="all_exhausted"
    )


class PackedTokenStream(IterableDataset):
    """Streams one or more HF datasets, tokenizes, and packs into fixed-length sequences.

    Multiple sources are weighted-interleaved with the config's fixed seed so the
    stream is identical on every rank — do NOT add per-node sharding here (see the
    note on ``CalibrationDataConfig.seed``).

    ``_streams`` is a test seam: pre-built (already "text"-keyed) iterables to use
    instead of ``load_dataset``, so the packing / interleave logic can be exercised
    without network.
    """

    def __init__(self, tokenizer, cfg: CalibrationDataConfig, *, _streams: list | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

        if _streams is not None:
            streams = list(_streams)
            weights = [s.weight for s in cfg.sources] if cfg.sources else [1.0] * len(streams)
        else:
            from datasets import load_dataset  # local: avoid hard dep at module load

            streams, weights = [], []
            for s in cfg.sources:
                ds = load_dataset(
                    s.dataset_name, s.dataset_config, split=s.split, streaming=True,
                )
                # Normalize every source to a single "text" column so they share a
                # feature schema (required by interleave_datasets).
                if s.text_key != "text":
                    ds = ds.rename_column(s.text_key, "text")
                ds = ds.select_columns(["text"])
                streams.append(ds)
                weights.append(s.weight)

        self.ds = _interleave(streams, weights, cfg.seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf: list[int] = []
        seq_len = self.cfg.seq_len
        # Skip the held-out prefix: counted on EVERY raw example (before the
        # empty-text check below) so the boundary is content-independent and
        # identical across ranks — the train stream (skip_docs=N) and a mirror
        # val stream (skip_docs=0) then read disjoint document ranges of the same
        # seeded sequence. The train loader is iterated once per run, so this is a
        # one-time advance; a mirror val (skip_docs=0) pays nothing.
        skip_remaining = self.cfg.skip_docs
        for example in self.ds:
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            text = example.get("text", "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            while len(buf) >= seq_len + 1:
                chunk = buf[: seq_len + 1]
                buf = buf[seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                # labels aligned with input_ids (HF convention): labels[i] == input_ids[i].
                # The fidelity NLL consumer does the next-token shift internally, so labels
                # MUST NOT be pre-shifted here — pre-shifting double-shifts and scores each
                # prediction against token p+2 (near-random ppl for any model).
                labels = input_ids.clone()
                yield {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones(seq_len, dtype=torch.long)}
