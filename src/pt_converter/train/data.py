"""Calibration / fine-tune data loader.

For the first conversion pass, perplexity-recovery distillation only needs
modest-quality next-token data. We default to streaming WikiText-103 raw,
tokenize on-the-fly, pack into fixed-length sequences. The user can swap in
SlimPajama or instruction data later by passing a different `dataset_name`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


@dataclass
class CalibrationDataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    split: str = "train"
    seq_len: int = 4096
    text_key: str = "text"


class PackedTokenStream(IterableDataset):
    """Streams a HF dataset, tokenizes, and packs into fixed-length sequences."""

    def __init__(self, tokenizer, cfg: CalibrationDataConfig):
        from datasets import load_dataset  # local import: avoid hard dep at module load

        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.ds = load_dataset(
            cfg.dataset_name, cfg.dataset_config, split=cfg.split, streaming=True
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf: list[int] = []
        seq_len = self.cfg.seq_len
        for example in self.ds:
            text = example.get(self.cfg.text_key, "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            while len(buf) >= seq_len + 1:
                chunk = buf[: seq_len + 1]
                buf = buf[seq_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones(seq_len, dtype=torch.long)}
