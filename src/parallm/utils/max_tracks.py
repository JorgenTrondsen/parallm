"""Detect the maximum natural track count for a given model config.

Track-count rule (KV-replicated max-parallelism):

    max_tracks is the largest N satisfying ALL of:
      1. num_attention_heads % N == 0      (each track gets >=1 q-head)
      2. N % num_key_value_heads == 0      (kv-groups split evenly across tracks,
                                            with replication within a group)
      3. linear_num_key_heads % N == 0
         and linear_num_value_heads % N == 0
      4. intermediate_size % N == 0

This is model-agnostic: each model_type registers a function that returns
a `ConstraintSet` and we scan N downward from num_attention_heads to the
first N that satisfies every constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ConstraintSet:
    """Sliceable-dim constraints used by the four-rule track-count check.

    `num_attention_heads` is the upper bound on N (each track gets >=1 q-head).
    `num_key_value_heads` is the dim that N must be a *multiple* of (the kv-group
    factor `tracks_per_kv_group = N // num_kv_heads` must be integer).
    All other entries are dims that N must *divide*.
    """

    num_attention_heads: int
    num_key_value_heads: int
    divides: tuple[int, ...]  # extra dims N must divide (linear_num_*, intermediate_size, ...)


_CONSTRAINT_PROVIDERS: dict[str, Callable[[object], ConstraintSet]] = {}


def register_constraints(model_type: str):
    def _wrap(fn: Callable[[object], ConstraintSet]):
        _CONSTRAINT_PROVIDERS[model_type] = fn
        return fn

    return _wrap


def _candidates_in_range(constraints: ConstraintSet) -> list[int]:
    """Enumerate every N that satisfies all four rules, in descending order."""
    out = []
    for n in range(constraints.num_attention_heads, 0, -1):
        if constraints.num_attention_heads % n != 0:
            continue
        if n % constraints.num_key_value_heads != 0:
            continue
        if any(d % n != 0 for d in constraints.divides):
            continue
        out.append(n)
    return out


def max_tracks_for_config(config) -> int:
    """Return the largest valid N under the KV-replicated rule."""
    cs = _constraints_from_config(config)
    cands = _candidates_in_range(cs)
    if not cands:
        raise ValueError(
            f"No valid track count for constraints {cs!r}. "
            f"Required: N | num_attention_heads, N multiple of num_key_value_heads, "
            f"N divides {cs.divides}."
        )
    return cands[0]


def valid_track_counts(config) -> list[int]:
    """All valid Ns (in descending order). Useful for CLI / ablation menus."""
    return _candidates_in_range(_constraints_from_config(config))


def _constraints_from_config(config) -> ConstraintSet:
    model_type = getattr(config, "model_type", None)
    if model_type in _CONSTRAINT_PROVIDERS:
        return _CONSTRAINT_PROVIDERS[model_type](config)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        text_model_type = getattr(text_cfg, "model_type", None)
        if text_model_type in _CONSTRAINT_PROVIDERS:
            return _CONSTRAINT_PROVIDERS[text_model_type](text_cfg)
    raise NotImplementedError(
        f"No constraint provider registered for model_type={model_type!r}. "
        f"Register one via @register_constraints in parallm.utils.max_tracks."
    )


@register_constraints("qwen3_5_text")
def _qwen3_5_constraints(cfg) -> ConstraintSet:
    return ConstraintSet(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        divides=(
            int(cfg.linear_num_key_heads),
            int(cfg.linear_num_value_heads),
            int(cfg.intermediate_size),
        ),
    )
