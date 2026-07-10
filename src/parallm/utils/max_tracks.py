"""Detect the maximum natural track count for a given model config.

Track-count rule (KV-replicated max-parallelism):

    max_tracks is the largest N satisfying ALL of:
      1. num_attention_heads % N == 0      (each track gets >=1 q-head)
      2. N % num_key_value_heads == 0      (kv-groups split evenly across tracks,
                                            with replication within a group)
      3. linear_num_key_heads % N == 0
         and linear_num_value_heads % N == 0
      4. intermediate_size % N == 0

This is model-agnostic: each family's `ModelAdapter` supplies a `constraints`
callback returning a `ConstraintSet`, and we scan N downward from
num_attention_heads to the first N that satisfies every constraint.
"""
from __future__ import annotations

from dataclasses import dataclass


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
    # The adapter registry is the single source of model knowledge: resolve the
    # right adapter (by config.model_type, else text_config.model_type) and ask it
    # for its `constraints`. Imported lazily to avoid an import cycle
    # (`parallm.adapters` imports this module for `ConstraintSet`).
    from parallm.adapters import get_adapter_for_config

    try:
        adapter = get_adapter_for_config(config)
    except KeyError as e:
        raise NotImplementedError(str(e)) from e
    if adapter.constraints is None:
        raise NotImplementedError(
            f"adapter {adapter.model_type!r} has no `constraints` for the max-tracks scan"
        )
    # Every adapter callback operates on the *text config*.
    text_cfg = config if getattr(config, "model_type", None) == adapter.model_type else config.text_config
    return adapter.constraints(text_cfg)
