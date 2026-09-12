"""Layer-list and sync-phase spec parsing.

One implementation of "which layers", because five CLI flags across four scripts
parse a layer list and `--sync-phase` now parses a per-layer schedule on top of it.
"""

from __future__ import annotations

# The three uniform sync placements. `PTWrappedModel.sync_sets` turns a phase into
# the two per-sublayer sets; these names are the three schedules that are uniform
# over the whole stack, and a `layers:phase` spec is how a non-uniform one is said.
PHASES = ("post-attn", "post-mlp", "exact")


def parse_layers(spec: str) -> list[int]:
    """``"32-62"`` or ``"32,40,43"`` -> a layer list.

    Ranges are INCLUSIVE of both ends, matching how the back band is named everywhere
    in this program (L32-62 is 31 layers, and every driver, artifact and log says so).
    One implementation because five flags parse it — a half-open copy would silently
    drop L62 from one arm and nothing would raise.
    """
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def format_layers(layers) -> str:
    """`parse_layers`' inverse: ``{0, 1, 2, 32}`` -> ``"0-2,32"`` (``"-"`` if empty)."""
    runs: list[list[int]] = []
    for i in sorted(set(layers)):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs) or "-"


def resolve_phases(spec: str, L: int) -> list[str]:
    """A sync-phase spec -> the phase of each of ``L`` layers.

    ``spec`` is either a bare phase name, which covers the whole stack::

        post-attn                       # what every run before this recorded

    or a comma list of ``layers:phase`` segments, applied IN ORDER (last write wins),
    so a leading bare phase is the default that later segments override::

        0-31:post-attn,32-63:post-mlp   # front half lever B, back half SPD
        exact,40-47:post-attn           # the isolation cell
        3,7,11:post-attn                # a segment's layers may themselves be a list

    Every layer must end up named — there is no implicit default. A spec that covers
    only part of the stack raises rather than silently scoring a different network
    than the one the flag describes (see `utils.checkpoint.train_meta_arg` for what
    that costs: a post-attn heal read back as post-mlp measured 0.553 vs its 0.700).
    """
    out: list[str | None] = [None] * L
    # Layer fragments seen since the last phase: a segment's layer list is itself
    # comma-separated, so "3,7,11:post-attn" arrives here as three parts and only the
    # last one carries the phase that governs all three.
    pending: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            layers_spec, _, phase = part.rpartition(":")
            targets = parse_layers(",".join(pending + [layers_spec]))
            pending = []
        elif part in PHASES:
            if pending:
                raise ValueError(
                    f"{spec!r}: layers {','.join(pending)!r} are followed by the bare "
                    f"phase {part!r}, which covers every layer and would drop them — "
                    f"give them their own 'layers:phase' segment"
                )
            phase, targets = part, range(L)
        else:
            pending.append(part)
            continue
        if phase not in PHASES:
            raise ValueError(
                f"{spec!r}: unknown sync phase {phase!r}, expected one of "
                f"{', '.join(PHASES)}"
            )
        for i in targets:
            if not 0 <= i < L:
                raise ValueError(f"{spec!r}: layer {i} is outside 0..{L - 1}")
            out[i] = phase
    if pending:
        trailing = ",".join(pending)
        # A trailing fragment is either a typo'd phase name or a layer list nobody
        # gave a phase to. Say which, or the message for "post-atn" reads as if the
        # user meant it as a layer spec.
        if any(c not in "0123456789-" for c in trailing):
            raise ValueError(
                f"{spec!r}: unknown sync phase {trailing!r}, expected one of "
                f"{', '.join(PHASES)}"
            )
        raise ValueError(
            f"{spec!r}: layers {trailing!r} name no phase (expected "
            f"'{trailing}:<phase>')"
        )
    missing = [i for i, p in enumerate(out) if p is None]
    if missing:
        raise ValueError(
            f"{spec!r} names no phase for layers {format_layers(missing)}; lead with a "
            f"bare phase to set a default (e.g. 'post-attn,{format_layers(missing)}:exact')"
        )
    return out
