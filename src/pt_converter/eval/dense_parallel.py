"""Window-parallel forwards for the DENSE Qwen3.5 model — the Lever-1 Gate-0
probe (no tracks, no training).

Question being gated: can the serial teacher's function be re-expressed as
"window-parallel + one post-attn sync per window" — the geometry an N-track
slice computes EXACTLY with one all-reduce per window — without (or before)
healing? Each D-window {a, b, …} entered at stream state ``W`` becomes

    y_j = attn_j(W)            for every layer j in the window (parallel)
    A   = W + Σ_j y_j          ← the ONE per-window sync point
    m_j = mlp_j(A)             for every layer j (parallel, reads REAL post-attn state)
    W'  = A + Σ_j m_j

The N=16 slice reproduces this with one sync event per window; its only
deviation is the next window's attention reading ``A + own-track m`` instead of
``A + Σm`` — the same 1-phase seam that was cheap at D=1 lever B. The dense
arms therefore bracket the slice:

* ``serial``                — stock forward (rail; must equal the teacher).
* ``parallel-attn``         — the target function above (exact seam: attn reads ``W``).
* ``parallel-attn-dropseam``— attn reads the PREVIOUS post-attn state ``A_prev``
                              (misses every m added since = worst-case seam; the
                              slice lands between this and ``parallel-attn``).
* ``parallel-full``         — PaLM-style: MLPs also read ``W`` (no post-attn sync
                              needed at all ⇒ no seam), measures what the phased
                              sync buys inside a window.

Windows are an ordered partition of the layer stack. ``serial`` mode ignores
window structure. In the parallel modes a size-1 window reduces algebraically
to the stock serial layer (attn on ``W``, MLP on ``W + y``) — the unit-test rail.
Evaluated under lm-harness loglikelihood (full-sequence scoring); no KV cache.
"""
from __future__ import annotations

from typing import Sequence

import torch

from pt_converter.model.cross_head_estimator import seam_mlp, seam_token_mixer

MODES = ("serial", "parallel-attn", "parallel-attn-dropseam", "parallel-full")


def build_windows(
    num_layers: int, group_size: int, first_parallel_layer: int = 1, last_solo: bool = True
) -> "list[list[int]]":
    """Ordered window partition: singleton windows for the first
    ``first_parallel_layer`` layers and (optionally) the final layer — those are
    real boundaries in the deployed schedule — and ``group_size`` chunks in
    between (the tail chunk may be short)."""
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    windows = [[i] for i in range(first_parallel_layer)]
    stop = num_layers - 1 if last_solo else num_layers
    body = list(range(first_parallel_layer, stop))
    windows += [body[i : i + group_size] for i in range(0, len(body), group_size)]
    if last_solo:
        windows.append([num_layers - 1])
    return [w for w in windows if w]


def _validate_windows(windows: Sequence[Sequence[int]], num_layers: int) -> None:
    flat = [i for w in windows for i in w]
    if flat != list(range(num_layers)):
        raise ValueError(
            f"windows must partition range({num_layers}) in order, got {list(map(list, windows))}"
        )


class WindowScaffold:
    """Per-batch forward scaffolding (masks, rope, position ids) shared by every
    window step. Carries no gradient; built once per (input_ids, attention_mask)."""

    def __init__(self, text_model, emb: torch.Tensor, attention_mask: "torch.Tensor | None"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask

        self.text_model = text_model
        # Stock-forward position handling: cache_position = arange (padding is
        # handled by the attention mask, exactly as when the teacher was scored).
        self.position_ids = torch.arange(emb.shape[1], device=emb.device).unsqueeze(0)
        self.causal_mask = create_causal_mask(
            config=text_model.config, inputs_embeds=emb, attention_mask=attention_mask,
            past_key_values=None, position_ids=self.position_ids,
        )
        self.linear_attn_mask = (
            None if (attention_mask is not None and torch.all(attention_mask == 1)) else attention_mask
        )
        self.position_embeddings = text_model.rotary_emb(emb, self.position_ids)

    def _mask(self, i: int):
        return (
            self.linear_attn_mask
            if self.text_model.config.layer_types[i] == "linear_attention"
            else self.causal_mask
        )

    def mixer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return seam_token_mixer(
            self.text_model.layers[i], x, self.position_embeddings, self._mask(i), self.position_ids
        )[1]


def window_step(
    scaf: WindowScaffold,
    window: Sequence[int],
    stream: torch.Tensor,
    attn_base: torch.Tensor,
    mode: str,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """One window of the rewired forward: ``(stream, attn_base) → (stream',
    attn_base')``. ``stream`` = exact state at window entry (W); ``attn_base`` =
    most recent post-attn synced state (A_prev, read only by dropseam)."""
    layers = scaf.text_model.layers
    if mode == "serial":
        for i in window:
            attn_base = stream + scaf.mixer(i, stream)
            stream = seam_mlp(layers[i], attn_base, None)
        return stream, attn_base
    attn_in = attn_base if mode == "parallel-attn-dropseam" else stream
    A = stream + sum(scaf.mixer(i, attn_in) for i in window)
    mlp_in = stream if mode == "parallel-full" else A
    stream = A + sum(seam_mlp(layers[i], mlp_in, None) - mlp_in for i in window)
    return stream, A


def dense_window_forward(
    text_model,
    input_ids: torch.Tensor,
    attention_mask: "torch.Tensor | None",
    *,
    windows: Sequence[Sequence[int]],
    mode: str,
) -> torch.Tensor:
    """Run the dense text model with the window-parallel rewiring. Returns the
    post-final-norm hidden ``(B, T, H)`` (pre-``lm_head``).

    ``text_model`` is a stock HF ``Qwen3_5TextModel`` (dense; NOT a PT track).
    Scaffolding (masks, rope, per-layer mask routing) mirrors ``eval/refine.py``;
    the block math goes through the shared ``seam_token_mixer`` / ``seam_mlp``
    helpers, which reproduce ``Qwen3_5DecoderLayer`` exactly. Grad-capable (the
    healing trainer backwards through it); eval callers run under no_grad.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    _validate_windows(windows, len(text_model.layers))
    emb = text_model.embed_tokens(input_ids)
    scaf = WindowScaffold(text_model, emb, attention_mask)
    stream, attn_base = emb, emb
    for window in windows:
        stream, attn_base = window_step(scaf, window, stream, attn_base, mode)
    return text_model.norm(stream)
