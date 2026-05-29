"""Frozen dense teacher with hooks at the PT sync layer indices.

Loads the original Qwen3.5 dense text model, registers forward hooks on
the decoder layers at every sync boundary, and exposes a `forward()` that
returns (final_logits, {layer_idx: hidden_state}).

The teacher's hidden states must be captured *after* the layer's residual
addition completes (i.e., the layer's output). HF's `Qwen3_5DecoderLayer.forward`
returns the post-residual hidden state directly, so a forward hook on the
DecoderLayer module captures exactly what we want.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class HookedTeacher:
    """Wraps a loaded dense text decoder + lm_head and captures hidden states at sync points."""

    def __init__(
        self,
        text_model: nn.Module,
        lm_head: nn.Linear,
        sync_layer_indices: Iterable[int],
    ):
        self.text_model = text_model
        self.lm_head = lm_head
        self.sync_indices = list(sync_layer_indices)
        self._captures: dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._install_hooks()

    def _install_hooks(self):
        layers = list(self.text_model.layers)
        for idx in self.sync_indices:
            layer = layers[idx]

            def make_hook(layer_idx: int):
                def _hook(_module, _inputs, output):
                    # Qwen3_5DecoderLayer returns a Tensor (post-residual hidden state).
                    out = output if isinstance(output, torch.Tensor) else output[0]
                    self._captures[layer_idx] = out
                return _hook

            self._handles.append(layer.register_forward_hook(make_hook(idx)))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        # Install a fresh dict the hooks write into this forward.
        self._captures = {}
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        # Hand the caller SOLE ownership of the captures and drop our reference,
        # so the caller can free each hidden state (pop/del) as it is consumed
        # — otherwise the 8 (B,T,H) bf16 tensors stay resident for the whole
        # step (grows ×B, the headroom we want back for larger batch sizes).
        captures = self._captures
        self._captures = {}
        return logits, captures
