"""Per-track gpt-oss text model — the shared body lives in `tracks.base`.

This family divides `intermediate_size` (the per-expert width) on top of the common
attention sizing, padded to `slicer.gpt_oss.EXPERT_WIDTH_ALIGN`. Two deliberate
overrides of the shared base:

- **`attn_implementation="eager"`.** `GptOssPreTrainedModel._supports_sdpa` is False,
  and `transformers.integrations.sdpa_attention.sdpa_attention_forward` has no
  ``s_aux`` parameter — under SDPA the attention SINKS would be dropped silently and
  the model would run, producing wrong numbers with no error. Eager is the only
  backend here that is correct out of the box.
- **`_resolve_position_ids`.** The base implementation expands to the 4-way
  (text/temporal/height/width) form Qwen3.5's mrope wants; gpt-oss rotary takes plain
  2-D position ids.

**Merged tracks (`fuse_size` > 1) are supported and are the fast path.** Unlike the
track-major stacking refuted for the 35B MoE in 2026-07-20, concatenating F shards
widens each EXPERT and leaves the expert COUNT alone — `experts.gate_up_proj` is
`Colwise(dim=2)` and `experts.down_proj` is `Rowwise(dim=1)` — so one merged
`grouped_mm` issues E group-dispatches at F times the width where the looped path
issues F*E at 1/F the width. It also collapses F redundant `router` evaluations into
one. The interleaved gate/up lanes survive because each member's slab has even width,
so `cat` keeps whole pairs and `[..., ::2]`/`[..., 1::2]` stay member-major on both
sides of the elementwise product.
"""
from __future__ import annotations

import torch

from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)

from parallm.model.tracks.base import PTTrackTextModelBase, apply_common_per_track_sizing
from parallm.slicer.base import align_chunk
from parallm.slicer.gpt_oss import EXPERT_WIDTH_ALIGN


class PTTrackTextModel(PTTrackTextModelBase):
    DECODER_LAYER_CLS = GptOssDecoderLayer
    RMSNORM_CLS = GptOssRMSNorm
    ROTARY_CLS = GptOssRotaryEmbedding

    def _resolve_position_ids(self, inputs_embeds, position_ids):
        """Plain 2-D position ids, used by both the rotary and the masks."""
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        return position_ids, position_ids


def build_per_track_text_config(text_config, n_tracks: int, fuse_size: int = 1):
    cfg = apply_common_per_track_sizing(
        text_config, n_tracks, fuse_size, attn_implementation="eager"
    )
    inter = int(cfg.intermediate_size)
    if inter % n_tracks != 0:
        # Zero-padding IS exact for the gate ((up+1)*glu(0) = 0) and the slicer does
        # pad, but the interleaved gate/up layout only keeps whole pairs when the
        # gate_up chunk is exactly twice the down chunk — which needs N | I. Refuse
        # rather than silently mis-lane the slab.
        raise ValueError(f"intermediate_size {inter} not divisible by {n_tracks}")
    # MUST match the slicer's `_even_chunk` or the shards won't load: the per-track
    # expert width is padded up to a multiple of 8 so `torch._grouped_mm` accepts the
    # contracted dim's stride (see `slicer.gpt_oss.EXPERT_WIDTH_ALIGN`). 2880/16 = 180
    # becomes 184; the 4 extra lanes are zeros and contribute exactly 0.0.
    #
    # `align_chunk(...) * fuse_size`, NOT `align_chunk(... * fuse_size)`: a merged
    # track is F ALIGNED per-track slabs concatenated, so its width must be F times
    # the width the shards on disk actually have. Aligning after multiplying would
    # give a slab the shards cannot fill (and would shift members 1..F-1 off their
    # gate/up lane boundaries).
    cfg.intermediate_size = align_chunk(inter // n_tracks, EXPERT_WIDTH_ALIGN) * fuse_size
    # The reference `GptOssExperts.forward` is a Python loop over every hit expert
    # (all 32/128 of them at training seq lengths). `grouped_mm` sorts tokens by expert
    # and does one grouped GEMM per expert: same math (it honours the class's own
    # `_apply_gate`, so the swiglu clamp is preserved — exact in fp32, relL2 ~3e-3 in
    # bf16 from the different accumulation order) and 4.7x faster than the loop
    # (measured 3.6 vs 16.9 ms/MLP call, 32 experts at hidden 2880, T=2048).
    # `batched_mm` was measured too and is WORSE than the loop (40 ms) and OOMs at
    # wider slabs — do not re-try it.
    cfg._experts_implementation = "grouped_mm"
    return cfg
