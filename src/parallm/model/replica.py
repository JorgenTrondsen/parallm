"""Cheap cross-track replicas — the deployable sparse-copy core.

Between sync boundaries a track has no access to the other tracks' residual
updates. The one estimator that recovers that content cheaply (and comm-free) is
**degraded local recomputation**: each rank re-derives the other tracks'
between-sync trajectory by replaying their sublayers from the last synced
residual through *cheap copies* of their weights. The copies that hold quality
are **activation-aware (Wanda) sparse** ones, optionally over an int-quantized
base (``qwanda``): score each weight ``|w_ij|·‖x_j‖`` with one dense calibration
pass and keep the top ``1−frac`` per output row (survivors exact). At N=16 this
holds ~90–99% of teacher downstream quality down to 4 sync events (D=8),
training-free, and composes with an already-quantized base.

This module is the deployable extract of that result (the probe harness that
mapped every refuted alternative lived in the old ``eval/intervention.py``):

- ``collect_input_norms`` — the one-pass Wanda calibration on the dense model.
- ``wanda_prune_weight`` / ``fake_quant_weight`` / ``block_wanda_prune_weight`` —
  the per-weight copy transforms.
- ``degrade_track_layers`` — build the degraded decoder-layer copies for one
  track (what the inference engine hosts as its replica pool).
"""
from __future__ import annotations

import copy

import torch


def fake_quant_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric absmax fake-quantization (fp32 compute,
    returned in ``w``'s dtype) — simulates an int-``bits`` weight copy."""
    qmax = float(2 ** (bits - 1) - 1)
    wf = w.float()
    flat = wf.reshape(wf.shape[0], -1)
    scale = (flat.abs().amax(dim=1) / qmax).clamp(min=1e-12)
    q = torch.round(flat / scale[:, None]).clamp(-qmax - 1.0, qmax)
    return (q * scale[:, None]).reshape_as(wf).to(w.dtype)


def wanda_prune_weight(w: torch.Tensor, frac: float, in_norms: torch.Tensor) -> torch.Tensor:
    """Activation-aware (Wanda-style) pruning per output row: score each weight
    ``|w_ij| * ||x_j||`` with calibration input norms and zero the ``frac``
    lowest-scored entries per row. Survivors keep their EXACT values."""
    if not (0.0 < frac < 1.0):
        raise ValueError(f"wanda frac must be in (0, 1), got {frac}")
    if in_norms.numel() != w.shape[-1]:
        raise ValueError(f"in_norms has {in_norms.numel()} entries for in_features {w.shape[-1]}")
    wf = w.float()
    score = wf.abs() * in_norms.to(device=wf.device, dtype=torch.float32)[None, :]
    k = int(round(frac * score.shape[1]))
    if k > 0:
        thresh = score.kthvalue(k, dim=1, keepdim=True).values
        wf = wf * (score > thresh)
    return wf.to(w.dtype)


def block_wanda_prune_weight(w: torch.Tensor, frac: float, in_norms: torch.Tensor,
                             block_size: int) -> torch.Tensor:
    """Wanda pruning at BLOCK granularity: score blocks of ``block_size``
    consecutive input weights per output row by their summed ``|w|·‖x‖`` and
    zero the ``frac`` lowest-scoring blocks. Index cost drops from ~1 bit/weight
    (unstructured bitmap) to ~1/block_size — the cheap-index variant."""
    if not (0.0 < frac < 1.0):
        raise ValueError(f"blockwanda frac must be in (0, 1), got {frac}")
    d = w.shape[-1]
    if d % block_size != 0:
        raise ValueError(f"in_features {d} not divisible by block_size {block_size}")
    if in_norms.numel() != d:
        raise ValueError(f"in_norms has {in_norms.numel()} entries for in_features {d}")
    wf = w.float()
    scored = (wf.abs() * in_norms.to(device=wf.device, dtype=torch.float32)[None, :])
    blocks = scored.reshape(wf.shape[0], d // block_size, block_size).sum(-1)
    k = int(round(frac * blocks.shape[1]))
    if k > 0:
        thresh = blocks.kthvalue(k, dim=1, keepdim=True).values
        keep = (blocks > thresh).unsqueeze(-1).expand(-1, -1, block_size)
        wf = wf * keep.reshape(wf.shape[0], d)
    return wf.to(w.dtype)


# One capture module per DISTINCT Linear input space in a dense decoder layer;
# siblings (k/v ≡ q; z/b/a ≡ qkv; up ≡ gate) reuse it via _WANDA_KEY.
_WANDA_CAPTURE = (
    "self_attn.q_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.down_proj",
)
_WANDA_KEY = {
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.q_proj",
    "self_attn.v_proj": "self_attn.q_proj",
    "self_attn.o_proj": "self_attn.o_proj",
    "linear_attn.in_proj_qkv": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_b": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_a": "linear_attn.in_proj_qkv",
    "linear_attn.out_proj": "linear_attn.out_proj",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.up_proj": "mlp.gate_proj",
    "mlp.down_proj": "mlp.down_proj",
}


def collect_input_norms(text_model, batches, device) -> "dict[str, torch.Tensor]":
    """Wanda calibration on the DENSE model: per-input-channel L2 norms for every
    distinct Linear input space, keyed ``'{layer_idx}.{capture_relpath}'``. The
    track slabs' input spaces are exactly the dense spaces (row-sliced weights
    read the full hidden; col-sliced weights read the track's contiguous slice
    of the dense intermediate — the conversion identity), so per-track norms are
    the dense vector or its slice (see ``norms_for``)."""
    acc: "dict[str, torch.Tensor]" = {}
    hooks = []

    def _mk(key: str):
        def hook(_mod, args):
            x = args[0]
            s = x.detach().float().pow(2).reshape(-1, x.shape[-1]).sum(0)
            acc[key] = s if key not in acc else acc[key] + s
        return hook

    for li, layer in enumerate(text_model.layers):
        for rel in _WANDA_CAPTURE:
            try:
                mod = layer.get_submodule(rel)
            except AttributeError:
                continue
            hooks.append(mod.register_forward_pre_hook(_mk(f"{li}.{rel}")))
    try:
        with torch.no_grad():
            for b in batches:
                ids = b["input_ids"].to(device)
                mask = b.get("attention_mask")
                text_model(input_ids=ids,
                           attention_mask=mask.to(device) if mask is not None else None)
    finally:
        for h in hooks:
            h.remove()
    return {k: v.sqrt().cpu() for k, v in acc.items()}


def norms_for(input_norms: "dict[str, torch.Tensor]", n_tracks: int,
              layer_idx: int, rel_name: str, in_features: int, track_id: int) -> torch.Tensor:
    """The per-track input-norm vector for one Linear. Row-sliced Linears read the
    full dense input space (norms match ``in_features`` directly); col-sliced
    Linears read the track's contiguous slice of the dense intermediate (the dense
    vector tiles ``n_tracks`` × ``in_features``, so take this track's slice)."""
    v = input_norms[f"{layer_idx}.{_WANDA_KEY[rel_name]}"]
    if v.numel() != in_features:
        slices, rem = divmod(v.numel(), in_features)
        if rem != 0 or slices != n_tracks:
            raise ValueError(
                f"norms ({v.numel()}) do not tile in_features {in_features} "
                f"× n_tracks {n_tracks} for {layer_idx}.{rel_name}"
            )
        v = v[track_id * in_features:(track_id + 1) * in_features]
    return v


def degrade_track_layers(track_text_model, input_norms: "dict[str, torch.Tensor]",
                         n_tracks: int, track_id: int, frac: float, *,
                         bits: "int | None" = None, block: "int | None" = None) -> "list":
    """Deep-copy a track's decoder layers and replace every ``nn.Linear`` weight
    with its cheap wanda copy — the deployable replica pool for one track.

    ``bits`` set ⇒ ``qwanda``: quantize the copy to int-``bits`` (per-output-row
    absmax) BEFORE pruning, so the copy composes with an already-quantized base.
    ``block`` set ⇒ block-granular index (``block_wanda_prune_weight``). Norms /
    biases / conv weights stay exact; only Linears are degraded. Parameters are
    detached and frozen (a replica is never trained)."""
    shadow = []
    for li, layer in enumerate(track_text_model.layers):
        clone = copy.deepcopy(layer)
        for rel, mod in clone.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            w = mod.weight.data
            if bits is not None:
                w = fake_quant_weight(w, bits)
            norms = norms_for(input_norms, n_tracks, li, rel, w.shape[-1], track_id)
            if block is not None:
                mod.weight.data = block_wanda_prune_weight(w, frac, norms, block)
            else:
                mod.weight.data = wanda_prune_weight(w, frac, norms)
        for prm in clone.parameters():
            prm.requires_grad_(False)
        shadow.append(clone)
    return shadow
