"""Price a local MLP read's channel selection in answer-letter KL, from a trace's attributions.

`scripts/trace_read_channels.py` writes, per item, ``p`` (the answer-letter distribution) and
``a [Y, layers, I]`` (the first-order change in ``log p_y`` if a read leaves channel ``c`` of a
layer out). Every config — which layers read, at what r, ranked how — is then a sum over
those, and its letter KL is the Fisher form ``½ Σ_y p_y Δlog p_y²``.
"""
from __future__ import annotations

import torch

# r = 0 is a post-attn layer with no read; 25600 is the whole MLP (an exact read).
LADDER = (0, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 25600)


def fisher_score(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``φ [layers, I] = mean_items Σ_y p_y a²`` (fp32 accumulation over items)."""
    out = torch.zeros(a.shape[2], a.shape[3], dtype=torch.float64)
    for n in range(a.shape[0]):
        out += (p[n].double()[:, None, None] * a[n].double().pow(2)).sum(0)
    return (out / a.shape[0]).float()


def missing_sums(a: torch.Tensor, orders: "list[torch.Tensor]", ladder=LADDER) -> torch.Tensor:
    """``M [items, Y, layers, len(ladder)]``: per layer, ``Σ`` of ``a`` over the channels a
    top-r read (that layer's order) leaves out."""
    N, Y, nL, _ = a.shape
    M = torch.empty(N, Y, nL, len(ladder), dtype=torch.float32)
    for k, order in enumerate(orders):
        cs = torch.cumsum(a[:, :, k, order].double(), dim=-1)           # [N, Y, I]
        total = cs[..., -1]
        for j, r in enumerate(ladder):
            M[:, :, k, j] = (total - (cs[..., r - 1] if r > 0 else 0.0)).float()
    return M


def config_kl(M: torch.Tensor, p: torch.Tensor, choice: "dict[int, int | None]",
              ladder=LADDER) -> torch.Tensor:
    """Per-item predicted letter KL of a config: ``choice[k]`` = the r of read layer k (an
    entry of ``ladder``), None for a layer that keeps its MLP sync (contributes nothing)."""
    d = torch.zeros(M.shape[0], M.shape[1], dtype=torch.float64)
    for k, r in choice.items():
        if r is not None:
            d -= M[:, :, k, ladder.index(r)].double()
    pd = p.double()
    d = d - (pd * d).sum(-1, keepdim=True)
    return 0.5 * (pd * d * d).sum(-1)
