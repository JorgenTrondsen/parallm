"""Dense expert evaluation — run every expert on every token, let routing zero the rest.

Sparse MoE dispatch is the right shape at production widths and the WRONG shape at max
tracks, which is the regime this program lives in. Slicing N ways divides the per-expert
width but not the expert COUNT, so at gpt-oss N=64 each of the 32 experts is 48 wide
against a hidden size of 2880. Three costs follow, and none of them is the arithmetic:

1. **`torch._grouped_mm` has no true grouped kernel below SM90 — it LOOPS.** On A100 one
   call issues E=32 separate cutlass GEMMs. Six calls per MLP (2 forward, 4 backward)
   is ~192 GEMM launches per MLP call, each one skinny enough to reach ~4% of peak.
2. **The gathers.** Sorting tokens by expert needs `hidden_states[perm]`, `bias[ids]`
   x2 and an inverse permutation, all at (S, hidden) = (8192, 2880) = 47 MB. Their
   backward is `indexing_backward_kernel`, measured at **39% of ALL GPU time** in a
   training step — the two bias gathers funnel 8192 rows into 32 slots, the worst
   duplicate count an atomics scatter-add can be handed.
3. **The graph breaks.** `sort` + `histc` + data-dependent offsets give `--compile` no
   single graph to fuse; dense is `fullgraph`-clean.

Dense pays E/top_k = 8x the FLOPs and buys all three back: two plain GEMMs, no sort, no
gather, no `indexing_backward`, no data-dependent shape. It also uses LESS memory — the
sparse path's (S, hidden) intermediates are 4x the dense (T, hidden) ones.

**Equivalent, not bit-equal.** Non-selected experts carry routing weight exactly 0.0, so
`gated * 0 = 0` exactly and their contribution vanishes; what differs is reduction order
(E exact zeros summed inside the down GEMM's K-reduction, instead of a separate top_k-term
sum). Verified in fp32 at 1e-7 on output and every parameter gradient — see
`tests/test_moe_dense.py`.

⚠ The win is width-dependent, not universal: the 8x FLOP bill is constant while the
sparse path's inefficiency shrinks as experts get wider. `DENSE_WIDTH_MAX` is the
measured crossover; `choose_experts_impl` applies it.
"""
from __future__ import annotations

import torch
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

#: Registered name, and the value `config._experts_implementation` takes.
IMPL_NAME = "parallm_dense"

#: Widest per-expert `intermediate_size` at which dense still beats `grouped_mm`,
#: measured fwd+bwd at hidden 2880 / 32 experts / top-k 4 / **2048 tokens** on A100.
#: See `scratchpad/moe_bwd_bench.py` for the sweep this comes from.
#:
#: ⚠ Training-shaped. A decode step is one token, where the sparse path's sort and
#: gathers cost the same launches over ~2048x less data — the crossover almost
#: certainly moves, and nobody has measured where. `PTWrappedModel` is shared with
#: the engine, so inference inherits this number; call `set_experts_policy` to
#: override rather than retuning the constant for a regime it was not measured in.
#:
#: Measured fwd+bwd ms, grouped_mm vs dense, at the widths the N ladder produces:
#:     48 -> 4.87x   96 -> 2.70x   144 -> 1.70x   184 -> 1.33x   192 -> 1.27x
#:    224 -> 1.10x  256 -> 1.00x   288 -> 0.91x   368 -> 0.75x
#: Break-even is 256, so 224 is the last rung that actually wins. This covers
#: N=64 at F=1/2/4 (48/96/192) and N=16 at F=1 (184); N=8 (360) stays sparse.
DENSE_WIDTH_MAX = 224

_POLICY: dict[str, str] = {"mode": "auto"}


def dense_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Signature-compatible with `transformers.integrations.moe.grouped_mm_experts_forward`."""
    E = self.num_experts
    T, H = hidden_states.size(0), hidden_states.size(-1)

    # (T, E) routing weights, zero everywhere routing did not pick. `scatter_add_`
    # rather than `scatter_`: under expert parallelism `top_k_index` carries sentinels
    # >= E whose weight is already 0, and clamping them collides with a real slot —
    # adding a zero there is harmless where overwriting a real weight would not be.
    #
    # Held at the PROMOTED dtype so a family with fp32 routing weights keeps them:
    # the sparse path multiplies in fp32 and casts once at the end, and rounding a
    # routing weight to bf16 up front would be a 4e-3 relative error, well above the
    # bf16 floor this change is allowed to sit inside.
    wdt = torch.promote_types(hidden_states.dtype, top_k_weights.dtype)
    w = torch.zeros(T, E, dtype=wdt, device=hidden_states.device).scatter_add_(
        1, top_k_index.clamp(max=E - 1).long(), top_k_weights.to(wdt)
    )

    # Flatten the WEIGHT to (H, E*2I) and use one plain GEMM. The obvious alternative —
    # broadcasting a bmm over the expert axis — needs no weight copy, but its backward
    # materializes grad (E, T, H) (377 MB at N=64) before the broadcast reduction.
    # Flattening pays one (H, E*2I) copy instead and hands back a (T, E, 2I) output whose
    # gated form feeds the down GEMM with no copy at all.
    up_w = self.gate_up_proj if self.has_gate else self.up_proj
    up_w = up_w if self.is_transposed else up_w.transpose(-2, -1)
    gu = hidden_states @ up_w.permute(1, 0, 2).reshape(H, -1)
    if self.has_bias:
        bias = self.gate_up_proj_bias if self.has_gate else self.up_proj_bias
        gu = gu + bias.reshape(-1)

    gu = gu.view(T, E, -1)
    gated = self._apply_gate(gu) if self.has_gate else self.act_fn(gu)  # (T, E, I)
    gated = (gated * w.unsqueeze(-1)).to(hidden_states.dtype)

    # sum_e w[t,e] * (gated[t,e] @ down[e]) is one GEMM over the flattened (E*I) axis.
    down_w = self.down_proj if self.is_transposed else self.down_proj.transpose(-2, -1)
    out = gated.reshape(T, -1) @ down_w.reshape(-1, H)
    if self.has_bias:
        # The sparse path adds this bias BEFORE the routing multiply, so the dense
        # equivalent is the weighted sum over experts, not a broadcast add.
        out = out + (w.to(hidden_states.dtype) @ self.down_proj_bias)
    return out.to(hidden_states.dtype)


ALL_EXPERTS_FUNCTIONS.register(IMPL_NAME, dense_experts_forward)


def set_experts_policy(mode: str) -> None:
    """``auto`` (default) applies `DENSE_WIDTH_MAX`; ``dense`` / ``grouped_mm`` force one.

    Module-level rather than threaded through `ModelAdapter.build_per_track_text_config`,
    whose signature every family shares — same pattern as `seam.enable_seam_compile`.
    Forcing exists so a step-time arm can be A/B'd against its own control.
    """
    if mode not in ("auto", "dense", "grouped_mm"):
        raise ValueError(f"experts policy must be auto|dense|grouped_mm, got {mode!r}")
    _POLICY["mode"] = mode


def choose_experts_impl(num_experts: int, intermediate_size: int) -> str:
    """The value to assign to `config._experts_implementation` for a per-track config."""
    if _POLICY["mode"] == "dense":
        return IMPL_NAME
    if _POLICY["mode"] == "grouped_mm":
        return "grouped_mm"
    return IMPL_NAME if intermediate_size <= DENSE_WIDTH_MAX else "grouped_mm"
