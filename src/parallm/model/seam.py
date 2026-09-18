"""Decoder-block seam: split a HF decoder layer at the post-attention point.

The post-attn (lever B) and exact sync schedules place a boundary INSIDE the
layer — after the token mixer's residual add, before the MLP — so the walk
must drive the two halves separately. Canonical home shared by the training
forward (`pt_model._run_stack`) and the distill TF block loop.
Ported from the pre-parallm-pivot `cross_head_estimator` module (the seam
split itself was rail-validated there; the estimator machinery was not).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import NamedTuple

import torch


class FoldRead(NamedTuple):
    """What an MLP reads at an attention-replica layer whose ``o_proj`` is FOLDED into
    this track's own gate/up slices (`parallm.model.attn_replica`, spec ``o:fold``).

    The plain read is one tensor, ``R + Â`` with ``Â = W_o a``. RMSNorm is a per-token
    scalar times ``γ``, so ``gate_t(norm(R + W_o a)) = [gate_t(γ⊙R) + (gate_t·diag(γ)·W_o) a]
    / s`` — the folded ``gate``/``up`` are ``[I/N, q_dim]`` per track and ``W_o`` is never
    stored. ``inv_s = 1/s`` is the one place ``Â`` still enters, from a sketch of it.

    ``gate``/``up`` are stacked ``[K, I/N, q_dim]`` over the tracks this rank walks;
    `track` picks one.
    """

    R: torch.Tensor       # the shared residual entering the layer, [B, T, H]
    a: torch.Tensor       # the replica's concatenated head outputs, [B, T, q_dim]
    inv_s: torch.Tensor   # 1/rms(R + Â_sketch) per token, fp32 [B, T, 1]
    gate: torch.Tensor
    up: torch.Tensor

    def track(self, k: int) -> "FoldRead":
        return self._replace(gate=self.gate[k], up=self.up[k])


def fold_mlp(mlp, norm, fr: FoldRead) -> torch.Tensor:
    """One track's MLP delta at a folded read: ``down(act(g)·u)`` with
    ``g = (gate(γ⊙R) + a·Gᵀ)/s``. The scalar multiply runs in fp32 and rounds once to the
    model dtype, as the norm's own ``hs * rsqrt(var)`` does."""
    dt = fr.R.dtype
    xr = norm.weight * fr.R
    g = mlp.gate_proj(xr).float() + torch.matmul(fr.a, fr.gate.transpose(-2, -1)).float()
    u = mlp.up_proj(xr).float() + torch.matmul(fr.a, fr.up.transpose(-2, -1)).float()
    g, u = (g * fr.inv_s).to(dt), (u * fr.inv_s).to(dt)
    return mlp.down_proj(mlp.act_fn(g) * u)


def seam_token_mixer(layer, x, position_embeddings, attention_mask, position_ids):
    """First half of a pre-norm decoder layer: ``input_layernorm`` → token mixer →
    residual add. Returns ``h_attn = x + Y`` where ``Y`` is the per-track mixer
    output (self-attn or gated-delta).

    ``block_type`` is Qwen3.5's hybrid marker; a family with only self-attention
    (gpt-oss) has no such attribute, hence the ``getattr`` — same guard
    `train.teacher.HookedTeacher._attn_submodule` uses.
    """
    h_ln = layer.input_layernorm(x)
    if getattr(layer, "block_type", None) == "linear_attention":
        y = layer.linear_attn(
            hidden_states=h_ln, cache_params=None, attention_mask=attention_mask
        )
    else:
        y, _ = layer.self_attn(
            hidden_states=h_ln,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
    return x + y


def seam_mlp(layer, h_attn: torch.Tensor, x_read: "torch.Tensor | None" = None) -> torch.Tensor:
    """Second half: ``post_attention_layernorm`` → ``mlp`` → residual add.

    A sparse MoE block may hand back ``(hidden_states, router_scores)`` (gpt-oss)
    rather than a bare tensor; only the hidden state joins the residual.

    ``x_read``: the tensor the MLP READS, when it differs from the residual its delta
    is added to — the attention replica's shared ``R + Â``. The carry does NOT hold it,
    so the post-MLP sync still sums the exact per-track deltas. ``None`` is the plain
    seam, bit for bit. A `FoldRead` is the replica's folded read (`fold_mlp`); the three
    are separate dynamo specializations, not a graph break.
    """
    if isinstance(x_read, FoldRead):
        y = fold_mlp(layer.mlp, layer.post_attention_layernorm, x_read)
    else:
        y = layer.mlp(layer.post_attention_layernorm(h_attn if x_read is None else x_read))
    return h_attn + (y[0] if isinstance(y, tuple) else y)


_COMPILED: dict[str, object] = {}


def enable_seam_compile(
    mode: str = "both", dynamic: bool | None = False, inductor_mode: str = "default"
) -> None:
    """Compile the two seam halves ONCE, for every caller.

    Compiled here rather than on the layer modules for three reasons: the walks call
    ``layer.self_attn`` / ``layer.mlp`` directly and never ``layer.forward``, so a
    module-level compile would not engage; ``torch.compile(module)`` renames
    state_dict keys to ``_orig_mod.*`` and breaks checkpoint I/O; and a free function
    is compiled once instead of once per layer instance.

    ``mode``: ``"mixer"`` compiles only the token-mixer half. A sparse-MoE MLP sorts
    tokens by expert, which is data-dependent and may not hold a graph — so the MLP
    half is separable, and a family whose MoE will not compile can still take the
    attention win.

    The profiled step is ~75% BACKWARD, which is the point: compiling the forward
    gives inductor the backward too, so this is the only built lever that touches the
    dominant phase.

    Compiles the BATCHED fold's halves as well (`model.batched.enable_batched_compile`,
    imported here rather than at module scope because that module imports the engine).
    Which of the two representations runs is decided by ``exec_groups``, never by the
    caller, so a flag that reached only these two functions compiled NOTHING for any
    family running the batched path — which is every family with
    ``supports_batched_exec``, at every F.
    """
    if mode not in ("mixer", "mlp", "both"):
        raise ValueError(f"compile mode must be mixer|mlp|both, got {mode!r}")
    if inductor_mode == "reduce-overhead":
        raise ValueError(
            "reduce-overhead (CUDA graphs) is not supported for the track walk: a "
            "graph's output pool is reused by the next per-layer invocation while the "
            "backward still needs those tensors."
        )
    kw = {"dynamic": dynamic}
    if inductor_mode != "default":
        kw["mode"] = inductor_mode
    if mode in ("mixer", "both"):
        _COMPILED["mixer"] = torch.compile(seam_token_mixer, **kw)
    if mode in ("mlp", "both"):
        _COMPILED["mlp"] = torch.compile(seam_mlp, **kw)

    from parallm.model.batched import enable_batched_compile

    enable_batched_compile(mode, dynamic=dynamic, inductor_mode=inductor_mode)


@contextmanager
def eager_for_eval():
    """Run BOTH track representations eager for the duration of an eval pass.

    Clearing the registries is enough — each walk resolves its compiled callable per
    `checkpointed_halves` / `batched_halves` call — and the training graphs survive,
    because `torch.compile` caches on the function object, which is handed straight
    back afterwards.

    ⚠ **This must cover the batched fold, not just the seam.** The compiled units are
    built with ``dynamic=False``, so every distinct shape is its own dynamo cache
    entry; an lm-eval pass arrives with a different batch size and a fresh sequence
    length per batch, and burns through ``recompile_limit`` (8) within the FIRST eval.
    Once that limit is hit dynamo marks the code object and runs it EAGER for the rest
    of the process — so a single missed registry silently disables ``--compile`` for
    the whole remainder of training, not just for the eval. Measured at 32B/N=64: 12
    training steps ran clean, and all 16 warnings (2 fold halves x 8 ranks) fired
    during the first eval.

    Pairs with `enable_seam_compile`, which likewise has to reach both registries —
    the same registry going missing there is the bug that made ``--compile`` a silent
    no-op on every ``supports_batched_exec`` family.
    """
    from parallm.model import batched

    saved = (dict(_COMPILED), dict(batched._COMPILED))
    _COMPILED.clear()
    batched._COMPILED.clear()
    try:
        yield
    finally:
        _COMPILED.update(saved[0])
        batched._COMPILED.update(saved[1])


def _mixer_fn():
    return _COMPILED.get("mixer", seam_token_mixer)


def _mlp_fn():
    return _COMPILED.get("mlp", seam_mlp)


def checkpointed_halves(use_ckpt: bool, position_embeddings, position_ids):
    """``(run_mixer, run_mlp)`` bound to this forward's no-grad scaffolding.

    ``use_ckpt`` wraps each half in an activation checkpoint, so the backward
    holds one recomputed sublayer instead of a whole own-carry window. The
    scaffolding is captured rather than passed through ``checkpoint``, which
    keeps the residual input the only checkpointed tensor. Shared by the model's
    lever-B walk and the distill TF block loop.
    """
    mixer_fn, mlp_fn = _mixer_fn(), _mlp_fn()

    def _mixer(layer, x, mask):
        return mixer_fn(layer, x, position_embeddings, mask, position_ids)

    if not use_ckpt:
        return _mixer, mlp_fn

    from torch.utils.checkpoint import checkpoint

    return (
        lambda layer, x, mask: checkpoint(_mixer, layer, x, mask, use_reentrant=False),
        lambda layer, x, x_read=None: checkpoint(
            mlp_fn, layer, x, x_read, use_reentrant=False),
    )
