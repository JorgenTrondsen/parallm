"""Pytest config: force the pure-torch gated-delta path for CPU tests.

The optional `flash-linear-attention` (FLA) extra provides fused Triton kernels
that the Qwen3.5 modeling code auto-uses when importable (`chunk_gated_delta_rule`,
`FusedRMSNormGated`, ...). Those kernels are GPU/Triton-only, but the test suite
builds tiny models and runs forwards on CPU. When FLA is installed in the same
venv (as it is for fast GPU training), the CPU forwards would fail with
"Pointer argument ... cannot be accessed from Triton (cpu tensor?)".

This autouse fixture monkeypatches the Qwen3.5 modeling module's FLA hooks back
to None before each test, so `Qwen3_5GatedDeltaNet.__init__` selects the
pure-torch fallback (`torch_chunk_gated_delta_rule`, `Qwen3_5RMSNormGated`,
`F.conv1d`). It only affects pytest; real training imports the module fresh and
uses FLA on GPU. The PT architecture correctness these tests gate (slicing,
sync, N=1 dense parity) is independent of which gated-delta kernel runs.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_torch_gated_delta(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m_moe

    for mod in (m, m_moe):
        for name in (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "FusedRMSNormGated",
        ):
            monkeypatch.setattr(mod, name, None, raising=False)
        monkeypatch.setattr(mod, "is_fast_path_available", False, raising=False)
