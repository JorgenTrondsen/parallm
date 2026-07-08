"""Rails for the block draft-verify greedy decode (``eval/draft_verify.py``).

* EXACTNESS: whatever the draft proposes, the emitted sequence is bit-exact
  the verifier's own greedy decode (the property the premise call rests on).
* Self-draft: draft == verifier ⇒ every proposal accepted, τ == k+1.
* Accept arithmetic on crafted logits (no model): agreeing prefix + bonus.
"""
from __future__ import annotations

import torch

torch.set_default_dtype(torch.float32)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from pt_converter.eval.draft_verify import (
    draft_verify_generate,
    greedy_verify_step_factory,
    naive_greedy_propose_factory,
)


def _tiny_config():
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
    )


def _build(seed: int) -> Qwen3_5ForCausalLM:
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(cfg).eval()


def _greedy_decode(model, prompt: torch.Tensor, n: int) -> torch.Tensor:
    """Reference: plain greedy decode by full re-forward."""
    ids = prompt
    with torch.no_grad():
        for _ in range(n):
            tok = model(input_ids=ids).logits[:, -1].argmax(-1)
            ids = torch.cat([ids, tok.view(1, 1)], dim=1)
    return ids


def test_emitted_is_verifier_greedy_decode():
    verifier = _build(13)
    draft = _build(99)  # different weights — proposals will disagree often
    prompt = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))

    propose = naive_greedy_propose_factory(draft)
    verify = greedy_verify_step_factory(lambda ids: verifier(input_ids=ids).logits)
    seq, tokens, events = draft_verify_generate(
        propose, verify, prompt, k=4, max_new=24
    )
    assert tokens >= 24
    ref = _greedy_decode(verifier, prompt, tokens)
    assert torch.equal(seq, ref), "draft-verify output diverged from verifier greedy decode"
    # τ must be in [1, k+1]; with a disagreeing draft it is strictly < k+1.
    assert 1.0 <= tokens / events <= 5.0


def test_self_draft_accepts_everything():
    model = _build(13)
    prompt = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))

    propose = naive_greedy_propose_factory(model)
    verify = greedy_verify_step_factory(lambda ids: model(input_ids=ids).logits)
    _seq, tokens, events = draft_verify_generate(
        propose, verify, prompt, k=4, max_new=20
    )
    assert tokens == events * 5, "draft == verifier must accept all k and emit k+1 per event"


def test_accept_arithmetic_crafted_logits():
    vocab, p_len, k = 16, 6, 4
    # Verifier greedy targets at joint positions p_len-1 .. p_len+k-1:
    targets = torch.tensor([3, 7, 2, 9, 11])

    def fake_logits(ids: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(1, ids.shape[1], vocab)
        for i, t in enumerate(targets):
            out[0, p_len - 1 + i, t] = 10.0
        return out

    verify = greedy_verify_step_factory(fake_logits)
    seq = torch.zeros(1, p_len, dtype=torch.long)

    # Proposals agree on the first 2 slots, diverge on the 3rd.
    emitted = verify(seq, torch.tensor([3, 7, 5, 9]))
    assert emitted.tolist() == [[3, 7, 2]]  # 2 accepted + correction token

    # Full agreement ⇒ k accepted + bonus.
    emitted = verify(seq, torch.tensor([3, 7, 2, 9]))
    assert emitted.tolist() == [[3, 7, 2, 9, 11]]

    # Immediate disagreement ⇒ single corrected token (the plain-decode floor).
    emitted = verify(seq, torch.tensor([4, 7, 2, 9]))
    assert emitted.tolist() == [[3]]
