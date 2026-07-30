"""Rails for the two levers that put a 122 B model on a 40 GB card.

Rail 1 — streamed teacher ≡ resident teacher: paging the frozen decoder layers
in from host DRAM one at a time must not perturb a single bit of the teacher's
output. Runs the forward twice, since every layer is released at pass end.

Rail 2 — optimizer-in-backward ≡ step-after-backward: with a single backward
(``lambda_block=0`` leaves only the free-running KL pass), stepping each param
from its ``post_accumulate_grad_hook`` is the same update as stepping the whole
model afterwards, so the params must land bit-identical and every grad freed.
This gates the mechanism; the recipe change it enables (one update per objective
instead of one summed update) is gated separately by the 35B calibration run.
"""
from __future__ import annotations

import torch
from transformers.optimization import Adafactor

from parallm.model.pt_model import HostResidentLayers, PTWrappedModel
from parallm.train.distill import (
    DistillConfig,
    capture_sets,
    distill_step,
    freeze_slice_teacher,
    teacher_forward,
)
from tests.test_train_distill import _batch, _build

N_LAYERS = 8


def _teacher(cfg, tracks, n_tracks=4):
    tpt = PTWrappedModel(
        text_config=cfg, n_tracks=n_tracks, local_track_ids=tuple(range(n_tracks)),
        sync_after_layers=list(range(N_LAYERS)), track_group=None,
    )
    tpt.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    return freeze_slice_teacher(tpt)


def test_streamed_teacher_matches_resident():
    """Paging layers in from host memory must not perturb a single bit of the
    teacher's output — it is the same math on the same weights, just resident
    for a shorter window. Runs the forward TWICE, since every layer is released
    at the end of a pass and has to be re-acquired for the next one."""
    cfg, _dense, tracks, _pt = _build(n_tracks=4, sync_after=list(range(N_LAYERS)))
    batch = _batch(cfg)
    pa, pm = capture_sets(tuple(range(N_LAYERS)), N_LAYERS)

    resident = _teacher(cfg, tracks)
    h_ref, caps_ref = teacher_forward(resident, batch["input_ids"], batch["attention_mask"], pa, pm)

    model = PTWrappedModel(
        text_config=cfg, n_tracks=4, local_track_ids=tuple(range(4)),
        sync_after_layers=list(range(N_LAYERS)), track_group=None,
    )
    model.load_track_state_dicts(dict(enumerate(tracks)), strict=False)
    for p in model.parameters():  # freeze before streaming — the streamer requires it
        p.requires_grad_(False)
    streamer = HostResidentLayers(model.text_models, N_LAYERS, "cpu")
    model.layer_stream = streamer
    streamed = freeze_slice_teacher(model)

    for pass_no in (1, 2):
        h, caps = teacher_forward(streamed, batch["input_ids"], batch["attention_mask"], pa, pm)
        assert torch.equal(h_ref, h), f"pass {pass_no}: streamed hidden drifts"
        assert caps_ref.keys() == caps.keys()
        for i in caps_ref:
            assert torch.equal(caps_ref[i], caps[i]), f"pass {pass_no}: capture {i} drifts"

    # The layers really are off-device between passes, not just copied around.
    assert all(p.data.numel() == 0 for ps in streamer._params for p in ps)
    assert streamer.host_bytes > 0


def _run(in_backward: bool):
    cfg, _dense, tracks, pt = _build(n_tracks=4, sync_after=list(range(N_LAYERS)))
    pt.set_sync_phase("post-attn")
    pt.train()
    teacher = _teacher(cfg, tracks)
    # lambda_block=0 ⇒ the TF loop never flushes, so the free-running KL pass is
    # the only backward and the two paths are exactly comparable.
    dcfg = DistillConfig(sync_layer_indices=tuple(range(N_LAYERS)), lambda_block=0.0)
    trainable = [p for p in pt.parameters() if p.requires_grad]

    def mk(params):
        return Adafactor(params, lr=1e-2, weight_decay=0.0, relative_step=False,
                         scale_parameter=False, warmup_init=False)

    if in_backward:
        opts = {id(p): mk([p]) for p in trainable}

        def hook(p):
            opts[id(p)].step()
            p.grad = None

        for p in trainable:
            p.register_post_accumulate_grad_hook(hook)

    distill_step(pt, teacher, pt.lm_head, _batch(cfg), dcfg)

    if in_backward:
        assert all(p.grad is None for p in trainable), "in-backward left grads resident"
    else:
        mk(trainable).step()
    return [p.detach().clone() for p in trainable]


def test_optim_in_backward_matches_step_after_backward():
    # _build seeds torch, so both runs start from identical weights.
    for a, b in zip(_run(in_backward=False), _run(in_backward=True)):
        assert torch.equal(a, b), f"param drifts by {(a - b).abs().max().item()}"
