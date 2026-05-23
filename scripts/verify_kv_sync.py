"""End-to-end verification of replicated-parameter gradient sync.

Launches the same init as train_qwen3_5_9b.py, runs N distillation steps,
and reports for representative parameters:

  - q_proj.weight (unique per track):   should CHANGE after each step.
  - k_proj.weight (kv-replicated):       should CHANGE AND stay bit-identical
                                          across all tracks in its kv-group.
  - input_layernorm.weight (Replicated): should CHANGE AND stay bit-identical
                                          across ALL tracks.

The "stay identical" claim is checked globally via
``assert_replicated_consistent(plan)`` — it MIN/MAX-all-reduces every
replicated parameter inside its process subgroup, so any drift across any
rank raises ``RuntimeError``.

Launch::

    torchrun --standalone --nproc-per-node=8 scripts/verify_kv_sync.py \\
        --hf-model /home2/jtr020/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha> \\
        --tracks-dir /home2/jtr020/pt_tracks/qwen3_5_9b_n16_d4 \\
        --steps 2
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.adapters import get_adapter_for_config
from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.data import CalibrationDataConfig, PackedTokenStream
from pt_converter.train.distill import DistillConfig, distill_step
from pt_converter.train.sync_grads import (
    assert_replicated_consistent,
    build_replication_plan,
    sync_replicated_grads,
)
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _find_param(student: PTWrappedModel, text_models_idx: int, dotted: str):
    tm = student.text_models[text_models_idx]
    module_path, _, name = dotted.rpartition(".")
    submod = tm.get_submodule(module_path) if module_path else tm
    return getattr(submod, name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True)
    p.add_argument("--tracks-dir", required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    manifest = load_manifest(args.tracks_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    _log(
        rank,
        f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
        f"K={layout.tracks_per_rank} D={manifest.sync_block_depth}",
    )

    cfg = AutoConfig.from_pretrained(args.hf_model)
    text_cfg = cfg.text_config
    num_kv = int(text_cfg.num_key_value_heads)
    tpg = manifest.n_tracks // num_kv  # tracks per kv-group
    _log(rank, f"[init] num_kv_heads={num_kv}  tracks_per_kv_group={tpg}")

    _log(rank, "[init] loading frozen dense teacher…")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    teacher_model.eval()
    for prm in teacher_model.parameters():
        prm.requires_grad = False
    text_model = (
        teacher_model.model.language_model
        if hasattr(teacher_model.model, "language_model")
        else teacher_model.model
    )
    teacher = HookedTeacher(
        text_model=text_model,
        lm_head=teacher_model.lm_head,
        sync_layer_indices=manifest.sync_layer_indices,
    )
    teacher_model = teacher_model.to(torch.cuda.current_device())
    wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)

    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}")
    student = PTWrappedModel(
        text_config=text_cfg,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=manifest.sync_layer_indices,
        track_group=layout.track_group,
    )
    track_states = {tid: load_track(args.tracks_dir, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)

    # ----- Build replication plan and assert initial consistency -----
    adapter = get_adapter_for_config(cfg)
    plan = build_replication_plan(student, adapter=adapter, text_cfg=text_cfg, layout=layout)
    _log(rank, f"[init] replication plan: {len(plan)} groups")
    assert_replicated_consistent(plan)
    _log(rank, "[init] startup consistency check PASSED")

    # ----- Snapshot a few representative tensors -----
    # Pick a full-attention layer to inspect. Layers ending in indices that
    # ARE in sync_layer_indices fire the sync; any full-attention layer works.
    target_layer = None
    for i, lt in enumerate(manifest.layer_types):
        if lt == "full_attention":
            target_layer = i
            break
    if target_layer is None:
        _log(rank, "[ERR] no full_attention layer found; aborting")
        dist.destroy_process_group()
        return 1
    _log(rank, f"[probe] inspecting layer {target_layer} (full_attention)")

    # On every rank, snapshot the *first* local track's copies of:
    #   - q_proj.weight   (unique per track)
    #   - k_proj.weight   (kv-replicated)
    #   - input_layernorm.weight (fully replicated)
    # Also: norm.weight (final, fully replicated) — global check.
    k0 = 0  # first local track index
    tid0 = layout.local_track_ids[k0]
    q_before = _find_param(student, k0, f"layers.{target_layer}.self_attn.q_proj.weight").detach().clone()
    k_before = _find_param(student, k0, f"layers.{target_layer}.self_attn.k_proj.weight").detach().clone()
    ln_before = _find_param(
        student, k0, f"layers.{target_layer}.input_layernorm.weight"
    ).detach().clone()
    final_norm_before = _find_param(student, k0, "norm.weight").detach().clone()

    # ----- Data + step -----
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    ds = PackedTokenStream(tok, CalibrationDataConfig(seq_len=args.seq_len))
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    optim = torch.optim.AdamW(
        [pp for pp in student.parameters() if pp.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    distill_cfg = DistillConfig(
        sync_layer_indices=tuple(manifest.sync_layer_indices),
        lambda_block=1.0,
        lambda_kl=1.0,
        lambda_ce=0.5,
        kl_temperature=1.0,
    )

    student.train()
    step = 0
    for batch in loader:
        if step >= args.steps:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        optim.zero_grad(set_to_none=True)
        losses = distill_step(student, teacher, batch, distill_cfg)
        losses["total"].backward()
        sync_replicated_grads(plan)
        optim.step()

        _log(
            rank,
            f"[step {step}] total={losses['total'].item():.4f} "
            f"block_mse={losses['block_mse'].item():.4f} "
            f"kl={losses['kl'].item():.4f} ce={losses['ce'].item():.4f}",
        )
        step += 1

    # ----- After-step checks -----
    q_after = _find_param(student, k0, f"layers.{target_layer}.self_attn.q_proj.weight").detach()
    k_after = _find_param(student, k0, f"layers.{target_layer}.self_attn.k_proj.weight").detach()
    ln_after = _find_param(
        student, k0, f"layers.{target_layer}.input_layernorm.weight"
    ).detach()
    final_norm_after = _find_param(student, k0, "norm.weight").detach()

    def _diff(a, b):
        return (a.float() - b.float()).abs().max().item()

    q_change = _diff(q_after, q_before)
    k_change = _diff(k_after, k_before)
    ln_change = _diff(ln_after, ln_before)
    fn_change = _diff(final_norm_after, final_norm_before)

    # Per-rank log
    print(
        f"[rank {rank} tid={tid0}] q_proj Δ={q_change:.3e}  k_proj Δ={k_change:.3e}  "
        f"input_layernorm Δ={ln_change:.3e}  final_norm Δ={fn_change:.3e}",
        flush=True,
    )

    # If this rank holds two local tracks AND they're in the same kv-group,
    # do a local bit-equality check on k_proj.
    if layout.tracks_per_rank > 1:
        k1_param = _find_param(
            student, 1, f"layers.{target_layer}.self_attn.k_proj.weight"
        ).detach()
        # Are track tid0 and tid1 in the same kv-group?
        tid1 = layout.local_track_ids[1]
        same_group = (tid0 // tpg) == (tid1 // tpg)
        if same_group:
            equal = torch.equal(k_after, k1_param)
            print(
                f"[rank {rank}] local k_proj tracks {tid0} vs {tid1} bit-identical: {equal}",
                flush=True,
            )

    # Cross-rank: the canonical check. After the step every replication group
    # member (across all ranks) must still be bit-identical.
    try:
        assert_replicated_consistent(plan)
        _log(rank, "[post-step] cross-rank consistency check PASSED")
    except RuntimeError as exc:
        _log(rank, f"[post-step] cross-rank consistency check FAILED: {exc}")
        dist.destroy_process_group()
        return 2

    # Gather verdict on rank 0
    q_changed = q_change > 0.0
    k_changed = k_change > 0.0
    ln_changed = ln_change > 0.0
    fn_changed = fn_change > 0.0
    _log(
        rank,
        f"[verdict on rank 0]\n"
        f"  q_proj   updated?     {q_changed}    (Δ={q_change:.3e})\n"
        f"  k_proj   updated?     {k_changed}    (Δ={k_change:.3e})\n"
        f"  input_LN updated?     {ln_changed}   (Δ={ln_change:.3e})\n"
        f"  final_norm updated?   {fn_changed}   (Δ={fn_change:.3e})\n"
        f"  cross-rank consistent: YES\n",
    )

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
