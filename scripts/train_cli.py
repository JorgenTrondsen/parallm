"""torchrun entry point: the d1b heal / progressive-D distillation trainer.

Trains the per-track slices at a lever-B (post-attn) schedule against the
frozen-slice exact-schedule teacher (see ``parallm.train.distill``). One rank
per GPU, one PTWrappedModel per rank holding that rank's tracks.

Stage-1 d1b heal of the 27B (every layer a boundary):

    setsid nohup .venv/bin/torchrun --standalone --nproc-per-node=8 \\
        scripts/train_cli.py \\
        --hf-model <Qwen3.6-27B-NVFP4 snapshot dir> \\
        --checkpoint-dir convert_out/qwen3_6_27b_n8 \\
        --out-dir train_out/d1b_heal \\
        --student-forcing-prob 0.25 --lambda-logit-mse 0.05 \\
        --max-steps 4001 --eval-every 500 > train_out/d1b_heal.log 2>&1 &

Progressive-D curriculum: warm-start ``--checkpoint-dir train_out/d1b_heal/best``
with ``--teacher-dir convert_out/qwen3_6_27b_n8`` (the teacher is ALWAYS the
original slices) and a ``--sync-indices`` subset.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from parallm.adapters import get_adapter_for_config
from parallm.dist.groups import build_groups
from parallm.model.pt_model import PTWrappedModel
from parallm.train.data import (
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
)
from parallm.train.distill import (
    DistillConfig,
    distill_step,
    freeze_slice_teacher,
    student_forcing_schedule,
    validate_step,
)
from parallm.train.sync_grads import (
    assert_replicated_consistent,
    build_replication_plan,
    compute_global_grad_norm,
    sync_replicated_grads,
)
from parallm.utils.checkpoint import load_manifest, load_track, load_track_keys, save_manifest


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _build_student(text_cfg, manifest, layout, ckpt_dir: str, sync_layers, sync_phase: str):
    model = PTWrappedModel(
        text_config=text_cfg,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=list(sync_layers),
        track_group=layout.track_group,
    )
    states = {tid: load_track(ckpt_dir, tid) for tid in layout.local_track_ids}
    model.load_track_state_dicts(states, strict=True)
    model.set_sync_phase(sync_phase)
    # bf16 on the HOST first: the module is built in fp32 and the bf16 track weights
    # load into fp32 params, so moving before casting asks the device for 2x the
    # slice. Invisible at 35B (16.4 GiB transient still fit); it is what OOMs 122B.
    return model.to(torch.bfloat16).to(torch.cuda.current_device())


def _build_teacher_streamed(text_cfg, manifest, layout, ckpt_dir: str, sync_layers):
    """Frozen exact-schedule teacher with its decoder layers paged from pinned
    host DRAM (``HostResidentLayers``): only the layers move off-device, which is
    what lets the teacher coexist with the student on a 40 GB card at 122 B."""
    from parallm.model.pt_model import HostResidentLayers

    model = PTWrappedModel(
        text_config=text_cfg,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=list(sync_layers),
        track_group=layout.track_group,
    )
    model.load_track_state_dicts(
        {tid: load_track(ckpt_dir, tid) for tid in layout.local_track_ids}, strict=True
    )
    model.set_sync_phase("exact")
    model = model.to(torch.bfloat16)  # still on the host
    for p in model.parameters():
        p.requires_grad_(False)
    model.layer_stream = HostResidentLayers(
        model.text_models, len(model.text_models[0].layers), torch.cuda.current_device()
    )
    return model.to(torch.cuda.current_device()), model.layer_stream


def _cpu_embed_hooks(embed, dev) -> None:
    """Embed on the host, in/out on the device. Hooks not a wrapper, so the
    state_dict keys (checkpoint format) are untouched; student and teacher share
    the weight through separate modules, so each module needs its own hooks."""
    embed.register_forward_pre_hook(lambda m, a: (a[0].to("cpu"),))
    embed.register_forward_hook(lambda m, a, out: out.to(dev))


def _extract_track_state(student: PTWrappedModel, k: int) -> dict[str, torch.Tensor]:
    """Inverse of ``load_track_state_dicts`` for local track index ``k``."""
    prefix = f"text_models.{k}."
    out: dict[str, torch.Tensor] = {}
    for key, val in student.state_dict().items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = val.detach().to("cpu", copy=True).contiguous()
        elif key == "lm_head.weight" and k == 0:
            out["lm_head.weight"] = val.detach().to("cpu", copy=True).contiguous()
    return out


def _save_checkpoint(student, manifest, layout, out_dir: Path, meta: dict, rank: int):
    """Weights-only per-track save: each rank writes its own tracks."""
    from safetensors.torch import save_file as save_safetensors

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, tid in enumerate(layout.local_track_ids):
        path = out_dir / f"track_{tid}.safetensors"
        # Unlink first: a best/ overwrite otherwise holds old+new (~108 GB at
        # 27B) transiently, which busts the 191 GB home quota. The ckpt is a
        # re-creatable training artifact, so the seconds-wide corruption window
        # is acceptable.
        path.unlink(missing_ok=True)
        save_safetensors(_extract_track_state(student, k), str(path))
    if rank == 0:
        # The saved manifest self-describes the trained schedule so eval picks
        # it up without --sync-indices.
        save_manifest(out_dir, replace(manifest, sync_layer_indices=list(meta["sync_layer_indices"])))
        (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    dist.barrier()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Original HF checkpoint dir (tokenizer + config)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Slices the STUDENT starts from (raw convert output, or a previous best/ for the curriculum)")
    p.add_argument("--teacher-dir", default=None,
                   help="Slices the frozen exact-schedule teacher loads (default: --checkpoint-dir; "
                        "MUST be the original convert output when warm-starting the student from a heal)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--teacher-stream", action="store_true",
                   help="Page the frozen teacher's decoder layers in from pinned host DRAM one "
                        "at a time (~1 layer resident, not the whole teacher) — the memory-for-"
                        "PCIe trade that fits a 122 B teacher next to the student")
    p.add_argument("--optim-in-backward", action="store_true",
                   help="Step + free each param's grad inside backward, so the grad buffer is "
                        "~2 layers instead of the whole model. Requires --grad-accum-steps 1, "
                        "drops global grad-norm clipping (adafactor's clip_threshold covers it), "
                        "and gives one update per objective per batch instead of one summed step")
    p.add_argument("--cpu-embed", action="store_true",
                   help="Keep the frozen embed table on CPU (~8 MB/step of H2D) so rank 0 does "
                        "not carry embed+lm_head over its peers' budget")
    p.add_argument("--sync-indices", default=None,
                   help="Comma-separated boundary layers (default: every layer = the d1b schedule)")
    p.add_argument("--sync-phase", default="post-attn", choices=["post-attn"],
                   help="Only lever B is built — the program's schedule")
    p.add_argument("--intra-window-mse", action="store_true",
                   help="Loss-only synced taps at non-boundary layers (the D>1 curriculum wants this)")
    # Recipe constants (the 9B-record defaults).
    p.add_argument("--max-steps", type=int, default=4001)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw",
                   help="adamw = fused bf16 default (27B/N=8, ~4 B/param state); adafactor "
                        "(factored 2nd moment, no 1st moment) ≈ 0 optimizer state (~0.5 GB/rank "
                        "vs 8 GB) — what fits N=16 K=2 on a 40 GB card (the 35B recipe) and "
                        "the memory-lean choice for scaling to 122 B.")
    p.add_argument("--lr-schedule-steps", type=int, default=None,
                   help="Cosine horizon for the LR schedule (default: --max-steps). Set it to the "
                        "REFERENCE run's --max-steps when comparing a short run against a "
                        "checkpoint taken part-way through a longer one; otherwise shortening "
                        "--max-steps silently anneals the LR and the two are not comparable")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lambda-block", type=float, default=1.0)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=0.5)
    p.add_argument("--lambda-logit-mse", type=float, default=0.05)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--block-mse-clamp", type=float, default=10.0)
    p.add_argument("--kl-ce-chunk-size", type=int, default=256)
    p.add_argument("--student-forcing-prob", type=float, default=0.25)
    p.add_argument("--student-forcing-warmup", type=int, default=0)
    p.add_argument("--train-embeddings", action="store_true",
                   help="Unfreeze embed_tokens (lm_head stays frozen — the replicated loss "
                        "requires an identical head on every rank)")
    p.add_argument("--sync-attention-heads", action="store_true",
                   help="force_sync the diverged-by-default attention head params (legacy A/B)")
    # Data.
    p.add_argument("--data-preset", default="open-mix", choices=preset_names())
    p.add_argument("--data-source", action="append", default=None,
                   help="NAME[:CONFIG[:TEXT_KEY[:WEIGHT]]] (repeatable; overrides --data-preset)")
    p.add_argument("--val-dataset-name", default="Salesforce/wikitext")
    p.add_argument("--val-dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--val-split", default="validation")
    p.add_argument("--val-batches", type=int, default=20)
    # Cadence.
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--best-name", default="best")
    p.add_argument("--no-save", action="store_true",
                   help="Skip ALL checkpoint writes (for smokes — a 16-track MoE best/ "
                        "is ~68GB, pointless when only measuring fit/speed).")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable per-sublayer activation checkpointing — no backward "
                        "recompute (~1.5-2x faster, launch-bound). Needs the VRAM "
                        "headroom --optim adafactor frees.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mem-report", action="store_true")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    torch.manual_seed(args.seed)

    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    teacher_dir = args.teacher_dir or args.checkpoint_dir

    cfg_hf = AutoConfig.from_pretrained(args.hf_model)
    text_cfg = cfg_hf.text_config if hasattr(cfg_hf, "text_config") else cfg_hf
    num_layers = text_cfg.num_hidden_layers
    if args.sync_indices is not None:
        sync_layers = sorted(int(x) for x in args.sync_indices.split(",") if x.strip())
    else:
        sync_layers = list(range(num_layers))  # d1b: every layer a boundary
    if sync_layers[-1] != num_layers - 1:
        sync_layers.append(num_layers - 1)  # the head needs the final post-MLP sync

    _log(rank, f"[init] n_tracks={manifest.n_tracks} world={layout.world_size} "
               f"boundaries={len(sync_layers)} phase={args.sync_phase}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    # Data: identical stream on every rank (the SyncBoundary contract).
    if args.data_source:
        sources = [parse_source_spec(s) for s in args.data_source]
        data_cfg = CalibrationDataConfig(sources=sources, seq_len=args.seq_len, seed=args.seed)
    else:
        data_cfg = CalibrationDataConfig.from_preset(args.data_preset, seq_len=args.seq_len,
                                                     seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        PackedTokenStream(tokenizer, data_cfg), batch_size=args.batch_size
    )
    val_cfg = CalibrationDataConfig.single(
        args.val_dataset_name, args.val_dataset_config, split=args.val_split,
        seq_len=args.seq_len, seed=args.seed,
    )
    val_batches: list[dict] = []
    for i, b in enumerate(torch.utils.data.DataLoader(
            PackedTokenStream(tokenizer, val_cfg), batch_size=args.batch_size)):
        if i >= args.val_batches:
            break
        val_batches.append({k: v.to(torch.cuda.current_device()) for k, v in b.items()})

    student = _build_student(text_cfg, manifest, layout, args.checkpoint_dir,
                             sync_layers, args.sync_phase)
    student.use_checkpoint = not args.no_checkpoint
    student.train()

    if args.teacher_stream:
        teacher_model, streamer = _build_teacher_streamed(
            text_cfg, manifest, layout, teacher_dir, sync_layers)
        teacher = teacher_val = freeze_slice_teacher(teacher_model)
        _log(rank, f"[init] teacher layers streamed from pinned host DRAM: "
                   f"{streamer.host_bytes / 2**30:.2f} GiB/rank off-device")
    else:
        teacher_model = _build_student(text_cfg, manifest, layout, teacher_dir,
                                       sync_layers, "exact")
        teacher = teacher_val = freeze_slice_teacher(teacher_model)

    # Freeze the dense-copy pieces; tie the teacher's to the student's frozen
    # storage (2×2.54 GB back on rank 0 at 27B).
    tm0 = student.text_models[0]
    if tm0.embed_tokens is not None:
        tm0.embed_tokens.weight.requires_grad_(args.train_embeddings)
        if not args.train_embeddings:
            teacher_model.text_models[0].embed_tokens.weight = tm0.embed_tokens.weight
            if args.cpu_embed:
                # Shared weight to the host, then hooks on each module that reads
                # it — the teacher embeds through its OWN module.
                dev = torch.cuda.current_device()
                tm0.embed_tokens.to("cpu")
                _cpu_embed_hooks(tm0.embed_tokens, dev)
                t_embed = teacher_model.text_models[0].embed_tokens
                if t_embed is not None:
                    _cpu_embed_hooks(t_embed, dev)
                _log(rank, "[init] embed_tokens on CPU")
    if student.lm_head is not None:
        student.lm_head.weight.requires_grad_(False)  # ponytail: frozen always — replicated loss needs an identical head on every rank; re-broadcast per step if this ever unfreezes
        teacher_model.lm_head.weight = student.lm_head.weight

    # The loss head on every rank: owner uses the student's; peers load a
    # frozen copy of the same tensor from the teacher slices.
    if student.lm_head is not None:
        lm_head = student.lm_head
    else:
        w = load_track_keys(teacher_dir, 0, ["lm_head.weight"])["lm_head.weight"]
        lm_head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        lm_head.weight = torch.nn.Parameter(w, requires_grad=False)
        lm_head = lm_head.to(torch.bfloat16).to(torch.cuda.current_device())

    adapter = get_adapter_for_config(text_cfg)
    plan = build_replication_plan(
        student, adapter=adapter, text_cfg=text_cfg, layout=layout,
        force_sync=args.sync_attention_heads,
    )
    assert_replicated_consistent(plan)

    trainable = [p_ for p_ in student.parameters() if p_.requires_grad]
    n_train = sum(p_.numel() for p_ in trainable)
    _log(rank, f"[init] trainable params/rank ≈ {n_train/1e9:.2f}B; replication groups={len(plan)}")
    # fused: single-kernel step, no _foreach transients — the multi-tensor path
    # materializes a full extra copy of the second-moment states (~5.7 GB at
    # 27B/N=8), which is exactly the 40 GB card's missing headroom.
    def _make_optim(params):
        if args.optim == "adafactor":
            from transformers.optimization import Adafactor
            # We drive LR (lr_at cosine), so disable relative-step + param scaling;
            # beta1=None ⇒ no first moment ⇒ ~0 optimizer state (the 122 B lever).
            return Adafactor(params, lr=args.lr, weight_decay=0.0,
                             relative_step=False, scale_parameter=False, warmup_init=False)
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)

    cur_lr = [args.lr]
    gsq = torch.zeros((), device=torch.cuda.current_device())
    # Per-hooked-param step counts: how often each fired last batch, used to
    # normalize its lr (see _step_in_backward).
    fire_prev: dict[int, int] = {}
    fire_cur: "collections.Counter[int]" = collections.Counter()
    if args.optim_in_backward:
        # Step + free each param as its grad lands ⇒ grad buffer is ~2 layers, not
        # the whole model (the 122 B lever). distill_step backwards several times a
        # batch (per boundary segment + the free-running KL pass), so a param steps
        # once per contribution, not once on the sum — a recipe change, 35B-gated.
        assert args.grad_accum_steps == 1, \
            "--optim-in-backward requires --grad-accum-steps 1 (accumulation needs the buffer it frees)"
        # Replicated params (norm scales, KV rows) keep the resident path: their
        # grads all-reduce across ranks first, which a per-param hook can't order.
        replicated = {id(p_) for cg in plan for p_ in cg.local_params}
        hooked = [p_ for p_ in trainable if id(p_) not in replicated]
        hooked_optims = {id(p_): _make_optim([p_]) for p_ in hooked}

        def _step_in_backward(p_):
            gsq.add_(torch.linalg.vector_norm(p_.grad, dtype=torch.float32) ** 2)
            o = hooked_optims[id(p_)]
            # Adafactor moves ~lr/step; a param steps once per grad (2x/batch at
            # d1b), so undivided it travels 2x too far — measured 0.586 vs 0.700.
            # Divide by last batch's firing count to restore single-step distance.
            for g_ in o.param_groups:
                g_["lr"] = cur_lr[0] / fire_prev.get(id(p_), 1)
            o.step()
            p_.grad = None
            fire_cur[id(p_)] += 1

        for p_ in hooked:
            p_.register_post_accumulate_grad_hook(_step_in_backward)
        # None when there are no replicated params — torch rejects an empty group.
        rep_params = [p_ for p_ in trainable if id(p_) in replicated]
        optim = _make_optim(rep_params) if rep_params else None
        _log(rank, f"[init] optimizer={args.optim} in-backward: {len(hooked)} params stepped "
                   f"on arrival, {len(trainable) - len(hooked)} replicated params on the "
                   f"collective path; global grad-norm clip OFF "
                   f"(adafactor clip_threshold=1.0 covers it)")
    else:
        optim = _make_optim(trainable)
        _log(rank, f"[init] optimizer={args.optim}")

    # Cosine horizon decoupled from run length — else --max-steps silently anneals
    # the LR and a short run's step-N checkpoint isn't comparable to a long run's.
    horizon = args.lr_schedule_steps or args.max_steps

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        frac = (step - args.warmup_steps) / max(1, horizon - args.warmup_steps)
        cos = 0.5 * (1.0 + torch.cos(torch.tensor(frac * 3.141592653589793)).item())
        return args.lr * (args.lr_min_ratio + (1 - args.lr_min_ratio) * cos)

    dcfg = DistillConfig(
        sync_layer_indices=tuple(sync_layers),
        lambda_block=args.lambda_block,
        lambda_kl=args.lambda_kl,
        lambda_ce=args.lambda_ce,
        lambda_logit_mse=args.lambda_logit_mse,
        kl_temperature=args.kl_temperature,
        normalize_block_mse=True,
        block_mse_clamp=args.block_mse_clamp,
        intra_window_mse=args.intra_window_mse,
        kl_ce_chunk_size=args.kl_ce_chunk_size,
    )
    out_dir = Path(args.out_dir)
    best_kl = float("inf")
    step = 0
    accum = 0
    t0 = time.perf_counter()
    train_iter = iter(train_loader)

    def run_validation() -> float:
        student.eval()
        with torch.no_grad():
            kls = [validate_step(student, teacher_val, lm_head, vb,
                                 kl_temperature=args.kl_temperature,
                                 chunk_size=args.kl_ce_chunk_size)["kl"].item()
                   for vb in val_batches]
        student.train()
        return sum(kls) / max(1, len(kls))

    while step < args.max_steps:
        batch = next(train_iter)
        batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
        sf_p = student_forcing_schedule(
            step, args.student_forcing_prob, args.student_forcing_warmup
        )
        cur_lr[0] = lr_at(step)  # the in-backward hooks read this as they fire
        gsq.zero_()
        losses = distill_step(
            student, teacher, lm_head, batch, dcfg,
            student_forcing_prob=sf_p,
            forcing_seed=(args.seed, step),
            loss_scale=1.0 / args.grad_accum_steps,
        )
        accum += 1
        if accum < args.grad_accum_steps:
            continue
        accum = 0

        sync_replicated_grads(plan)
        if args.optim_in_backward:
            # Hooked params already stepped and freed, so no moment holds the whole
            # gradient to clip against; gsq is this rank's running sum, diagnostic
            # only (not deduplicated across replication groups).
            gnorm = gsq.sqrt()
        else:
            gnorm = compute_global_grad_norm(student, plan)
            clip = (args.max_grad_norm / (gnorm + 1e-6)).clamp(max=1.0)
            if clip < 1.0:
                for p_ in trainable:
                    if p_.grad is not None:
                        p_.grad.mul_(clip)
        if optim is not None:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            optim.step()
            optim.zero_grad(set_to_none=True)
        if args.optim_in_backward:
            fire_prev = dict(fire_cur)  # this batch's counts normalize the next
            fire_cur.clear()

        if step % args.log_every == 0:
            dt = (time.perf_counter() - t0) / max(1, step + 1)
            _log(rank, f"step {step} total={losses['total'].item():.4f} "
                       f"block={losses['block_mse'].item():.4f} kl={losses['kl'].item():.4f} "
                       f"ce={losses['ce'].item():.4f} lmse={losses['logit_mse'].item():.4f} "
                       f"sf={sf_p:.3f} gnorm={gnorm.item():.2f} lr={lr_at(step):.2e} "
                       f"{dt:.2f}s/step")
        if args.mem_report and step == 2:
            _log(rank, f"[mem] rank0 peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                       f"resident={torch.cuda.memory_allocated()/2**30:.2f}GiB")

        step += 1
        if args.eval_every and step % args.eval_every == 0:
            val_kl = run_validation()
            _log(rank, f"[val] step {step} val_kl={val_kl:.4f} (best {best_kl:.4f})")
            if val_kl < best_kl:
                best_kl = val_kl
                if not args.no_save:
                    _save_checkpoint(student, manifest, layout, out_dir / args.best_name,
                                     {"step": step, "val_kl": val_kl,
                                      "sync_layer_indices": sync_layers,
                                      "sync_phase": args.sync_phase,
                                      "args": vars(args)}, rank)
                    _log(rank, f"[save] best → {out_dir / args.best_name}")
        if args.save_every and not args.no_save and step % args.save_every == 0:
            _save_checkpoint(student, manifest, layout, out_dir / f"step_{step}",
                             {"step": step, "sync_layer_indices": sync_layers,
                              "sync_phase": args.sync_phase, "args": vars(args)}, rank)

    val_kl = run_validation()
    _log(rank, f"[final] step {step} val_kl={val_kl:.4f} best={best_kl:.4f}")
    if val_kl < best_kl and not args.no_save:
        _save_checkpoint(student, manifest, layout, out_dir / args.best_name,
                         {"step": step, "val_kl": val_kl,
                          "sync_layer_indices": sync_layers,
                          "sync_phase": args.sync_phase, "args": vars(args)}, rank)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
