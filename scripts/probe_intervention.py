"""torchrun entry point: cross-track *intervention* probe.

Runs the free-running student forward with a candidate comm-free cross-track
channel substituted at every partial-read layer (see
``pt_converter.eval.intervention``) and reports the REAL end-to-end metrics —
KL/top-1 vs the teacher and (optionally) downstream lm-eval accuracy — for each
channel, **forward-only, no training**.

It is self-calibrating. The ``zero`` channel reproduces the deployed D-window
forward (the floor) and the ``oracle`` channel reproduces the D=1 sync-every-layer
forward (the ceiling); both are exact by construction, so they are a built-in
correctness check. Any cheap channel (``stale`` / ``avg:W`` / ``lowrank:r``) lands
between, and the printed ``%head`` column is its share of the ``zero → oracle``
headroom — the honest answer to "would this channel help?", in minutes.

Point ``--checkpoint-dir`` at a TRAINED ``best`` so the compounding confound is
already removed by distillation; then the ``zero → oracle`` headroom isolates the
cross-track structural value alone.

    torchrun --standalone --nproc-per-node=8 scripts/probe_intervention.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir train_out/qwen3_5_9b_n16/best \\
        --channels zero,oracle,stale,avg:8,lowrank:64,lowrank:256 \\
        --downstream-tasks arc_challenge,winogrande --num-batches 16
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.eval.fidelity import _LOGIT_METRIC_NAMES, _fidelity_logit_metrics
from pt_converter.eval.intervention import (
    CalibratedFixedLowRankChannel,
    MaskedOracleChannel,
    intervention_forward,
    parse_channel,
)
from pt_converter.train.distill import _block_ranges
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.train.data import (
    DEFAULT_PRESET,
    CalibrationDataConfig,
    PackedTokenStream,
    parse_source_spec,
    preset_names,
    preset_sources,
)
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _kl_pass(student, teacher, batches, channel, sync_indices, chunk_size, device):
    """Mean KL/top-1/ppl over the materialized ``batches`` under ``channel``.

    Peer ranks emit zero placeholders for the logit metrics; the SUM all-reduce
    lands on the owner's value. Every rank still runs the teacher forward and the
    intervention forward (whose per-layer all-reduces must line up)."""
    sums: dict[str, torch.Tensor] = {}
    for batch in batches:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch.get("attention_mask")
        attn = attn.to(device, non_blocking=True) if attn is not None else None
        labels = batch["labels"].to(device, non_blocking=True)

        teacher_logits, _ = teacher.forward(input_ids, attention_mask=attn)
        hidden = intervention_forward(student, input_ids, attn, sync_indices, channel)
        if student.lm_head is not None:
            student_logits = student.lm_head(hidden)
            m = _fidelity_logit_metrics(student_logits, teacher_logits, labels, attn, chunk_size)
        else:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            m = {name: zero.clone() for name in _LOGIT_METRIC_NAMES}
        for name, val in m.items():
            sums[name] = sums.get(name, torch.zeros((), device=device, dtype=torch.float32)) + val.float()

    n = max(1, len(batches))
    for name in sums:
        dist.all_reduce(sums[name], op=dist.ReduceOp.SUM)
        sums[name] = sums[name] / n
    return sums


@torch.no_grad()
def _calibrate(student, calib_batches, channel, sync_indices, device):
    """Fit a CalibratedFixedLowRankChannel's frozen basis over a calibration set.

    Runs ``intervention_forward`` with the channel in observing mode (it records the
    D=2-trajectory ``other_k`` per layer/track and returns zeros), then finalizes the
    per-(layer,track) PCA bases. Every rank runs it in lockstep (the forward all-reduces)."""
    channel.start_observing()
    for batch in calib_batches:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch.get("attention_mask")
        attn = attn.to(device, non_blocking=True) if attn is not None else None
        intervention_forward(student, input_ids, attn, sync_indices, channel)
    channel.finalize()


def _downstream_pass(student, tok, channel, sync_indices, args, rank, device):
    """lm-eval downstream accuracy of the intervention forward (broadcast score)."""
    from lm_eval import simple_evaluate

    from pt_converter.eval.downstream import _pick_metric, aggregate_downstream_score
    from pt_converter.eval.lm_eval_adapter import PTLM, is_lm_head_owner

    tasks = [t.strip() for t in args.downstream_tasks.split(",") if t.strip()]

    def _fn(input_ids, attention_mask):
        hidden = intervention_forward(student, input_ids, attention_mask, sync_indices, channel)
        return student.lm_head(hidden) if student.lm_head is not None else None

    lm = PTLM(
        forward_fn=_fn, tokenizer=tok,
        max_length=args.downstream_max_length, batch_size=args.downstream_batch_size,
        device=device, is_owner_rank=is_lm_head_owner(student),
    )
    res = simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=args.downstream_num_fewshot,
        limit=args.downstream_limit, batch_size=args.downstream_batch_size,
        random_seed=args.seed, numpy_random_seed=args.seed,
        torch_random_seed=args.seed, fewshot_random_seed=args.seed,
    )
    score, per_task = 0.0, {}
    if rank == 0:
        score = aggregate_downstream_score(res, tasks)
        table = (res or {}).get("results", {})
        for t in tasks:
            v = _pick_metric(table.get(t, {}), ["acc_norm", "acc"])
            if v is not None:
                per_task[t] = v
    score_t = torch.tensor([score], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.broadcast(score_t, src=0)
    return score_t.item(), per_task


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path (same as training)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir (track_*.safetensors + manifest.json). Point at a "
                        "TRAINED best/ so distillation has already removed the compounding confound.")
    p.add_argument("--channels", default="zero,oracle,stale,avg:8,lowrank:64,lowrank:256",
                   help="Comma-separated channels. Always include 'zero' and 'oracle' (the floor/ceiling "
                        "anchors that calibrate the %%head column). stale=1-token cache; avg:W=W-token "
                        "causal-mean cache; lowrank:r=rank-r exchange ceiling.")
    p.add_argument("--oracle-sweep", default=None, choices=["single", "loo", "band"],
                   help="Localize WHERE the zero->oracle headroom lives: REPLACE --channels with "
                        "[zero, oracle, <one masked-oracle per probe unit>]. single=oracle at exactly "
                        "one mid-window layer (marginal value on the D2 floor); loo=oracle everywhere "
                        "EXCEPT one layer (that layer's contribution to the ceiling); band=oracle on one "
                        "contiguous depth-band of mid-window layers (see --oracle-sweep-bands).")
    p.add_argument("--oracle-sweep-bands", type=int, default=4,
                   help="For --oracle-sweep band: number of contiguous depth-bands the mid-window "
                        "(partial-read) layers are split into.")
    p.add_argument("--num-batches", type=int, default=16, help="Packed sequences for the KL/top-1 pass.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=128, help="Seq-chunk for the fp32 vocab expansion.")
    p.add_argument("--sync-indices", default=None,
                   help="Override the manifest schedule (comma-separated). The intervention windows "
                        "and the real boundaries both follow it.")
    # Data (mirrors eval_fidelity.py: held-out front of the training mixture by default).
    p.add_argument("--data-preset", default=DEFAULT_PRESET, choices=preset_names())
    p.add_argument("--data-source", action="append", default=None, metavar="NAME[:CONFIG[:KEY[:WEIGHT]]]")
    p.add_argument("--dataset-name", default=None, help="Legacy single-dataset override.")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="validation")
    p.add_argument("--text-key", default="text")
    p.add_argument("--skip-docs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # Downstream (optional; off unless --downstream-tasks given).
    p.add_argument("--downstream-tasks", default="",
                   help="Comma-separated lm-eval tasks (e.g. arc_challenge,winogrande). Empty = skip "
                        "downstream (KL/top-1 only).")
    p.add_argument("--downstream-limit", type=int, default=200, help="Requests per task.")
    p.add_argument("--downstream-batch-size", type=int, default=4)
    p.add_argument("--downstream-max-length", type=int, default=2048)
    p.add_argument("--downstream-num-fewshot", type=int, default=0)
    p.add_argument("--fixed-calib-batches", type=int, default=32,
                   help="For calib-fixed-lowrank channels: data batches used to PCA-fit the "
                        "frozen basis (a collect pass on the D=2 trajectory) before scoring.")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_count = torch.cuda.device_count()
    if local_rank >= gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} >= visible GPU count {gpu_count}. "
            f"Launch with --nproc-per-node <= #GPUs."
        )
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    if args.sync_indices is not None:
        sync_layers = [int(x) for x in args.sync_indices.split(",") if x.strip() != ""]
    elif manifest.sync_layer_indices is not None:
        sync_layers = list(manifest.sync_layer_indices)
    else:
        raise SystemExit(
            "[error] checkpoint manifest has no sync_layer_indices. Pass --sync-indices, or point "
            "--checkpoint-dir at a trained checkpoint."
        )
    if args.oracle_sweep:
        # Mid-window (partial-read) layers = every layer that is NOT a window end
        # (the harness only substitutes at those; a window end is the real boundary).
        mid = [
            L
            for (start, end) in _block_ranges(manifest.num_layers, tuple(sync_layers))
            for L in range(start, end)
        ]
        if args.oracle_sweep == "band":
            n = max(1, args.oracle_sweep_bands)
            bands = [mid[k * len(mid) // n:(k + 1) * len(mid) // n] for k in range(n)]
            probes = [MaskedOracleChannel(set(b), invert=False) for b in bands if b]
        else:
            invert = args.oracle_sweep == "loo"
            probes = [MaskedOracleChannel({L}, invert=invert) for L in mid]
        channels = [parse_channel("zero"), parse_channel("oracle"), *probes]
    else:
        channels = [parse_channel(s) for s in args.channels.split(",") if s.strip()]
    run_kl = args.num_batches > 0
    run_ds = bool(args.downstream_tasks.strip())
    if not run_kl and not run_ds:
        raise SystemExit(
            "[error] nothing to do: set --num-batches > 0 (KL vs teacher) and/or --downstream-tasks. "
            "Downstream-only (--num-batches 0) skips the teacher entirely — much lower memory."
        )
    _log(rank, f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
               f"K={layout.tracks_per_rank} num_layers={manifest.num_layers}")
    _log(rank, f"[init] sync schedule: {len(sync_layers)} syncs at {sync_layers}"
               + ("  (OVERRIDE)" if args.sync_indices is not None else ""))
    _log(rank, f"[init] channels: {[c.name for c in channels]}")

    # ----- Teacher (frozen, FSDP-sharded). Needed ONLY for the KL pass; skipped
    # for downstream-only runs (--num-batches 0), which saves the teacher shard +
    # all-gather transients — the lever for fitting a shared/constrained GPU. -----
    cfg = AutoConfig.from_pretrained(args.hf_model)
    teacher = None
    if run_kl:
        _log(rank, "[init] loading frozen dense teacher…")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        text_model = (
            teacher_model.model.language_model
            if hasattr(teacher_model.model, "language_model")
            else teacher_model.model
        )
        teacher = HookedTeacher(
            text_model=text_model, lm_head=teacher_model.lm_head,
            sync_layer_indices=[manifest.num_layers - 1],
        )
        teacher_model = teacher_model.to(device)
        wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)
    else:
        _log(rank, "[init] downstream-only (--num-batches 0): skipping teacher load.")

    # ----- Student (legacy full-logits path; owner holds the full lm_head). -----
    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        sync_after_layers=sync_layers,
        track_group=layout.track_group,
    )
    track_states = {tid: load_track(args.checkpoint_dir, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    student = student.to(device).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)
    student.eval()

    tok = AutoTokenizer.from_pretrained(args.hf_model)

    # ----- Materialize batches from the seed-deterministic stream (every rank reads
    # the identical sequence). KL batches feed the teacher-vs-student pass; calib
    # batches PCA-fit the calib-fixed-lowrank bases. Both are skippable. -----
    need_calib = any(isinstance(ch, CalibratedFixedLowRankChannel) for ch in channels)

    def _materialize(n: int):
        if args.data_source:
            data_cfg = CalibrationDataConfig(
                sources=[parse_source_spec(s) for s in args.data_source],
                seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
            )
        elif args.dataset_name:
            data_cfg = CalibrationDataConfig.single(
                dataset_name=args.dataset_name, dataset_config=args.dataset_config,
                split=args.split, text_key=args.text_key, seq_len=args.seq_len, seed=args.seed,
            )
        else:
            data_cfg = CalibrationDataConfig(
                sources=preset_sources(args.data_preset),
                seq_len=args.seq_len, seed=args.seed, skip_docs=args.skip_docs,
            )
        loader = DataLoader(PackedTokenStream(tok, data_cfg), batch_size=args.batch_size, num_workers=0)
        out = []
        for batch in loader:
            if len(out) >= n:
                break
            if batch["input_ids"].ndim == 1:
                batch = {k: v.unsqueeze(0) for k, v in batch.items()}
            out.append(batch)
        return out

    batches = _materialize(args.num_batches) if run_kl else []
    if run_kl:
        _log(rank, f"[data] materialized {len(batches)} KL batches (seq_len={args.seq_len})")
    calib_batches = _materialize(args.fixed_calib_batches) if need_calib else []
    if need_calib:
        _log(rank, f"[data] materialized {len(calib_batches)} calibration batches (seq_len={args.seq_len})")

    sync_indices = tuple(sync_layers)
    results: dict[str, dict] = {}
    for ch in channels:
        if isinstance(ch, CalibratedFixedLowRankChannel):
            _log(rank, f"[run] channel={ch.name} — PCA-fitting basis on {len(calib_batches)} batches…")
            _calibrate(student, calib_batches, ch, sync_indices, device)
        kl_m = {"kl": math.nan, "top1": math.nan, "ppl": math.nan}
        if run_kl:
            _log(rank, f"[run] channel={ch.name} — KL pass…")
            kl = _kl_pass(student, teacher, batches, ch, sync_indices, args.chunk_size, device)
            kl_m = {
                "kl": kl["kl_forward"].item(),
                "top1": kl["top1_agree"].item(),
                "ppl": math.exp(kl["student_nll"].item()),
            }
        ds_score, per_task = (math.nan, {})
        if run_ds:
            _log(rank, f"[run] channel={ch.name} — downstream…")
            ds_score, per_task = _downstream_pass(student, tok, ch, sync_indices, args, rank, device)
        results[ch.name] = {**kl_m, "ds": ds_score, "per_task": per_task}

    if rank == 0:
        zero = results.get("zero")
        orac = results.get("oracle")

        def _head(metric: str, name: str, lower_better: bool) -> str:
            if zero is None or orac is None or name in ("zero", "oracle"):
                return "    —"
            v, vz, vo = results[name][metric], zero[metric], orac[metric]
            denom = (vz - vo) if lower_better else (vo - vz)
            if abs(denom) < 1e-9 or any(math.isnan(x) for x in (v, vz, vo)):
                return "    —"
            frac = ((vz - v) / denom) if lower_better else ((v - vz) / denom)
            return f"{100 * frac:6.1f}%"

        print()
        print("===== Cross-track intervention probe "
              f"({len(batches)} batches, {len(sync_layers)} syncs, ckpt={args.checkpoint_dir}) =====")
        have_ds = run_ds
        have_kl = run_kl
        print(f"{'channel':>14} "
              + (f"{'KL':>9} {'top1':>7} {'ppl':>9} " if have_kl else "")
              + (f"{'downstr':>9} {'%head(ds)':>10} " if have_ds else "")
              + (f"{'%head(KL)':>10}" if have_kl else ""))
        for ch in channels:
            r = results[ch.name]
            row = f"{ch.name:>14} "
            if have_kl:
                row += f"{r['kl']:>9.4f} {r['top1']:>7.4f} {r['ppl']:>9.3f} "
            if have_ds:
                ds = "    nan" if math.isnan(r["ds"]) else f"{r['ds']:>9.4f}"
                row += f"{ds:>9} {_head('ds', ch.name, lower_better=False):>10} "
            if have_kl:
                row += f"{_head('kl', ch.name, lower_better=True):>10}"
            print(row)
        if have_ds:
            print()
            print("  per-task downstream (acc_norm/acc):")
            for ch in channels:
                pt = results[ch.name]["per_task"]
                if pt:
                    print(f"    {ch.name:>14}: " + "  ".join(f"{k}={v:.4f}" for k, v in pt.items()))
        print()
        print("  Read: %head = share of the zero→oracle headroom recovered (oracle=D1 ceiling, "
              "zero=D2 floor). oracle≈zero ⇒ no cross-track headroom; oracle≫zero but every cheap "
              "channel≈zero ⇒ content unreachable cheaply.")
        print()

    if teacher is not None:
        teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
