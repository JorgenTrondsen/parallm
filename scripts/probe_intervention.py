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
    PhasedMode,
    collect_input_covs,
    collect_input_norms,
    intervention_forward,
    parse_channel,
    phased_intervention_forward,
    seam_intervention_forward,
    seam_predictability_analysis,
)
from pt_converter.eval.refine import RefineSpec, refine_forward, refine_intervention_forward
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
from pt_converter.utils.checkpoint import load_cross_head, load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _print_seam_analysis(res: "dict[int, dict]", ckpt: str) -> None:
    """Per-layer predictability table + a verdict aggregate. All columns are relMSE
    (1.0 = predicting zero) except ``d<-hln`` (fraction of the 1-token drift that
    h_ln linearly explains = the complementarity signal) and the SVD energy fractions."""
    layers = sorted(res)
    print()
    print(f"===== Seam predictability of ΣY (the chp target), per layer  ckpt={ckpt} =====")
    print("  relMSE (1=predict-zero): stale=1-tok cache | hln=best linear h_ln→ΣY | "
          "hybrid=stale+linear-h_ln-correction | d<-hln=frac of drift h_ln explains")
    print(f"{'L':>3} {'stale':>7} {'avg4':>7} {'avg16':>7} {'hln':>7} {'hybrid':>7} "
          f"{'d<-hln':>7} | {'ΣY r64':>7} {'ΣY r256':>8} {'drift r64':>9} {'drift r256':>10}")
    import statistics
    agg = {k: [] for k in ("stale", "hln", "hybrid", "drift_by_hln")}
    sv = {"sumy64": [], "sumy256": [], "drift64": [], "drift256": []}
    for L in layers:
        r = res[L]
        a4 = r["avg"].get(4, float("nan"))
        a16 = r["avg"].get(16, float("nan"))
        s64 = r["svd_sumy"].get(64, float("nan"))
        s256 = r["svd_sumy"].get(256, float("nan"))
        d64 = r["svd_drift"].get(64, float("nan"))
        d256 = r["svd_drift"].get(256, float("nan"))
        print(f"{L:>3} {r['stale']:>7.3f} {a4:>7.3f} {a16:>7.3f} {r['hln']:>7.3f} "
              f"{r['hybrid']:>7.3f} {r['drift_by_hln']:>7.3f} | "
              f"{s64:>7.3f} {s256:>8.3f} {d64:>9.3f} {d256:>10.3f}")
        for k in agg:
            agg[k].append(r[k])
        sv["sumy64"].append(s64); sv["sumy256"].append(s256)
        sv["drift64"].append(d64); sv["drift256"].append(d256)
    mean = lambda xs: statistics.fmean([x for x in xs if x == x]) if xs else float("nan")
    print()
    print("  layer-mean: "
          + "  ".join(f"{k}={mean(v):.3f}" for k, v in agg.items())
          + f"  | ΣY-energy@r64={mean(sv['sumy64']):.3f} r256={mean(sv['sumy256']):.3f}"
          + f"  drift-energy@r64={mean(sv['drift64']):.3f} r256={mean(sv['drift256']):.3f}")
    print("  Read: hybrid ≪ min(stale,hln) ⇒ h_ln & stale are COMPLEMENTARY ⇒ a better predictor is "
          "worth building. hybrid ≈ stale (d<-hln≈0) AND drift-energy spread over many ranks ⇒ the "
          "drift is high-rank/orthogonal ⇒ the wall ⇒ go to D>1. Compare hybrid to the trained chp "
          "(~0.43/block): hybrid<chp ⇒ training/arch headroom; hybrid≈chp ⇒ predictor already optimal.")
    print()


def _kl_pass(student, teacher, batches, channel, sync_indices, chunk_size, device, fwd):
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
        hidden = fwd(student, input_ids, attn, sync_indices, channel)
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
def _calibrate(student, calib_batches, channel, sync_indices, device, fwd):
    """Fit a CalibratedFixedLowRankChannel's frozen basis over a calibration set.

    Runs the intervention forward with the channel in observing mode (it records the
    D=2-trajectory ``other_k`` per layer/track and returns zeros), then finalizes the
    per-(layer,track) PCA bases. Every rank runs it in lockstep (the forward all-reduces)."""
    channel.start_observing()
    for batch in calib_batches:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch.get("attention_mask")
        attn = attn.to(device, non_blocking=True) if attn is not None else None
        fwd(student, input_ids, attn, sync_indices, channel)
    channel.finalize()


def _downstream_pass(student, tok, channel, sync_indices, args, rank, device, fwd):
    """lm-eval downstream accuracy of the intervention forward (broadcast score)."""
    from lm_eval import simple_evaluate

    from pt_converter.eval.downstream import _pick_metric, aggregate_downstream_score
    from pt_converter.eval.lm_eval_adapter import PTLM, is_lm_head_owner

    tasks = [t.strip() for t in args.downstream_tasks.split(",") if t.strip()]

    def _fn(input_ids, attention_mask):
        hidden = fwd(student, input_ids, attention_mask, sync_indices, channel)
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
    p.add_argument("--seam", action="store_true",
                   help="Intra-block (post-attention) intervention: substitute the missing cross-HEAD "
                        "attention output Sigma_others(Y) at the MLP input of every layer, instead of the "
                        "whole-layer residual substitution. For the D=1 study (oracle==dense teacher, "
                        "zero==current D=1). Channels reuse the same specs on other_k = SigmaY - Y_self.")
    p.add_argument("--sync-phase", choices=["boundary", "post-attn"], default="boundary",
                   help="'post-attn' runs the PHASED intervention forward (mirrors the lever-B deployed "
                        "regime: post-attn sync at every boundary except the last layer, post-MLP at the "
                        "last, non-boundary layers fully partial) and interprets --channels as phased "
                        "modes: zero (deployed floor), oracle (perfect-delivery ceiling), kv (write-side "
                        "memory-correction gate: k/v + gated-delta write gate/decay from the exact full "
                        "residual), q (read-side complement, diagnostic), and replica:{exact|int8|int4|"
                        "svd:<r>|prune:<frac>} (degraded LOCAL-RECOMPUTATION estimator: shadow replay of "
                        "all tracks from the last synced residual with degraded weight copies; "
                        "prune:<frac> = magnitude-sparse copies, the precision-orthogonal cheapness that "
                        "composes with an already-quantized base; replica:exact must "
                        "reproduce oracle — the rail). Use on a post-attn-trained "
                        "checkpoint (or the untrained slice for de-confounded replica gates). "
                        "Default 'boundary' = the whole-layer/--seam harnesses.")
    p.add_argument("--refine-iters", default=None,
                   help="Comma-separated refinement pass counts x (e.g. '0,1,2'). Activates the "
                        "Jacobi/iterative-refinement harness (eval/refine.py) and REPLACES "
                        "--channels: pass 0 runs the full stack comm-free per track, then x "
                        "refinement passes each fed by ONE bulk all-layer exchange. Sync events "
                        "per token = x+1 (+ any --refine-base-syncs).")
    p.add_argument("--refine-carry", default="own-fresh",
                   help="Comma-separated carry rules for the refinement passes: 'own-fresh' (each "
                        "track keeps its own residual fresh and fills with others' previous-pass "
                        "content) and/or 'shared' (every sublayer computes on the previous pass's "
                        "reconstructed trunk). Cross-product with --refine-iters.")
    p.add_argument("--refine-base-syncs", default=None,
                   help="Optional comma-separated layer indices given REAL boundary syncs during "
                        "pass 0 (the hybrid arm; +1 event each; the last layer is subsumed by the "
                        "final combine).")
    p.add_argument("--refine-exactness-check", action="store_true",
                   help="One-batch correctness rail instead of the eval loop: relMSE between the "
                        "refine hidden at each requested --refine-iters and the phased-oracle "
                        "hidden (perfect per-sublayer delivery == dense on a fresh slice). Run "
                        "with --refine-iters 63 (=2L-1, where the forward is provably exact) to "
                        "validate the NCCL path + bf16 drift before spending eval hours.")
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
    p.add_argument("--wanda-calib-batches", type=int, default=16,
                   help="For replica:wanda channels: batches run through the DENSE model to "
                        "collect per-input-channel L2 norms (the |w|*||x|| pruning criterion).")
    p.add_argument("--seam-analyze", action="store_true",
                   help="Training-free predictability decomposition of the missing ΣY at the D=1 seam "
                        "(NOT the channel/KL/downstream loop): per-layer ceilings for stale/avg/h_ln "
                        "prediction, the drift-vs-h_ln COMPLEMENTARITY, and SVD structure. Use the "
                        "trained seam stream if the checkpoint has a cross_head. Pair with "
                        "--num-batches 0 and no --downstream-tasks.")
    p.add_argument("--analyze-batches", type=int, default=8,
                   help="Calibration batches for --seam-analyze (token pool for the ridge fit + SVD).")
    p.add_argument("--analyze-ridge-lambda", type=float, default=1e-2,
                   help="Ridge regularization (relative to mean diag of FᵀF) for the linear ceilings.")
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
    elif args.refine_iters is not None:
        # Refinement has no boundary schedule; the wrapper just needs a valid one.
        sync_layers = [manifest.num_layers - 1]
    else:
        raise SystemExit(
            "[error] checkpoint manifest has no sync_layer_indices. Pass --sync-indices, or point "
            "--checkpoint-dir at a trained checkpoint."
        )
    if args.sync_phase == "post-attn" and (args.seam or args.oracle_sweep or args.seam_analyze):
        raise SystemExit(
            "[error] --sync-phase post-attn is its own harness (phased modes zero/oracle/kv/q) "
            "and is incompatible with --seam / --oracle-sweep / --seam-analyze."
        )
    if args.refine_iters is not None and (
        args.seam or args.oracle_sweep or args.seam_analyze or args.sync_phase == "post-attn"
    ):
        raise SystemExit(
            "[error] --refine-iters is its own harness (Jacobi iterative refinement) and is "
            "incompatible with --seam / --oracle-sweep / --seam-analyze / --sync-phase post-attn."
        )
    if args.refine_exactness_check and args.refine_iters is None:
        raise SystemExit("[error] --refine-exactness-check requires --refine-iters.")
    if args.seam_analyze:
        channels = []  # analyze mode bypasses the channel/KL/downstream loop
    elif args.refine_iters is not None:
        iters_list = [int(x) for x in args.refine_iters.split(",") if x.strip() != ""]
        carries = [c.strip() for c in args.refine_carry.split(",") if c.strip()]
        base_syncs = (
            tuple(int(x) for x in args.refine_base_syncs.split(",") if x.strip() != "")
            if args.refine_base_syncs else None
        )
        channels = [RefineSpec(x, c, base_syncs) for c in carries for x in iters_list]
    elif args.sync_phase == "post-attn":
        channels = [PhasedMode(s.strip()) for s in args.channels.split(",") if s.strip()]
    elif args.oracle_sweep:
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
    run_kl = args.num_batches > 0 and not args.seam_analyze
    run_ds = bool(args.downstream_tasks.strip()) and not args.seam_analyze
    if not run_kl and not run_ds and not args.seam_analyze and not args.refine_exactness_check:
        raise SystemExit(
            "[error] nothing to do: set --num-batches > 0 (KL vs teacher) and/or --downstream-tasks, "
            "or --seam-analyze. Downstream-only (--num-batches 0) skips the teacher entirely."
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

    # ----- Wanda/SparseGPT calibration: one dense-model pass collecting per-input-
    # channel L2 norms (wanda/qwanda/chanwanda) and/or input covariances H = XᵀX
    # (sparsegpt), then freed (each rank computes identical stats from the
    # identical seed-deterministic batches). -----
    def _needs_norms(ch) -> bool:
        # A wanda/lsparse-family base spec, or such a :mlp: sub-spec on any base.
        sub = getattr(ch, "mlp_mode", None)
        return (getattr(ch, "wanda", False) or getattr(ch, "chanwanda", False)
                or getattr(ch, "lsparse", False)
                or getattr(ch, "shared_lsparse", False)
                or getattr(sub, "wanda", False) or getattr(sub, "lsparse", False))

    need_norms = any(_needs_norms(ch) for ch in channels)
    need_covs = any(getattr(ch, "sparsegpt", False) for ch in channels)
    if need_norms or need_covs:
        wbatches = _materialize(args.wanda_calib_batches)
        _log(rank, f"[data] materialized {len(wbatches)} calibration batches "
                   f"(seq_len={args.seq_len})")
        _log(rank, "[init] loading dense model for copy calibration…")
        calib_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).eval()
        c_text = (
            calib_model.model.language_model
            if hasattr(calib_model.model, "language_model")
            else calib_model.model
        )
        calib_model = calib_model.to(device)
        if need_norms:
            norms = collect_input_norms(c_text, wbatches, device)
            for ch in channels:
                if _needs_norms(ch):
                    ch.set_input_norms(norms, manifest.n_tracks)
            _log(rank, f"[init] wanda calibration done ({len(norms)} input spaces).")
            for ch in channels:
                if getattr(ch, "shared_lsparse", False):
                    from pt_converter.eval.intervention import compute_shared_lsparse_slices

                    _log(rank, f"[init] {ch.name}: dense-level shared-L decomposition…")
                    ch.set_shared_slices(compute_shared_lsparse_slices(
                        c_text, student, ch.lsparse_rank, ch.wanda_frac, norms,
                        quant_bits=ch.pre_quant_bits,
                    ))
                    _log(rank, f"[init] {ch.name}: shared slices ready.")
        if need_covs:
            covs = collect_input_covs(
                c_text, wbatches, device,
                slice_keys={"self_attn.o_proj": manifest.n_tracks,
                            "linear_attn.out_proj": manifest.n_tracks,
                            "mlp.down_proj": manifest.n_tracks},
            )
            for ch in channels:
                if getattr(ch, "sparsegpt", False):
                    ch.set_input_covs(covs, manifest.n_tracks)
            _log(rank, f"[init] sparsegpt covariance calibration done ({len(covs)} input spaces).")
        del calib_model, c_text
        torch.cuda.empty_cache()
        for ch in channels:
            if getattr(ch, "profiled", False):
                from pt_converter.eval.intervention import allocate_layer_fracs

                fracs = allocate_layer_fracs(
                    student, ch, sync_layers, ch.wanda_frac,
                    group=layout.track_group if dist.get_world_size() > 1 else None,
                )
                ch.set_layer_fracs(fracs)
                _log(rank, f"[init] {ch.name} layer fracs: "
                           + ",".join(f"{f:.2f}" for f in fracs))

    sync_indices = tuple(sync_layers)

    # ----- Training-free predictability decomposition (bypasses the channel loop). -----
    if args.seam_analyze:
        abatches = _materialize(args.analyze_batches)
        _log(rank, f"[data] materialized {len(abatches)} analyze batches (seq_len={args.seq_len})")
        ch = load_cross_head(args.checkpoint_dir)
        if ch is not None:
            student.cross_head = ch.to(device).to(torch.bfloat16)
            _log(rank, f"[init] analysing the TRAINED seam stream (cross_head backend={ch.backend}).")
        else:
            _log(rank, "[init] no cross_head in checkpoint; analysing the plain-D1 stream.")
        res = seam_predictability_analysis(
            student, abatches, sync_indices, ridge_lambda=args.analyze_ridge_lambda,
        )
        if rank == 0:
            _print_seam_analysis(res, args.checkpoint_dir)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    # ----- Refine exactness rail (bypasses the channel/KL/downstream loop). -----
    if args.refine_exactness_check:
        xb = _materialize(1)[0]
        ids = xb["input_ids"].to(device, non_blocking=True)
        attn = xb.get("attention_mask")
        attn = attn.to(device, non_blocking=True) if attn is not None else None
        # Phased oracle = perfect per-sublayer delivery = the dense forward on a
        # fresh slice (the validated rail) — the fixed point refine converges to.
        ref = phased_intervention_forward(student, ids, attn, sync_indices, PhasedMode("oracle"))
        ref_sq = ref.float().pow(2).sum()
        for ch in channels:
            out = refine_forward(
                student, ids, attn,
                iters=ch.iters, carry=ch.carry, base_sync_indices=ch.base_sync_indices,
            )
            rel = ((out.float() - ref.float()).pow(2).sum() / ref_sq).item()
            _log(rank, f"[exactness] {ch.name}: relMSE vs phased-oracle(=dense on fresh slice) "
                       f"= {rel:.3e}" + ("  [OK]" if rel < 1e-3 else "  [DRIFT]"))
        dist.barrier()
        dist.destroy_process_group()
        return 0

    if args.refine_iters is not None:
        fwd = refine_intervention_forward
        _log(rank, "[init] REFINE mode (--refine-iters): Jacobi iterative refinement — pass 0 "
                   "comm-free, then x refinement passes each fed by one bulk all-layer exchange "
                   "(sync events/token = x+1).")
    elif args.sync_phase == "post-attn":
        fwd = phased_intervention_forward
        _log(rank, "[init] PHASED mode (--sync-phase post-attn): write-side memory-correction "
                   "gate (zero==deployed phased forward, oracle==perfect delivery).")
    else:
        fwd = seam_intervention_forward if args.seam else intervention_forward
        if args.seam:
            _log(rank, "[init] SEAM mode: intra-block post-attention substitution "
                       "(oracle==dense teacher, zero==current D=1).")
    results: dict[str, dict] = {}
    for ch in channels:
        if isinstance(ch, CalibratedFixedLowRankChannel):
            _log(rank, f"[run] channel={ch.name} — PCA-fitting basis on {len(calib_batches)} batches…")
            _calibrate(student, calib_batches, ch, sync_indices, device, fwd)
        kl_m = {"kl": math.nan, "top1": math.nan, "ppl": math.nan}
        if run_kl:
            _log(rank, f"[run] channel={ch.name} — KL pass…")
            kl = _kl_pass(student, teacher, batches, ch, sync_indices, args.chunk_size, device, fwd)
            kl_m = {
                "kl": kl["kl_forward"].item(),
                "top1": kl["top1_agree"].item(),
                "ppl": math.exp(kl["student_nll"].item()),
            }
        ds_score, per_task = (math.nan, {})
        if run_ds:
            _log(rank, f"[run] channel={ch.name} — downstream…")
            ds_score, per_task = _downstream_pass(student, tok, ch, sync_indices, args, rank, device, fwd)
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
