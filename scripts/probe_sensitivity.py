"""torchrun entry point: per-layer partial-residual sensitivity probe.

Phase-1 diagnostic for the uniform-D≥2 recovery work. Loads a trained (or
freshly sliced) PT student + the dense teacher, then for each requested uniform
sync depth measures, **per layer**, how wrong each track's un-synced residual is
and why — see ``pt_converter.eval.sensitivity``. Use it to decide which
structural recovery lever (residual-gain / head-scattering / intra-window
per-layer MSE) is worth implementing, before writing one.

The probe is independent of the checkpoint's own manifest schedule: the student
weights come from ``--checkpoint-dir`` (point it at the recovered D=1 ``best/``
to probe near-dense weights, or a raw ``convert_out`` dir for the untrained
floor), while the windows are generated from ``--probe-depths`` (uniform, must
divide num_layers). The teacher is hooked at EVERY layer.

    torchrun --standalone --nproc-per-node=8 scripts/probe_sensitivity.py \\
        --hf-model <teacher path> \\
        --checkpoint-dir train_out/qwen3_5_9b_n16_d1/best \\
        --probe-depths 2,4 --num-batches 20
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_converter.dist.fsdp_setup import wrap_student_with_fsdp, wrap_teacher_with_fsdp
from pt_converter.dist.groups import build_groups
from pt_converter.eval.sensitivity import partial_residual_probe
from pt_converter.model.pt_model import PTWrappedModel
from pt_converter.slicer.convert import _resolve_sync_schedule
from pt_converter.train.data import CalibrationDataConfig, PackedTokenStream
from pt_converter.train.teacher import HookedTeacher
from pt_converter.utils.checkpoint import load_manifest, load_track


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="Dense teacher model path (same as training)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Per-rank checkpoint dir (track_*.safetensors + manifest.json). Supplies "
                        "the student WEIGHTS only; the probe schedule comes from --probe-depths. "
                        "Point at a recovered D=1 best/ (near-dense) or a raw convert_out dir "
                        "(untrained floor).")
    p.add_argument("--probe-depths", default="2,4",
                   help="Comma-separated uniform sync depths to probe (each must divide num_layers).")
    p.add_argument("--dataset-name", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--split", default="validation")
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--svd-energy", action="store_true",
                   help="Also report the SVD cumulative-energy spectrum of the cross-track delta "
                        "entering each partial-read layer (low-rank cross-track-exchange viability "
                        "gate): the fraction of energy a rank-r projection recovers. Adds one SVD "
                        "per partial-read layer per batch.")
    args = p.parse_args()

    depths = [int(d) for d in args.probe_depths.split(",") if d.strip()]

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

    manifest = load_manifest(args.checkpoint_dir)
    layout = build_groups(n_tracks=manifest.n_tracks)
    num_layers = manifest.num_layers
    layer_types = manifest.layer_types
    _log(rank, f"[init] world={layout.world_size} n_tracks={manifest.n_tracks} "
               f"K={layout.tracks_per_rank} num_layers={num_layers}")

    # Validate the probe depths up front (uniform rule).
    test_schedules: dict[int, tuple[int, ...]] = {}
    for d in depths:
        test_schedules[d] = tuple(_resolve_sync_schedule(num_layers, d))

    # ----- Teacher hooked at EVERY layer (the probe needs a hidden per depth). -----
    cfg = AutoConfig.from_pretrained(args.hf_model)
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
        text_model=text_model,
        lm_head=teacher_model.lm_head,
        sync_layer_indices=list(range(num_layers)),
    )
    teacher_model = teacher_model.to(torch.cuda.current_device())
    wrap_teacher_with_fsdp(text_model, teacher_model.lm_head)

    # ----- Student (legacy non-vocab-parallel path; we only read hidden states). -----
    _log(rank, f"[init] building PT student for tracks {layout.local_track_ids}…")
    student = PTWrappedModel(
        text_config=cfg.text_config,
        n_tracks=manifest.n_tracks,
        local_track_ids=layout.local_track_ids,
        # The probe builds its own windows from --probe-depths, so the student's
        # own schedule is irrelevant; fall back to a final-layer placeholder when
        # the manifest carries none (a raw convert output).
        sync_after_layers=(manifest.sync_layer_indices or [num_layers - 1]),
        track_group=layout.track_group,
    )
    track_states = {tid: load_track(args.checkpoint_dir, tid) for tid in layout.local_track_ids}
    student.load_track_state_dicts(track_states, strict=True)
    student = student.to(torch.cuda.current_device()).to(torch.bfloat16)
    wrap_student_with_fsdp(student, layout)
    student.eval()

    tok = AutoTokenizer.from_pretrained(args.hf_model)

    # Accumulate per-depth, per-layer metric sums across batches.
    metric_keys = ("rel_err_partial", "rel_err_synced", "delta_imbalance", "gain_cos",
                   "delta_norm_mean", "delta_staleness_ratio")
    sums: dict[int, dict[int, dict[str, float]]] = {
        d: {L: {k: 0.0 for k in metric_keys} for L in range(num_layers)} for d in depths
    }
    SVD_R_GRID = (8, 16, 32, 64, 128, 256)
    svd_sums = (
        {d: {L: {r: 0.0 for r in SVD_R_GRID} for L in range(num_layers)} for d in depths}
        if args.svd_energy else None
    )
    n_batches = 0

    # One data pass; probe every depth on the same batch so the comparison is paired.
    ds = PackedTokenStream(
        tok,
        CalibrationDataConfig.single(
            dataset_name=args.dataset_name, dataset_config=args.dataset_config,
            split=args.split, seq_len=args.seq_len,
        ),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    for batch in loader:
        if n_batches >= args.num_batches:
            break
        batch = {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        for d in depths:
            res = partial_residual_probe(
                student, teacher, batch, test_schedules[d],
                svd_energy=args.svd_energy, svd_r_grid=SVD_R_GRID,
            )
            for L, m in res.items():
                for k in metric_keys:
                    sums[d][L][k] += m[k]
                if args.svd_energy:
                    for r in SVD_R_GRID:
                        svd_sums[d][L][r] += m[f"svd_r{r}"]
        n_batches += 1

    if rank == 0:
        denom = max(1, n_batches)
        for d in depths:
            ranges = _block_window_index(num_layers, test_schedules[d])
            print()
            print(f"===== Partial-residual probe @ uniform D={d} "
                  f"({n_batches} batches, {args.dataset_name}/{args.split}, "
                  f"weights={args.checkpoint_dir}) =====")
            print(f"{'layer':>5} {'ty':>2} {'pos':>3} {'rel_err_partial':>16} "
                  f"{'rel_err_synced':>15} {'delta_imbalance':>16} {'gain_cos':>9} "
                  f"{'stale_ratio':>11}")
            for L in range(num_layers):
                ty = "F" if layer_types[L] == "full_attention" else "L"
                m = sums[d][L]
                print(f"{L:>5} {ty:>2} {ranges[L]:>3} "
                      f"{m['rel_err_partial'] / denom:>16.4e} "
                      f"{m['rel_err_synced'] / denom:>15.4e} "
                      f"{m['delta_imbalance'] / denom:>16.4f} "
                      f"{m['gain_cos'] / denom:>9.4f} "
                      f"{m['delta_staleness_ratio'] / denom:>11.4f}")
            if args.svd_energy:
                print()
                print(f"===== Cross-track-delta SVD cumulative energy @ uniform D={d} "
                      f"(fraction of the missing cross-track content a rank-r exchange "
                      f"recovers; partial-read layers) =====")
                print(f"{'layer':>5} {'ty':>2} {'pos':>3}"
                      + "".join(f"{'r' + str(r):>9}" for r in SVD_R_GRID))
                for L in range(num_layers):
                    ty = "F" if layer_types[L] == "full_attention" else "L"
                    print(f"{L:>5} {ty:>2} {ranges[L]:>3}"
                          + "".join(f"{svd_sums[d][L][r] / denom:>9.4f}" for r in SVD_R_GRID))
        print()

    teacher.remove_hooks()
    dist.destroy_process_group()
    return 0


def _block_window_index(num_layers: int, sync_indices: tuple[int, ...]) -> dict[int, int]:
    """Map each layer → its 1-based position within its sync window."""
    from pt_converter.train.distill import _block_ranges

    pos: dict[int, int] = {}
    for start, end in _block_ranges(num_layers, sync_indices):
        for i, L in enumerate(range(start, end + 1)):
            pos[L] = i + 1
    return pos


if __name__ == "__main__":
    raise SystemExit(main())
