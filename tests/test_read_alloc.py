"""Answer-letter KL, the read-channel trace, and per-layer read allocation."""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(path, rows):
    path.write_text(json.dumps({"student": {"samples": {"mmlu_math_mc": [
        {"doc_id": i, "acc": acc, "ll": ll} for i, (ll, acc) in enumerate(rows)]}}}))
    return str(path)


def test_letter_kl_is_zero_for_identical_and_matches_a_hand_value():
    lk = _script("letter_kl")
    assert lk.letter_kl([-1.0, -2.0, -3.0, -4.0], [-1.0, -2.0, -3.0, -4.0]) == pytest.approx(0.0)
    # Shift-invariant: a constant on every option is the same distribution.
    assert lk.letter_kl([-1.0, -2.0], [-11.0, -12.0]) == pytest.approx(0.0, abs=1e-12)
    p = [0.5, 0.5]
    q = [math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.0)),
         math.exp(-2.0) / (math.exp(-0.1) + math.exp(-2.0))]
    want = sum(a * math.log(a / b) for a, b in zip(p, q))
    assert lk.letter_kl([-1.0, -1.0], [-0.1, -2.0]) == pytest.approx(want, rel=1e-12)


def test_letter_kl_cli_reads_dumps_and_refuses_one_without_ll(tmp_path, capsys):
    lk = _script("letter_kl")
    ref = _dump(tmp_path / "ref.json", [([-1.0, -2.0], 1.0), ([-2.0, -1.0], 0.0)])
    arm = _dump(tmp_path / "arm.json", [([-1.0, -2.0], 1.0), ([-1.0, -2.0], 1.0)])
    assert lk.main(["--cut", "1", "ref=" + ref, "arm=" + arm]) == 0
    out = capsys.readouterr().out
    assert "lost   0 won   1" in out
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"student": {"samples": {"mmlu_math_mc": [
        {"doc_id": 0, "acc": 1.0}]}}}))
    with pytest.raises(SystemExit, match="no per-option"):
        lk.main(["ref=" + ref, "bare=" + str(bare)])


# ----- the read-channel trace -----

N_TINY = 4


@pytest.fixture(scope="module")
def tiny():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=N_TINY, num_key_value_heads=2, head_dim=16, vocab_size=128,
                      max_position_embeddings=64, rms_norm_eps=1e-6, tie_word_embeddings=False)
    torch.manual_seed(3)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight"):
                p.normal_(mean=1.0, std=0.2)
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(1))
    return model, ids, [5, 17, 33, 90]


def test_trace_zero_leaves_are_bit_identical_and_letters_centre(tiny):
    import torch
    tr = _script("trace_read_channels")
    model, ids, letters = tiny
    with torch.no_grad():
        ref = tr.letter_logp(model, ids, letters)
    lp, a = tr.item_attributions(model, ids, letters, [0, 1, 2])
    assert torch.equal(lp, ref)
    assert a.shape == (4, 3, 128)
    # log-softmax over the letters: Σ_y p_y Δlog p_y ≡ 0, so the p-weighted attributions vanish.
    assert float((lp.exp()[:, None, None] * a).sum(0).abs().max()) < 1e-5 * float(a.abs().max())


def test_trace_attribution_predicts_a_real_dropped_channel_set(tiny):
    """Leave channels out of layer 1's read only (attention 2's input, not the residual):
    the measured change in log p matches −Σ_{dropped} a to first order."""
    import torch
    tr = _script("trace_read_channels")
    model, ids, letters = tiny
    lp, a = tr.item_attributions(model, ids, letters, [0, 1, 2])
    drop = torch.randperm(128, generator=torch.Generator().manual_seed(7))[:80]
    layers, alpha, h = model.model.layers, 1e-2, {}
    hd = layers[1].mlp.down_proj.register_forward_pre_hook(lambda _m, x: h.update(x=x[0][0]))
    with torch.no_grad():
        tr.letter_logp(model, ids, letters)
    hd.remove()
    W = layers[1].mlp.down_proj.weight
    err = -alpha * (h["x"][:, drop] @ W[:, drop].T)
    hd = layers[2].input_layernorm.register_forward_pre_hook(lambda _m, x: (x[0] + err,))
    with torch.no_grad():
        lq = tr.letter_logp(model, ids, letters)
    hd.remove()
    pred = -alpha * a[:, 1, drop].sum(-1)
    meas = lq - lp
    assert float(meas.abs().max()) > 0
    assert float((meas - pred).norm() / pred.norm()) < 0.05


def test_trace_channel_map_rail_on_real_sliced_tracks(tiny, tmp_path):
    import shutil

    from safetensors.torch import save_file

    from parallm.slicer.convert import slice_model_to_tracks
    tr = _script("trace_read_channels")
    model, _, _ = tiny
    shards, _ = slice_model_to_tracks(model.model, n_tracks=N_TINY, sync_block_depth=1,
                                      text_config_attr="config")
    for t, sd in enumerate(shards):
        save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / f"track_{t}.safetensors"))
    tr.check_channel_map(model, str(tmp_path), [0, 1, 2], N_TINY)
    shutil.copy(tmp_path / "track_0.safetensors", tmp_path / "track_1.safetensors")
    with pytest.raises(SystemExit, match="not contiguous"):
        tr.check_channel_map(model, str(tmp_path), [0, 1, 2], N_TINY)


def test_selection_cost_algebra():
    import torch
    tr = _script("trace_read_channels")
    g = torch.Generator().manual_seed(0)
    N, Y, nL, I = 5, 4, 2, 25600
    p = torch.softmax(torch.randn(N, Y, generator=g), -1)
    a = torch.randn(N, Y, nL, I, generator=g) * 1e-3
    a = a - (p[:, :, None, None] * a).sum(1, keepdim=True)        # centred, as a real trace is
    orders = [torch.randperm(I, generator=g) for _ in range(nL)]
    M = tr.missing_sums(a, orders)
    full = tr.config_kl(M, p, {0: 25600, 1: 25600})
    assert float(full.abs().max()) < 1e-12                         # full width ≡ exact
    none0 = tr.config_kl(M, p, {0: 0, 1: None})                     # layer 1 keeps its sync
    want = 0.5 * (p.double() * a[:, :, 0].double().sum(-1).pow(2)).sum(-1)
    assert torch.allclose(none0, want, rtol=1e-6)
    keep = orders[0][:2048]
    miss = torch.ones(I, dtype=torch.bool)
    miss[keep] = False
    want = 0.5 * (p.double() * a[:, :, 0][:, :, miss].double().sum(-1).pow(2)).sum(-1)
    assert torch.allclose(tr.config_kl(M, p, {0: 2048}), want, rtol=1e-5)
    phi = tr.fisher_score(a, p)
    assert torch.allclose(phi, (p[:, :, None, None] * a.pow(2)).sum(1).mean(0), rtol=1e-5)


# ----- allocation -----

def _curves(tmp_path, nL=3, N=3, seed=0):
    import torch

    from parallm.eval.read_cost import LADDER, missing_sums
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(N, 4, generator=g), -1)
    # Layers of very different sensitivity, energy-like orders (big |a| first, noisily).
    scale = torch.tensor([1e-4, 1e-3, 1e-2])[:nL]
    a = torch.randn(N, 4, nL, 25600, generator=g) * scale[None, None, :, None]
    a = a * torch.linspace(3, 0.1, 25600)[None, None, None, :]
    a = a - (p[:, :, None, None] * a).sum(1, keepdim=True)
    orders = [torch.arange(25600) for _ in range(nL)]
    M = missing_sums(a, orders)
    path = tmp_path / "curves.pt"
    torch.save({"ladder": LADDER, "read_layers": list(range(nL)), "doc_id": torch.arange(N), "p": p,
                "fisher_xfit": M}, path)
    return path, M, p


def test_layer_costs_are_the_single_layer_joint_kl(tmp_path):
    import torch

    from parallm.eval.read_cost import LADDER, config_kl
    al = _script("alloc_read")
    _, M, p = _curves(tmp_path)
    C = al.layer_costs(M, p, LADDER)
    assert float(config_kl(M[:, :, 1:2], p, {0: 512}).mean()) == pytest.approx(float(C[1, LADDER.index(512)]))
    assert torch.all(C[:, -1].abs() < 1e-12)                      # full width costs nothing
    assert al.schedule([0, 1, 2], [0, 512, None]) == "exact,0-1:post-attn"
    assert al.schedule([0, 1, 2], [None] * 3) == "exact"


def test_alloc_exact_sync_count_syncs_the_sensitive_layer_and_fits_the_joint_budget(tmp_path):
    from parallm.eval.read_cost import LADDER, config_kl
    al = _script("alloc_read")
    _, M, p = _curves(tmp_path)
    C = al.layer_costs(M, p, LADDER)
    budget = float(config_kl(M, p, {k: 2048 for k in range(3)}).mean())
    sizes = []
    for n in range(3):
        picks = al.solve_n(C, M, p, LADDER, budget, n)
        assert sum(r is al.SYNC for r in picks) == n
        assert float(config_kl(M, p, dict(enumerate(picks))).mean()) <= budget * (1 + 1e-9)
        sizes.append(sum(r for r in picks if r))
    assert picks[2] is al.SYNC and al.solve_n(C, M, p, LADDER, budget, 1)[2] is al.SYNC
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_alloc_channel_budget_fits_the_cap_and_never_loses_kl_to_a_bigger_cap(tmp_path):
    from parallm.eval.read_cost import LADDER, config_kl
    al = _script("alloc_read")
    _, M, p = _curves(tmp_path)
    C = al.layer_costs(M, p, LADDER)
    for n in range(3):
        kls = []
        for cap in (512, 4096, 32768):
            picks = al.solve_channels(C, M, p, LADDER, cap, n)
            assert sum(r is al.SYNC for r in picks) == n and sum(r for r in picks if r) <= cap
            kls.append(float(config_kl(M, p, dict(enumerate(picks))).mean()))
        assert kls[0] >= kls[1] >= kls[2]
    assert al.solve_channels(C, M, p, LADDER, 4096, 1)[2] is al.SYNC


def test_alloc_cli_emits_the_config(tmp_path):
    al = _script("alloc_read")
    path, _, _ = _curves(tmp_path)
    out = tmp_path / "alloc.json"
    assert al.main(["--curves", str(path), "--channels", "4096", "--syncs", "1-3",
                    "--emit", "3", "--out", str(out)]) == 0
    alloc = json.loads(out.read_text())
    assert alloc["sync_phase"] == "exact" and alloc["r"] == {}
    assert al.main(["--curves", str(path), "--channels", "4096", "--syncs", "1",
                    "--emit", "1", "--out", str(out)]) == 0
    alloc = json.loads(out.read_text())
    assert alloc["sync_phase"] == "exact,0-1:post-attn" and sum(alloc["r"].values()) <= 4096
    with pytest.raises(SystemExit, match="not among"):
        al.main(["--curves", str(path), "--channels", "4096", "--syncs", "1",
                 "--emit", "2", "--out", str(out)])
    with pytest.raises(SystemExit, match="exactly one"):
        al.main(["--curves", str(path), "--syncs", "1"])


def test_builder_cuts_each_layer_s_top_r_by_score(tiny, tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    from parallm.slicer.convert import slice_model_to_tracks
    model, _, _ = tiny
    shards, _ = slice_model_to_tracks(model.model, n_tracks=N_TINY, sync_block_depth=1,
                                      text_config_attr="config")
    for t, sd in enumerate(shards):
        save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / f"track_{t}.safetensors"))
    inter = 128 // N_TINY
    score = {f"layer.{i}": torch.rand(N_TINY, inter, generator=torch.Generator().manual_seed(i))
             for i in range(4)}
    save_file(score, str(tmp_path / "score.safetensors"))
    (tmp_path / "alloc.json").write_text(json.dumps({"r": {"0": 64, "1": 128, "2": 32}}))
    b = _script("build_trunc_estimator")
    assert b.main(["--checkpoint-dir", str(tmp_path), "--score", str(tmp_path / "score.safetensors"),
                   "--alloc", str(tmp_path / "alloc.json"), "--out", str(tmp_path / "a.safetensors")]) == 0
    got = load_file(str(tmp_path / "a.safetensors"))
    for li, r in ((0, 64), (1, 128), (2, 32)):
        top = torch.argsort(score[f"layer.{li}"].reshape(-1), descending=True)[:r]
        want = torch.stack([top // inter, top % inter], 1).int()
        assert torch.equal(got[f"layer.{li}.index"], want)
        for col, (t, j) in enumerate(want.tolist()):
            sd = shards[t]
            assert torch.equal(got[f"layer.{li}.gate"][:, col], sd[f"layers.{li}.mlp.gate_proj.weight"][j])
            assert torch.equal(got[f"layer.{li}.up"][:, col], sd[f"layers.{li}.mlp.up_proj.weight"][j])
            assert torch.equal(got[f"layer.{li}.down"][col], sd[f"layers.{li}.mlp.down_proj.weight"][:, j])
    from parallm.model.mlp_read import load_mlp_read_base
    base = load_mlp_read_base(str(tmp_path / "a.safetensors"), None, [0], N_TINY, "cpu")
    assert base.rank == "32-128" and base.ranks == {0: 64, 1: 128, 2: 32}
