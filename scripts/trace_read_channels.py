"""Answer-letter trace of every MLP channel a local read might leave out, at every layer.

A post-attn-only layer ``i`` leaves its MLP sum unsynced; attention ``i+1`` reads it through
the local estimate, which carries a top-r subset ``S_i`` of the 25600 exact channels. Energy
(``Σ_t ‖h_c d_c‖²``) picks ``S_i`` and was never what accuracy needed: a 48% swing in energy
capture moved accuracy by nothing, and the band that mattered most held 31% of its energy.
This trace prices a channel by what it does to the ANSWER instead — the one signal that
tracked which math items a schedule flips (letter KL AUC .962, every hidden-state tap ≈ chance).

On the dense teacher (≡ the exact schedule), each layer ``i+1``'s ``input_layernorm`` gets a
zero leaf ``ε`` added to its input. ``Qwen3DecoderLayer`` keeps ``residual`` from before the
norm, so ``ε`` moves only the attention branch — exactly where a read enters (read-only,
never the carry). With ``p`` the softmax over the answer letters' logits and
``g_y = ∂ log p_y / ∂ε``, channel ``c`` of layer ``i`` gets

    a_{y,i,c} = Σ_t h_{i,c}(t) · (g_y(t) · d_{i,c})        (d = down_proj column)

the first-order change in ``log p_y`` if the read leaves it out. Any selection is then priced
in letter KL, ``K = ½ Σ_y p_y (Σ_i Σ_{c∉S_i} a_{y,i,c})²`` (the Fisher form, cross-layer terms
included), and ``φ_{i,c} = Σ_y p_y a²`` is an output-aware channel score. One error vector
stands for every track's read (a track's own non-base channels, 1/64, are ignored).

    python scripts/trace_read_channels.py requests --hf-model <teacher> --out-dir D
    python scripts/trace_read_channels.py trace --hf-model <teacher> --checkpoint-dir <convert> \\
        --out-dir D [--max-doc 300]
    python scripts/trace_read_channels.py summarize --out-dir D

``trace`` writes ``attr_*.safetensors`` (per item ``p`` [4], ``a`` [4, layers, I] bf16), after
a rail that the convert's track t channel j IS the teacher's channel t*I/N+j. ``summarize``
writes ``fisher_score.safetensors`` (``layer.{i}`` [tracks, channels], the ranking
`build_trunc_estimator.py --score` cuts bases from) and ``curves.pt`` (per item and letter,
each layer's missing sum at every ladder r under the cross-fitted Fisher ranking: an item is
priced with the order of the OTHER doc_id parity) that `alloc_read.py` prices configs with.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from parallm.eval.read_cost import LADDER, config_kl, fisher_score, missing_sums


# --------------------------------------------------------------------------- #
# the trace itself (model-agnostic over HF Qwen3-style decoder layers)
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def read_taps(model, read_layers):
    """Hooks for one forward: ``h[i]`` = layer i's down_proj input (detached) and, for each
    read layer i, a zero leaf ``eps[i]`` added to layer i+1's attention-branch input."""
    layers = model.model.layers
    h: dict[int, torch.Tensor] = {}
    eps: dict[int, torch.Tensor] = {}
    handles = []

    def save_h(i):
        def pre(_m, a):
            h[i] = a[0][0].detach()
        return pre

    def add_eps(i):
        def pre(_m, a):
            x = a[0]
            e = torch.zeros_like(x, requires_grad=True)
            eps[i] = e
            return (x + e,)
        return pre

    for i in read_layers:
        handles.append(layers[i].mlp.down_proj.register_forward_pre_hook(save_h(i)))
        handles.append(layers[i + 1].input_layernorm.register_forward_pre_hook(add_eps(i)))
    try:
        yield h, eps
    finally:
        for hd in handles:
            hd.remove()


def letter_logp(model, ids: torch.Tensor, letters: "list[int]") -> torch.Tensor:
    logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
    return F.log_softmax(logits.float()[letters], dim=-1)


def item_attributions(model, ids, letters, read_layers):
    """``(lp [Y], a [Y, len(read_layers), I] fp32)`` for one prompt, both on the CPU."""
    layers = model.model.layers
    with read_taps(model, read_layers) as (h, eps):
        lp = letter_logp(model, ids, letters)
    Y = lp.shape[0]
    leaves = [eps[i] for i in read_layers]
    grads = [torch.autograd.grad(lp[y], leaves, retain_graph=y < Y - 1) for y in range(Y)]
    out = []
    for k, i in enumerate(read_layers):
        W = layers[i].mlp.down_proj.weight                         # [H, I]
        G = torch.stack([g[k][0] for g in grads]).to(W.device)      # [Y, T, H]
        a = ((G.to(W.dtype) @ W).float() * h[i].float()).sum(1)    # [Y, I]
        out.append(a.cpu())                                        # layers live on different GPUs
    return lp.detach().cpu(), torch.stack(out, 1)


# --------------------------------------------------------------------------- #
# requests: the eval's own prompts
# --------------------------------------------------------------------------- #

def cmd_requests(args) -> None:
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    from parallm.eval.downstream import EVAL_TASK_PATH
    from parallm.eval.lm_eval_adapter import PTLM

    class _Recorder(PTLM):
        def __init__(self, tok):
            super().__init__(forward_fn=None, tokenizer=tok, max_length=args.max_length,
                             device="cpu")
            self.seen = []

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            for r in requests:
                ctx_enc, cont_enc = self._encode_pair(*r.args)
                self.seen.append((int(r.doc_id), int(r.idx), ctx_enc, cont_enc))
            return [(0.0, False)] * len(requests)

    lm = _Recorder(AutoTokenizer.from_pretrained(args.hf_model))
    simple_evaluate(model=lm, tasks=[args.task], task_manager=TaskManager(include_path=EVAL_TASK_PATH),
                    limit=args.limit, log_samples=False, random_seed=args.seed,
                    numpy_random_seed=args.seed, torch_random_seed=args.seed,
                    fewshot_random_seed=args.seed)
    docs: dict[int, dict] = {}
    for doc_id, idx, ctx_enc, cont_enc in lm.seen:
        if len(cont_enc) != 1:
            raise SystemExit(f"doc {doc_id} choice {idx}: continuation is {len(cont_enc)} tokens")
        d = docs.setdefault(doc_id, {"doc_id": doc_id, "ctx": ctx_enc, "letters": {}})
        if d["ctx"] != ctx_enc:
            raise SystemExit(f"doc {doc_id}: choices do not share one context")
        d["letters"][idx] = cont_enc[0]
    # PTLM keeps the continuation and truncates the context from the left.
    out = [{"doc_id": i, "ctx": docs[i]["ctx"][-(args.max_length - 1):],
            "letters": [docs[i]["letters"][k] for k in sorted(docs[i]["letters"])]}
           for i in sorted(docs)]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(out, Path(args.out_dir) / "requests.pt")
    lens = [len(r["ctx"]) for r in out]
    print(f"[requests] {len(out)} docs, ctx tokens {min(lens)}-{max(lens)} -> {args.out_dir}/requests.pt")


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #

def check_channel_map(model, convert_dir: str, read_layers, n_tracks: int) -> None:
    """Rail: track t's local channel j IS the teacher's channel t*per+j (contiguous slicing).
    Every row at the first and the middle read layer, three rows per track elsewhere."""
    from safetensors import safe_open

    layers = model.model.layers
    inter = layers[0].mlp.gate_proj.weight.shape[0]
    per = inter // n_tracks
    full = {read_layers[0], read_layers[len(read_layers) // 2]}
    for t in range(n_tracks):
        with safe_open(f"{convert_dir}/track_{t}.safetensors", "pt") as f:
            for i in read_layers:
                sl = f.get_slice(f"layers.{i}.mlp.gate_proj.weight")
                W = layers[i].mlp.gate_proj.weight
                for j in ((0,) if i in full else (0, per // 2, per - 1)):
                    n = per if i in full else 1
                    mine = sl[j:j + n].to(W.device, W.dtype)
                    if not torch.equal(mine, W[t * per + j:t * per + j + n]):
                        raise SystemExit(f"[map] layer {i} track {t} channel {j} is not teacher "
                                         f"channel {t * per + j} — slicing is not contiguous")
    print(f"[map] contiguous (track t, channel j) = teacher t*{per}+j: OK over {n_tracks} tracks, "
          f"{len(read_layers)} layers", flush=True)


def cmd_trace(args) -> None:
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    reqs = [r for r in torch.load(Path(args.out_dir) / "requests.pt") if r["doc_id"] < args.max_doc]
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = len(model.model.layers)
    read_layers = list(range(L - 1))
    check_channel_map(model, args.checkpoint_dir, read_layers, args.n_tracks)
    dev0 = model.get_input_embeddings().weight.device
    out = Path(args.out_dir)
    done = {int(x) for f in sorted(out.glob("attr_*.safetensors")) for x in _doc_ids(f)}
    buf = {"a": [], "p": [], "doc_id": []}

    def flush():
        if not buf["doc_id"]:
            return
        name = out / f"attr_{buf['doc_id'][0]:04d}.safetensors"
        save_file({"a": torch.stack(buf["a"]).to(torch.bfloat16).contiguous(),
                   "p": torch.stack(buf["p"]).contiguous(),
                   "doc_id": torch.tensor(buf["doc_id"])}, str(name),
                  metadata={"read_layers": json.dumps(read_layers)})
        for v in buf.values():
            v.clear()

    for k, r in enumerate(reqs):
        if r["doc_id"] in done:
            continue
        ids = torch.tensor([r["ctx"]], device=dev0)
        if k == 0:  # rail: zero leaves leave the letters bit-identical
            with torch.no_grad():
                ref = letter_logp(model, ids, r["letters"]).cpu()
        lp, a = item_attributions(model, ids, r["letters"], read_layers)
        if k == 0:
            if not torch.equal(lp, ref):
                raise SystemExit("[rail] the zero-ε hooks changed the answer letters")
            p = lp.exp()
            centred = float((p[:, None, None] * a).sum(0).abs().max() / a.abs().max())
            print(f"[rail] zero-ε letters bit-identical; Σ_y p_y a_y / max|a| = {centred:.2e}", flush=True)
        buf["a"].append(a)
        buf["p"].append(lp.exp())
        buf["doc_id"].append(r["doc_id"])
        if len(buf["doc_id"]) == args.chunk:
            flush()
        if k % 25 == 0:
            print(f"[trace] {k + 1}/{len(reqs)} doc {r['doc_id']} T={ids.shape[1]} "
                  f"p={[round(float(x), 3) for x in lp.exp()]}", flush=True)
    flush()


def _doc_ids(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return f.get_tensor("doc_id").tolist()


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #

def load_attr(out_dir: str):
    from safetensors import safe_open
    a, p, ids, read_layers = [], [], [], None
    for f in sorted(Path(out_dir).glob("attr_*.safetensors")):
        with safe_open(str(f), "pt") as s:
            a.append(s.get_tensor("a"))
            p.append(s.get_tensor("p"))
            ids.append(s.get_tensor("doc_id"))
            read_layers = json.loads(s.metadata()["read_layers"])
    if not a:
        raise SystemExit(f"no attr_*.safetensors in {out_dir}")
    return torch.cat(a), torch.cat(p), torch.cat(ids), read_layers


def cmd_summarize(args) -> None:
    from safetensors.torch import save_file

    a, p, ids, read_layers = load_attr(args.out_dir)
    N, Y, nL, I = a.shape
    out = Path(args.out_dir)
    odd = (ids % 2 == 1)
    phi = {"all": fisher_score(a, p), "odd": fisher_score(a[odd], p[odd]),
           "even": fisher_score(a[~odd], p[~odd])}
    save_file({f"layer.{i}": phi["all"][k].reshape(args.n_tracks, -1).contiguous()
               for k, i in enumerate(read_layers)},
              str(out / "fisher_score.safetensors"), metadata={"items": str(N)})
    # Cross-fitted: an item is priced with the OTHER parity's order.
    M = torch.empty(N, Y, nL, len(LADDER), dtype=torch.float32)
    for mask, src in ((odd, "even"), (~odd, "odd")):
        orders = [torch.argsort(phi[src][k], descending=True) for k in range(nL)]
        M[mask] = missing_sums(a[mask], orders)
    torch.save({"ladder": LADDER, "read_layers": read_layers, "doc_id": ids, "p": p,
                "fisher_xfit": M}, out / "curves.pt")

    print(f"[summary] {N} items, {nL} read layers, ladder {LADDER}")
    print(f"{'layer':>5} " + " ".join(f"{'K r' + str(r):>10}" for r in (0, 512, 2048, 6144)))
    for k, i in enumerate(read_layers):
        print(f"{i:>5} " + " ".join(
            f"{float(config_kl(M[:, :, k:k + 1], p, {0: r}).mean()):10.3e}" for r in (0, 512, 2048, 6144)),
            flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["requests", "trace", "summarize"])
    p.add_argument("--hf-model")
    p.add_argument("--checkpoint-dir", help="trace: the convert, for the channel-map rail")
    p.add_argument("--n-tracks", type=int, default=64)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task", default="mmlu_math_mc")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-doc", type=int, default=300, help="trace docs with doc_id below this")
    p.add_argument("--chunk", type=int, default=25)
    args = p.parse_args(argv)
    {"requests": cmd_requests, "trace": cmd_trace, "summarize": cmd_summarize}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
