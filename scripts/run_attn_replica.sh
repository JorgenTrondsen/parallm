#!/usr/bin/env bash
# Score ONE attention-replica arm and pair it against a reference dump.
#
# Default arm = the frontier: q rank-1024 pruned 50%, k/v full rank 60%, o FOLDED into each
# track's own gate/up and pruned 60%, rank-32 norm sketch with the bias constant. 0.952 GiB,
# macro4 0.7628 — the teacher (+0.0008) and the exact 5.625 GiB replica (−0.0042).
#
# REF pairs the new dump against a stored one. A re-run of the same arm is BIT-EXACT on this
# harness, so the bar is per-doc identity; McNemar is the fallback that says how far off a
# non-identical run is (it resolves ~0.010 and nothing smaller).
set -euo pipefail
cd "$(dirname "$0")/.."

CONVERT=${CONVERT:-convert_out/qwen3/32b_n64_even}
TEACHER=${TEACHER:-/mnt/nas2/jtr020/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}
OUT=${OUT:-logs/qwen3/attn_replica}
CELL=${CELL:-0-31:exact,32-63:post-mlp}
LAYERS=${LAYERS:-32-63}
LIMIT=${LIMIT:-1000}
NPROC=${NPROC:-8}
ARM=${ARM:-q:1024/s50,k:s60,v:s60,o:fold/s60,norm:r32+c}
# A pruned q/k/v reads x_rms, a pruned o or fold reads o_rms, a pruned factor A reads z_rms,
# and a `+c` scalar reads norm_c — each kept apart from the multi-GiB bases file.
BASES=${BASES:-"$OUT/bases.safetensors,$OUT/x_rms.safetensors,$OUT/o_rms.safetensors,$OUT/z_rms.safetensors,$OUT/norm_c.safetensors"}
TAG=$(t=${ARM//:/}; t=${t//\//}; echo "${t//,/_}__read")
REF=${REF:-$OUT/$TAG.json}
JSON=${JSON:-$OUT/$TAG.rerun.json}

export TMPDIR=${TMPDIR:-$HOME/.cache/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.triton}
PY=.venv/bin/python
mkdir -p "$OUT" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# A full / quota makes EVERY command exit 1, so gate on it before a 20-minute run.
mb=$(quota -s | awk '/md0/{gsub(/M/,"",$2); gsub(/M/,"",$3); print int($3-$2)}')
echo "[gate] root-quota headroom ${mb}M"
[ "${mb:-0}" -ge "${MIN_ROOT_MB:-1200}" ] || { echo "[gate] ABORT: under ${MIN_ROOT_MB:-1200}M"; exit 1; }
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024')
if [ -n "$busy" ] && [ -z "${ALLOW_BUSY:-}" ]; then
  echo "[gate] GPUs are not idle:"; nvidia-smi --query-gpu=index,memory.used --format=csv
  echo "[gate] set ALLOW_BUSY=1 to override"; exit 1
fi

if [ -f "$JSON" ]; then echo "[1/2] $JSON present — skipping the arm"; else
  echo "[1/2] arm $ARM → $JSON"
  .venv/bin/torchrun --standalone --nproc-per-node="$NPROC" scripts/eval_lm_harness.py \
    --target student --hf-model "$TEACHER" --checkpoint-dir "$CONVERT" \
    --sync-indices 0-63 --sync-phase "$CELL" \
    --read-attn-replica "$ARM" --read-attn-replica-layers "$LAYERS" \
    --read-attn-replica-bases "$BASES" \
    --limit "$LIMIT" --output-json "$JSON" 2>&1 | tee "${JSON%.json}.log"
fi

echo
[ -f "$REF" ] || { echo "[2/2] no reference at $REF — nothing to pair against"; exit 0; }
$PY - "$REF" "$JSON" <<'PYEOF'
import json, sys
sys.path.insert(0, "scripts")
from paired_macro import TASKS, mcnemar
from parallm.eval.downstream import macro_metrics

def dump(p):
    return json.load(open(p))["student"]
def per_doc(d):
    return {t: {r["doc_id"]: float(r.get("acc", 0.0)) for r in d["samples"][t]} for t in TASKS}
def macro(d):
    m = macro_metrics(d, TASKS)
    return sum(m.values()) / len(m), m

ref, new = dump(sys.argv[1]), dump(sys.argv[2])
a, b = per_doc(ref), per_doc(new)
mr, dr = macro(ref)
mn, dn = macro(new)
print(f"{'task':<16} {'reference':>10} {'rerun':>10} {'flipped docs':>13}")
flips = 0
for t in TASKS:
    f = sum(1 for i in a[t] if a[t][i] != b[t].get(i))
    flips += f
    print(f"{t:<16} {dr.get(t, 0):10.4f} {dn.get(t, 0):10.4f} {f:13d}")
print(f"{'MACRO':<16} {mr:10.4f} {mn:10.4f} {flips:13d}")
if flips == 0:
    print("\n[rail] PASS — every doc_id identical: the re-implementation IS the stored arm.")
    sys.exit(0)
_r, md, se, _z, p = mcnemar(a, b)
ok = p >= 0.05 and abs(md) < 0.005
print(f"\n[rail] {flips} docs flipped. paired {md:+.4f} SE {se:.4f} p={p:.3f} -> "
      f"{'within resolution, but EXPLAIN the flips' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
PYEOF
echo "[done] $JSON"
