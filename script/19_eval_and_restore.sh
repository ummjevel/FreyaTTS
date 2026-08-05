#!/bin/bash
# -----------------------------------------------------------------------------
# Evaluate every new distill, then give the GPUs back to vLLM.
#
#   setsid nohup bash script/19_eval_and_restore.sh > script/logs/eval_restore.log 2>&1 &
#
# Waits for 18_distill_queue.sh, then for each checkpoint: dump 300 wavs once and
# score them three ways off that single synthesis pass -- CER (the Korean
# metric), UTMOS (quality), and syllable-interval CV (the rhythm measure that
# separated FreyaTTS from Matcha). Running benchmark.py separately would
# synthesize the same audio a second time for no reason.
#
# The seed is the speaker in FreyaTTS, so each voice uses its locked seed from
# confirmed_voices/best_seeds.json. A wrong seed here produces a different person
# and the numbers would be meaningless.
#
# vLLM comes back only at the very end, once nothing is holding a GPU.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
UTMOS=/data/users/voice/zoey/utmos-eval
# The container that has to be stopped to free the GPUs and started again
# afterwards. It belongs to another account on this node, so the name is not
# baked in: set VLLM_CONTAINER in the environment. Left unset, every step that
# touches it is skipped and the rest of the script runs unchanged.
VLLM="${VLLM_CONTAINER:-}"
cd "$REPO"
say() { echo "[evalrestore $(date '+%m-%d %H:%M:%S')] $*"; }

declare -A SEED=( [A]=9 [B]=9 [C]=1 [D]=11 [E]=9 )

say "waiting for the distill queue"
while pgrep -f "18_distill_queue\.sh|training/sft\.py" > /dev/null 2>&1; do sleep 60; done
say "distills done"

source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

eval_ckpt() {   # $1=size $2=voice $3=gpu
    local size="$1" v="$2" gpu="$3"
    local ck="checkpoints/distill${size}_voice${v}"
    local tag="distill${size}_voice${v}"
    [ -f "$ck/final/model.pt" ] || { say "  $tag: no checkpoint, skip"; return; }
    [ -f "eval/results/bench_${tag}.json" ] && { say "  $tag: already scored, skip"; return; }

    python training/convert_ckpt.py "$ck/final/model.pt" --out "$ck/final/hf" > /dev/null 2>&1
    local out="synth/${tag}"
    if [ ! -f "$out/manifest.jsonl" ]; then
        say "  $tag: synthesizing (seed ${SEED[$v]})"
        CUDA_VISIBLE_DEVICES="$gpu" python eval/dump_wavs.py --model "$ck/final/hf" \
            --outdir "$out" --seed "${SEED[$v]}" > "script/logs/dump_${tag}.log" 2>&1
    fi
    [ -f "$out/manifest.jsonl" ] || { say "  $tag: synthesis failed"; return; }

    CUDA_VISIBLE_DEVICES="$gpu" python eval/score_wavs.py --manifest "$out/manifest.jsonl" \
        --system "$tag" --out "eval/results/bench_${tag}.json" > "script/logs/cer_${tag}.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$UTMOS/.venv/bin/python" "$UTMOS/score_utmos.py" \
        --dir "$out" --name "$tag" --out "$UTMOS/results/${tag}.json" > "script/logs/utmos_${tag}.log" 2>&1

    local line
    line=$(python - "$tag" <<'PY' 2>/dev/null
import glob, json, os, random, statistics as st, sys
import numpy as np, soundfile as sf, librosa
tag = sys.argv[1]
b = json.load(open(f"eval/results/bench_{tag}.json"))
try:
    u = json.load(open(f"/data/users/voice/zoey/utmos-eval/results/{tag}.json"))["utmos_mean"]
except Exception:
    u = None
paths = sorted(glob.glob(f"synth/{tag}/*.wav"))
random.seed(0); paths = random.sample(paths, min(60, len(paths)))
cvs = []
for p in paths:
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(on) >= 4:
        d = np.diff(on); d = d[(d > 0.03) & (d < 1.0)]
        if len(d) >= 3: cvs.append(float(np.std(d) / np.mean(d)))
cv = round(st.mean(cvs), 3) if cvs else None
print(json.dumps({"tag": tag, "cer": b["cer"], "wer": b["wer"], "utmos": u, "rhythm_cv": cv},
                 ensure_ascii=False))
PY
)
    [ -n "$line" ] && { echo "$line" >> eval/results/distill_summary.jsonl; say "  $line"; }
}

for size in 183M 88M; do
    for v in B C D; do
        eval_ckpt "$size" "$v" 0
    done
done

say "=== 요약 (한국어는 CER 기준) ==="
python - <<'PY'
import json, os
rows = []
p = "eval/results/distill_summary.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
for r in rows:
    u = f"{r['utmos']:.3f}" if r.get("utmos") else "—"
    c = f"{r['rhythm_cv']:.3f}" if r.get("rhythm_cv") else "—"
    print(f"  {r['tag']:26s} CER {r['cer']:.4f}  UTMOS {u}  리듬CV {c}")
PY

say "restoring vLLM"
if [ -z "$VLLM" ]; then
    say "VLLM_CONTAINER unset -- vLLM 복구 건너뜀"
elif [ "$(docker inspect -f '{{.State.Running}}' "$VLLM" 2>/dev/null)" = "true" ]; then
    say "vLLM already running -- leaving it alone"
else
    docker start "$VLLM" && docker update --restart=unless-stopped "$VLLM" > /dev/null 2>&1 \
        && say "vLLM started, restart policy back to unless-stopped (~9 min to serve)" \
        || say "ERROR: docker start failed -- run: docker start $VLLM"
fi
say "done"
