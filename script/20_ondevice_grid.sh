#!/bin/bash
# -----------------------------------------------------------------------------
# On-device grid for FreyaTTS: size x ODE steps x clause splitting, all measured
# the same way, then ONNX export + CPU RTF for the survivors.
#
#   setsid nohup bash script/20_ondevice_grid.sh > script/logs/ondevice_grid.log 2>&1 &
#
# What each knob is worth, from earlier single-point measurements:
#   splitting  mw11 -> mw4   CER 0.123 -> 0.072 on 337M, rhythm CV unchanged
#   steps      32 -> 16      1.6x faster; on 88M the CER did not get worse
#   size       337M -> 88M   ONNX 1.6 GB -> 562 MB, rhythm CV unchanged
# This fills in the combinations so the choice is made on measurements rather
# than on extrapolation from voiceA alone.
#
# Rhythm CV is reported alongside CER because it is the axis that separated
# FreyaTTS from Matcha; a config that wins on CER but collapses to ~0.44 has
# thrown away the reason to use FreyaTTS at all.
#
# vLLM is restored at the end ONLY if no other account is holding a GPU -- it
# needs all eight, so bringing it up against someone else's job just crash-loops.
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
say() { echo "[grid $(date '+%m-%d %H:%M:%S')] $*"; }

declare -A SEED=( [A]=9 [B]=9 [C]=1 [D]=11 [E]=9 )
STEPS="${STEPS:-16}"
MW="${MW:-4}"
OUT=eval/results/ondevice_grid.jsonl

source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

one_config() {   # $1=size $2=voice $3=gpu
    local size="$1" v="$2" gpu="$3"
    local tag="${size}_voice${v}_s${STEPS}_mw${MW}"
    local hf="checkpoints/distill${size}_voice${v}/final/hf"
    [ -d "$hf" ] || { say "  $tag: no hf dir, skip"; return; }
    grep -q "\"tag\": \"$tag\"" "$OUT" 2>/dev/null && { say "  $tag: done, skip"; return; }

    local d="synth/grid_${tag}"
    if [ ! -f "$d/manifest.jsonl" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" python - "$hf" "$d" "${SEED[$v]}" "$STEPS" "$MW" <<'PY' \
            > "script/logs/grid_${tag}.synth.log" 2>&1
import json, os, sys, soundfile as sf
sys.path.insert(0, "/data/users/voice/zoey/FreyaTTS")
from freyatts import FreyaTTS
hf, outdir, seed, steps, mw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.makedirs(outdir, exist_ok=True)
tts = FreyaTTS.from_pretrained(hf, device="cuda")
tts.max_words = mw
texts = [json.loads(l)["text"] for l in open("eval/eval_ko_dev.jsonl") if l.strip()]
man = []
for i, t in enumerate(texts):
    w = tts.synthesize(t, steps=steps, seed=seed)
    p = f"{outdir}/{i:04d}.wav"
    sf.write(p, w, 48000, subtype="PCM_16")
    man.append({"wav": os.path.abspath(p), "text": t})
with open(f"{outdir}/manifest.jsonl", "w") as f:
    for m in man:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
PY
    fi
    [ -f "$d/manifest.jsonl" ] || { say "  $tag: synthesis failed"; return; }

    CUDA_VISIBLE_DEVICES="$gpu" python eval/score_wavs.py --manifest "$d/manifest.jsonl" \
        --system "$tag" --out "eval/results/bench_grid_${tag}.json" > "script/logs/grid_${tag}.cer.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$UTMOS/.venv/bin/python" "$UTMOS/score_utmos.py" \
        --dir "$d" --name "grid_${tag}" --out "$UTMOS/results/grid_${tag}.json" \
        > "script/logs/grid_${tag}.utmos.log" 2>&1

    local row
    row=$(python - "$tag" "$size" "$v" "$d" <<'PY' 2>/dev/null
import glob, json, os, random, statistics as st, sys
import numpy as np, soundfile as sf, librosa
tag, size, v, d = sys.argv[1:5]
b = json.load(open(f"eval/results/bench_grid_{tag}.json"))
try:
    u = json.load(open(f"/data/users/voice/zoey/utmos-eval/results/grid_{tag}.json"))["utmos_mean"]
except Exception:
    u = None
paths = sorted(glob.glob(f"{d}/*.wav")); random.seed(0)
paths = random.sample(paths, min(60, len(paths)))
cvs = []
for p in paths:
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(on) >= 4:
        dd = np.diff(on); dd = dd[(dd > 0.03) & (dd < 1.0)]
        if len(dd) >= 3: cvs.append(float(np.std(dd) / np.mean(dd)))
print(json.dumps({"tag": tag, "size": size, "voice": v, "cer": b["cer"], "wer": b["wer"],
                  "utmos": u, "rhythm_cv": round(st.mean(cvs), 3) if cvs else None},
                 ensure_ascii=False))
PY
)
    [ -n "$row" ] && { echo "$row" >> "$OUT"; say "  $row"; }
    rm -f "$d"/*.wav          # keep the manifest, drop the bulk
}

say "grid: 183M/88M x voices A-E, steps=$STEPS, max_words=$MW"
gpu=0
for size in 183M 88M; do
    for v in A B C D E; do
        one_config "$size" "$v" "$gpu" &
        gpu=$(( (gpu + 1) % 4 ))
        # keep at most 4 concurrent so each has a GPU to itself
        while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 20; done
    done
done
wait
say "grid finished"

say "=== 결과 (CER 오름차순) ==="
python - <<'PY'
import json, os
p = "eval/results/ondevice_grid.jsonl"
rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for r in sorted(rows, key=lambda x: x["cer"]):
    u = f"{r['utmos']:.3f}" if r.get("utmos") else "—"
    c = f"{r['rhythm_cv']:.3f}" if r.get("rhythm_cv") else "—"
    print(f"  {r['size']:5s} voice{r['voice']}  CER {r['cer']:.4f}  UTMOS {u}  리듬CV {c}")
PY

# --- vLLM: only if nobody else is on the GPUs -------------------------------
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
         | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done \
         | grep -v "^$(id -un)$" | sort -u | tr '\n' ' ')
if [ -z "$VLLM" ]; then
    say "VLLM_CONTAINER unset -- vLLM 복구 건너뜀"
elif [ -n "$others" ]; then
    say "다른 계정이 GPU 사용 중($others) -- vLLM 올리지 않음"
elif [ "$(docker inspect -f '{{.State.Running}}' "$VLLM" 2>/dev/null)" = "true" ]; then
    say "vLLM already running"
else
    say "GPU 비어 있음 -- vLLM 복구"
    docker start "$VLLM" && docker update --restart=unless-stopped "$VLLM" > /dev/null 2>&1 \
        && say "vLLM started (~9 min)" || say "ERROR: docker start failed"
fi
say "done"
