#!/bin/bash
# -----------------------------------------------------------------------------
# Reflow distillation on voiceD: fewer ODE steps without losing what makes
# FreyaTTS worth using.
#
#   setsid nohup bash script/22_reflow_distill.sh > script/logs/reflow.log 2>&1 &
#
# Order matters here:
#   1. measure 8-step CER first. Without it there is no baseline to judge the
#      distilled 4-step model against -- "4 steps works" only means something
#      next to what 8 steps does undistilled.
#   2. generate (noise, teacher-latent) pairs with the 32-step teacher
#   3. fine-tune on those pairs (straightens the trajectory)
#   4. evaluate the student at 4 and 8 steps
#
# Every evaluation reports rhythm CV alongside CER. Step reduction that buys
# speed by flattening delivery is not a win: rhythm variety is the one axis
# where FreyaTTS beats Matcha, and 183M voiceD currently sits at 0.523.
#
# voiceD is the test case: best CER and UTMOS of the five, and the cleanest
# teacher (UTMOS 3.606), so the signal is not muddied by a weak voice.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
UTMOS=/data/users/voice/zoey/utmos-eval
# GPU 4,5,6 only. Another account trains on 0-3 and vLLM holds 7, so this work
# stays off both. Reflow trains on three GPUs (effective batch 64x3=192 rather
# than the 256 the distills used) -- it is a fine-tune of an existing checkpoint,
# not a run being compared against those baselines on training config.
V=D
SIZE=183M
cd "$REPO"
say() { echo "[reflow $(date '+%m-%d %H:%M:%S')] $*"; }

source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1   # this node's NVLink peer access is unreliable

say "waiting for the precision measurements to finish"
# Matched on the python executable + script path, not on "bench_onnx_cpu.py".
# The bare filename also appears inside a watcher launched with bash -c, so
# pgrep -f found the watcher itself and it waited on its own existence. Three
# chained watchers sat in that deadlock for 21 hours on 08-04.
while pgrep -f "[.]venv/bin/python eval/bench_onnx_cpu" > /dev/null 2>&1; do sleep 60; done
while pgrep -f "bash script/[2]1_precision_quality" > /dev/null 2>&1; do sleep 60; done
say "GPUs free"

TEACHER_HF="checkpoints/distill${SIZE}_voice${V}/final/hf"
TEACHER_PT="checkpoints/distill${SIZE}_voice${V}/final/model.pt"

# --- 1. baseline: what 8 steps does without distillation ----------------------
evaluate() {   # $1=hf dir  $2=steps  $3=tag  $4=gpu
    local hf="$1" st="$2" tag="$3" gpu="$4"
    local d="synth/reflow_${tag}"
    if [ ! -f "$d/manifest.jsonl" ]; then
        say "  $tag: 300문장 합성 (${st}스텝)"
        CUDA_VISIBLE_DEVICES="$gpu" python - "$hf" "$d" "$st" <<'PY' \
            > "script/logs/reflow_${tag}.synth.log" 2>&1
import json, os, sys, soundfile as sf
sys.path.insert(0, "/data/users/voice/zoey/FreyaTTS")
from freyatts import FreyaTTS
hf, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
tts = FreyaTTS.from_pretrained(hf, device="cuda")
tts.max_words = 4                      # the split setting the grid was measured with
texts = [json.loads(l)["text"] for l in open("eval/eval_ko_dev.jsonl") if l.strip()]
man = []
for i, t in enumerate(texts):
    w = tts.synthesize(t, steps=steps, seed=11)   # voiceD's locked seed
    p = f"{out}/{i:04d}.wav"; sf.write(p, w, 48000, subtype="PCM_16")
    man.append({"wav": os.path.abspath(p), "text": t})
with open(f"{out}/manifest.jsonl", "w") as f:
    for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
PY
    fi
    [ -f "$d/manifest.jsonl" ] || { say "  $tag: 합성 실패"; return; }
    CUDA_VISIBLE_DEVICES="$gpu" python eval/score_wavs.py --manifest "$d/manifest.jsonl" \
        --system "$tag" --out "eval/results/bench_reflow_${tag}.json" \
        > "script/logs/reflow_${tag}.cer.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$UTMOS/.venv/bin/python" "$UTMOS/score_utmos.py" \
        --dir "$d" --name "reflow_${tag}" --out "$UTMOS/results/reflow_${tag}.json" \
        > "script/logs/reflow_${tag}.utmos.log" 2>&1
    python - "$tag" "$d" <<'PY'
import glob, json, random, statistics as st, sys
import numpy as np, soundfile as sf, librosa
tag, d = sys.argv[1], sys.argv[2]
b = json.load(open(f"eval/results/bench_reflow_{tag}.json"))
try: u = json.load(open(f"/data/users/voice/zoey/utmos-eval/results/reflow_{tag}.json"))["utmos_mean"]
except Exception: u = None
paths = sorted(glob.glob(f"{d}/*.wav")); random.seed(0)
paths = random.sample(paths, min(60, len(paths)))
cv = []
for p in paths:
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(on) >= 4:
        dd = np.diff(on); dd = dd[(dd > 0.03) & (dd < 1.0)]
        if len(dd) >= 3: cv.append(float(np.std(dd) / np.mean(dd)))
row = {"tag": tag, "cer": b["cer"], "utmos": u,
       "rhythm_cv": round(st.mean(cv), 3) if cv else None}
print(json.dumps(row, ensure_ascii=False))
open("eval/results/reflow_summary.jsonl", "a").write(json.dumps(row, ensure_ascii=False) + "\n")
PY
    rm -f "$d"/*.wav
}

say "1) 기준선: distill 없이 8스텝"
evaluate "$TEACHER_HF" 8 "teacher_s8" 4

# --- 2. reflow pairs ----------------------------------------------------------
PAIRS="data/reflow_voice${V}"
if [ ! -f "$PAIRS/reflow_00000.pt" ]; then
    say "2) reflow 쌍 생성 (16스텝 teacher — 평가 대상과 같은 스텝)"
    CUDA_VISIBLE_DEVICES=4 python training/gen_reflow_pairs.py \
        --model "$TEACHER_HF" --latents "data/latents_distill_voice${V}" \
        --out "$PAIRS" --steps 16 > script/logs/reflow_pairs.log 2>&1
fi
n=$(ls "$PAIRS"/*.pt 2>/dev/null | wc -l)
pairs=$(python training/count_reflow_pairs.py "$PAIRS" 2>/dev/null)
want=$(wc -l < text_data/clean.txt)
say "   쌍 ${pairs:-0}개 / 기대 ${want}개 (샤드 $n)"
# An interrupted generation leaves a valid but short set. The first run trained
# on 4000 of 11554 pairs and produced a student worse than the undistilled
# teacher at every step count, because the check only asked whether any shard
# existed. Require most of the corpus before spending an hour of training on it.
if [ "${pairs:-0}" -lt $(( want * 9 / 10 )) ]; then
    say "쌍이 부족합니다 (${pairs:-0} < $(( want * 9 / 10 ))) -- 중단"; exit 1
fi

# --- 3. train -----------------------------------------------------------------
OUT="checkpoints/reflow${SIZE}_voice${V}"
if [ ! -f "$OUT/final/model.pt" ]; then
    say "3) reflow 학습 (GPU 0-3, 3000 step)"
    CUDA_VISIBLE_DEVICES=4,5,6 accelerate launch --multi_gpu --num_processes 3 \
        --mixed_precision bf16 --main_process_port 29850 \
        training/train_reflow.py --init "$TEACHER_PT" --data "$PAIRS" --out "$OUT" \
        --steps 3000 --batch_size 64 --save_every 500 > script/logs/reflow_train.log 2>&1
fi
[ -f "$OUT/final/model.pt" ] || { say "학습 실패 -- script/logs/reflow_train.log 확인"; exit 1; }
python training/convert_ckpt.py "$OUT/final/model.pt" --out "$OUT/final/hf" > /dev/null 2>&1

# --- 4. evaluate the student --------------------------------------------------
say "4) 학생 평가"
for st in 4 8 16; do
    evaluate "$OUT/final/hf" "$st" "student_s${st}" 4
done

say "=== 요약 (183M voiceD, mw4 분할) ==="
python - <<'PY'
import json, os
p = "eval/results/reflow_summary.jsonl"
rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
print(f"  {'구성':16s} {'CER':>8s} {'UTMOS':>8s} {'리듬CV':>8s}")
for r in rows:
    u = f"{r['utmos']:.3f}" if r.get("utmos") else "—"
    c = f"{r['rhythm_cv']:.3f}" if r.get("rhythm_cv") else "—"
    print(f"  {r['tag']:16s} {r['cer']:8.4f} {u:>8s} {c:>8s}")
print("  (참고) 16스텝 distill 전: CER 0.0860  UTMOS 3.063  리듬CV 0.523")
PY
say "done"
