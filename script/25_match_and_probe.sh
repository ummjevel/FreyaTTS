#!/bin/bash
# Two loose ends, both cheap:
#
# 1. A matched PyTorch baseline for the ONNX split number. The ONNX runs score
#    the first 150 sentences of eval_ko_dev; the 0.0860 grid figure is over all
#    300. Comparing them was not valid, so re-score PyTorch on the same 150.
#
# 2. Whether reflow's damage is over-training. The 3000-step student is worse
#    than its own teacher at every step count (16-step: 0.1377 vs 0.0860), so
#    the fine-tune hurt the model rather than straightening its trajectory.
#    If step500 is closer to the teacher, the recipe is salvageable with a
#    shorter schedule; if it is already broken, the objective is wrong.
set -uo pipefail
REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO"
say() { echo "[match $(date '+%m-%d %H:%M:%S')] $*"; }

synth() {   # $1=hf  $2=steps  $3=tag  $4=limit
    local d="synth/$3"
    [ -f "$d/manifest.jsonl" ] && return 0
    say "  $3: $4문장 (${2}스텝)"
    CUDA_VISIBLE_DEVICES=4 python - "$1" "$d" "$2" "$4" <<'PY' > "script/logs/$3.synth.log" 2>&1
import json, os, sys, soundfile as sf
sys.path.insert(0, "/data/users/voice/zoey/FreyaTTS")
from freyatts import FreyaTTS
hf, out, steps, lim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
tts = FreyaTTS.from_pretrained(hf, device="cuda")
tts.max_words = 4
texts = [json.loads(l)["text"] for l in open("eval/eval_ko_dev.jsonl") if l.strip()][:lim]
man = []
for i, t in enumerate(texts):
    w = tts.synthesize(t, steps=steps, seed=11)
    p = f"{out}/{i:04d}.wav"; sf.write(p, w, 48000, subtype="PCM_16")
    man.append({"wav": os.path.abspath(p), "text": t})
with open(f"{out}/manifest.jsonl", "w") as f:
    for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
PY
}

score() {   # $1=tag
    CUDA_VISIBLE_DEVICES=4 python eval/score_wavs.py --manifest "synth/$1/manifest.jsonl" \
        --system "$1" --out "eval/results/bench_$1.json" > "script/logs/$1.cer.log" 2>&1
    python -c "
import json; d=json.load(open('eval/results/bench_$1.json'))
print(f\"  $1: CER {d['cer']:.4f}\")"
}

TEACHER="checkpoints/distill183M_voiceD/final/hf"

say "1) PyTorch, 같은 150문장 (ONNX와 맞춘 기준선)"
synth "$TEACHER" 16 pt_150_s16 150
score pt_150_s16

say "2) reflow step500 (과학습 여부 확인)"
CK=checkpoints/reflow183M_voiceD/step500
if [ -f "$CK/model.pt" ] && [ ! -d "$CK/hf" ]; then
    python training/convert_ckpt.py "$CK/model.pt" --out "$CK/hf" > /dev/null 2>&1
fi
if [ -d "$CK/hf" ]; then
    synth "$CK/hf" 8 reflow_s500_s8 300
    score reflow_s500_s8
else
    say "  step500 체크포인트 없음 -- 건너뜀"
fi

say "=== 정리 ==="
python - <<'PY'
import json, os
rows = [("PyTorch 16스텝 (150문장)", "bench_pt_150_s16"),
        ("ONNX 분할 16스텝 (150문장)", "bench_onnx_split"),
        ("reflow step500 8스텝 (300문장)", "bench_reflow_s500_s8")]
for name, f in rows:
    p = f"eval/results/{f}.json"
    if os.path.exists(p):
        print(f"  {name:32s} CER {json.load(open(p))['cer']:.4f}")
print("  참고: reflow 3000스텝 8스텝 CER 0.1323 / distill 없는 8스텝 0.0987")
PY
say done
