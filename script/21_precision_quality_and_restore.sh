#!/bin/bash
# -----------------------------------------------------------------------------
# After the RTF runs: check what fp16 and int8 cost in accuracy, then restore vLLM.
#
#   setsid nohup bash script/21_precision_quality_and_restore.sh > script/logs/prec_quality.log 2>&1 &
#
# Size and speed are only half the question. This model runs the same weights 16
# times inside one ODE solve, so precision error is applied repeatedly and
# compounds along the trajectory -- a quantized graph can measure fine on a
# single forward pass and still drift audibly by the last step. CER is the check.
#
# All three precisions are scored on the SAME 150-sentence subset, including
# fp32. Comparing a 150-sentence quantized number against the existing
# 300-sentence fp32 number (0.0860) would confound precision with sample.
#
# Synthesis runs on CPU through ONNX Runtime (that is the deployment path);
# Whisper scoring runs on GPU because it is free and much faster there.
#
# vLLM comes back at the end, unless another account is on the GPUs.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
# The container that has to be stopped to free the GPUs and started again
# afterwards. It belongs to another account on this node, so the name is not
# baked in: set VLLM_CONTAINER in the environment. Left unset, every step that
# touches it is skipped and the rest of the script runs unchanged.
VLLM="${VLLM_CONTAINER:-}"
# GPU 4,5,6 only. Another account trains on 0-3 and vLLM holds 7, so this work
# stays off both. Reflow trains on three GPUs (effective batch 64x3=192 rather
# than the 256 the distills used) -- it is a fine-tune of an existing checkpoint,
# not a run being compared against those baselines on training config.
N=150
THREADS=4
cd "$REPO"
say() { echo "[prec $(date '+%m-%d %H:%M:%S')] $*"; }

say "waiting for the RTF benchmarks (timing runs must not overlap)"
# Matched on the python executable + script path, not on "bench_onnx_cpu.py".
# The bare filename also appears inside a watcher launched with bash -c, so
# pgrep -f found the watcher itself and it waited on its own existence. Three
# chained watchers sat in that deadlock for 21 hours on 08-04.
while pgrep -f "[.]venv/bin/python eval/bench_onnx_cpu" > /dev/null 2>&1; do sleep 60; done
say "RTF runs finished"

source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

head -"$N" eval/eval_ko_dev.jsonl > /tmp/prec_eval_${N}.jsonl

for prec in fp32 fp16 int8; do
    case "$prec" in
        fp32) d=onnx_export/distill183M_voiceD_s16 ;;
        *)    d=onnx_export/distill183M_voiceD_s16_${prec} ;;
    esac
    [ -d "$d" ] || { say "  $prec: 그래프 없음, 건너뜀"; continue; }
    out="synth/prec_${prec}"
    if [ ! -f "$out/manifest.jsonl" ]; then
        say "  $prec: ${N}문장 합성 (CPU ${THREADS}스레드)"
        python - "$d" "$out" "$THREADS" "/tmp/prec_eval_${N}.jsonl" <<'PY' \
            > "script/logs/prec_${prec}.synth.log" 2>&1
import json, os, sys, numpy as np, soundfile as sf
sys.path.insert(0, "/data/users/voice/zoey/FreyaTTS")
from eval.bench_onnx_cpu import OnnxFreya
from freyatts.hangul import decompose_hangul
d, out, threads, data = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
vocab = json.load(open("freyatts/char_vocab.json")); UNK = vocab.get("<unk>", 1)
eng = OnnxFreya(d, threads=threads)
os.makedirs(out, exist_ok=True)
man = []
for i, line in enumerate(open(data)):
    if not line.strip(): continue
    text = json.loads(line)["text"]
    ids = np.array([[vocab.get(c, UNK) for c in decompose_hangul(text)]], dtype=np.int64)
    wav = eng.synthesize(ids)
    p = f"{out}/{i:04d}.wav"
    sf.write(p, wav, 48000, subtype="PCM_16")
    man.append({"wav": os.path.abspath(p), "text": text})
with open(f"{out}/manifest.jsonl", "w") as f:
    for m in man:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(f"{len(man)} wavs")
PY
    fi
    [ -f "$out/manifest.jsonl" ] || { say "  $prec: 합성 실패"; continue; }

    CUDA_VISIBLE_DEVICES=4 python eval/score_wavs.py --manifest "$out/manifest.jsonl" \
        --system "183M-s16-$prec" --out "eval/results/bench_prec_${prec}.json" \
        > "script/logs/prec_${prec}.cer.log" 2>&1
    cer=$(python -c "
import json;print(f\"{json.load(open('eval/results/bench_prec_${prec}.json'))['cer']:.4f}\")" 2>/dev/null)
    say "  $prec: CER ${cer:-?}"
    rm -f "$out"/*.wav
done

say "=== 정밀도별 요약 (183M, 16스텝, ${N}문장) ==="
python - <<'PY'
import json, os
for prec, d in (("fp32", "onnx_export/distill183M_voiceD_s16"),
                ("fp16", "onnx_export/distill183M_voiceD_s16_fp16"),
                ("int8", "onnx_export/distill183M_voiceD_s16_int8")):
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
               if f.endswith(".onnx")) / 1e6 if os.path.isdir(d) else None
    try:
        cer = json.load(open(f"eval/results/bench_prec_{prec}.json"))["cer"]
    except Exception:
        cer = None
    print(f"  {prec:5s}  크기 {size:6.0f} MB" if size else f"  {prec:5s}  크기 —",
          f" CER {cer:.4f}" if cer is not None else " CER —")
PY

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
        && say "vLLM started (~9분 후 서빙)" || say "ERROR: docker start 실패"
fi
say "done"
