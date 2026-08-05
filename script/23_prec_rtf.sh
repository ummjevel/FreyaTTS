#!/bin/bash
# fp16 / int8 CPU RTF. A separate file rather than an inline `bash -c`, because
# an inline command carries the whole script text in its command line and any
# pgrep -f on a string inside it then matches this process too.
set -uo pipefail
cd /data/users/voice/zoey/FreyaTTS
source .venv/bin/activate
export PYTHONPATH=/data/users/voice/zoey/FreyaTTS
for prec in fp16 int8; do
  d=onnx_export/distill183M_voiceD_s16_${prec}
  [ -d "$d" ] || { echo "$prec 없음"; continue; }
  for th in 1 4; do
    python eval/bench_onnx_cpu.py --onnx "$d" --threads "$th" --runs 5 --warmup 2 \
      --out "eval/results/speed_onnx_cpu_183M_s16_${prec}_t${th}.json" 2>&1 | grep -E "^(short|medium|long)"
    echo "=== ${prec} ${th}스레드 완료"
  done
done
echo PREC_RTF_DONE
