#!/bin/bash
# ONNX path with clause splitting: re-measure CER and RTF.
# The exported graphs are the model only; normalization and splitting live in
# the python pipeline and had to be reproduced in the ONNX caller. Without them
# the deployed path scored CER 0.1317 against 0.0860 for the same weights driven
# from PyTorch -- the gap was entirely the missing split.
set -uo pipefail
cd /data/users/voice/zoey/FreyaTTS
source .venv/bin/activate
export PYTHONPATH=/data/users/voice/zoey/FreyaTTS
say() { echo "[onnxsplit $(date '+%m-%d %H:%M:%S')] $*"; }

rm -rf synth/prec_fp32 eval/results/bench_prec_fp32.json

say "150문장 합성 (분할 적용, CPU 4스레드)"
python - <<'PY' > script/logs/onnx_split_synth.log 2>&1
import json, os, sys, numpy as np, soundfile as sf
sys.path.insert(0, "/data/users/voice/zoey/FreyaTTS")
from eval.bench_onnx_cpu import OnnxFreya
from freyatts.hangul import decompose_hangul
vocab = json.load(open("freyatts/char_vocab.json")); UNK = vocab.get("<unk>", 1)
to_ids = lambda s: np.array([[vocab.get(c, UNK) for c in decompose_hangul(s)]], dtype=np.int64)
eng = OnnxFreya("onnx_export/distill183M_voiceD_s16", threads=4)
out = "synth/onnx_split"; os.makedirs(out, exist_ok=True)
texts = [json.loads(l)["text"] for l in open("eval/eval_ko_dev.jsonl") if l.strip()][:150]
man = []
for i, t in enumerate(texts):
    w = eng.synth_text(t, to_ids)
    p = f"{out}/{i:04d}.wav"; sf.write(p, w, 48000, subtype="PCM_16")
    man.append({"wav": os.path.abspath(p), "text": t})
with open(f"{out}/manifest.jsonl", "w") as f:
    for m in man: f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(len(man))
PY

say "CER 채점 (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python eval/score_wavs.py --manifest synth/onnx_split/manifest.jsonl \
  --system "onnx-split-183M-s16" --out eval/results/bench_onnx_split.json 2>&1 | grep -E '"cer"'

say "RTF 재측정 (분할 포함)"
for th in 1 4; do
  python eval/bench_onnx_cpu.py --onnx onnx_export/distill183M_voiceD_s16 --threads "$th" \
    --runs 5 --warmup 2 --out "eval/results/speed_onnx_split_t${th}.json" 2>&1 | grep -E "^(short|medium|long)"
  echo "=== ${th}스레드 완료"
done
echo ONNX_SPLIT_DONE
