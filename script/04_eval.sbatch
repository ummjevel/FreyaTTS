#!/bin/bash
#SBATCH --job-name=freya_eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/eval_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/eval_%j.err

# -----------------------------------------------------------------------------
# Eval a training checkpoint: model.pt -> from_pretrained dir -> WER/CER + speed.
# Uses ONE GPU. Usable GPUs: 4,5,6,7.
#
#   CKPT=checkpoints/pretrain/step30000/model.pt GPUID=4 sbatch script/04_eval.sbatch
#   sbatch script/04_eval.sbatch checkpoints/pretrain/final/model.pt 5
#
# NOTE on the pretrain stage: the model has no speaker conditioning (seed = voice),
# so a pretrain-only checkpoint is judged on INTELLIGIBILITY (WER/CER) + SPEED,
# not on hitting a target voice. Voice lock comes after SFT.
# First run downloads faster-whisper large-v3 (~3GB) into the HF cache.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

CKPT="${CKPT:-checkpoints/pretrain/final/model.pt}"
[ -n "${1:-}" ] && CKPT="$1"
GPUID="${GPUID:-4}"
[ -n "${2:-}" ] && GPUID="$2"

DATA="${DATA:-eval/eval_ko_dev.jsonl}"     # held-out dev sentences ({"text":...})
LIMIT="${LIMIT:-0}"                          # 0 = all; e.g. 100 for a quick pass

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$GPUID  CKPT=$CKPT  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUID"
# eval/benchmark.py & eval/speed.py `import freyatts` but don't add the repo root
# to sys.path (unlike training/*.py), so run with the repo on PYTHONPATH.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

# tag results by the checkpoint dir name (e.g. step30000, final)
TAG=$(basename "$(dirname "$CKPT")")
HFDIR="$(dirname "$CKPT")/hf"
mkdir -p eval/results

echo "--- 1/3 convert model.pt -> $HFDIR ---"
python training/convert_ckpt.py "$CKPT" --out "$HFDIR"

echo "--- 2/3 WER/CER (Whisper large-v3, ko) on $DATA ---"
BENCH_ARGS="--system freyatts --model $HFDIR --data $DATA --device cuda --out eval/results/bench_${TAG}.json"
[ "$LIMIT" -gt 0 ] && BENCH_ARGS="$BENCH_ARGS --limit $LIMIT"
python eval/benchmark.py $BENCH_ARGS

echo "--- 3/3 speed / RTF / VRAM ---"
python eval/speed.py --system freyatts --model "$HFDIR" --out "eval/results/speed_${TAG}.json"

echo "=== done $(date) ==="
echo "results: eval/results/bench_${TAG}.json , eval/results/speed_${TAG}.json"
