#!/bin/bash
#SBATCH --job-name=freya_latents
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/latents_%A_%a.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/latents_%A_%a.err

# -----------------------------------------------------------------------------
# Step 2/3 : encode audio -> frozen VoxCPM2 AudioVAE latents (25 Hz, 64-dim fp16)
# PARALLEL: a 4-task job array, one GPU per task.
#   array task 0 -> GPU 4,  1 -> GPU 5,  2 -> GPU 6,  3 -> GPU 7
# Each task encodes its own manifest_train.part<N>.jsonl (from step 01) into the
# shared data/latents/ dir with a distinct --prefix, so shards never collide and
# pretrain's *.pt glob picks up all of them.
#
# To use a different GPU set, change --array and BASE_GPU below (task -> BASE_GPU+id).
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

BASE_GPU="${BASE_GPU:-4}"                 # array task i runs on GPU (BASE_GPU + i)
PART="$SLURM_ARRAY_TASK_ID"
GPUID=$((BASE_GPU + PART))

MANIFEST="data/manifest_train.part${PART}.jsonl"
OUT="${OUT:-data/latents}"

echo "=== Job $SLURM_ARRAY_JOB_ID task $PART on $SLURM_NODELIST  GPU=$GPUID  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUID"
cd "$REPO"

python training/precompute_latents.py \
    --manifest "$MANIFEST" \
    --out "$OUT" \
    --prefix "part${PART}" \
    --device cuda

echo "=== task $PART done $(date) ==="
