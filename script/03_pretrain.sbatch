#!/bin/bash
#SBATCH --job-name=freya_pretrain
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/pretrain_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/pretrain_%j.err

# -----------------------------------------------------------------------------
# Step 3/3 : from-scratch FreyaTTS pretraining on the precomputed latents.
# Multi-GPU via Accelerate (bf16). Pick GPUs with GPUS (comma list).
# Usable GPUs: 4,5,6,7.
#   GPUS=4,5,6,7        sbatch script/03_pretrain.sbatch
#   sbatch script/03_pretrain.sbatch 5,6,7     # arg also works
#   sbatch script/03_pretrain.sbatch 6         # single GPU
# Default = 4,5,6,7.
# Resume:  RESUME=checkpoints/pretrain/step30000  GPUS=... sbatch ...
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

# physical GPU ids to use (usable: 4,5,6,7). env GPUS or first positional arg.
GPUS="${GPUS:-4,5,6,7}"
[ -n "${1:-}" ] && GPUS="$1"
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

DATA="${DATA:-data/latents}"
OUT="${OUT:-checkpoints/pretrain}"
CONFIG="${CONFIG:-training/configs/pretrain.yaml}"
RESUME="${RESUME:-}"

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST  GPUS=$GPUS ($NGPU)  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# This node (gpu-1.novalocal) is an OpenStack GPU-passthrough VM where NVLink
# peer-to-peer memory access is unreliable -> NCCL all-reduce in backward() dies
# with "Invalid access of peer GPU memory over nvlink". Route NCCL via shared
# memory/host instead of P2P. Single node, so no InfiniBand.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

cd "$REPO"
mkdir -p "$OUT"

# unique rendezvous port per job so concurrent runs don't collide
PORT=$((29500 + SLURM_JOB_ID % 2000))

ACC_ARGS="--num_machines 1 --num_processes $NGPU --mixed_precision bf16 --main_process_port $PORT"
[ "$NGPU" -gt 1 ] && ACC_ARGS="--multi_gpu $ACC_ARGS"

# checkpoint earlier than the 10k default so a mid-run crash loses less progress
SAVE_EVERY="${SAVE_EVERY:-5000}"
TRAIN_ARGS="--config $CONFIG --data $DATA --out $OUT --save_every $SAVE_EVERY"
# extend training past the config's 150000: STEPS stretches the cosine LR
# schedule (keeps a meaningful LR longer). e.g. STEPS=250000 RESUME=.../final ...
[ -n "${STEPS:-}" ] && TRAIN_ARGS="$TRAIN_ARGS --steps $STEPS"
[ -n "$RESUME" ] && TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"

echo "accelerate launch $ACC_ARGS training/pretrain.py $TRAIN_ARGS"
accelerate launch $ACC_ARGS training/pretrain.py $TRAIN_ARGS

echo "=== done $(date) ==="
