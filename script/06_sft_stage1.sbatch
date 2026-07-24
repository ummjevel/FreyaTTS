#!/bin/bash
#SBATCH --job-name=freya_sft1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/sft1_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/sft1_%j.err

# -----------------------------------------------------------------------------
# SFT stage 1 (voice lock): full-param fine-tune of pretrain/final on ONE speaker.
# Multi-GPU (Accelerate bf16). Usable GPUs: 4,5,6,7.
#   GPUS=4,5,6,7 sbatch script/06_sft_stage1.sbatch
#   sbatch script/06_sft_stage1.sbatch 6,7
# Short stage (3000 steps); checkpoints every 500 so you can pick the best
# (README: released checkpoints were selected ~step 1000). After this stage,
# synthesize a few seeds and pick the one that locked onto the target voice.
# Resume:  RESUME=checkpoints/sft_stage1/step1000 GPUS=... sbatch ...
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

GPUS="${GPUS:-4,5,6,7}"
[ -n "${1:-}" ] && GPUS="$1"
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

CONFIG="${CONFIG:-training/configs/sft_stage1.yaml}"
INIT="${INIT:-checkpoints/pretrain/final/model.pt}"
DATA="${DATA:-data/latents_sft}"
OUT="${OUT:-checkpoints/sft_stage1}"
SAVE_EVERY="${SAVE_EVERY:-500}"
RESUME="${RESUME:-}"

echo "=== Job $SLURM_JOB_ID  GPUS=$GPUS ($NGPU)  INIT=$INIT  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# same OpenStack-VM NVLink-P2P workaround as pretrain
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
cd "$REPO"
mkdir -p "$OUT"

[ -f "$INIT" ] || { echo "ERROR: init checkpoint not found: $INIT (finish pretrain first)"; exit 1; }
[ -d "$DATA" ] || { echo "ERROR: sft latents not found: $DATA (run 05_build_sft_data first)"; exit 1; }

PORT=$((29600 + SLURM_JOB_ID % 2000))
ACC_ARGS="--num_machines 1 --num_processes $NGPU --mixed_precision bf16 --main_process_port $PORT"
[ "$NGPU" -gt 1 ] && ACC_ARGS="--multi_gpu $ACC_ARGS"

TRAIN_ARGS="--config $CONFIG --init $INIT --data $DATA --out $OUT --save_every $SAVE_EVERY"
[ -n "$RESUME" ] && TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"

echo "accelerate launch $ACC_ARGS training/sft.py $TRAIN_ARGS"
accelerate launch $ACC_ARGS training/sft.py $TRAIN_ARGS

echo "=== done $(date) ==="
