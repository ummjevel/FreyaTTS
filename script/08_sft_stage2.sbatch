#!/bin/bash
#SBATCH --job-name=freya_sft2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/sft2_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/sft2_%j.err

# -----------------------------------------------------------------------------
# SFT stage 2 (short-utterance coverage): continue from stage 1 on the short+full
# mix, lower LR + stronger duration loss (see sft_stage2.yaml). Multi-GPU bf16.
#   GPUS=4,5,6,7 sbatch script/08_sft_stage2.sbatch
#   sbatch script/08_sft_stage2.sbatch 6,7
# save_every 500 -> pick the best checkpoint (~step 1000) by ear/eval afterwards.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

GPUS="${GPUS:-4,5,6,7}"
[ -n "${1:-}" ] && GPUS="$1"
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

CONFIG="${CONFIG:-training/configs/sft_stage2.yaml}"
INIT="${INIT:-checkpoints/sft_stage1/final/model.pt}"
DATA="${DATA:-data/latents_sft_short}"
OUT="${OUT:-checkpoints/sft_stage2}"
SAVE_EVERY="${SAVE_EVERY:-500}"
RESUME="${RESUME:-}"

echo "=== Job $SLURM_JOB_ID  GPUS=$GPUS ($NGPU)  INIT=$INIT  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
cd "$REPO"
mkdir -p "$OUT"

[ -f "$INIT" ] || { echo "ERROR: stage-1 checkpoint not found: $INIT (run 06_sft_stage1 first)"; exit 1; }
[ -d "$DATA" ] || { echo "ERROR: short-mix latents not found: $DATA (run 07_extract_short first)"; exit 1; }

PORT=$((29700 + SLURM_JOB_ID % 2000))
ACC_ARGS="--num_machines 1 --num_processes $NGPU --mixed_precision bf16 --main_process_port $PORT"
[ "$NGPU" -gt 1 ] && ACC_ARGS="--multi_gpu $ACC_ARGS"

TRAIN_ARGS="--config $CONFIG --init $INIT --data $DATA --out $OUT --save_every $SAVE_EVERY"
[ -n "$RESUME" ] && TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"

echo "accelerate launch $ACC_ARGS training/sft.py $TRAIN_ARGS"
accelerate launch $ACC_ARGS training/sft.py $TRAIN_ARGS

echo "=== done $(date) ==="
