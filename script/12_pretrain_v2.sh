#!/bin/bash
# -----------------------------------------------------------------------------
# From-scratch pretrain of a size variant, run directly (GPUs are hand-allocated
# on this node, so no sbatch / SLURM_JOB_ID).
#
#   SIZE=88M  GPUS=0,1,2,3 DATA=data/latents_133_015 \
#     OUT=checkpoints/pretrain_88M_v2_133_015 bash script/12_pretrain_v2.sh
#
# Dims come from training/configs/pretrain_<SIZE>.yaml; DATA/OUT override the
# config on the CLI (argparse defaults are seeded from the YAML, so CLI wins).
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

SIZE="${SIZE:-88M}"
GPUS="${GPUS:-0,1,2,3}"
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

CONFIG="${CONFIG:-training/configs/pretrain_${SIZE}.yaml}"
DATA="${DATA:-data/latents}"
OUT="${OUT:-checkpoints/pretrain_${SIZE}_v2}"
STEPS="${STEPS:-150000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
RESUME="${RESUME:-}"
# per-device batch. Keep BATCH x NGPU = 256 to match the 4x64 runs everything
# else in this repo is compared against; changing the effective batch changes
# the optimization, not just the speed.
BATCH="${BATCH:-64}"

[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[ -d "$DATA" ]   || { echo "ERROR: latents dir not found: $DATA"; exit 1; }

echo "=== pretrain $SIZE  GPUS=$GPUS ($NGPU)  $(date) ==="
echo "    config=$CONFIG  data=$DATA  out=$OUT  steps=$STEPS"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# OpenStack-VM NVLink P2P workaround (see 03_pretrain.sbatch)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
cd "$REPO"
mkdir -p "$OUT"

PORT=$((29800 + $(cut -d',' -f1 <<< "$GPUS") * 7))

ACC_ARGS="--num_machines 1 --num_processes $NGPU --mixed_precision bf16 --main_process_port $PORT"
[ "$NGPU" -gt 1 ] && ACC_ARGS="--multi_gpu $ACC_ARGS"

TRAIN_ARGS="--config $CONFIG --data $DATA --out $OUT --steps $STEPS --save_every $SAVE_EVERY --batch_size $BATCH"
[ -n "$RESUME" ] && TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"

accelerate launch $ACC_ARGS training/pretrain.py $TRAIN_ARGS

echo "=== done $(date) ==="
