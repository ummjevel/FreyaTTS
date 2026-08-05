#!/bin/bash
# -----------------------------------------------------------------------------
# Distill a *smaller* pretrain checkpoint onto one of the Qwen3-TTS-synthesized
# voices, so the 88M/127M/183M variants can be compared against the shipped 337M
# voices on equal footing (same corpus, same 3000 steps, same lr).
#
# Unlike 06_sft_stage1.sbatch this passes the model dims explicitly: sft.py's
# defaults (and sft_stage1.yaml) hardcode the 337M shape (768/22/12/2048), which
# would silently build the wrong architecture and drop every weight on --init.
#
#   VOICE=A SIZE=183M GPUS=4,5,6,7 bash script/09_distill_small.sh
#
# Runs directly (not via sbatch) because the GPUs are allocated by hand here.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

VOICE="${VOICE:-A}"
SIZE="${SIZE:-183M}"
GPUS="${GPUS:-4,5,6,7}"
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

# model dims per size -- must match the pretrain checkpoint being initialized from
case "$SIZE" in
  88M)  DIMS="--d_model 512 --depth 12 --heads 8  --ff 1536" ;;
  127M) DIMS="--d_model 512 --depth 16 --heads 8  --ff 2048" ;;
  183M) DIMS="--d_model 640 --depth 16 --heads 10 --ff 2048" ;;
  337M) DIMS="--d_model 768 --depth 22 --heads 12 --ff 2048" ;;
  *) echo "ERROR: unknown SIZE=$SIZE (want 88M|127M|183M|337M)"; exit 1 ;;
esac

INIT="${INIT:-checkpoints/pretrain_${SIZE}/final/model.pt}"
DATA="${DATA:-data/latents_distill_voice${VOICE}}"
OUT="${OUT:-checkpoints/distill${SIZE}_voice${VOICE}}"
STEPS="${STEPS:-3000}"        # same as the shipped 337M voices (final = step3000)
SAVE_EVERY="${SAVE_EVERY:-500}"
LR="${LR:-1.0e-4}"            # sft_stage1.yaml

[ -f "$INIT" ] || { echo "ERROR: init checkpoint not found: $INIT"; exit 1; }
[ -d "$DATA" ] || { echo "ERROR: distill latents not found: $DATA"; exit 1; }

# Refuse to start if another run is already training into the same OUT dir.
# Two concurrent runs sharing an output dir interleave their checkpoint writes,
# and (before the dynamic port below) shared a rendezvous port, which makes both
# runs' weights untrustworthy. Happened once on 2026-07-27; caught by eye only.
if pgrep -f "training/sft\.py.*--out $OUT( |\$)" > /dev/null 2>&1; then
    echo "ERROR: a run is already training into $OUT -- refusing to start a second one"
    pgrep -fa "training/sft\.py.*--out $OUT( |\$)" | head -2
    exit 1
fi

echo "=== distill $SIZE -> voice$VOICE  GPUS=$GPUS ($NGPU)  $(date) ==="
echo "    init=$INIT  data=$DATA  out=$OUT  steps=$STEPS"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# same OpenStack-VM NVLink-P2P workaround as pretrain/sft
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
cd "$REPO"
mkdir -p "$OUT"

# Ask the kernel for a free rendezvous port. Deriving it from the GPU ids (as
# this script used to) hands the SAME port to two runs on the same GPUs, so their
# torch.distributed groups collide instead of staying independent.
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

ACC_ARGS="--num_machines 1 --num_processes $NGPU --mixed_precision bf16 --main_process_port $PORT"
[ "$NGPU" -gt 1 ] && ACC_ARGS="--multi_gpu $ACC_ARGS"

accelerate launch $ACC_ARGS training/sft.py \
    --init "$INIT" --data "$DATA" --out "$OUT" \
    $DIMS --steps "$STEPS" --save_every "$SAVE_EVERY" --lr "$LR" \
    --lambda_dur 0.1 --batch_size 64 --max_frames 500 --num_workers 3

echo "=== done $(date) ==="
