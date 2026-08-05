#!/bin/bash
# -----------------------------------------------------------------------------
# The two clean-corpus pretrains, run back to back on GPU 0-3, then vLLM back up.
#
#   setsid nohup bash script/16_pretrain_v3.sh > script/logs/pretrain_v3.log 2>&1 &
#
# The encode this depends on is already finished (686 shards / 1673 h in
# data/latents_133_015_clean), so this only has to train and evaluate.
#
# GPU 7 belongs to the Korean Matcha-TTS run; GPU 4-6 are free for short jobs.
# Pretrains run one after the other -- two concurrent runs measured 250-850
# samp/s on 07-27 against 3470 samp/s for a single run.
#
# No trap restores vLLM. On 07-28 a TERM trap did exactly that against busy GPUs
# and the container's restart policy then took all 8 GPUs the moment the encode
# finished, OOM-killing both pretrains. vLLM comes back only at the clean end.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"

# The container that has to be stopped to free the GPUs and started again
# afterwards. It belongs to another account on this node, so the name is not
# baked in: set VLLM_CONTAINER in the environment. Left unset, every step that
# touches it is skipped and the rest of the script runs unchanged.
VLLM_CONTAINER="${VLLM_CONTAINER:-}"
LATENTS=data/latents_133_015_clean
GPUS=0,1,2,3
BATCH=64
EVAL133=eval/eval_ko_dev.jsonl
EVAL015=eval/eval_ko_015dev.jsonl

say() { echo "[v3 $(date '+%m-%d %H:%M:%S')] $*"; }
trap 'say "interrupted -- vLLM NOT touched"' INT TERM

source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

eval_one() {
    local ckpt="$1/final/model.pt" hf="$1/final/hf"
    [ -f "$ckpt" ] || { say "  eval $2: SKIP ($ckpt missing)"; return 1; }
    say "  eval $2 on GPU $3"
    python training/convert_ckpt.py "$ckpt" --out "$hf" || return 1
    CUDA_VISIBLE_DEVICES="$3" python eval/benchmark.py --system freyatts --model "$hf" \
        --data "$EVAL133" --device cuda --out "eval/results/bench_${2}.json"
    CUDA_VISIBLE_DEVICES="$3" python eval/benchmark.py --system freyatts --model "$hf" \
        --data "$EVAL015" --device cuda --out "eval/results/bench_${2}_015dev.json"
    CUDA_VISIBLE_DEVICES="$3" python eval/speed.py --system freyatts --model "$hf" \
        --out "eval/results/speed_${2}.json"
    say "  eval $2 done"
}

latest_ckpt() {
    ls -d "$1"/step* 2>/dev/null | sed 's/.*step//' | sort -n | tail -1 \
        | while read -r n; do [ -n "$n" ] && echo "$1/step$n"; done
}

pretrain_retry() {
    local size="$1" out="$2" log="$3" attempt=1 resume=""
    while [ "$attempt" -le 5 ]; do
        say "  $size attempt $attempt${resume:+ (resume $resume)}"
        SIZE="$size" GPUS="$GPUS" BATCH="$BATCH" DATA="$LATENTS" OUT="$out" RESUME="$resume" \
            bash script/12_pretrain_v2.sh >> "$log" 2>&1
        local rc=$?
        [ "$rc" -eq 0 ] && [ -f "$out/final/model.pt" ] && { say "  $size done (attempt $attempt)"; return 0; }
        resume=$(latest_ckpt "$out")
        say "  $size FAILED rc=$rc; checkpoint: ${resume:-none}"
        # retry even with no checkpoint: an immediate failure is usually transient
        # (a GPU still held by something else), and restarting costs 2 minutes.
        attempt=$((attempt + 1))
        sleep 120
    done
    say "  $size: retries exhausted"; return 1
}

say "start: 183M then 88M on GPU $GPUS, data $LATENTS ($(ls $LATENTS/*.pt | wc -l) shards)"

pretrain_retry 183M checkpoints/pretrain_183M_v3_clean script/logs/pretrain_183M_v3.log \
    && eval_one checkpoints/pretrain_183M_v3_clean pretrain_183M_v3_clean 0

pretrain_retry 88M checkpoints/pretrain_88M_v3_clean script/logs/pretrain_88M_v3.log \
    && eval_one checkpoints/pretrain_88M_v3_clean pretrain_88M_v3_clean 0

say "all FreyaTTS work finished"
if [ -z "$VLLM_CONTAINER" ]; then
    say "VLLM_CONTAINER unset -- vLLM 복구 건너뜀"
elif [ "$(docker inspect -f '{{.State.Running}}' "$VLLM_CONTAINER" 2>/dev/null)" = "true" ]; then
    say "vLLM already running -- leaving it alone"
else
    say "restoring vLLM"
    docker start "$VLLM_CONTAINER" && say "docker start issued (~9 min)" || say "ERROR: docker start failed"
fi
say "pipeline complete"
