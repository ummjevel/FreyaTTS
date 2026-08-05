#!/bin/bash
# -----------------------------------------------------------------------------
# Unattended overnight pipeline (2026-07-27 evening -> next morning).
#
#   setsid nohup bash script/13_overnight.sh > script/logs/overnight.log 2>&1 &
#
#   stage 0  wait for the voiceE distill eval (11_eval_after_distill.sh)
#   stage 1  precompute parts 4-7 on GPU 0-3, then join the first half
#            (parts 0-3, already running on GPU 4-7 since 18:10)   ~2.2 h each
#   stage 2  two independent chains, one per GPU half:
#              GPU 0-3 : pretrain 88M v2  -> eval -> distill onto voiceA -> eval
#              GPU 4-7 : pretrain 183M v2 -> eval
#            The 88M finishes ~2 h before the 183M, so its chain keeps that half
#            busy instead of idling until the 183M lands.
#   stage 3  hand the GPUs back: hand the GPUs back (restart the vLLM container)
#
# Stage 3 runs even if an earlier stage fails -- if the pipeline stops early the
# GPUs are idle and the coding-agent server should be up before people arrive.
# It never runs while training is live (vLLM needs ~76 GB on every GPU), and it
# leaves the container alone if someone has already brought it up.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"
mkdir -p script/logs

# The container that has to be stopped to free the GPUs and started again
# afterwards. It belongs to another account on this node, so the name is not
# baked in: set VLLM_CONTAINER in the environment. Left unset, every step that
# touches it is skipped and the rest of the script runs unchanged.
VLLM_CONTAINER="${VLLM_CONTAINER:-}"
LATENTS=data/latents_133_015
OUT88=checkpoints/pretrain_88M_v2_133_015
OUT183=checkpoints/pretrain_183M_v2_133_015
DISTILL88=checkpoints/distill88Mv2_voiceA

say() { echo "[overnight $(date '+%m-%d %H:%M:%S')] $*"; }

restore_vllm() {
    if [ -z "${VLLM_CONTAINER}" ]; then
        say "VLLM_CONTAINER unset -- nothing to restart"
        return 0
    fi
    local state
    state=$(docker inspect -f '{{.State.Running}}' "$VLLM_CONTAINER" 2>/dev/null)
    if [ "$state" = "true" ]; then
        say "vLLM is already running -- someone brought it back; leaving it alone"
        return
    fi
    say "restoring vLLM ($VLLM_CONTAINER)"
    if docker start "$VLLM_CONTAINER"; then
        say "docker start issued; startup takes ~9 min"
    else
        say "ERROR: docker start failed -- restore by hand: docker start $VLLM_CONTAINER"
    fi
}

# convert -> WER/CER -> speed for one checkpoint dir. $1=dir $2=tag $3=gpu
eval_one() {
    local ckpt="$1/final/model.pt" hf="$1/final/hf"
    if [ ! -f "$ckpt" ]; then say "  eval $2: SKIP ($ckpt missing)"; return 1; fi
    say "  eval $2 on GPU $3"
    python training/convert_ckpt.py "$ckpt" --out "$hf" \
      && CUDA_VISIBLE_DEVICES="$3" python eval/benchmark.py --system freyatts \
           --model "$hf" --data eval/eval_ko_dev.jsonl --device cuda \
           --out "eval/results/bench_${2}.json" \
      && CUDA_VISIBLE_DEVICES="$3" python eval/speed.py --system freyatts \
           --model "$hf" --out "eval/results/speed_${2}.json"
    say "  eval $2 finished (exit $?)"
}

# --- stage 0: the two distill evals -------------------------------------------
say "stage 0: waiting for the 183M distill evals"
for v in E; do
    f="eval/results/bench_distill183M_voice${v}.json"
    waited=0
    until [ -f "$f" ]; do
        sleep 60
        waited=$((waited + 1))
        if [ "$waited" -gt 240 ]; then
            say "WARNING: $f never appeared (4 h); continuing without it"
            break
        fi
    done
    [ -f "$f" ] && say "  got $f"
done
while pgrep -f "eval/benchmark.py|eval/speed.py" > /dev/null 2>&1; do
    say "  an eval is still running; waiting for its GPU"
    sleep 60
done

# --- stage 1: precompute the merged corpus ------------------------------------
say "stage 1: precompute parts 4-7 on GPU 0-3 (parts 0-3 already running on GPU 4-7)"
PARTS=4,5,6,7 GPUS=0,1,2,3 bash script/10_precompute_133_015.sh
say "stage 1: second half done; waiting for the first half to land"
while pgrep -f "precompute_latents\.py" > /dev/null 2>&1; do sleep 60; done
ok=$(grep -l "^done:" script/logs/precompute_133_015_part*.log 2>/dev/null | wc -l)
shards=$(ls "$LATENTS"/*.pt 2>/dev/null | wc -l)
say "stage 1 finished: $ok/8 parts reported done, $shards shards on disk"
if [ "$ok" -lt 8 ]; then
    say "ERROR: precompute incomplete -- NOT starting pretrain (would train on partial data)"
    restore_vllm
    exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# --- stage 2: two chains, one per GPU half ------------------------------------
chain_88M() {
    say "chain 88M: pretrain (GPU 0-3)"
    SIZE=88M GPUS=0,1,2,3 DATA="$LATENTS" OUT="$OUT88" \
        bash script/12_pretrain_v2.sh > script/logs/pretrain_88M_v2.log 2>&1
    say "chain 88M: pretrain exit=$?"
    eval_one "$OUT88" pretrain_88M_v2_133_015 0

    # fill this half until the 183M lands: distill the fresh 88M onto voiceA.
    # 88M is the on-device candidate (1.13 GB peak VRAM), so a voice-locked 88M
    # is the thing to compare against the shipped 337M voiceA (WER 0.224).
    if [ -f "$OUT88/final/model.pt" ]; then
        say "chain 88M: distill -> voiceA (GPU 0-3)"
        VOICE=A SIZE=88M GPUS=0,1,2,3 INIT="$OUT88/final/model.pt" OUT="$DISTILL88" \
            bash script/09_distill_small.sh > script/logs/distill88Mv2_voiceA.log 2>&1
        say "chain 88M: distill exit=$?"
        eval_one "$DISTILL88" distill88Mv2_voiceA 0
    fi
    say "chain 88M: done"
}

chain_183M() {
    say "chain 183M: pretrain (GPU 4-7)"
    SIZE=183M GPUS=4,5,6,7 DATA="$LATENTS" OUT="$OUT183" \
        bash script/12_pretrain_v2.sh > script/logs/pretrain_183M_v2.log 2>&1
    say "chain 183M: pretrain exit=$?"
    eval_one "$OUT183" pretrain_183M_v2_133_015 4
    say "chain 183M: done"
}

say "stage 2: launching both chains"
chain_88M  & C88=$!
chain_183M & C183=$!
wait $C88 $C183
say "stage 2: both chains finished"

# --- stage 3: give the GPUs back ----------------------------------------------
say "stage 3: all FreyaTTS work finished"
restore_vllm
say "pipeline complete"
