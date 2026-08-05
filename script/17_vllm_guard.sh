#!/bin/bash
# -----------------------------------------------------------------------------
# Two jobs in one watcher:
#
#  1. Stop 16_pretrain_v3.sh from restoring vLLM itself. That script ends with a
#     `docker start`, which was correct when it owned the whole node -- but the
#     Korean Matcha run holds GPU 7 now, and vLLM needs ~76 GB on ALL 8 GPUs. It
#     would fail, and (before `--restart=no`) crash-loop, exactly as on 07-28.
#     So the orchestrator is killed the moment its last eval finishes, before it
#     reaches that line. Editing a running bash script is not safe; killing it
#     between stages is.
#
#  2. Restore vLLM only when every GPU is actually free -- no pretrain, no SFT,
#     no eval, and no Matcha. Then put the restart policy back to unless-stopped.
#
#   setsid nohup bash script/17_vllm_guard.sh > script/logs/vllm_guard.log 2>&1 &
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"
# The container that has to be stopped to free the GPUs and started again
# afterwards. It belongs to another account on this node, so the name is not
# baked in: set VLLM_CONTAINER in the environment. Left unset, every step that
# touches it is skipped and the rest of the script runs unchanged.
VLLM="${VLLM_CONTAINER:-}"

say() { echo "[guard $(date '+%m-%d %H:%M:%S')] $*"; }

# --- 1. disarm the orchestrator's own restore ---------------------------------
say "watching for the 88M eval to finish so the orchestrator can be stopped first"
while pgrep -f "bash script/16_pretrain_v3.sh" > /dev/null 2>&1; do
    if grep -q "eval pretrain_88M_v3_clean done" script/logs/pretrain_v3.log 2>/dev/null; then
        pid=$(pgrep -f "bash script/16_pretrain_v3.sh" | head -1)
        say "88M eval finished -- stopping orchestrator (pid $pid) before its vLLM restore"
        [ -n "$pid" ] && kill "$pid"
        break
    fi
    sleep 20
done
say "orchestrator no longer holds the restore"

# --- 2. wait for the node to be genuinely idle --------------------------------
busy() {
    pgrep -f "training/pretrain\.py|training/sft\.py|eval/benchmark\.py|eval/speed\.py" > /dev/null 2>&1 && return 0
    pgrep -f "matcha/train\.py|matcha/utils/generate_data_statistics\.py" > /dev/null 2>&1 && return 0
    pgrep -f "precompute_latents\.py|synth_matcha" > /dev/null 2>&1 && return 0
    return 1
}

say "waiting until no FreyaTTS/Matcha GPU work is left"
idle_rounds=0
while true; do
    if busy; then
        idle_rounds=0
    else
        idle_rounds=$((idle_rounds + 1))
        # three quiet minutes in a row, so a gap between two chained jobs does
        # not read as "all done"
        [ "$idle_rounds" -ge 3 ] && break
    fi
    sleep 60
done

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
say "node idle; total GPU memory in use: ${used} MiB"

if [ -z "$VLLM" ]; then
    say "VLLM_CONTAINER unset -- vLLM 복구 건너뜀"
elif [ "$(docker inspect -f '{{.State.Running}}' "$VLLM" 2>/dev/null)" = "true" ]; then
    say "vLLM already running -- nothing to do"
else
    say "restoring vLLM"
    if docker start "$VLLM"; then
        docker update --restart=unless-stopped "$VLLM" > /dev/null 2>&1 \
            && say "restart policy set back to unless-stopped"
        say "docker start issued (~9 min to serve); check: docker logs --tail 5 $VLLM"
    else
        say "ERROR: docker start failed -- run it by hand: docker start $VLLM"
    fi
fi
say "guard done"
