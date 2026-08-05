#!/bin/bash
# -----------------------------------------------------------------------------
# 2026-07-28 daytime pipeline: redo the merged-corpus experiment on CLEAN data.
#
#   setsid nohup bash script/14_day.sh > script/logs/day.log 2>&1 &
#
# Why a redo: the 07-27 merged manifest included all 112,157 clips of the 015
# *dev* split, so nothing held out from 015 could be used to evaluate. The
# corrected manifest (manifest_train_133_015_clean.jsonl) drops them, and the
# latents have to be re-encoded from it.
#
#   stage 0  wait for the 88M voiceA distill, eval it           ~20 min
#   stage 1  precompute clean merged latents                    ~2.4 h
#            7-way split: parts 0-3 here on GPU 0-3, parts 4-6 launched
#            separately on GPU 4-6. GPU 7 is reserved for Matcha-TTS training,
#            so the encode gets 7 GPUs, not 8. Waits for all 7.
#   stage 2  pin training to GPU 0-3
#   stage 3  pretrain 183M on clean data, then eval             ~4-5 h
#   stage 4  pretrain 88M  on clean data, then eval             ~3 h
#   stage 5  hand the GPUs back (restart the vLLM container)
#
# Training stays on 4 GPUs by request. Data prep is not training, so it uses
# whatever is idle -- on 4 GPUs alone the encode would take 4.4 h instead of 2.2.
#
# Pretrains run SEQUENTIALLY, not side by side: on 07-27 two concurrent runs got
# 250-850 samp/s while a single run got 3470, so splitting the node across two
# jobs cost ~6 h of wall clock.
#
# Every pretrain is wrapped in retry-with-resume. The 183M died at step 6100 on
# 07-27 with the node's known NVLink peer-access fault, and with nothing watching
# it the GPUs idled for 6.5 h.
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
MANIFEST=data/manifest_train_133_015_clean
LATENTS=data/latents_133_015_clean
EVAL133=eval/eval_ko_dev.jsonl
EVAL015=eval/eval_ko_015dev.jsonl

say() { echo "[day $(date '+%m-%d %H:%M:%S')] $*"; }

restore_vllm() {
    if [ -z "${VLLM_CONTAINER}" ]; then
        say "VLLM_CONTAINER unset -- nothing to restart"
        return 0
    fi
    local state
    state=$(docker inspect -f '{{.State.Running}}' "$VLLM_CONTAINER" 2>/dev/null)
    if [ "$state" = "true" ]; then
        say "vLLM already running -- leaving it alone"; return
    fi
    say "restoring vLLM ($VLLM_CONTAINER)"
    docker start "$VLLM_CONTAINER" && say "docker start issued (~9 min to serve)" \
        || say "ERROR: docker start failed -- run: docker start $VLLM_CONTAINER"
}
# NO trap that restores vLLM. An earlier version restored it on INT/TERM so the
# GPUs would not sit idle if the pipeline was killed. On 2026-07-28 that fired
# twice while I was re-planning GPU allocation: vLLM came up against busy GPUs,
# crash-looped 111 times under `restart: unless-stopped`, and then grabbed all 8
# GPUs 9 seconds after the encode finished -- killing both pretrains with OOM.
# Restoring someone else's service is only safe once our work is genuinely done.
trap 'say "pipeline interrupted -- vLLM NOT touched; restore it by hand when done"' INT TERM

# convert + WER/CER on both eval sets + speed. $1=ckpt dir $2=tag $3=gpu
eval_one() {
    local ckpt="$1/final/model.pt" hf="$1/final/hf"
    if [ ! -f "$ckpt" ]; then say "  eval $2: SKIP ($ckpt missing)"; return 1; fi
    say "  eval $2 on GPU $3"
    python training/convert_ckpt.py "$ckpt" --out "$hf" || return 1
    CUDA_VISIBLE_DEVICES="$3" python eval/benchmark.py --system freyatts --model "$hf" \
        --data "$EVAL133" --device cuda --out "eval/results/bench_${2}.json"
    CUDA_VISIBLE_DEVICES="$3" python eval/benchmark.py --system freyatts --model "$hf" \
        --data "$EVAL015" --device cuda --out "eval/results/bench_${2}_015dev.json"
    CUDA_VISIBLE_DEVICES="$3" python eval/speed.py --system freyatts --model "$hf" \
        --out "eval/results/speed_${2}.json"
    say "  eval $2 finished"
}

# newest stepN checkpoint under $1, empty if none
latest_ckpt() {
    ls -d "$1"/step* 2>/dev/null | sed 's/.*step//' | sort -n | tail -1 \
        | while read -r n; do [ -n "$n" ] && echo "$1/step$n"; done
}

# pretrain with retry-from-latest-checkpoint. $1=size $2=gpus $3=batch $4=out $5=log
pretrain_retry() {
    local size="$1" gpus="$2" batch="$3" out="$4" log="$5" attempt=1 resume=""
    while [ "$attempt" -le 5 ]; do
        say "  $size attempt $attempt${resume:+ (resume $resume)}"
        SIZE="$size" GPUS="$gpus" BATCH="$batch" DATA="$LATENTS" OUT="$out" RESUME="$resume" \
            bash script/12_pretrain_v2.sh >> "$log" 2>&1
        local rc=$?
        if [ "$rc" -eq 0 ] && [ -f "$out/final/model.pt" ]; then
            say "  $size finished on attempt $attempt"; return 0
        fi
        resume=$(latest_ckpt "$out")
        say "  $size FAILED (rc=$rc); latest checkpoint: ${resume:-none}"
        if [ -z "$resume" ]; then
            say "  $size: no checkpoint to resume from -- giving up"; return 1
        fi
        attempt=$((attempt + 1))
        sleep 30
    done
    say "  $size: retries exhausted"; return 1
}

# --- stage 0: the 88M distills ------------------------------------------------
say "stage 0: waiting for the 88M distills"
for v in A; do
    log="script/logs/distill88M_voice${v}.log"
    seen=0; waited=0
    until grep -q "^\[done\]" "$log" 2>/dev/null; do
        if pgrep -f "training/sft.py.*distill88M_voice${v}" > /dev/null 2>&1; then
            seen=1
        elif [ "$seen" = "1" ]; then
            sleep 20
            grep -q "^\[done\]" "$log" 2>/dev/null || { say "  voice$v died; skipping"; break; }
        fi
        sleep 30
        waited=$((waited + 1)); [ "$waited" -gt 240 ] && { say "  voice$v timeout"; break; }
    done
done
while pgrep -f "training/sft\.py" > /dev/null 2>&1; do sleep 30; done
source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
eval_one checkpoints/distill88M_voiceA distill88M_voiceA 0
say "stage 0 done"

# --- stage 1: clean latents ---------------------------------------------------
# The 7-way split is made up front (parts 4-6 start on GPU 4-6 before this
# pipeline reaches stage 1), so only verify it here.
say "stage 1: checking the 7-way manifest split"
for i in 0 1 2 3 4 5 6; do
    [ -s "${MANIFEST}.part${i}.jsonl" ] || { say "ERROR: missing ${MANIFEST}.part${i}.jsonl"; restore_vllm; exit 1; }
done

say "stage 1: precompute clean latents, parts 0-3 on GPU 0-3 -> $LATENTS"
PARTS=0,1,2,3 GPUS=0,1,2,3 OUT="$LATENTS" MANIFEST_BASE="$MANIFEST" \
    bash script/10_precompute_133_015.sh
say "stage 1: parts 0-3 done; waiting for parts 4-6 (GPU 4-6)"
while pgrep -f "precompute_latents\.py" > /dev/null 2>&1; do sleep 60; done
ok=$(grep -l "^done:" script/logs/precompute_133_015_part*.log 2>/dev/null | wc -l)
shards=$(ls "$LATENTS"/*.pt 2>/dev/null | wc -l)
say "stage 1 finished: $ok/7 parts done, $shards shards"
if [ "$ok" -lt 7 ]; then
    say "ERROR: precompute incomplete -- stopping before pretrain"; restore_vllm; exit 1
fi

# --- stage 2: training GPUs -------------------------------------------------
# Pinned to GPU 0-3 by request: the other half stays free for the sherpa-onnx
# Matcha speed study and whatever follows it. Batch 64 x 4 = the effective 256
# every other run in this repo is compared against.
PGPUS=0,1,2,3
PBATCH=64
say "stage 2: training on GPU $PGPUS (batch $PBATCH/device)"

# --- stage 3 / 4: the two pretrains, one after the other ----------------------
say "stage 3: pretrain 183M on clean merged data"
pretrain_retry 183M "$PGPUS" "$PBATCH" checkpoints/pretrain_183M_v3_clean script/logs/pretrain_183M_v3.log \
    && eval_one checkpoints/pretrain_183M_v3_clean pretrain_183M_v3_clean 0

say "stage 4: pretrain 88M on clean merged data"
pretrain_retry 88M "$PGPUS" "$PBATCH" checkpoints/pretrain_88M_v3_clean script/logs/pretrain_88M_v3.log \
    && eval_one checkpoints/pretrain_88M_v3_clean pretrain_88M_v3_clean 0

# --- stage 5 ------------------------------------------------------------------
say "stage 5: all work finished"
restore_vllm
say "pipeline complete"
