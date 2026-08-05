#!/bin/bash
# -----------------------------------------------------------------------------
# Second half of the clean-corpus encode: parts 4-6 on GPU 4-6. GPU 7 is left
# free for Matcha-TTS training. 14_day.sh runs parts 0-3 on GPU 0-3 once the
# voiceA distill releases them, and blocks until no precompute process is left,
# so the two halves join up.
#
#   setsid nohup bash script/15_precompute_second_half.sh > script/logs/precompute_half2.log 2>&1 &
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"

MANIFEST=data/manifest_train_133_015_clean
LATENTS=data/latents_133_015_clean

say() { echo "[half2 $(date '+%m-%d %H:%M:%S')] $*"; }

say "waiting for the ODE-step sweep to release GPU 4-7"
while pgrep -f "eval/benchmark\.py|eval/speed\.py" > /dev/null 2>&1; do sleep 60; done
say "step sweep done"

# 14_day.sh writes the part files; don't race it
say "waiting for the 8-way manifest split"
until [ -s "${MANIFEST}.part6.jsonl" ]; do sleep 30; done
sleep 10
say "launching parts 4-6 on GPU 4-6 (GPU 7 is running Matcha-TTS)"

PARTS=4,5,6 GPUS=4,5,6 OUT="$LATENTS" MANIFEST_BASE="$MANIFEST" \
    bash script/10_precompute_133_015.sh
say "parts 4-6 finished (rc=$?)"
