#!/bin/bash
# -----------------------------------------------------------------------------
# Fill in the missing FreyaTTS distills so every size has all five voices.
#
#   setsid nohup bash script/18_distill_queue.sh > script/logs/distill_queue.log 2>&1 &
#
# Existing: 337M has A-E; 183M and 88M have only A and E. This runs the six
# missing combinations, two at a time (GPU 0-3 and 4-7), each with the same
# 4 GPU x batch 64 setup every other distill in this repo used -- effective batch
# 256, so the results stay comparable to the ones already measured.
#
# 127M is deliberately excluded: it scored worse than 88M on a single-seed run
# and that inversion was never explained, so it is not a size worth shipping.
#
# Waits for any running eval to release its GPU before starting.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"
say() { echo "[queue $(date '+%m-%d %H:%M:%S')] $*"; }

say "waiting for in-flight synthesis/scoring to release the GPUs"
while pgrep -f "eval/score_wavs\.py|eval/benchmark\.py|eval/export_onnx\.py" > /dev/null 2>&1; do
    sleep 60
done
say "GPUs free"

# size:voice pairs still missing
JOBS=("183M:B" "183M:C" "183M:D" "88M:B" "88M:C" "88M:D")

run_one() {   # $1=SIZE $2=VOICE $3=GPUS
    local size="$1" v="$2" gpus="$3"
    local out="checkpoints/distill${size}_voice${v}"
    if [ -f "$out/final/model.pt" ]; then
        say "  $size voice$v already done, skipping"; return 0
    fi
    say "  start $size voice$v on GPU $gpus"
    VOICE="$v" SIZE="$size" GPUS="$gpus" \
        INIT="checkpoints/pretrain_${size}/final/model.pt" OUT="$out" \
        bash script/09_distill_small.sh > "script/logs/distill${size}_voice${v}.log" 2>&1
    if [ -f "$out/final/model.pt" ]; then
        say "  done $size voice$v"
    else
        say "  FAILED $size voice$v -- see script/logs/distill${size}_voice${v}.log"
    fi
}

# two at a time: one on each GPU half
i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
    a="${JOBS[$i]}"; b="${JOBS[$((i+1))]:-}"
    run_one "${a%%:*}" "${a##*:}" "0,1,2,3" &
    p1=$!
    if [ -n "$b" ]; then
        sleep 10   # stagger so the two runs do not collide picking a port
        run_one "${b%%:*}" "${b##*:}" "4,5,6,7" &
        p2=$!
        wait $p1 $p2
    else
        wait $p1
    fi
    i=$((i + 2))
done

say "all distills finished"
for size in 183M 88M; do
    for v in A B C D E; do
        d="checkpoints/distill${size}_voice${v}/final/model.pt"
        [ -f "$d" ] && echo "  ${size} voice${v}: ok" || echo "  ${size} voice${v}: MISSING"
    done
done
