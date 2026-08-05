#!/bin/bash
# -----------------------------------------------------------------------------
# Watch the running 183M distill jobs and submit 04_eval.sbatch for each as soon
# as it finishes.
#
#   setsid nohup bash script/11_eval_after_distill.sh > script/logs/eval_watch.log 2>&1 &
#
# Why a watcher and not `sbatch --dependency=afterok:<jobid>`: the distills were
# launched directly with accelerate (the GPUs are hand-allocated on this node),
# so there is no slurm job id to depend on. The eval itself IS submitted with
# sbatch, so it shows up in squeue normally.
#
# Trigger is the `[done]` line sft.py prints *after* writing final/model.pt, so
# the checkpoint is guaranteed complete -- waiting on the file alone would race
# a partially written save.
#
# TAG is passed explicitly: 04_eval.sbatch would otherwise name the results after
# the checkpoint dir ("final") and overwrite eval/results/bench_final.json, which
# holds the 337M pretrain result.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
cd "$REPO"

# voice:gpu pairs to watch (the eval runs on that GPU, free by then).
# Override to watch a subset:  WATCH_JOBS="E:0" bash script/11_eval_after_distill.sh
read -ra JOBS <<< "${WATCH_JOBS:-A:4 E:0}"

for job in "${JOBS[@]}"; do
    V="${job%%:*}"
    GPU="${job##*:}"
    LOG="script/logs/distill183M_voice${V}.log"
    CKPT="checkpoints/distill183M_voice${V}/final/model.pt"
    TAG="distill183M_voice${V}"

    echo "[watch] waiting for voice$V -> $LOG"
    # "no process" only means death AFTER the run has been seen alive at least
    # once. Without that latch, a restart window (or a slow launch) reads as a
    # dead job and the eval is skipped -- which is exactly what happened to
    # voiceE at 17:48 on 2026-07-27, while its run was being relaunched.
    seen_alive=0
    until grep -q "^\[done\]" "$LOG" 2>/dev/null; do
        if pgrep -f "training/sft.py.*distill183M_voice${V}" > /dev/null 2>&1; then
            seen_alive=1
        elif [ "$seen_alive" = "1" ]; then
            sleep 20   # grace period: the marker may land as the process exits
            if ! grep -q "^\[done\]" "$LOG" 2>/dev/null; then
                echo "[watch] ERROR: voice$V died after running; skipping eval"
                continue 2
            fi
        fi
        sleep 60
    done

    [ -f "$CKPT" ] || { echo "[watch] ERROR: $CKPT missing after [done]; skipping"; continue; }
    echo "[watch] voice$V done $(date) -- submitting eval on GPU $GPU"
    TAG="$TAG" CKPT="$CKPT" GPUID="$GPU" sbatch script/04_eval.sbatch
done

echo "[watch] all evals submitted $(date)"
echo "        results -> eval/results/bench_distill183M_voice{A,E}.json"
echo "                   eval/results/speed_distill183M_voice{A,E}.json"
