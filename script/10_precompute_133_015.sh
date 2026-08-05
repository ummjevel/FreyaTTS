#!/bin/bash
# -----------------------------------------------------------------------------
# Encode the MERGED 133+015 corpus (1,487,965 utterances, 2.7x the 133-only set)
# to AudioVAE2 latents. The manifest is pre-split into 8 round-robin parts; this
# runs any subset of them, one GPU per part, pairing PARTS to GPUS positionally.
#
#   PARTS=0,1,2,3 GPUS=4,5,6,7 CLEAN=1 bash script/10_precompute_133_015.sh
#   PARTS=4,5,6,7 GPUS=0,1,2,3 bash script/10_precompute_133_015.sh
#
# Splitting the run in two halves lets each half start the moment its GPUs free
# up, instead of idling until all 8 are available at once.
#
# Measured throughput: ~86.6 s per 2000-clip shard on one H100. Each part is
# ~186k utterances (~93 shards), so a part takes ~2.2 h regardless of how many
# run in parallel.
#
# CLEAN=1 removes the shards left by the aborted single-GPU run of 2026-07-27
# (shard_000{00..10}.pt). Those cover utterances that the part splits also cover,
# and training globs *.pt, so leaving them would feed ~22k utterances twice.
# Pass CLEAN=1 on the FIRST half only -- it would delete good output otherwise.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

PARTS="${PARTS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CLEAN="${CLEAN:-0}"
OUT="${OUT:-data/latents_133_015}"
MANIFEST_BASE="${MANIFEST_BASE:-data/manifest_train_133_015}"

cd "$REPO"
source "$VENV/bin/activate"
mkdir -p "$OUT" script/logs

IFS=',' read -ra P <<< "$PARTS"
IFS=',' read -ra G <<< "$GPUS"
if [ "${#P[@]}" -ne "${#G[@]}" ]; then
    echo "ERROR: PARTS (${#P[@]}) and GPUS (${#G[@]}) must have the same length"
    exit 1
fi

if [ "$CLEAN" = "1" ]; then
    stale=$(ls "$OUT"/shard_*.pt 2>/dev/null | wc -l)
    if [ "$stale" -gt 0 ]; then
        echo "removing $stale stale shard_*.pt from the aborted 2026-07-27 run"
        rm -f "$OUT"/shard_*.pt "$OUT"/manifest.json
    fi
fi

pids=()
for i in "${!P[@]}"; do
    part="${P[$i]}"; gpu="${G[$i]}"
    m="${MANIFEST_BASE}.part${part}.jsonl"
    [ -f "$m" ] || { echo "ERROR: missing $m"; exit 1; }
    if pgrep -f "precompute_latents\.py.*--prefix part${part}( |\$)" > /dev/null 2>&1; then
        echo "part$part already running -- skipping"
        continue
    fi
    echo "part$part -> GPU $gpu  ($(wc -l < "$m") utterances)"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python training/precompute_latents.py \
        --manifest "$m" --out "$OUT" --prefix "part${part}" --device cuda \
        > "script/logs/precompute_133_015_part${part}.log" 2>&1 &
    pids+=($!)
done

echo "=== ${#pids[@]} parts launched $(date); waiting ==="
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== parts [$PARTS] finished $(date), rc=$rc ==="
ls "$OUT"/*.pt 2>/dev/null | wc -l
exit $rc
