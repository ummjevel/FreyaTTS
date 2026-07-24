#!/bin/bash
#SBATCH --job-name=freya_short
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/short_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/short_%j.err

# -----------------------------------------------------------------------------
# SFT stage 2 prep: extract short (1-3s, 1-3 eojeol) utterances from the SAME
# single-speaker manifest, force-aligned via MMS_FA (romanized through uroman),
# re-encoded to latents. Uses ONE GPU (usable: 4,5,6,7).
#   SPEAKER=005 GPUID=4 sbatch script/07_extract_short.sbatch
#
# Builds a MIX dir data/latents_sft_short = short segments + the stage-1 latents
# (symlinked), which is what sft_stage2.yaml reads. Korean-tuned args match
# training/run_extract_short_ko.sh (--max_words 3 --max_s 3.0 --max_repeats 40).
#   GPUID=4 sbatch script/07_extract_short.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

GPUID="${GPUID:-4}"
[ -n "${1:-}" ] && GPUID="$1"

# same single-speaker manifest that 05_build_sft_data produced
MANIFEST="${MANIFEST:-data/manifest_sft.jsonl}"
SFT_LATENTS="${SFT_LATENTS:-data/latents_sft}"
OUT="${OUT:-data/latents_sft_short}"

echo "=== Job $SLURM_JOB_ID  GPU=$GPUID  MANIFEST=$MANIFEST  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUID"
cd "$REPO"

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST (run 05_build_sft_data first)"; exit 1; }
mkdir -p "$OUT"

echo "--- extracting short segments -> $OUT ---"
python training/extract_short_segments.py \
    --manifest "$MANIFEST" \
    --out "$OUT" \
    --max_words 3 \
    --max_s 3.0 \
    --max_repeats 40 \
    --device cuda

# make the mix: symlink the stage-1 full latents alongside the short shards so
# stage 2 trains on both (LatentDataset globs *.pt in this one dir).
if [ -d "$SFT_LATENTS" ]; then
    echo "--- linking $SFT_LATENTS/*.pt into $OUT (mix) ---"
    for f in "$SFT_LATENTS"/*.pt; do
        ln -sf "$(readlink -f "$f")" "$OUT/main_$(basename "$f")"
    done
fi

echo "=== done $(date) ==="
ls "$OUT"/*.pt 2>/dev/null | wc -l | sed 's/^/total shards in mix: /'
