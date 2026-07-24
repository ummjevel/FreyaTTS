#!/bin/bash
#SBATCH --job-name=freya_manifest
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/manifest_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/manifest_%j.err

# -----------------------------------------------------------------------------
# Step 1/3 : AIHub 동시고려 tsv -> FreyaTTS {audio, text} manifest (train + dev)
# CPU only. Applies freyatts.pipeline.normalize() (digit spelling + jamo
# decomposition) so training text matches inference tokenization.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
DS="/data/datasets/voice/133.감성_및_발화_스타일_동시_고려_음성합성_데이터"
VENV=$REPO/.venv

# tsv columns are tab-separated: <id>\t<text>\t<broken wavpath>.
# The 3rd column path is stale, so we resolve audio from the id (col 0)
# against the real wavs dir; build_manifest_ko appends ".wav" automatically.
WAV_DIR="$DS/wavs"

echo "=== Job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=""   # CPU only, never grab a GPU
cd "$REPO"
mkdir -p data

for split in train dev; do
    echo "--- building $split manifest ---"
    python training/build_manifest_ko.py \
        --metadata "$DS/train_tsv/custom_dongsi_${split}.tsv" \
        --sep $'\t' \
        --audio-col 0 \
        --text-col 1 \
        --wav-dir "$WAV_DIR" \
        --out "data/manifest_${split}.jsonl"
done

# split the train manifest into NSHARD round-robin parts so 02's job array
# can encode them in parallel, one GPU each (balanced audio duration per part).
NSHARD="${NSHARD:-4}"
echo "--- splitting train manifest into $NSHARD parts (round-robin) ---"
rm -f data/manifest_train.part*.jsonl
for i in $(seq 0 $((NSHARD - 1))); do
    awk -v n="$NSHARD" -v r="$i" 'NR % n == r' data/manifest_train.jsonl \
        > "data/manifest_train.part${i}.jsonl"
done

echo "=== done $(date) ==="
wc -l data/manifest_train.jsonl data/manifest_dev.jsonl data/manifest_train.part*.jsonl
