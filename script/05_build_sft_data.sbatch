#!/bin/bash
#SBATCH --job-name=freya_sftdata
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/data/users/voice/zoey/FreyaTTS/script/logs/sftdata_%j.out
#SBATCH --error=/data/users/voice/zoey/FreyaTTS/script/logs/sftdata_%j.err

# -----------------------------------------------------------------------------
# SFT prep: build a SINGLE-SPEAKER latent set for voice-lock SFT.
#
# FreyaTTS SFT collapses the model onto whichever speaker's corpus you fine-tune
# on, so stage 1 needs ONE 성우 (and one style), not the full pretrain set.
# AIHub id format: <STYLE>-<f2>-<emotion>-<SPEAKER>-<utt>
#   field1 STYLE  = 발화 스타일: D=대화체 M=독백체 F=구연체 A=애니체 K=친절체 S=중계체 N=낭독체
#   field4 SPEAKER = 3-digit speaker id (89 speakers)
# We filter the already-normalized data/manifest_train.jsonl by STYLE+SPEAKER,
# then encode -> data/latents_sft.
#
#   STYLE=D SPEAKER=010 GPUID=4 sbatch script/05_build_sft_data.sbatch   # 대화체, 화자 010
#   sbatch script/05_build_sft_data.sbatch 010 5                          # (SPEAKER, GPU) args
#
# 대화체 coverage by speaker (train):
#   awk -F'\t' '{split($1,a,"-");if(a[1]=="D")print a[4]}' <train.tsv> | sort | uniq -c | sort -rn
# → 010 (5394), 005 (5186), 063 (4485) are the best-covered 대화체 speakers.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO=/data/users/voice/zoey/FreyaTTS
VENV=$REPO/.venv

STYLE="${STYLE-D}"                  # field1 발화 스타일 (D=대화체); STYLE="" = 전 스타일
SPEAKER="${SPEAKER:-010}"           # field4 speaker id (010 = most 대화체 data)
[ -n "${1:-}" ] && SPEAKER="$1"
GPUID="${GPUID:-4}"
[ -n "${2:-}" ] && GPUID="$2"

SRC="${SRC:-data/manifest_train.jsonl}"
MANIFEST="${MANIFEST:-data/manifest_sft.jsonl}"
OUT="${OUT:-data/latents_sft}"

echo "=== Job $SLURM_JOB_ID  STYLE=${STYLE:-any}  SPEAKER=$SPEAKER  GPU=$GPUID  $(date) ==="
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPUID"
cd "$REPO"

# keep lines whose audio basename is <STYLE>-<f2>-<emotion>-<SPEAKER>-<utt>.wav
# (e.g. .../wavs/D-A1-D-010-0114.wav). STYLE="" matches any style ([A-Z]).
S="${STYLE:-[A-Z]}"
echo "--- filtering $SRC -> $MANIFEST (style ${STYLE:-any}, speaker $SPEAKER) ---"
grep -E "/wavs/${S}-[A-Z][A-Z0-9]-[A-Z]-${SPEAKER}-[0-9]+\.wav" "$SRC" > "$MANIFEST" || true
n=$(wc -l < "$MANIFEST")
echo "style=${STYLE:-any} speaker=$SPEAKER: $n utterances"
[ "$n" -eq 0 ] && { echo "ERROR: no utterances for style=${STYLE:-any} speaker=$SPEAKER"; exit 1; }

echo "--- encoding -> $OUT ---"
python training/precompute_latents.py --manifest "$MANIFEST" --out "$OUT" --device cuda

echo "=== done $(date) ==="
