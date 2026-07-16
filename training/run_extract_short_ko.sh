#!/usr/bin/env bash
# Korean short-utterance extraction for FreyaTTS SFT stage 2 (voice: single
# 성우, style: 대화체). Wraps extract_short_segments.py with settings tuned
# for Korean, not the script's original Turkish defaults.
#
# Usage:
#   training/run_extract_short_ko.sh [manifest.jsonl]
#
# The manifest must be a JSONL file of {"audio": path, "text": str}, already
# filtered down to one voice actor's 대화체 (conversational-style) recordings.
# That filtering script is still pending -- it needs the actual AI-Hub
# metadata schema (speaker id / style / emotion field names) to write
# correctly, so this file just takes the manifest path as an argument for now.

set -euo pipefail

MANIFEST="${1:?usage: $0 <manifest.jsonl>}"

python training/extract_short_segments.py \
  --manifest "$MANIFEST" \
  --out data/latents_short_ko \
  --max_words 3 \
  --max_s 3.0 \
  --max_repeats 40 \
  --device cuda

# ---------------------------------------------------------------------------
# Why these numbers (vs. the script's original Turkish defaults):
#
# --max_words 3 (was 2)
#   Spans are counted in eojeol (space-separated words), not syllables. Short
#   Korean conversational replies are commonly 1-3 eojeol -- "네", "알겠습니다",
#   "네, 알겠습니다, 감사합니다" -- and capping at 2 eojeol under-uses the 3s
#   window below, since 2 eojeol rarely reaches 3s at natural speaking rate.
#   3 was picked to actually reach the target range, not because Korean has
#   some fixed "3-word" unit.
#
# --max_s 3.0 (was 1.6)
#   Upper bound for the "short utterance" target range (1-3s), itself a round
#   number, not a measured value -- adjust if eval shows over/under-coverage
#   at either end.
#   Important: this is an upper bound only. The *lower* bound is a hardcoded
#   0.2s floor inside extract_short_segments.py (duration < 0.2 → dropped),
#   not a CLI flag, so segments under 1s already pass through untouched.
#   That matters here because some TTS pipelines (e.g. ZipVoice) discard
#   everything under 1s outright -- which would throw away exactly the
#   single-eojeol acknowledgements ("응", "알았어") this stage exists to add
#   coverage for. Do not raise the floor to "fix" short clips; the floor is
#   deliberately low.
#
# --max_repeats 40 (unchanged)
#   Caps how many copies of the same phrase get kept, so high-frequency
#   fillers ("네", "음") don't dominate the shard. This isn't a Korean-specific
#   number -- no language-specific reason found yet to move it off the
#   inherited default. Revisit only if the output manifest.json shows a
#   filler phrase eating a disproportionate share of `kept`.
#
# Romanization for the MMS forced aligner (not a flag here, but relevant):
#   extract_short_segments.py used to romanize via a Turkish diacritic-strip
#   table (TR_TO_ASCII), which doesn't generalize to Hangul syllable blocks.
#   It now romanizes through `uroman`, the tool the MMS aligner's own
#   multilingual examples are built around -- the extracted `text` field
#   still keeps the original Hangul, romanization is alignment-only.
# ---------------------------------------------------------------------------
