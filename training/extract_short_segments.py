#!/usr/bin/env python3
"""Cut short (1-3 word) segments out of a speaker corpus via MMS forced alignment.

Long-form training data underrepresents very short utterances (single words,
numbers, acknowledgements), so the model struggles with them. This script
force-aligns each clip with torchaudio's MMS aligner, extracts contiguous short
word spans, re-encodes them to AudioVAE latents, and writes shards that can be
mixed into the SFT stage 2 data.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
import torch
from uroman import Uroman

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freyatts.vae import load_audio_vae

VAE_SAMPLE_RATE = 16000

# the MMS aligner vocabulary is plain a-z, so Hangul is romanized via uroman
# (syllable-block-aware, unlike a simple diacritic strip) for alignment only —
# extracted text keeps the original Hangul spelling
_UROMANIZER = Uroman()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="JSONL manifest, one {\"audio\": path, \"text\": str} per line")
    parser.add_argument("--out", default="data/latents_short",
                        help="output directory for latent shards")
    parser.add_argument("--n", type=int, default=8000,
                        help="number of source clips to scan")
    parser.add_argument("--max_words", type=int, default=3,
                        help="longest span to extract, in eojeol (space-separated words)")
    parser.add_argument("--max_s", type=float, default=3.0,
                        help="drop segments longer than this many seconds")
    parser.add_argument("--max_repeats", type=int, default=40,
                        help="cap on extracted copies of the same phrase")
    parser.add_argument("--shard_size", type=int, default=3000,
                        help="number of segments per .pt shard")
    parser.add_argument("--device", default="cuda",
                        help="device for the aligner and the AudioVAE encoder")
    return parser.parse_args()


def romanize(word):
    return re.sub(r"[^a-z]", "", _UROMANIZER.romanize_string(word).lower())


def load_clip(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != VAE_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=VAE_SAMPLE_RATE)
    return audio


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    from torchaudio.pipelines import MMS_FA as bundle
    aligner_model = bundle.get_model().to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    vae = load_audio_vae(device)

    shard = []
    shard_index = 0
    kept = 0
    seen = 0
    t0 = time.time()
    phrase_counts = Counter()

    def flush():
        nonlocal shard, shard_index
        if not shard:
            return
        torch.save(shard, os.path.join(args.out, f"short_{shard_index:05d}.pt"))
        shard = []
        shard_index += 1

    with open(args.manifest) as f:
        for line in f:
            if seen >= args.n:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = entry.get("text")
            audio_path = entry.get("audio")
            if not text or not audio_path:
                continue
            seen += 1

            try:
                audio = load_clip(audio_path)
            except Exception:
                continue

            words = re.findall(r"\S+", text)
            roman = [romanize(w) for w in words]
            keep = [i for i, r in enumerate(roman) if r]
            if not keep:
                continue

            try:
                with torch.inference_mode():
                    emission, _ = aligner_model(torch.from_numpy(audio).to(device).unsqueeze(0))
                tokens = tokenizer([roman[i] for i in keep])
                spans = aligner(emission[0], tokens)
            except Exception:
                continue

            samples_per_frame = len(audio) / emission.shape[1]
            word_spans = []
            for word_index, span in zip(keep, spans):
                if not span:
                    continue
                s0 = int(span[0].start * samples_per_frame)
                s1 = int(span[-1].end * samples_per_frame)
                word_spans.append((word_index, s0, s1))

            for length in range(1, args.max_words + 1):
                for k in range(len(word_spans) - length + 1):
                    indices = [word_spans[k + j][0] for j in range(length)]
                    if indices != list(range(indices[0], indices[0] + length)):
                        continue
                    s0 = word_spans[k][1]
                    s1 = word_spans[k + length - 1][2]
                    if s1 <= s0:
                        continue

                    # small pad around the aligned span to avoid clipped phones
                    segment = audio[max(0, s0 - 400):s1 + 400]
                    duration = len(segment) / VAE_SAMPLE_RATE
                    if duration < 0.2 or duration > args.max_s:
                        continue

                    phrase = " ".join(words[indices[0]:indices[0] + length])
                    key = phrase.lower()
                    if phrase_counts[key] >= args.max_repeats:
                        continue
                    phrase_counts[key] += 1

                    with torch.no_grad():
                        wav = torch.from_numpy(segment.copy()).to(device).view(1, 1, -1)
                        z = vae.encode(wav, VAE_SAMPLE_RATE)
                        latent = z.squeeze(0).transpose(0, 1).contiguous().to(torch.float16).cpu()
                    if latent.shape[0] < 4:
                        continue

                    shard.append(dict(latent=latent, text=phrase))
                    kept += 1
                    if len(shard) >= args.shard_size:
                        flush()

            if seen % 1000 == 0:
                rate = seen / max(1.0, time.time() - t0)
                print(f"seen {seen} kept {kept} unique {len(phrase_counts)} {rate:.0f} clip/s", flush=True)

    flush()

    summary = dict(kept=kept, seen=seen, unique=len(phrase_counts),
                   top=phrase_counts.most_common(15))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"done: kept={kept} unique={len(phrase_counts)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
