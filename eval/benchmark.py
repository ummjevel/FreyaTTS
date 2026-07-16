#!/usr/bin/env python3
"""Evaluate a TTS system with band-matched WER/CER on a Korean sentence set.

Protocol: synthesize each sentence, downsample audio to 8 kHz so every system
is judged in the same band, transcribe with Whisper large-v3 (faster-whisper,
Korean, beam 5), normalize both reference and hypothesis, then report
WER/CER (jiwer) and RTF.

There is no Korean equivalent of Freya-TR-Eval yet -- pass your own held-out
sentences via --data (JSONL, one {"text": ...} per line; a natural source is
a slice of utterances excluded from training/build_manifest_ko.py's input).

Usage:
  python3 eval/benchmark.py --system freyatts --data my_ko_eval.jsonl --out results/freyatts.json
"""

import argparse
import json
import os
import re
import time

import numpy as np

JUDGE_SR = 8000


# ----------------------------- text normalization --------------------------
# Self-contained (no `torch`-importing freyatts.pipeline dependency) so this
# script stays usable for scoring any TTS system, not just FreyaTTS. Mirrors
# freyatts/pipeline.py's Sino-Korean spelling -- keep the two in sync.

_UNITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_SCALES = [(10 ** 12, "조"), (10 ** 8, "억"), (10 ** 4, "만")]


def _spell_group(n):
    parts = []
    if n >= 1000:
        d = n // 1000
        if d > 1:
            parts.append(_UNITS[d])
        parts.append("천")
        n %= 1000
    if n >= 100:
        d = n // 100
        if d > 1:
            parts.append(_UNITS[d])
        parts.append("백")
        n %= 100
    if n >= 10:
        d = n // 10
        if d > 1:
            parts.append(_UNITS[d])
        parts.append("십")
        n %= 10
    if n > 0:
        parts.append(_UNITS[n])
    return "".join(parts)


def digits_to_words(n):
    """Spell a non-negative integer in Sino-Korean (0 -> '영'), grouped by 만/억/조."""
    if n == 0:
        return "영"
    parts = []
    remaining = n
    for scale_val, scale_name in _SCALES:
        if remaining >= scale_val:
            head = remaining // scale_val
            remaining %= scale_val
            parts.append(scale_name if head == 1 else _spell_group(head) + scale_name)
    if remaining > 0 or not parts:
        parts.append(_spell_group(remaining))
    return "".join(parts)


def normalize_ko(text):
    """Spell out digit runs, strip punctuation/whitespace. Applied to both sides."""
    t = re.sub(r"\d+", lambda m: digits_to_words(int(m.group(0))), text)
    t = re.sub(r"[^\wㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------------------- system adapters -----------------------------
# An adapter wraps one TTS system behind a single method:
#   synthesize(text) -> (waveform np.float32, sample_rate)
# Add your own system by writing a class with that method and registering it
# in ADAPTERS below.


class FreyaTTSAdapter:
    """Default system: the FreyaTTS pipeline shipped in this repo."""

    def __init__(self, args):
        from freyatts import FreyaTTS

        self.tts = FreyaTTS.from_pretrained(args.model, device=args.device)

    def synthesize(self, text):
        wav = self.tts.synthesize(text)
        return wav, 48000


class XTTSAdapter:
    """Example third-party adapter: Coqui XTTS-v2 voice cloning.

    Requires the `TTS` package and a reference clip (--ref). Kept here as a
    template for plugging in other systems.
    """

    def __init__(self, args):
        from TTS.api import TTS as CoquiTTS

        if not args.ref:
            raise SystemExit("xtts needs --ref pointing to a reference wav")
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(args.device)
        self.ref = args.ref

    def synthesize(self, text):
        wav = self.tts.tts(text=text, speaker_wav=self.ref, language="ko")
        return np.asarray(wav, dtype=np.float32), 24000


ADAPTERS = {"freyatts": FreyaTTSAdapter, "xtts": XTTSAdapter}


# ----------------------------- data ----------------------------------------

def load_sentences(args):
    """Return the list of reference sentences to synthesize."""
    if args.data:
        sentences = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(json.loads(line)["text"])
        return sentences

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    return [row["text"] for row in ds]


# ----------------------------- evaluation ----------------------------------

def transcribe(asr, wav, sr):
    """Band-match to 8 kHz, then transcribe with Whisper large-v3."""
    import librosa

    wav8k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=JUDGE_SR)
    # whisper expects 16 kHz input; upsampling from 8 kHz keeps the band limit
    wav16k = librosa.resample(wav8k, orig_sr=JUDGE_SR, target_sr=16000)
    segments, _ = asr.transcribe(wav16k, language="ko", beam_size=5)
    return " ".join(seg.text for seg in segments).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="freyatts", choices=sorted(ADAPTERS),
                    help="which TTS system to evaluate")
    ap.add_argument("--model", default="freyavoice/freya-tts",
                    help="FreyaTTS model id or local directory")
    ap.add_argument("--dataset", default="",
                    help="HF dataset with a 'text' column (no Korean eval set is bundled -- "
                         "point this at your own, or use --data)")
    ap.add_argument("--split", default="test", help="dataset split to use")
    ap.add_argument("--data", default="",
                    help="local JSONL ({'text': ...} per line, Korean sentences); overrides --dataset")
    ap.add_argument("--ref", default="", help="reference wav for cloning systems (e.g. xtts)")
    ap.add_argument("--device", default="cuda", help="torch device for synthesis")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N sentences")
    ap.add_argument("--out", default="", help="write results JSON here")
    args = ap.parse_args()

    sentences = load_sentences(args)
    if args.limit:
        sentences = sentences[:args.limit]

    from faster_whisper import WhisperModel
    from jiwer import cer, wer

    asr = WhisperModel("large-v3", device=args.device, compute_type="float16")
    engine = ADAPTERS[args.system](args)

    refs = []
    hyps = []
    synth_time = 0.0
    audio_time = 0.0

    for i, text in enumerate(sentences):
        t0 = time.time()
        wav, sr = engine.synthesize(text)
        synth_time += time.time() - t0
        audio_time += len(wav) / sr

        ref = normalize_ko(text)
        hyp = normalize_ko(transcribe(asr, wav, sr))
        refs.append(ref)
        hyps.append(hyp)
        print(f"[{i + 1}/{len(sentences)}] wer={wer(ref, hyp):.2f} | {text[:48]!r} -> {hyp[:48]!r}",
              flush=True)

    results = {
        "system": args.system,
        "sentences": len(sentences),
        "wer": round(wer(refs, hyps), 4),
        "cer": round(cer(refs, hyps), 4),
        "rtf": round(synth_time / max(1e-6, audio_time), 3),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
