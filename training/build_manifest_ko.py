#!/usr/bin/env python3
"""Build a FreyaTTS {audio, text} manifest from a Korean speech corpus.

Bridges a raw (wav, transcript) corpus -- KSS/LJSpeech-style metadata.csv,
Zeroth-Korean/Kaldi-style *.trans.txt, or your own recordings -- to the
`data/manifest.jsonl` that training/precompute_latents.py expects. Applies
the same normalize() (digit spelling + acronym transliteration + jamo
decomposition) that freyatts.pipeline uses at inference time, so training
text and inference text are tokenized identically against char_vocab.json.

Expected input: a delimited text file, one utterance per line, e.g.

    1/1_0032|그는 괜찮은 척하려고 애쓰는 것 같았다.
    1/1_0033|이제 와서 후회해도 소용없는 일이었다.

(KSS's own metadata.csv has extra pipe-separated columns after the text --
only --text-col is read, the rest are ignored.)

Usage:
    python training/build_manifest_ko.py \\
        --metadata /path/to/corpus/metadata.csv \\
        --wav-dir /path/to/corpus/wavs \\
        --out data/manifest.jsonl

Then, as usual:
    python training/precompute_latents.py --manifest data/manifest.jsonl --output-dir data/latents
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freyatts.pipeline import normalize

AUDIO_EXTS = (".wav", ".flac")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True,
                        help="delimited text file, one utterance per line")
    parser.add_argument("--wav-dir", default="",
                        help="directory to resolve relative audio paths against "
                             "(leave empty if --audio-col already holds full/absolute paths)")
    parser.add_argument("--sep", default="|", help="field delimiter (default '|')")
    parser.add_argument("--audio-col", type=int, default=0, help="0-indexed column with the audio filename/path")
    parser.add_argument("--text-col", type=int, default=1, help="0-indexed column with the transcript")
    parser.add_argument("--out", default="data/manifest.jsonl", help="output JSONL manifest path")
    parser.add_argument("--encoding", default="utf-8", help="metadata file encoding")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip normalize() -- use only if the metadata text is "
                             "already digit-expanded and jamo-decomposed")
    parser.add_argument("--debug-readable", default="",
                        help="optional path to also dump {audio, text} with the "
                             "pre-decomposition (human-readable) text, for spot-checking")
    return parser.parse_args()


def resolve_audio_path(raw, wav_dir):
    path = raw.strip()
    if wav_dir and not os.path.isabs(path):
        path = os.path.join(wav_dir, path)
    if os.path.exists(path):
        return path
    for ext in AUDIO_EXTS:
        if os.path.exists(path + ext):
            return path + ext
    return None


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_lines = 0
    kept = 0
    missing_audio = 0
    empty_text = 0

    readable_f = open(args.debug_readable, "w", encoding="utf-8") if args.debug_readable else None

    with open(args.metadata, encoding=args.encoding) as f, open(args.out, "w", encoding="utf-8") as out_f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            n_lines += 1

            fields = line.split(args.sep)
            if len(fields) <= max(args.audio_col, args.text_col):
                print(f"[skip] line {n_lines}: not enough columns for sep={args.sep!r}: {line[:80]!r}", flush=True)
                continue

            raw_text = fields[args.text_col].strip()
            if not raw_text:
                empty_text += 1
                continue

            audio_path = resolve_audio_path(fields[args.audio_col], args.wav_dir)
            if audio_path is None:
                missing_audio += 1
                if missing_audio <= 10:
                    print(f"[skip] audio not found for {fields[args.audio_col]!r}", flush=True)
                continue

            text = raw_text if args.no_normalize else normalize(raw_text)
            if not text:
                empty_text += 1
                continue

            out_f.write(json.dumps({"audio": audio_path, "text": text}, ensure_ascii=False) + "\n")
            if readable_f:
                readable_f.write(json.dumps({"audio": audio_path, "text": raw_text}, ensure_ascii=False) + "\n")
            kept += 1

    if readable_f:
        readable_f.close()

    print(f"done: {kept}/{n_lines} kept, {missing_audio} missing audio, {empty_text} empty text -> {args.out}",
          flush=True)
    if missing_audio > 10:
        print(f"({missing_audio - 10} more missing-audio lines not shown; check --wav-dir)", flush=True)


if __name__ == "__main__":
    main()
