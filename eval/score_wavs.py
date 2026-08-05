"""Score pre-synthesized wavs with the same ASR protocol as eval/benchmark.py.

benchmark.py synthesizes and scores in one process, which only works for systems
importable here. Matcha-TTS lives in its own venv (different torch, its own
frontend), so it writes wavs first and this scores them -- same Whisper model,
same Korean normalization, same jiwer metrics, so the numbers land on the same
scale as bench_distill_voice*.json.

  python eval/score_wavs.py --manifest .../synth/eval_ko_dev/manifest.jsonl \
      --system matcha-ko --out eval/results/bench_matcha_voiceA.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf  # noqa: E402

from eval.benchmark import normalize_ko, transcribe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help='JSONL of {"wav": path, "text": reference}')
    ap.add_argument("--system", default="matcha-ko")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from jiwer import cer, wer

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    asr = WhisperModel("large-v3", device=args.device, compute_type="float16")

    refs, hyps, audio_time = [], [], 0.0
    t0 = time.time()
    for i, row in enumerate(rows):
        wav, sr = sf.read(row["wav"], dtype="float32")
        audio_time += len(wav) / sr
        # benchmark.py band-limits every system to 8 kHz before the ASR so that
        # systems with different output rates are judged on the same band. Score
        # the same way or these numbers are not comparable to bench_*.json --
        # full-band audio scores noticeably better.
        hyp = transcribe(asr, wav, sr)
        ref_n, hyp_n = normalize_ko(row["text"]), normalize_ko(hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(rows)}] wer={wer(refs, hyps):.4f}", flush=True)

    out = {
        "system": args.system,
        "sentences": len(rows),
        "wer": round(wer(refs, hyps), 4),
        "cer": round(cer(refs, hyps), 4),
        "audio_s": round(audio_time, 1),
        "scoring_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
