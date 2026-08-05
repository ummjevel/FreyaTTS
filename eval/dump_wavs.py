"""Synthesize the eval sentences with a FreyaTTS checkpoint and keep the wavs.

eval/benchmark.py throws its audio away after scoring, so there has never been a
FreyaTTS wav set to run a quality metric (DNSMOS/UTMOS) on, or to listen to
next to another system. This writes them out with the same seed the checkpoint
was locked to, plus a manifest that eval/score_wavs.py can read.

  python eval/dump_wavs.py --model checkpoints/distill_voiceA/final/hf \
      --outdir synth/freyatts_337M_voiceA --seed 9
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="eval/eval_ko_dev.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=None,
                    help="speaker seed; leave unset to use the package default")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from freyatts import FreyaTTS

    os.makedirs(args.outdir, exist_ok=True)
    tts = FreyaTTS.from_pretrained(args.model, device=args.device)
    texts = [json.loads(l)["text"] for l in open(args.data) if l.strip()]
    if args.limit:
        texts = texts[: args.limit]

    kwargs = {"steps": args.steps}
    if args.seed is not None:
        kwargs["seed"] = args.seed

    manifest, total, t0 = [], 0.0, time.time()
    for i, text in enumerate(texts):
        wav = tts.synthesize(text, **kwargs)
        path = os.path.join(args.outdir, f"{i:04d}.wav")
        sf.write(path, wav, 48000, subtype="PCM_16")
        total += len(wav) / 48000
        manifest.append({"wav": os.path.abspath(path), "text": text})
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(texts)}  {total/max(1e-6, time.time()-t0):.1f}x realtime", flush=True)

    with open(os.path.join(args.outdir, "manifest.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(manifest)} wavs ({total/60:.1f} min) -> {args.outdir}")


if __name__ == "__main__":
    main()
