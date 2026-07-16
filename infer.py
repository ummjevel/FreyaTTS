"""Synthesize Korean speech with FreyaTTS from the command line.

Example:
    python infer.py --text "안녕하세요, 어떻게 도와드릴까요?" --out output.wav
"""

import argparse

from freyatts import FreyaTTS


def main():
    parser = argparse.ArgumentParser(description="FreyaTTS inference")
    parser.add_argument("--text", required=True, help="Korean text to synthesize")
    parser.add_argument("--out", default="output.wav", help="output wav path (48 kHz)")
    parser.add_argument("--model", default="freyavoice/freya-tts",
                        help="Hugging Face repo id or a local directory with config.json + model.safetensors")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--steps", type=int, default=32, help="Euler ODE steps")
    args = parser.parse_args()

    tts = FreyaTTS.from_pretrained(args.model, device=args.device)
    wav = tts.synthesize(args.text, steps=args.steps)
    tts.save_wav(wav, args.out)
    print(f"wrote {args.out} ({len(wav) / 48000:.2f} s)")


if __name__ == "__main__":
    main()
