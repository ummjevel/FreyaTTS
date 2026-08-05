"""Export a FreyaTTS checkpoint to ONNX so it can run on CPU.

FreyaTTS has only ever run as PyTorch on a GPU, which made every speed
comparison against Matcha (ONNX Runtime, CPU) an apples-to-oranges one: two
different runtimes on two different processors. This produces the missing half.

Three graphs, because the pipeline has a data-dependent shape in the middle:

  dur.onnx  text_ids            -> log frame count   (tiny MLP over pooled text)
  dit.onnx  text_ids, x0        -> latents           (Euler ODE loop unrolled)
  vae.onnx  latents [B,64,T]    -> waveform 48 kHz   (frozen AudioVAE2 decoder)

The frame count T comes out of `dur`, so the caller sizes x0 and the DiT graph
never has to predict its own output length. x0 is an input rather than sampled
inside the graph: the seed *is* the speaker in this model, and RNG cannot be
made to match across PyTorch and ONNX Runtime, so the caller draws it with torch
and the exported model reproduces the same voice exactly.

  python eval/export_onnx.py --model checkpoints/distill_voiceA/final/hf \
      --out onnx/distill_voiceA --steps 32
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


class DurationWrapper(torch.nn.Module):
    """text ids -> predicted log frame count."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text_ids):
        te = self.model.text_encode(text_ids)
        pooled = te.mean(1)          # cmask is all-ones for a single clause
        return self.model.dur(pooled).squeeze(-1)


class DiTWrapper(torch.nn.Module):
    """text ids + initial noise -> latents, with the ODE loop unrolled."""

    def __init__(self, model, steps: int):
        super().__init__()
        self.model = model
        self.steps = steps

    def forward(self, text_ids, x0):
        ctx = self.model.text_encode(text_ids)
        x = x0
        B = x0.shape[0]
        for i in range(self.steps):
            t = torch.full((B,), i / self.steps, dtype=x.dtype, device=x.device)
            x = x + self.model(x, t, ctx, None, None) / self.steps
        return x


class VaeWrapper(torch.nn.Module):
    """latents [B, 64, T] -> waveform."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a from_pretrained directory (config.json + safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from freyatts import FreyaTTS

    os.makedirs(args.out, exist_ok=True)
    tts = FreyaTTS.from_pretrained(args.model, device="cpu")
    model = tts.model.eval()
    vae = tts.vae

    L, T = 40, 120   # tracing shapes only; both axes are dynamic below
    text_ids = torch.randint(1, 100, (1, L), dtype=torch.int64)
    x0 = torch.randn(1, T, model.feat)
    z = torch.randn(1, model.feat, T)

    with torch.no_grad():
        torch.onnx.export(
            DurationWrapper(model), (text_ids,), f"{args.out}/dur.onnx",
            input_names=["text_ids"], output_names=["log_frames"],
            dynamic_axes={"text_ids": {0: "B", 1: "L"}},
            opset_version=args.opset, dynamo=False)
        print("dur.onnx 완료", flush=True)

        torch.onnx.export(
            DiTWrapper(model, args.steps), (text_ids, x0), f"{args.out}/dit.onnx",
            input_names=["text_ids", "x0"], output_names=["latents"],
            dynamic_axes={"text_ids": {0: "B", 1: "L"}, "x0": {0: "B", 1: "T"},
                          "latents": {0: "B", 1: "T"}},
            opset_version=args.opset, dynamo=False)
        print("dit.onnx 완료", flush=True)

        torch.onnx.export(
            VaeWrapper(vae), (z,), f"{args.out}/vae.onnx",
            input_names=["z"], output_names=["wav"],
            dynamic_axes={"z": {0: "B", 2: "T"}, "wav": {0: "B", 2: "S"}},
            opset_version=args.opset, dynamo=False)
        print("vae.onnx 완료", flush=True)

    meta = {
        "steps": args.steps, "feat": model.feat, "sample_rate": 48000,
        "t_floor": tts.t_floor, "seed": tts.seed,
        "source_model": os.path.abspath(args.model),
        "sizes_mb": {f: round(os.path.getsize(f"{args.out}/{f}") / 1e6, 1)
                     for f in ("dur.onnx", "dit.onnx", "vae.onnx")},
    }
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
