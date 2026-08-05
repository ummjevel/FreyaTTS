"""Generate (noise, teacher-latent) pairs for reflow distillation.

The step count is expensive because the ODE trajectory is curved: a Euler solver
needs many small steps to follow it. Reflow (rectified flow) straightens it.
Instead of pairing a random noise with a random real latent -- which is what
`cfm_loss` does, and which produces crossing, curved paths -- it pairs each noise
with the latent that *this teacher* actually produces from it. Retraining on
those pairs makes the learned field close to a straight line between them, and a
straight line is exactly what a 2- or 4-step Euler solver can follow.

This is the data half of the ZipVoice-Distill idea; the training half reuses
`cfm_loss` unchanged, with x0 supplied instead of resampled.

The teacher runs with a *random* seed per pair, not the locked voice seed: the
student has to straighten the whole noise distribution, not one point of it.

  python training/gen_reflow_pairs.py --model checkpoints/distill183M_voiceD/final/hf \
      --latents data/latents_distill_voiceD --out data/reflow_voiceD --steps 32
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="teacher, a from_pretrained dir")
    ap.add_argument("--latents", required=True,
                    help="precomputed latent shards; used for the text and the frame counts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32, help="teacher ODE steps")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json

    from freyatts import FreyaTTS

    os.makedirs(args.out, exist_ok=True)
    tts = FreyaTTS.from_pretrained(args.model, device=args.device)
    model = tts.model.eval()
    vocab = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "freyatts", "char_vocab.json")))
    unk = vocab.get("<unk>", 1)

    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    shard, shard_idx, made = [], 0, 0

    def flush():
        nonlocal shard, shard_idx
        if not shard:
            return
        torch.save(shard, os.path.join(args.out, f"reflow_{shard_idx:05d}.pt"))
        shard, shard_idx = [], shard_idx + 1

    files = sorted(glob.glob(os.path.join(args.latents, "*.pt")))
    print(f"{len(files)} shards in {args.latents}", flush=True)
    for f in files:
        for entry in torch.load(f, weights_only=False):
            text = str(entry.get("text", ""))
            T = int(entry["latent"].shape[0])
            if not text or T < 8:
                continue
            ids = torch.tensor([[vocab.get(c, unk) for c in text]],
                               dtype=torch.long, device=args.device)
            if ids.shape[1] < 1:
                continue
            # a fresh noise draw per pair, so the student sees the whole
            # distribution rather than the one seed a voice is locked to
            x0 = torch.randn(1, T, model.feat, device=args.device, generator=gen)
            with torch.no_grad():
                ctx = model.text_encode(ids)
                x = x0
                for i in range(args.steps):
                    t = torch.full((1,), i / args.steps, device=args.device)
                    x = x + model(x, t, ctx, None, None) / args.steps
            shard.append(dict(x0=x0.squeeze(0).half().cpu(),
                              x1=x.squeeze(0).half().cpu(),
                              text=text))
            made += 1
            if len(shard) >= args.shard_size:
                flush()
                print(f"  {made} pairs", flush=True)
            if args.limit and made >= args.limit:
                flush()
                print(f"done (limit): {made} pairs in {shard_idx} shards -> {args.out}")
                return
    flush()
    print(f"done: {made} pairs in {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
