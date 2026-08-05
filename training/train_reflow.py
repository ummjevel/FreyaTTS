"""Reflow distillation: fine-tune on teacher-generated (noise, latent) pairs.

Same conditional flow-matching objective as `cfm_loss`, with one change that is
the whole point: x0 is the noise the teacher actually started from for this
target, not a fresh random draw. Training on those pairs pulls the velocity
field toward the straight line between them, and a straight field is what lets a
4-step Euler solver match what previously needed 16 or 32.

Initialized from the teacher, so this is a refinement rather than a new run --
the voice, the text frontend and the latent space are all unchanged, and the
result drops into the existing export and evaluation path.

  accelerate launch --num_processes 4 --mixed_precision bf16 training/train_reflow.py \
      --init checkpoints/distill183M_voiceD/final/model.pt --data data/reflow_voiceD \
      --out checkpoints/reflow183M_voiceD --steps 3000
"""
import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from freyatts.model import FreyaDiT  # noqa: E402

UNK_ID = 1
FILL_ID = 0


class ReflowDataset(Dataset):
    """(x0, x1, text ids) triples held in RAM as fp16, like LatentDataset."""

    def __init__(self, data_dir, char_to_id, max_frames):
        self.items = []
        for path in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
            for e in torch.load(path, weights_only=False):
                x0, x1 = e["x0"], e["x1"]
                if x1.shape[0] < 8 or x1.shape[0] > max_frames:
                    continue
                ids = [char_to_id.get(c, UNK_ID) for c in str(e.get("text", ""))[:300]]
                if not 1 <= len(ids) <= 250:
                    continue
                self.items.append((x0.half(), x1.half(),
                                   torch.tensor(ids, dtype=torch.long)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    max_t = max(x1.shape[0] for _, x1, _ in batch)
    max_l = max(ids.shape[0] for _, _, ids in batch)
    B, feat = len(batch), batch[0][1].shape[1]
    x0 = torch.zeros(B, max_t, feat)
    x1 = torch.zeros(B, max_t, feat)
    text = torch.full((B, max_l), FILL_ID, dtype=torch.long)
    fmask = torch.zeros(B, max_t, dtype=torch.bool)
    cmask = torch.zeros(B, max_l, dtype=torch.bool)
    for i, (a, b, ids) in enumerate(batch):
        t, l = b.shape[0], ids.shape[0]
        x0[i, :t] = a.float()
        x1[i, :t] = b.float()
        fmask[i, :t] = True
        text[i, :l] = ids
        cmask[i, :l] = True
    return dict(x0=x0, x1=x1, text=text, fmask=fmask, cmask=cmask)


def reflow_loss(model, x0, x1, text_ids, fmask, cmask):
    """cfm_loss with the paired noise instead of a fresh sample."""
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    ctx = model.text_encode(text_ids)
    v = model(xt, t, ctx, fmask, cmask)
    target = x1 - x0
    m = fmask[..., None].float()
    return (((v - target) ** 2) * m).sum() / (m.sum() * model.feat + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="teacher checkpoint (model.pt)")
    ap.add_argument("--data", required=True, help="reflow pair shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab_json", default="freyatts/char_vocab.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json

    acc = Accelerator()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.init, map_location="cpu")
    cfg = ckpt["cfg"]
    model = FreyaDiT(vocab=cfg["vocab"], feat=64, d=cfg["d"], depth=cfg["depth"],
                     heads=cfg["heads"], ff=cfg["ff"])
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if acc.is_main_process:
        n = sum(p.numel() for p in model.parameters())
        print(f"[init] {args.init} missing={len(missing)} unexpected={len(unexpected)} "
              f"{n/1e6:.1f}M params", flush=True)

    with open(args.vocab_json, encoding="utf-8") as f:
        char_to_id = json.load(f)
    ds = ReflowDataset(args.data, char_to_id, args.max_frames)
    if acc.is_main_process:
        print(f"[data] {len(ds)} reflow pairs", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=collate, num_workers=args.num_workers,
                        persistent_workers=(args.num_workers > 0))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01, eps=1e-8)
    model, opt, loader = acc.prepare(model, opt, loader)

    os.makedirs(args.out, exist_ok=True)
    step, t0, running, nrun = 0, time.time(), 0.0, 0
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        lr *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr

        loss = reflow_loss(acc.unwrap_model(model), batch["x0"], batch["x1"],
                           batch["text"], batch["fmask"], batch["cmask"])
        opt.zero_grad(set_to_none=True)
        acc.backward(loss)
        gnorm = acc.clip_grad_norm_(model.parameters(), 1.0)
        # one bad batch must not poison the weights, same guard as pretrain
        if gnorm is None or torch.isfinite(gnorm):
            opt.step()
        step += 1
        running += float(loss.detach())
        nrun += 1

        if acc.is_main_process and step % args.log_every == 0:
            sps = nrun * args.batch_size * acc.num_processes / (time.time() - t0)
            print(f"step {step:6d}/{args.steps}  reflow {running/max(nrun,1):.4f}  "
                  f"lr {lr:.2e}  {sps:.0f} samp/s", flush=True)
            running, nrun, t0 = 0.0, 0, time.time()

        if step % args.save_every == 0 or step == args.steps:
            acc.wait_for_everyone()
            if acc.is_main_process:
                d = os.path.join(args.out, "final" if step == args.steps else f"step{step}")
                os.makedirs(d, exist_ok=True)
                torch.save(dict(state_dict=acc.unwrap_model(model).state_dict(), cfg=cfg),
                           os.path.join(d, "model.pt"))
                with open(os.path.join(d, "step.txt"), "w") as f:
                    f.write(str(step))
                print(f"[ckpt] {d}", flush=True)

    if acc.is_main_process:
        print(f"[done] {step}", flush=True)


if __name__ == "__main__":
    main()
