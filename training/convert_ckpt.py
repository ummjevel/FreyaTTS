#!/usr/bin/env python3
"""Convert a training checkpoint (model.pt) into a from_pretrained() directory.

pretrain.py / sft.py save `model.pt` = torch.save({"state_dict", "cfg"}).
freyatts.FreyaTTS.from_pretrained() (used by eval/benchmark.py, eval/speed.py,
infer.py) instead expects a directory holding `config.json` + `model.safetensors`.
This bridges the two.

Usage:
    python training/convert_ckpt.py checkpoints/pretrain/step30000/model.pt \
        --out checkpoints/pretrain/step30000/hf
    # then: python eval/benchmark.py --model checkpoints/pretrain/step30000/hf ...
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="path to a training model.pt")
    ap.add_argument("--out", required=True, help="output directory for config.json + model.safetensors")
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" not in obj or "cfg" not in obj:
        raise SystemExit(f"{args.ckpt} is not a training model.pt ({{'state_dict','cfg'}} expected)")

    cfg = obj["cfg"]
    # from_pretrained reads vocab/d/depth/heads/ff; keep the rest of cfg too.
    for key in ("vocab", "d", "depth", "heads", "ff"):
        if key not in cfg:
            raise SystemExit(f"cfg is missing required key {key!r}: {cfg}")

    # safetensors forbids shared storage -> detach + clone every tensor.
    state = {k: v.detach().cpu().clone().contiguous() for k, v in obj["state_dict"].items()}

    os.makedirs(args.out, exist_ok=True)
    save_file(state, os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    n = sum(v.numel() for v in state.values())
    print(f"[convert] {args.ckpt} -> {args.out}  ({n / 1e6:.1f}M params, vocab={cfg['vocab']})")


if __name__ == "__main__":
    main()
