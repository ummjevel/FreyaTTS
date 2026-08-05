"""Verify the exported FreyaTTS ONNX against PyTorch, then measure CPU RTF.

Same protocol as eval/speed.py so the numbers sit beside the existing
speed_*.json: the Korean prompt buckets, 3 untimed warmups + 10 timed runs per
sentence, per-sentence median then median across sentences.

The verification step matters more than usual here. The ODE loop is unrolled 32
times in the graph, and the speaker is decided by the initial noise, so a silent
numerical drift would show up as a different voice rather than an error. Feeding
the identical x0 to both runtimes and comparing waveforms catches that.

  python eval/bench_onnx_cpu.py --onnx onnx_export/distill_voiceA \
      --model checkpoints/distill_voiceA/final/hf --threads 1
"""
import argparse
import json
import re
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from freyatts.pipeline import normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def split_clauses(text, max_words):
    """Same rule as FreyaTTS._clauses: split on punctuation, then hard-split
    anything still longer than max_words + 4 words."""
    parts = re.split(r"(?<=[\.\?\!,:;])\s+", text)
    out, cur = [], ""
    for p in parts:
        if len((cur + " " + p).split()) <= max_words:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    final = []
    for c in out:
        w = c.split()
        if len(w) <= max_words + 4:
            final.append(c)
        else:
            for i in range(0, len(w), max_words):
                final.append(" ".join(w[i:i + max_words]))
    return [c for c in final if c.strip()]


class OnnxFreya:
    """The three exported graphs wired back into one synthesize() call."""

    def __init__(self, onnx_dir, threads=1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        mk = lambda n: ort.InferenceSession(os.path.join(onnx_dir, n), so,
                                            providers=["CPUExecutionProvider"])
        self.dur, self.dit, self.vae = mk("dur.onnx"), mk("dit.onnx"), mk("vae.onnx")
        self.meta = json.load(open(os.path.join(onnx_dir, "meta.json")))
        self.feat = self.meta["feat"]
        self.t_floor = self.meta["t_floor"]
        self.seed = self.meta["seed"]
        self.max_words = int(self.meta.get("max_words", 4))

    def frames(self, ids):
        logT = self.dur.run(None, {"text_ids": ids})[0]
        T = int(round(math.exp(float(np.asarray(logT).reshape(-1)[0]))))
        return max(self.t_floor, ids.shape[1] + 4, min(300, T))

    def synthesize(self, ids, x0=None, seed=None):
        """One clause. Callers wanting a whole utterance should use synth_text."""
        T = self.frames(ids)
        if x0 is None:
            gen = torch.Generator().manual_seed(int(self.seed if seed is None else seed))
            x0 = torch.randn(1, T, self.feat, generator=gen).numpy().astype(np.float32)
        lat = self.dit.run(None, {"text_ids": ids, "x0": x0})[0]
        wav = self.vae.run(None, {"z": np.ascontiguousarray(lat.transpose(0, 2, 1))})[0]
        return np.asarray(wav).reshape(-1)

    def synth_text(self, text, to_ids, seed=None):
        """Normalize, split into clauses, synthesize each, join with short gaps.

        The exported graphs cover the model only; normalization and clause
        splitting live in freyatts/pipeline.py and have to be reproduced here or
        the deployed path is not the measured one. Skipping the split cost
        CER 0.086 -> 0.132 on 183M voiceD: this model is accurate on short
        inputs and degrades with length, so the split is doing real work, not
        cosmetics.
        """
        t = normalize(text)
        chunks = split_clauses(t, self.max_words) if len(t.split()) > self.max_words else [t]
        gap = np.zeros(int(0.12 * 48000), dtype=np.float32)
        out = []
        for c in chunks:
            ids = to_ids(c)
            if ids.shape[1] < 1:
                continue
            out.append(self.synthesize(ids, seed=seed).astype(np.float32))
            out.append(gap)
        return np.concatenate(out[:-1]) if len(out) > 1 else (
            out[0] if out else np.zeros(1, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model", default="", help="from_pretrained dir; enables the PyTorch cross-check")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts_ko.json"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from freyatts import FreyaTTS
    from freyatts.hangul import decompose_hangul

    vocab = json.load(open(os.path.join(os.path.dirname(HERE), "freyatts", "char_vocab.json")))
    UNK = vocab.get("<unk>", 1)

    def to_ids(text):
        jamo = decompose_hangul(text)
        return np.array([[vocab.get(c, UNK) for c in jamo]], dtype=np.int64)

    engine = OnnxFreya(args.onnx, threads=args.threads)
    out = {
        "system": "freyatts-onnx-cpu",
        "onnx_dir": os.path.abspath(args.onnx),
        "provider": "cpu",
        "num_threads": args.threads,
        "ode_steps": engine.meta["steps"],
        "buckets": {},
    }

    if args.model:
        tts = FreyaTTS.from_pretrained(args.model, device="cpu")
        text = "안녕하세요, 오늘 날씨가 참 좋네요."
        ids = to_ids(text)
        T = engine.frames(ids)
        gen = torch.Generator().manual_seed(int(engine.seed))
        x0 = torch.randn(1, T, engine.feat, generator=gen)
        with torch.no_grad():
            lat = tts.model.sample(torch.from_numpy(ids), T, steps=engine.meta["steps"],
                                   cmask=None, seed=None) if False else None
            # reuse the exact same x0 in both runtimes
            ctx = tts.model.text_encode(torch.from_numpy(ids))
            x = x0
            for i in range(engine.meta["steps"]):
                t = torch.full((1,), i / engine.meta["steps"])
                x = x + tts.model(x, t, ctx, None, None) / engine.meta["steps"]
            ref = tts.vae.decode(x.transpose(1, 2).float()).squeeze().numpy()
        got = engine.synthesize(ids, x0=x0.numpy().astype(np.float32))
        n = min(len(ref), len(got))
        diff = float(np.abs(ref[:n] - got[:n]).max())
        rel = float(np.sqrt(((ref[:n] - got[:n]) ** 2).mean()) / (np.sqrt((ref[:n] ** 2).mean()) + 1e-9))
        out["verification"] = {"max_abs_diff": round(diff, 6), "rel_rms_error": round(rel, 6),
                               "len_pytorch": int(len(ref)), "len_onnx": int(len(got))}
        print("검증:", json.dumps(out["verification"]), flush=True)

    prompts = json.load(open(args.prompts))
    for bucket, sentences in prompts.items():
        lat_s, rtf_s, dur_s = [], [], []
        for text in sentences:
            ids = to_ids(text)
            for _ in range(args.warmup):
                engine.synth_text(text, to_ids)
            runs_l, runs_r = [], []
            for _ in range(args.runs):
                t0 = time.time()
                wav = engine.synth_text(text, to_ids)
                wall = time.time() - t0
                d = len(wav) / 48000
                runs_l.append(wall)
                runs_r.append(wall / max(1e-6, d))
            lat_s.append(statistics.median(runs_l))
            rtf_s.append(statistics.median(runs_r))
            dur_s.append(len(wav) / 48000)
        out["buckets"][bucket] = {
            "latency_s": round(statistics.median(lat_s), 3),
            "rtf": round(statistics.median(rtf_s), 3),
            "audio_s": round(statistics.median(dur_s), 2),
        }
        print(bucket, out["buckets"][bucket], flush=True)

    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
