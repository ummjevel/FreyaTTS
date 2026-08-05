"""Convert the exported FreyaTTS graphs to fp16.

int8 dynamic quantization only reached 1.21x on this model: it rewrites MatMul
and Gemm weights, and the DiT keeps most of its parameters in ops that pass
untouched (LayerNorm, embeddings, AdaLN modulation), while the VAE is Conv1d
throughout. fp16 does not care what the op is, so the 2x is reliable.

What fp16 buys depends on where the model runs. On a CPU it is a size win only
-- there is no fp16 arithmetic, so ONNX Runtime casts back to fp32 and the extra
casts can make it slower. On an NPU or a GPU fp16 is usually the native format
and the win is real. Measure before assuming.

  python eval/convert_fp16.py --src onnx_export/distill183M_voiceD_s16 \
      --dst onnx_export/distill183M_voiceD_s16_fp16
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep-io-fp32", action="store_true", default=True,
                    help="keep graph inputs/outputs fp32 so callers need no change")
    ap.add_argument("--block-ops", default="Einsum,Range",
                    help="ops to leave in fp32; the converter inserts casts around them")
    args = ap.parse_args()

    import onnx
    from onnxconverter_common import float16

    os.makedirs(args.dst, exist_ok=True)
    report = {}
    for name in ("dur", "dit", "vae"):
        src = os.path.join(args.src, f"{name}.onnx")
        dst = os.path.join(args.dst, f"{name}.onnx")
        if not os.path.exists(src):
            continue
        before = os.path.getsize(src) / 1e6
        model = onnx.load(src)
        # keep_io_types leaves the graph boundary in fp32: the benchmark and the
        # pipeline feed float32 arrays, and this avoids touching every caller.
        # Einsum stays fp32. These are the i,j->ij outer products in the time and
        # position embedding, one per ODE step, and the converter only casts one
        # of the two operands -- the resulting graph fails to load with
        # "Type parameter (T) of Optype (Einsum) bound to different types".
        block = [o for o in args.block_ops.split(",") if o]
        # Shape inference must stay ON. With it disabled the converter cannot
        # see the types either side of a blocked node and skips the casts, and
        # the graph then fails to load on the Einsum operands.
        model16 = float16.convert_float_to_float16(
            model, keep_io_types=args.keep_io_fp32, disable_shape_infer=False,
            op_block_list=block)
        onnx.save(model16, dst)
        after = os.path.getsize(dst) / 1e6
        report[name] = {"fp32_mb": round(before, 1), "fp16_mb": round(after, 1),
                        "ratio": round(before / max(after, 1e-9), 2)}
        print(f"{name}: {before:.1f} MB -> {after:.1f} MB ({before/after:.2f}x)", flush=True)

    meta_src = os.path.join(args.src, "meta.json")
    if os.path.exists(meta_src):
        meta = json.load(open(meta_src))
        meta["precision"] = {"scheme": "fp16 weights, fp32 graph io", "report": report}
        json.dump(meta, open(os.path.join(args.dst, "meta.json"), "w"), indent=1)

    tb = sum(v["fp32_mb"] for v in report.values())
    ta = sum(v["fp16_mb"] for v in report.values())
    print(json.dumps({"total_fp32_mb": round(tb, 1), "total_fp16_mb": round(ta, 1),
                      "ratio": round(tb / max(ta, 1e-9), 2)}, indent=1))


if __name__ == "__main__":
    main()
