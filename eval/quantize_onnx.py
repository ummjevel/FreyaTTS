"""Dynamic int8 quantization of the exported FreyaTTS graphs.

Size is the binding constraint on device: the 183M export is 918 MB, against
124 MB for the whole Matcha stack. int8 should cut roughly 4x without retraining.

The risk is specific to this architecture. The DiT graph runs the same weights
16 times inside one ODE solve, so any quantization error is applied repeatedly
and compounds along the trajectory -- unlike a feed-forward net where it is
applied once. That is exactly why the quantized model has to be scored on CER,
not just measured for size and speed.

The VAE decoder is quantized separately and can be kept in fp32 if it turns out
to be the sensitive part; they are independent files.

  python eval/quantize_onnx.py --src onnx_export/distill183M_voiceD_s16 \
      --dst onnx_export/distill183M_voiceD_s16_int8
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--graphs", default="dit,vae,dur",
                    help="which graphs to quantize; the rest are copied as fp32")
    args = ap.parse_args()

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Shape inference cannot type every intermediate in this graph (the AdaLN
    # Gemm outputs come back untyped), and the quantizer refuses to guess.
    # Telling it everything untyped is fp32 is correct here: the export is a
    # plain fp32 graph with no pre-quantized subgraphs.
    extra = {"DefaultTensorType": int(onnx.TensorProto.FLOAT)}

    os.makedirs(args.dst, exist_ok=True)
    todo = set(args.graphs.split(","))
    report = {}

    for name in ("dur", "dit", "vae"):
        src = os.path.join(args.src, f"{name}.onnx")
        dst = os.path.join(args.dst, f"{name}.onnx")
        if not os.path.exists(src):
            continue
        before = os.path.getsize(src) / 1e6
        if name in todo:
            quantize_dynamic(src, dst, weight_type=QuantType.QInt8, extra_options=extra)
            after = os.path.getsize(dst) / 1e6
            report[name] = {"fp32_mb": round(before, 1), "int8_mb": round(after, 1),
                            "ratio": round(before / max(after, 1e-9), 2)}
            print(f"{name}: {before:.1f} MB -> {after:.1f} MB ({before/after:.2f}x)", flush=True)
        else:
            shutil.copy2(src, dst)
            report[name] = {"fp32_mb": round(before, 1), "int8_mb": round(before, 1),
                            "ratio": 1.0, "note": "not quantized"}
            print(f"{name}: {before:.1f} MB (fp32 유지)", flush=True)

    meta_src = os.path.join(args.src, "meta.json")
    if os.path.exists(meta_src):
        meta = json.load(open(meta_src))
        meta["quantization"] = {"scheme": "dynamic int8 (QInt8 weights)", "graphs": sorted(todo),
                                "report": report}
        json.dump(meta, open(os.path.join(args.dst, "meta.json"), "w"), indent=1)

    total_before = sum(v["fp32_mb"] for v in report.values())
    total_after = sum(v["int8_mb"] for v in report.values())
    print(json.dumps({"total_fp32_mb": round(total_before, 1),
                      "total_int8_mb": round(total_after, 1),
                      "ratio": round(total_before / max(total_after, 1e-9), 2)}, indent=1))


if __name__ == "__main__":
    main()
