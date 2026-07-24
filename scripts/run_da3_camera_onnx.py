#!/usr/bin/env python3
# =============================================================================
# Run a DA3 multi-view camera-pose ONNX model (exported with
# `export_da3_onnx.py --camera`) and print / save the results.
# =============================================================================
#
# Consumes the 4-output camera graph:
#   input  "image"      : float32 (N, 3, H, W)   N views of one scene
#   output "depth"      : float32 (N, 1, H, W)
#   output "confidence" : float32 (N, 1, H, W)
#   output "extrinsics" : float32 (N, 3, 4)      world-to-camera [R|t]
#   output "intrinsics" : float32 (N, 3, 3)      pinhole K (pixels)
#
# Preprocessing matches the exporter contract:
#   RGB, resized to the model's (H, W) (÷14), /255, ImageNet mean/std.
#
# The graph is static: pass exactly N images (the baked view count). Each image
# is resized to the model's input resolution.
#
# Requirements:
#   pip install onnxruntime numpy pillow
#
# Examples:
#   python scripts/run_da3_camera_onnx.py --model models/da3_large_cam.onnx \
#       view0.jpg view1.jpg --save-npz scene.npz
# =============================================================================

import argparse
import sys

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_preprocess(path: str, h: int, w: int) -> np.ndarray:
    """Read an image -> (3, H, W) float32, RGB, /255, ImageNet-normalized."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)   # (3, H, W)


def resolve_hw(sess, fallback_res: int):
    """Determine the model's expected (H, W) from its input shape."""
    shape = sess.get_inputs()[0].shape  # [N, 3, H, W], entries may be str/None
    h, w = shape[2], shape[3]
    h = fallback_res if not isinstance(h, int) else h
    w = fallback_res if not isinstance(w, int) else w
    return h, w


def main():
    ap = argparse.ArgumentParser(
        description="Run a DA3 multi-view camera-pose ONNX model.")
    ap.add_argument("--model", required=True, help="Path to the --camera .onnx.")
    ap.add_argument("images", nargs="+",
                    help="View images of ONE scene (exactly the baked N).")
    ap.add_argument("--process-res", type=int, default=504,
                    help="Fallback (H, W) if the model input is dynamic.")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Execution provider.")
    ap.add_argument("--save-npz", default=None,
                    help="Save depth/confidence/extrinsics/intrinsics to .npz.")
    args = ap.parse_args()

    import onnxruntime as ort

    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ort.get_available_providers()
    sess = ort.InferenceSession(args.model, providers=providers)

    h, w = resolve_hw(sess, args.process_res)
    exp_n = sess.get_inputs()[0].shape[0]
    if isinstance(exp_n, int) and exp_n != len(args.images):
        sys.exit(f"model expects {exp_n} views but {len(args.images)} images "
                 "were given (the camera graph is static in view count).")

    print(f"[run] {len(args.images)} views at {h}x{w} on {sess.get_providers()}")
    batch = np.stack([load_and_preprocess(p, h, w) for p in args.images], axis=0)

    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: batch})
    names = [o.name for o in sess.get_outputs()]
    result = dict(zip(names, outs))

    depth = result["depth"]
    conf = result.get("confidence")
    extr = result["extrinsics"]
    intr = result["intrinsics"]

    print(f"[run] depth      {depth.shape}  range=[{depth.min():.3f}, {depth.max():.3f}]")
    if conf is not None:
        print(f"[run] confidence {conf.shape}  mean={conf.mean():.3f}")
    np.set_printoptions(precision=4, suppress=True)
    for i, path in enumerate(args.images):
        print(f"\n[view {i}] {path}")
        print("  extrinsics (world->cam) [R|t]:\n", extr[i])
        print("  intrinsics K:\n", intr[i])

    if args.save_npz:
        np.savez(
            args.save_npz,
            depth=depth,
            confidence=conf if conf is not None else np.empty(0),
            extrinsics=extr,
            intrinsics=intr,
            images=np.array(args.images),
        )
        print(f"\n[run] saved -> {args.save_npz}")


if __name__ == "__main__":
    main()
