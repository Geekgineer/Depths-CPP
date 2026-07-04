#!/usr/bin/env python3
# =============================================================================
# Export Depth Anything 3 -> ONNX for Depths-CPP
# =============================================================================
#
# Produces a slim, inference-only ONNX graph that Depths-CPP's C++ engine
# (`include/depth_anything.hpp`) can consume directly.
#
# Graph contract (matches include/depth_anything.hpp):
#   input  "image" : float32 (N, 3, H, W)
#            - RGB channel order
#            - ALREADY normalized: (pixel/255 - mean) / std
#              with ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
#            - H and W divisible by 14 (DINOv2 patch size)
#   output "depth" : float32 (N, 1, H, W)   relative (inverse) or metric depth
#   output "sky"   : float32 (N, 1, H, W)   ONLY for metric models (--metric)
#
# The exported graph deliberately stops at (backbone + depth head); the Python
# API's data-dependent post-processing (sky in-painting, camera/pose decode,
# Gaussian-splat branch) is skipped because it does not trace to a static ONNX
# graph and is not needed for depth inference.
#
# Requirements (run on a machine with the DA3 stack installed):
#   pip install depth-anything-3 onnx onnxsim
#   # torch/torchvision per your CUDA. Weights are pulled from Hugging Face.
#
# Examples:
#   # Relative monocular depth, small & fast, 504 px:
#   python scripts/export_da3_onnx.py --model da3mono-large \
#       --process-res 504 --fp16 --output models/da3mono_large.onnx
#
#   # Metric depth (adds the "sky" output):
#   python scripts/export_da3_onnx.py --model da3metric-large --metric \
#       --process-res 504 --output models/da3metric_large.onnx
#
# Exporter compatibility (IMPORTANT):
#   DA3 variants differ in which ops they use, and PyTorch's two ONNX exporters
#   support different op sets. If one path fails, try the other:
#     * default (legacy TorchScript) may reject `aten::cartesian_prod`.
#     * `--dynamo` (torch.export) may reject data-dependent ops in some
#       backbones' RoPE layers (e.g. `int(positions.max())`), especially with
#       dynamic shapes; `--static` sometimes helps.
#   These are limitations of the model code x torch version, not of this script.
#   Verified working recipe depends on your torch build; if both paths fail on a
#   given variant, pin the torch version used by the community exporters
#   (MoonCodeMaster / devin-lai, linked below) or export a non-RoPE variant.
#
# Notes:
#   * --process-res is only the tracing/reference resolution. Dynamic axes are
#     enabled so the model still accepts other (÷14) sizes at runtime; TensorRT
#     will build shape-specific engines on first use.
#   * bfloat16 is not fully supported by ONNX; weights are cast to fp16/fp32.
#   * Community references: github.com/MoonCodeMaster/Depth-Anything-3-Onnx and
#     github.com/devin-lai/Depth-Anything-3-Onnx
# =============================================================================

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthExportWrapper(nn.Module):
    """Wraps a DepthAnything3Net into a clean single-image -> depth graph.

    Input : (N, 3, H, W) ImageNet-normalized RGB, treated as N independent
            single-view scenes (B=N, S=1) so there is no cross-view mixing.
    Output: depth (N, 1, H, W) and, for metric models, sky (N, 1, H, W).
    """

    def __init__(self, net: nn.Module, metric: bool):
        super().__init__()
        self.net = net
        self.metric = metric

    def forward(self, image: torch.Tensor):
        n = image.shape[0]
        h, w = image.shape[-2], image.shape[-1]

        # (N, 3, H, W) -> (B=N, S=1, 3, H, W)
        x = image.unsqueeze(1)

        # Backbone + depth head only (deterministic, traceable).
        feats, _aux = self.net.backbone(
            x,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy="first",
        )
        out = self.net.head(feats, h, w, patch_start_idx=0)

        depth = out["depth"].reshape(n, 1, h, w).contiguous()

        if self.metric:
            sky_key = next((k for k in ("sky", "sky_logits", "mask") if k in out), None)
            if sky_key is None:
                raise KeyError(
                    "Requested --metric but the model head produced no sky output; "
                    f"available head keys: {list(out.keys())}"
                )
            sky = out[sky_key].reshape(n, 1, h, w).contiguous()
            return depth, sky

        return depth


def load_model(model_name: str, device: str):
    """Load a DA3 model via the official API and return its inner nn.Module."""
    from depth_anything_3.api import DepthAnything3

    # Accept either a short alias ("da3mono-large") or a full HF repo id.
    repo = model_name if "/" in model_name else f"depth-anything/{model_name.upper()}"
    print(f"[export] loading {repo} ...")
    api = DepthAnything3.from_pretrained(repo).to(device).eval()

    # The inner network exposes .backbone and .head.
    net = getattr(api, "model", api)
    if not (hasattr(net, "backbone") and hasattr(net, "head")):
        raise AttributeError(
            "Loaded object does not expose .backbone/.head; API layout may have "
            "changed. Inspect depth_anything_3.api.DepthAnything3."
        )
    return net


def main():
    ap = argparse.ArgumentParser(description="Export Depth Anything 3 to ONNX.")
    ap.add_argument("--model", default="da3mono-large",
                    help="DA3 alias (da3-small/base/large, da3mono-large, "
                         "da3metric-large) or full HF repo id.")
    ap.add_argument("--output", default="models/da3.onnx", help="Output .onnx path.")
    ap.add_argument("--process-res", type=int, default=504,
                    help="Reference tracing resolution (longest side, ÷14).")
    ap.add_argument("--metric", action="store_true",
                    help="Also export the metric 'sky' output.")
    ap.add_argument("--fp16", action="store_true", help="Cast weights to float16.")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    ap.add_argument("--no-simplify", action="store_true",
                    help="Skip onnxsim graph simplification.")
    ap.add_argument("--static", action="store_true",
                    help="Export a fixed input shape (no dynamic axes). Useful "
                         "when dynamic-shape tracing fails; the C++ engine still "
                         "resizes inputs to the model's fixed resolution.")
    ap.add_argument("--dynamo", action="store_true",
                    help="Use the TorchDynamo ONNX exporter (decomposes ops the "
                         "legacy exporter rejects, e.g. aten::cartesian_prod). "
                         "Requires `pip install onnxscript`.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Round the reference resolution to a multiple of 14.
    res = max(14, round(args.process_res / 14) * 14)
    if res != args.process_res:
        print(f"[export] process-res rounded {args.process_res} -> {res} (÷14)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    net = load_model(args.model, args.device)
    wrapper = DepthExportWrapper(net, metric=args.metric).to(args.device).eval()

    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        wrapper = wrapper.half()

    dummy = torch.randn(1, 3, res, res, device=args.device, dtype=dtype)

    output_names = ["depth"] + (["sky"] if args.metric else [])
    dynamic_axes = None
    if not args.static:
        dynamic_axes = {"image": {0: "batch", 2: "height", 3: "width"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch", 2: "height", 3: "width"}

    print(f"[export] tracing {args.model} at {res}x{res} ({dtype}) ...")
    export_kwargs = dict(
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    with torch.no_grad():
        if args.dynamo:
            torch.onnx.export(wrapper, dummy, args.output, dynamo=True, **export_kwargs)
        else:
            try:
                # Legacy TorchScript exporter: no onnxscript dependency. Some DA3
                # variants use ops it rejects (aten::cartesian_prod) - use
                # --dynamo in that case.
                torch.onnx.export(wrapper, dummy, args.output, dynamo=False, **export_kwargs)
            except TypeError:
                # Older torch without the `dynamo` kwarg.
                torch.onnx.export(wrapper, dummy, args.output, **export_kwargs)
    print(f"[export] wrote {args.output}")

    if not args.no_simplify:
        try:
            import onnx
            from onnxsim import simplify

            model = onnx.load(args.output)
            model_sim, ok = simplify(model)
            if ok:
                onnx.save(model_sim, args.output)
                print("[export] simplified with onnxsim")
            else:
                print("[export] onnxsim reported the model could not be simplified")
        except Exception as e:  # noqa: BLE001
            print(f"[export] onnxsim skipped: {e}")

    # Preprocessing reminder for the C++ side.
    print("\n[export] DONE. C++ preprocessing contract:")
    print(f"         RGB, /255, mean={IMAGENET_MEAN}, std={IMAGENET_STD}, ÷14")
    print("         (this is the default in include/depth_anything.hpp)")


if __name__ == "__main__":
    main()
