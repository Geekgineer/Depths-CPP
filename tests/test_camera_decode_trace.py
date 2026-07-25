#!/usr/bin/env python3
# =============================================================================
# Verify that DA3's camera-pose decode traces to ONNX and matches PyTorch.
# =============================================================================
#
# `scripts/export_da3_onnx.py --camera` reuses the model's own camera-pose math
# (cam_dec -> pose_encoding_to_extri_intri -> affine_inverse). This test proves
# that isolatable path exports to a static ONNX graph and is numerically
# faithful, WITHOUT needing the multi-GB DA3 weights - the decode is
# dim-agnostic, so a random-init CameraDec is enough.
#
# The code below is copied verbatim from the DA3 source it mirrors:
#   src/depth_anything_3/model/cam_dec.py            (CameraDec)
#   src/depth_anything_3/model/utils/transform.py    (pose_encoding_to_extri_intri, quat_to_mat)
#   src/depth_anything_3/utils/geometry.py           (affine_inverse)
#
# It does NOT exercise the multi-view backbone (that needs weights); the export
# script keeps that path traceable by fixing ref_view_strategy="first" (constant
# reference index) and exporting a static graph.
#
# Requirements (dev only, not needed to build/run the C++ engine):
#   pip install torch onnx onnxruntime onnxscript numpy
#
# Run:
#   python tests/test_camera_decode_trace.py
# =============================================================================

import numpy as np
import torch
import torch.nn as nn

DIM = 384  # any embed dim works; the decode path is dimension-agnostic


class CameraDec(nn.Module):
    """Verbatim from depth_anything_3/model/cam_dec.py (inference path)."""

    def __init__(self, dim_in=DIM):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim), nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        out_t = self.fc_t(feat.float()).reshape(B, N, 3)
        out_qvec = self.fc_qvec(feat.float()).reshape(B, N, 4)
        out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        return torch.cat([out_t, out_qvec, out_fov], dim=-1)


def quat_to_mat(quaternions):
    """Verbatim from depth_anything_3/model/utils/transform.py."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_encoding_to_extri_intri(pose_encoding, image_size_hw):
    """Verbatim from depth_anything_3/model/utils/transform.py."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def affine_inverse(A):
    """Verbatim from depth_anything_3/utils/geometry.py."""
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


class DecodeWrapper(nn.Module):
    """cam_token (B,N,dim) -> w2c extrinsics (B,N,3,4), intrinsics (B,N,3,3)."""

    def __init__(self, h, w):
        super().__init__()
        self.cam_dec = CameraDec()
        self.h, self.w = h, w

    def forward(self, cam_token):
        pose_enc = self.cam_dec(cam_token)
        c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (self.h, self.w))
        return affine_inverse(c2w), ixt


def test_camera_decode_traces_and_matches():
    H, W, N = 504, 504, 2
    torch.manual_seed(0)
    m = DecodeWrapper(H, W).eval()
    dummy = torch.randn(1, N, DIM)

    with torch.no_grad():
        ref_w2c, ref_ixt = m(dummy)
    assert tuple(ref_w2c.shape) == (1, N, 3, 4)
    assert tuple(ref_ixt.shape) == (1, N, 3, 3)

    import tempfile
    import os

    import onnx
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "camera_decode.onnx")
        with torch.no_grad():
            # Static graph (matches `--camera`, which forces --static).
            torch.onnx.export(
                m, dummy, out,
                input_names=["cam_token"],
                output_names=["extrinsics", "intrinsics"],
                opset_version=17, do_constant_folding=True,
            )
        onnx.checker.check_model(onnx.load(out))
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        ort_w2c, ort_ixt = sess.run(None, {"cam_token": dummy.numpy()})

    # Relative tolerance: a random-init cam_dec can drive fov->0 (=> huge focal),
    # where float32 gives a large absolute but tiny relative diff.
    def rel(a, b):
        return np.abs(a - b).max() / (np.abs(a).max() + 1e-9)

    r_extr = rel(ref_w2c.numpy(), ort_w2c)
    r_intr = rel(ref_ixt.numpy(), ort_ixt)
    assert r_extr < 1e-5, f"extrinsics rel diff too large: {r_extr}"
    assert r_intr < 1e-5, f"intrinsics rel diff too large: {r_intr}"
    return r_extr, r_intr


if __name__ == "__main__":
    print("torch", torch.__version__)
    r_extr, r_intr = test_camera_decode_traces_and_matches()
    print(f"extrinsics rel|diff|={r_extr:.2e}  intrinsics rel|diff|={r_intr:.2e}")
    print("PASS: camera-pose decode traces to ONNX and matches PyTorch. ✅")
