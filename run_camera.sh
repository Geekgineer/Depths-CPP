#!/bin/bash
# Usage: camera_depth_estimation <model.onnx> <camera_id> [display_mode] [skip_frames] [resolution]
#   display_mode: original | depth | both   resolution: engine input longest-side (÷14), e.g. 266
# Depth Anything 3 (export first: see scripts/export_da3_onnx.py):
cd build/ && ./camera_depth_estimation ../models/da3mono_large.onnx 0 both 2
# Depth Anything 2:
# cd build/ && ./camera_depth_estimation ../models/vits.onnx 0 both 2
