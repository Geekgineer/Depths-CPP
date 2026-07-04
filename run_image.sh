#!/bin/bash
# Depth Anything 3 (export first: see scripts/export_da3_onnx.py):
cd build/ && ./image_depth_estimation ../models/da3mono_large.onnx ../data/indoor.jpg
# Depth Anything 2:
# cd build/ && ./image_depth_estimation ../models/vits.onnx ../data/indoor.jpg
