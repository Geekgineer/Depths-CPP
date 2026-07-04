#!/bin/bash
# Depth Anything 3 (export first: see scripts/export_da3_onnx.py):
cd build/ && ./video_depth_estimation ../models/da3mono_large.onnx ../data/ferris_wheel.mp4 ../data/ferris_wheel_depth.mp4
# Depth Anything 2:
# cd build/ && ./video_depth_estimation ../models/vits.onnx ../data/ferris_wheel.mp4 ../data/ferris_wheel_depth.mp4
# Override the output codec if your OpenCV supports H.264: DEPTH_FOURCC=H264 ./run_video.sh
