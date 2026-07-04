#include "depth_anything.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>   // std::getenv
#include <exception> // Needed for std::exception

void processAndWriteFrame(const cv::Mat& frame, const cv::Mat& depthMap, cv::VideoWriter& writer) {
    if (depthMap.empty()) {
        std::cerr << "Warning: Skipping empty depth map.\n";
        return;
    }

    // Normalize and colorize the depth map
    cv::Mat depthVis, depthColor, combined;
    cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);

    // Resize depthColor to match the frame if needed
    if (depthColor.size() != frame.size()) {
        cv::resize(depthColor, depthColor, frame.size());
    }

    // Horizontally concatenate the original frame and the colorized depth map
    cv::hconcat(frame, depthColor, combined);
    writer.write(combined);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_model.onnx> <path_to_input_video> <path_to_output_video>\n";
        return -1;
    }

    std::string modelPath = argv[1];
    std::string inputVideoPath = argv[2];
    std::string outputVideoPath = argv[3];

    // Ensure the output file has an .mp4 extension
    if (outputVideoPath.substr(outputVideoPath.find_last_of(".") + 1) != "mp4") {
        outputVideoPath += ".mp4";
        std::cout << "Output video path adjusted to: " << outputVideoPath << std::endl;
    }

    try {
        // === Initialize the depth engine ===
        depth::Config cfg;
        cfg.modelPath = modelPath;
        cfg.provider = depth::Provider::Auto;      // TensorRT -> CUDA -> CPU
        cfg.precision = depth::Precision::FP16;
        cfg.resizeMode = depth::ResizeMode::AspectLongest;
        cfg.process_res = 518;                     // quality-oriented for offline video
        cfg.maxBatchSize = 16;
        DepthAnything depthEstimator(cfg);

        // === Open the input video ===
        cv::VideoCapture cap(inputVideoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open the video file: " << inputVideoPath << std::endl;
            return -1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0) fps = 30.0;

        // The writer is opened lazily once the first frame's size is known so
        // the original aspect ratio is preserved (output = frame | depth).
        // mp4v is used by default because it is the most portable H.264-free
        // encoder across OpenCV/FFMPEG builds (H.264 via FFMPEG frequently
        // reports success yet writes nothing). Set DEPTH_FOURCC=H264 to override.
        cv::VideoWriter writer;
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (const char *env = std::getenv("DEPTH_FOURCC"); env && std::string(env).size() == 4)
            fourcc = cv::VideoWriter::fourcc(env[0], env[1], env[2], env[3]);

        std::cout << "Processing video: " << inputVideoPath << std::endl;
        std::cout << "Output video: " << outputVideoPath << " (Codec: " << fourcc << ")" << std::endl;

        // === Batch processing loop ===
        const int batchSize = 16;
        std::vector<cv::Mat> batchFrames;
        std::vector<cv::Mat> batchDepths;
        int frameCount = 0;

        auto ensureWriter = [&](const cv::Mat &frame) {
            if (writer.isOpened()) return;
            const cv::Size outSize(frame.cols * 2, frame.rows);
            writer.open(outputVideoPath, fourcc, fps, outSize, true);
            if (!writer.isOpened()) {
                // Requested encoder unavailable in this OpenCV build; fall back.
                fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                writer.open(outputVideoPath, fourcc, fps, outSize, true);
            }
            if (writer.isOpened())
                writer.set(cv::VIDEOWRITER_PROP_QUALITY, 95);
        };

        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            // Feed the original frame; the engine resizes internally with the
            // correct aspect ratio.
            batchFrames.push_back(frame);

            if (static_cast<int>(batchFrames.size()) == batchSize) {
                auto start = std::chrono::high_resolution_clock::now();
                batchDepths = depthEstimator.predictBatch(batchFrames);
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "Batch of " << batchSize << " processed in "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                          << " ms\n";

                for (size_t i = 0; i < batchFrames.size(); ++i) {
                    frameCount++;
                    ensureWriter(batchFrames[i]);
                    if (frameCount == 1 && !writer.isOpened()) {
                        std::cerr << "Error: Cannot open output writer: " << outputVideoPath << std::endl;
                        return -1;
                    }
                    processAndWriteFrame(batchFrames[i], batchDepths[i], writer);
                }

                batchFrames.clear();
                batchDepths.clear();
            }
        }

        // Process any leftover frames in the final batch
        if (!batchFrames.empty()) {
            batchDepths = depthEstimator.predictBatch(batchFrames);
            for (size_t i = 0; i < batchFrames.size(); ++i) {
                frameCount++;
                ensureWriter(batchFrames[i]);
                if (frameCount == 1 && !writer.isOpened()) {
                    std::cerr << "Error: Cannot open output writer: " << outputVideoPath << std::endl;
                    return -1;
                }
                processAndWriteFrame(batchFrames[i], batchDepths[i], writer);
            }
        }

        cap.release();
        writer.release();

        std::cout << "Done. Total frames processed: " << frameCount << std::endl;
        std::cout << "Output saved to: " << outputVideoPath << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception during processing: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
