#include "depth_anything.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
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
        // === Initialize DepthAnything ===
        bool useCuda = true;  // Set to false to disable GPU
        DepthAnything depthEstimator(modelPath, useCuda);

        // === Open the input video ===
        cv::VideoCapture cap(inputVideoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open the video file: " << inputVideoPath << std::endl;
            return -1;
        }

        // Define the target dimensions for processing (required by the depth model)
        int targetWidth = 518;
        int targetHeight = 518;
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        // Choose codec: try H.264 first; fallback to mp4v if needed
        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        if (fourcc == -1) {
            fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        }

        // Create the VideoWriter using the target dimensions.
        // Note: the output frame is the horizontal concatenation of two frames.
        cv::VideoWriter writer;
        writer.open(outputVideoPath, fourcc, fps, cv::Size(targetWidth * 2, targetHeight), true);
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot open output writer: " << outputVideoPath << std::endl;
            return -1;
        }
        writer.set(cv::VIDEOWRITER_PROP_QUALITY, 95); // Set quality (0-100)

        std::cout << "Processing video: " << inputVideoPath << std::endl;
        std::cout << "Output video: " << outputVideoPath << " (Codec: " << fourcc << ")" << std::endl;

        // === Batch processing loop ===
        const int batchSize = 16;
        std::vector<cv::Mat> batchFrames;
        std::vector<cv::Mat> batchDepths;
        int frameCount = 0;

        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            // Resize the frame to the target dimensions required by the depth model
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(targetWidth, targetHeight));
            batchFrames.push_back(resized);

            if (batchFrames.size() == batchSize) {
                auto start = std::chrono::high_resolution_clock::now();
                batchDepths = depthEstimator.predictBatch(batchFrames);
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "Batch of " << batchSize << " processed in "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                          << " ms\n";

                for (size_t i = 0; i < batchFrames.size(); ++i) {
                    frameCount++;
                    std::cout << "Writing frame " << frameCount << "\n";
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
                std::cout << "Writing frame " << frameCount << "\n";
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
