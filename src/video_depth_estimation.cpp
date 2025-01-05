// video_inference.cpp

#include "depth_anything.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <path_to_model.onnx> <path_to_input_video> <path_to_output_video>" 
                  << std::endl;
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
        // Initialize the DepthAnything object with the path to your ONNX model
        // Set useCuda to true to enable GPU acceleration (if available)
        bool useCuda = false; // Set to true if you have a CUDA-enabled GPU and want to use it
        DepthAnything depthEstimator(modelPath, useCuda);

        // Open the input video
        cv::VideoCapture cap(inputVideoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open the video file: " << inputVideoPath << std::endl;
            return -1;
        }

        // Retrieve video properties
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        // Explicitly set the codec to 'mp4v' for MP4 files
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

        // Define the codec and create VideoWriter object
        // Since we're concatenating frames horizontally, the width doubles
        cv::VideoWriter writer;
        writer.open(outputVideoPath, fourcc, fps, cv::Size(frameWidth * 2, frameHeight), true);

        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot open the video writer with path: " << outputVideoPath << std::endl;
            return -1;
        }

        std::cout << "Processing video: " << inputVideoPath << std::endl;
        std::cout << "Output will be saved to: " << outputVideoPath << std::endl;

        cv::Mat frame;
        int frameCount = 0;

        while (true) {
            bool ret = cap.read(frame);
            if (!ret) {
                std::cout << "End of video stream or cannot read the frame." << std::endl;
                break;
            }

            frameCount++;
            std::cout << "Processing frame " << frameCount << "..." << std::endl;

            // Perform depth estimation
            cv::Mat depthMap = depthEstimator.predict(frame);

            // Check if depthMap is valid
            if (depthMap.empty()) {
                std::cerr << "Error: Depth map is empty for frame " << frameCount << ". Skipping frame." << std::endl;
                continue;
            }

            // Normalize depth map for visualization
            cv::Mat depthVis;
            cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);

            // Apply a color map for better visualization
            cv::Mat depthColor;
            cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);

            // Ensure both images have the same height and number of channels
            if (depthColor.size() != frame.size()) {
                cv::resize(depthColor, depthColor, frame.size());
            }

            // Optionally, combine the original frame and depth map side by side
            cv::Mat combined;
            cv::hconcat(frame, depthColor, combined);

            // Write the combined frame to the output video
            writer.write(combined);
        }

        // Release resources
        cap.release();
        writer.release();

        std::cout << "Video processing completed. Total frames processed: " << frameCount << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred during inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
