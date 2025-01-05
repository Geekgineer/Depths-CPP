// image_inference.cpp

#include "depth_anything.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <path_to_input_image>" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    try {
        // Initialize the DepthAnything object with the path to your ONNX model
        // Set useCuda to true to enable GPU acceleration (if available)
        bool useCuda = false; // Set to true if you have CUDA-enabled GPU and want to use it
        DepthAnything depthEstimator(modelPath, useCuda);
        
        // Load an image using OpenCV
        cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            std::cerr << "Error: Failed to load image from " << imagePath << std::endl;
            return -1;
        }

        std::cout << "Image loaded successfully: " << imagePath << std::endl;

        // Perform depth estimation
        std::cout << "Running depth estimation..." << std::endl;
        cv::Mat depthMap = depthEstimator.predict(inputImage);
        std::cout << "Depth estimation completed." << std::endl;

        // Check if depthMap is valid
        if (depthMap.empty()) {
            std::cerr << "Error: Depth map is empty. Depth estimation might have failed." << std::endl;
            return -1;
        }

        // Normalize depth map for visualization
        cv::Mat depthVis;
        cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);

        // Apply a color map for better visualization
        cv::Mat depthColor;
        cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);

        // Display the original image and the depth map
        cv::imshow("Original Image", inputImage);
        cv::imshow("Depth Map", depthColor);

        // Save the depth map to disk
        std::string outputPath = "depth_map.png";
        if (cv::imwrite(outputPath, depthColor)) {
            std::cout << "Depth map saved successfully to " << outputPath << std::endl;
        } else {
            std::cerr << "Error: Failed to save depth map to " << outputPath << std::endl;
        }

        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0); // Wait for a key press to close the windows

    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred during inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
