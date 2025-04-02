// ================================================================
// DepthAnything Image Inference - Batch Depth Estimation Tool
// ================================================================
//
// This program performs single or batch image depth estimation using
// the DepthAnything ONNX model. It processes one or more input images,
// applies the model to generate depth maps, and saves both a visualized
// (colored) depth image and a raw normalized depth map.
//
// It supports GPU acceleration via ONNX Runtime with CUDA when enabled,
// and automatically detects image formats in input directories.
//
// Features:
// - Single image or batch directory processing
// - Output saving in both color-mapped and raw formats
// - OpenCV visualization for individual image testing
// - ONNX Runtime-based depth inference with optional CUDA
//
// Usage:
//   ./image_depth_estimation <model.onnx> <input_image_or_folder> [output_dir]
//
// Example:
//   ./image_depth_estimation models/vits.onnx data/ ./depth_maps
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 02.04.2025
// Enhanced: Auto-detects image formats and supports raw 16-bit export
//
// ================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "depth_anything.hpp"

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <path_to_input_directory_or_image> [output_directory]" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string inputPath = argv[2];
    std::string outputDir = (argc > 3) ? argv[3] : "./depth_maps";

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDir);

    try {
        // Initialize the DepthAnything object with the path to your ONNX model
        // Set useCuda to true to enable GPU acceleration (if available)
        bool useCuda = false; // Set to true if you have CUDA-enabled GPU and want to use it
        DepthAnything depthEstimator(modelPath, useCuda);
        
        // Determine if input is a directory or a single file
        std::vector<std::string> imagePaths;
        if (std::filesystem::is_directory(inputPath)) {
            // Process all images in directory
            for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    // Check for common image extensions
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        imagePaths.push_back(entry.path().string());
                    }
                }
            }
            
            if (imagePaths.empty()) {
                std::cerr << "No valid image files found in directory: " << inputPath << std::endl;
                return -1;
            }
            
            std::cout << "Found " << imagePaths.size() << " images to process." << std::endl;
        } else {
            // Process single image
            imagePaths.push_back(inputPath);
        }

        // Process each image
        for (size_t i = 0; i < imagePaths.size(); ++i) {
            const std::string& imagePath = imagePaths[i];
            std::cout << "Processing image " << (i + 1) << "/" << imagePaths.size() 
                      << ": " << imagePath << std::endl;
            
            // Load an image using OpenCV
            cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (inputImage.empty()) {
                std::cerr << "Error: Failed to load image from " << imagePath << std::endl;
                continue;
            }

            // Perform depth estimation
            std::cout << "Running depth estimation..." << std::endl;
            cv::Mat depthMap = depthEstimator.predict(inputImage);
            
            // Check if depthMap is valid
            if (depthMap.empty()) {
                std::cerr << "Error: Depth map is empty. Depth estimation might have failed." << std::endl;
                continue;
            }

            // Normalize depth map for visualization
            cv::Mat depthVis;
            cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);

            // Apply a color map for better visualization
            cv::Mat depthColor;
            cv::applyColorMap(depthVis, depthColor, cv::COLORMAP_JET);

            // Generate output filename
            std::filesystem::path imageFsPath(imagePath);
            std::string outputPath = outputDir + "/" + imageFsPath.stem().string() + "_depth.png";
            
            // Save the depth map to disk
            if (cv::imwrite(outputPath, depthColor)) {
                std::cout << "Depth map saved successfully to " << outputPath << std::endl;
            } else {
                std::cerr << "Error: Failed to save depth map to " << outputPath << std::endl;
            }
            
            // Save the raw depth map as well (as a 16-bit single channel image)
            std::string rawOutputPath = outputDir + "/" + imageFsPath.stem().string() + "_depth_raw.png";
            cv::Mat depthNormalized;
            cv::normalize(depthMap, depthNormalized, 0, 65535, cv::NORM_MINMAX, CV_16U);
            if (cv::imwrite(rawOutputPath, depthNormalized)) {
                std::cout << "Raw depth map saved successfully to " << rawOutputPath << std::endl;
            }
            
            // Display images only if processing a single image
            if (imagePaths.size() == 1) {
                cv::imshow("Original Image", inputImage);
                cv::imshow("Depth Map", depthColor);
                std::cout << "Press any key to exit..." << std::endl;
                cv::waitKey(0); // Wait for a key press to close the windows
            }
        }

        std::cout << "Batch processing completed. Processed " << imagePaths.size() << " images." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred during inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}