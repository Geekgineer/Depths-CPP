// =========================================
// DepthAnythingV2 Detector Header File
// =========================================
//
// This header defines the DepthAnythingV2 for performing depth estimation
// using a deep learning model. It includes necessary libraries, utility functions,
// and methods to handle model inference and depth map postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
// Enhanced: 12.03.2025 - Added batch processing support
//
// =========================================

#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <memory>
#include <thread>
#include <stdexcept>

namespace DepthUtils
{
    /**
     * @brief Resizes an image to the target dimensions without padding.
     *
     * @param img Input image.
     * @param target_w Target width.
     * @param target_h Target height.
     * @return cv::Mat Resized image.
     */
    inline cv::Mat resize_no_padding(const cv::Mat &img, int target_w, int target_h)
    {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(target_w, target_h));
        return resized;
    }

    /**
     * @brief Returns a rounded dimension that's a multiple of the given factor
     * 
     * @param dimension Original dimension
     * @param factor Factor to round to (e.g., 32)
     * @return int Rounded dimension
     */
    inline int round_to_multiple(int dimension, int factor = 32)
    {
        return (dimension + factor - 1) & ~(factor - 1);
    }

} // namespace DepthUtils

class DepthAnything
{
public:
    DepthAnything(const std::string &modelPath, bool useCuda = true, int maxBatchSize = 8);
    
    /**
     * @brief Process a single image and return its depth map
     * 
     * @param image Input image
     * @return cv::Mat Depth map
     */
    cv::Mat predict(const cv::Mat &image);
    
    /**
     * @brief Process a batch of images and return their depth maps
     * 
     * @param images Vector of input images
     * @return std::vector<cv::Mat> Vector of depth maps
     */
    std::vector<cv::Mat> predictBatch(const std::vector<cv::Mat> &images);
    
    /**
     * @brief Set the maximum batch size
     * 
     * @param batchSize New maximum batch size
     */
    void setMaxBatchSize(int batchSize);
    
    /**
     * @brief Get the current maximum batch size
     * 
     * @return int Current maximum batch size
     */
    int getMaxBatchSize() const;
    
    ~DepthAnything() = default;

private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DepthAnything"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    bool isDynamicInputShape;
    cv::Size inputImageShape;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    size_t numInputs;
    size_t numOutputs;
    int maxBatchSize;

    /**
     * @brief Preprocess a single image
     */
    cv::Mat preprocessSingle(const cv::Mat &image, std::vector<float> &blob, std::vector<int64_t> &inputTensorShape);
    
    /**
     * @brief Preprocess a batch of images
     */
    std::vector<cv::Mat> preprocessBatch(const std::vector<cv::Mat> &images, std::vector<float> &blob, std::vector<int64_t> &inputTensorShape);
    
    /**
     * @brief Postprocess a single depth map
     */
    cv::Mat postprocessSingle(const cv::Size &originalImageSize, const Ort::Value &outputTensor);
    
    /**
     * @brief Postprocess a batch of depth maps
     */
    std::vector<cv::Mat> postprocessBatch(const std::vector<cv::Size> &originalImageSizes, const Ort::Value &outputTensor);
    
    /**
     * @brief Determine optimal dimensions for a batch of images
     * 
     * @param images Batch of input images
     * @return cv::Size Common size to use for the batch
     */
    cv::Size determineBatchDimensions(const std::vector<cv::Mat> &images);
};

DepthAnything::DepthAnything(const std::string &modelPath, bool useCuda, int maxBatchSize)
    : maxBatchSize(maxBatchSize)
{
    try
    {
        sessionOptions.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);


        
        // Enable batch processing optimization
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        // Retrieve available execution providers (e.g., CPU, CUDA, TRT)
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
        OrtCUDAProviderOptions cudaOption;

        // Configure CUDA options for better batch processing
        if (useCuda && cudaAvailable != availableProviders.end())
        {
            cudaOption.device_id = 0;
            cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cudaOption.gpu_mem_limit = 0; // No limit
            cudaOption.arena_extend_strategy = 0; // Default strategy
            
            std::cout << "Inference device: GPU" << std::endl;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
        }
        else
        {
            if (useCuda)
            {
                std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
            }
            std::cout << "Inference device: CPU" << std::endl;
        }

#ifdef _WIN32
        std::wstring w_modelPath(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(env, w_modelPath.c_str(), sessionOptions);
#else
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
        std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        isDynamicInputShape = false;

        if (inputTensorShapeVec.size() >= 4)
        {
            bool batch_dynamic = (inputTensorShapeVec[0] == -1 || inputTensorShapeVec[0] == 0);
            bool height_dynamic = (inputTensorShapeVec[2] == -1 || inputTensorShapeVec[2] == 0);
            bool width_dynamic = (inputTensorShapeVec[3] == -1 || inputTensorShapeVec[3] == 0);
            isDynamicInputShape = batch_dynamic || height_dynamic || width_dynamic;
        }

        numInputs = session->GetInputCount();
        for (size_t i = 0; i < numInputs; ++i)
        {
            Ort::AllocatedStringPtr inputName(session->GetInputNameAllocated(i, allocator));
            inputNodeNameAllocatedStrings.push_back(std::move(inputName));
            inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
        }

        numOutputs = session->GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i)
        {
            Ort::AllocatedStringPtr outputName(session->GetOutputNameAllocated(i, allocator));
            outputNodeNameAllocatedStrings.push_back(std::move(outputName));
            outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
        }

        if (inputTensorShapeVec.size() >= 4)
        {
            if (!isDynamicInputShape)
            {
                inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
            }
            else
            {
                int default_w = 512;
                int default_h = 512;
                inputImageShape = cv::Size(default_w, default_h);
            }
        }
        else
        {
            throw std::runtime_error("Invalid input tensor shape.");
        }
    }
    catch (const Ort::Exception &e)
    {
        throw;
    }
}

cv::Mat DepthAnything::preprocessSingle(const cv::Mat &image, std::vector<float> &blob, std::vector<int64_t> &inputTensorShape)
{
    if (image.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    cv::Size currentInputShape = inputImageShape;
    if (isDynamicInputShape)
    {
        int rounded_w = DepthUtils::round_to_multiple(image.cols, 32);
        int rounded_h = DepthUtils::round_to_multiple(image.rows, 32);
        currentInputShape = cv::Size(rounded_w, rounded_h);
        inputTensorShape = {1, 3, currentInputShape.height, currentInputShape.width};
    }

    cv::Mat resizedImage = DepthUtils::resize_no_padding(image, currentInputShape.width, currentInputShape.height);

    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32FC3, 1.0f / 255.0f);

    for (int c = 0; c < 3; ++c)
    {
        floatImage.forEach<cv::Vec3f>([c, this](cv::Vec3f &pixel, const int *) -> void
                                      { pixel[c] = (pixel[c] - mean[c]) / std[c]; });
    }

    std::vector<cv::Mat> chw;
    cv::split(floatImage, chw);
    for (auto &channel : chw)
    {
        blob.insert(blob.end(), (float *)channel.datastart, (float *)channel.dataend);
    }

    return resizedImage;
}

std::vector<cv::Mat> DepthAnything::preprocessBatch(const std::vector<cv::Mat> &images, std::vector<float> &blob, std::vector<int64_t> &inputTensorShape)
{
    if (images.empty())
    {
        throw std::runtime_error("Input images vector is empty.");
    }

    // Determine common dimensions for all images in the batch
    cv::Size batchSize = determineBatchDimensions(images);
    
    // Setup input tensor shape for the batch
    int batchCount = static_cast<int>(images.size());
    inputTensorShape = {batchCount, 3, batchSize.height, batchSize.width};
    
    // Pre-allocate memory for the blob
    blob.reserve(batchCount * 3 * batchSize.height * batchSize.width);
    
    std::vector<cv::Mat> preprocessedImages;
    preprocessedImages.reserve(images.size());
    
    // Process each image
    for (const cv::Mat &img : images)
    {
        if (img.empty())
        {
            throw std::runtime_error("One of the input images is empty.");
        }
        
        // Resize image to match batch dimensions
        cv::Mat resizedImage = DepthUtils::resize_no_padding(img, batchSize.width, batchSize.height);
        preprocessedImages.push_back(resizedImage);
        
        // Convert to float and normalize
        cv::Mat floatImage;
        resizedImage.convertTo(floatImage, CV_32FC3, 1.0f / 255.0f);
        
        // Apply mean/std normalization
        for (int c = 0; c < 3; ++c)
        {
            floatImage.forEach<cv::Vec3f>([c, this](cv::Vec3f &pixel, const int *) -> void
                                          { pixel[c] = (pixel[c] - mean[c]) / std[c]; });
        }
        
        // Convert from HWC to CHW and add to blob
        std::vector<cv::Mat> chw;
        cv::split(floatImage, chw);
        for (auto &channel : chw)
        {
            blob.insert(blob.end(), (float *)channel.datastart, (float *)channel.dataend);
        }
    }
    
    return preprocessedImages;
}

cv::Size DepthAnything::determineBatchDimensions(const std::vector<cv::Mat> &images)
{
    if (images.empty())
    {
        return inputImageShape;
    }
    
    if (!isDynamicInputShape)
    {
        return inputImageShape;
    }
    
    // Find maximum dimensions across all images
    int maxWidth = 0;
    int maxHeight = 0;
    
    for (const auto &img : images)
    {
        maxWidth = std::max(maxWidth, img.cols);
        maxHeight = std::max(maxHeight, img.rows);
    }
    
    // Round up to multiple of 32 for better GPU performance
    int rounded_w = DepthUtils::round_to_multiple(maxWidth, 32);
    int rounded_h = DepthUtils::round_to_multiple(maxHeight, 32);
    
    return cv::Size(rounded_w, rounded_h);
}

cv::Mat DepthAnything::postprocessSingle(const cv::Size &originalImageSize, const Ort::Value &outputTensor)
{
    const float *rawOutput = outputTensor.GetTensorData<float>();
    auto outputTypeInfo = outputTensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputShape = outputTypeInfo.GetShape();

    if (outputShape.size() == 3 && outputShape[0] == 1)
    {
        int H = static_cast<int>(outputShape[1]);
        int W = static_cast<int>(outputShape[2]);
        cv::Mat depthMap(H, W, CV_32FC1, const_cast<float *>(rawOutput));
        cv::Mat resizedDepth;
        cv::resize(depthMap, resizedDepth, originalImageSize, 0, 0, cv::INTER_LINEAR);
        return resizedDepth;
    }
    else
    {
        throw std::runtime_error("Unhandled output tensor shape.");
    }
}

std::vector<cv::Mat> DepthAnything::postprocessBatch(const std::vector<cv::Size> &originalImageSizes, const Ort::Value &outputTensor)
{
    const float *rawOutput = outputTensor.GetTensorData<float>();
    auto outputTypeInfo = outputTensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputShape = outputTypeInfo.GetShape();
    
    if (outputShape.size() < 3 || outputShape[0] <= 0)
    {
        throw std::runtime_error("Invalid output tensor shape for batch processing.");
    }
    
    int batchSize = static_cast<int>(outputShape[0]);
    int H = static_cast<int>(outputShape[1]);
    int W = static_cast<int>(outputShape[2]);
    size_t singleImageSize = H * W;
    
    std::vector<cv::Mat> depthMaps;
    depthMaps.reserve(batchSize);
    
    for (int i = 0; i < batchSize; i++)
    {
        // Extract each depth map from the batch output
        cv::Mat depthMap(H, W, CV_32FC1);
        memcpy(depthMap.data, rawOutput + i * singleImageSize, singleImageSize * sizeof(float));
        
        // Resize to original image dimensions
        cv::Mat resizedDepth;
        if (i < originalImageSizes.size()) {
            cv::resize(depthMap, resizedDepth, originalImageSizes[i], 0, 0, cv::INTER_LINEAR);
        } else {
            resizedDepth = depthMap.clone();
        }
        
        depthMaps.push_back(resizedDepth);
    }
    
    return depthMaps;
}

cv::Mat DepthAnything::predict(const cv::Mat &image)
{
    try
    {
        std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};
        std::vector<float> blob;
        cv::Mat preprocessedImage = preprocessSingle(image, blob, inputTensorShape);

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            blob.data(),
            blob.size(),
            inputTensorShape.data(),
            inputTensorShape.size());

        std::vector<Ort::Value> outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            numInputs,
            outputNames.data(),
            numOutputs);

        return postprocessSingle(image.size(), outputTensors[0]);
    }
    catch (const Ort::Exception &e)
    {
        throw;
    }
}

std::vector<cv::Mat> DepthAnything::predictBatch(const std::vector<cv::Mat> &images)
{
    if (images.empty()) {
        return {};
    }
    
    try
    {
        // Process images in smaller batches if necessary
        const int actualBatchSize = std::min(static_cast<int>(images.size()), maxBatchSize);
        std::vector<cv::Mat> allResults;
        allResults.reserve(images.size());
        
        for (size_t batchStart = 0; batchStart < images.size(); batchStart += actualBatchSize)
        {
            // Create a batch of appropriate size
            size_t currentBatchSize = std::min(static_cast<size_t>(actualBatchSize), images.size() - batchStart);
            std::vector<cv::Mat> currentBatch(images.begin() + batchStart, images.begin() + batchStart + currentBatchSize);
            
            // Store original sizes for post-processing
            std::vector<cv::Size> originalSizes;
            originalSizes.reserve(currentBatchSize);
            for (const auto& img : currentBatch) {
                originalSizes.push_back(img.size());
            }
            
            // Preprocess batch
            std::vector<int64_t> inputTensorShape;
            std::vector<float> blob;
            preprocessBatch(currentBatch, blob, inputTensorShape);
            
            // Create input tensor
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                blob.data(),
                blob.size(),
                inputTensorShape.data(),
                inputTensorShape.size());
            
            // Run inference
            std::vector<Ort::Value> outputTensors = session->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(),
                &inputTensor,
                numInputs,
                outputNames.data(),
                numOutputs);
            
            // Postprocess results
            std::vector<cv::Mat> batchResults = postprocessBatch(originalSizes, outputTensors[0]);
            
            // Add to overall results
            allResults.insert(allResults.end(), batchResults.begin(), batchResults.end());
        }
        
        return allResults;
    }
    catch (const Ort::Exception &e)
    {
        throw;
    }
}

void DepthAnything::setMaxBatchSize(int batchSize)
{
    if (batchSize <= 0) {
        throw std::invalid_argument("Batch size must be greater than zero");
    }
    this->maxBatchSize = batchSize;
}

int DepthAnything::getMaxBatchSize() const
{
    return maxBatchSize;
}