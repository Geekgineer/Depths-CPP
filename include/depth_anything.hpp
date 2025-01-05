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

} // namespace DepthUtils

class DepthAnything
{
public:
    DepthAnything(const std::string &modelPath, bool useCuda = true);
    cv::Mat predict(const cv::Mat &image);
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

    cv::Mat preprocess(const cv::Mat &image, std::vector<float> &blob, std::vector<int64_t> &inputTensorShape);
    cv::Mat postprocess(const cv::Size &originalImageSize, const std::vector<Ort::Value> &outputTensors);
};

DepthAnything::DepthAnything(const std::string &modelPath, bool useCuda)
{
    try
    {
        sessionOptions.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Retrieve available execution providers (e.g., CPU, CUDA)
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
        OrtCUDAProviderOptions cudaOption;

        // Configure session options based on whether GPU is to be used and available
        if (useCuda && cudaAvailable != availableProviders.end())
        {
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
            bool height_dynamic = (inputTensorShapeVec[2] == -1 || inputTensorShapeVec[2] == 0);
            bool width_dynamic = (inputTensorShapeVec[3] == -1 || inputTensorShapeVec[3] == 0);
            isDynamicInputShape = height_dynamic || width_dynamic;
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

cv::Mat DepthAnything::preprocess(const cv::Mat &image, std::vector<float> &blob, std::vector<int64_t> &inputTensorShape)
{
    if (image.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    cv::Size currentInputShape = inputImageShape;
    if (isDynamicInputShape)
    {
        int rounded_w = (image.cols + 31) & ~31;
        int rounded_h = (image.rows + 31) & ~31;
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

cv::Mat DepthAnything::postprocess(const cv::Size &originalImageSize, const std::vector<Ort::Value> &outputTensors)
{
    if (outputTensors.empty())
    {
        throw std::runtime_error("No output tensors received from the model.");
    }

    const float *rawOutput = outputTensors[0].GetTensorData<float>();
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    std::vector<int64_t> outputShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

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

cv::Mat DepthAnything::predict(const cv::Mat &image)
{
    try
    {
        std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};
        std::vector<float> blob;
        cv::Mat preprocessedImage = preprocess(image, blob, inputTensorShape);

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

        return postprocess(image.size(), outputTensors);
    }
    catch (const Ort::Exception &e)
    {
        throw;
    }
}
