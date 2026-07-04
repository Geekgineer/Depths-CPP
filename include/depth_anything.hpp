// =============================================================================
// Depths-CPP : Unified Depth Anything V2 / V3 inference engine (single header)
// =============================================================================
//
// High-performance, real-time monocular depth estimation in C++ built on
// ONNX Runtime + OpenCV. A single class, `DepthAnything`, transparently drives
// both Depth Anything V2 and Depth Anything V3 (relative, mono and metric)
// ONNX models.
//
// Key design points
// -----------------
//  * Model-family auto-detection from the ONNX graph:
//       - patch size (DINOv2 = 14 for V2/V3, legacy 32 fallback),
//       - output rank: V3 emits 4D depth (B,1,H,W); V2 emits 3D (B,H,W),
//       - metric models expose an extra "sky" output -> the "depth" output is
//         selected by name automatically.
//  * Correct preprocessing that mirrors the official DA3 InputProcessor:
//       RGB order, /255, ImageNet mean/std, aspect-preserving resize to
//       `process_res` rounded to a multiple of the patch size
//       (INTER_AREA when downscaling, INTER_CUBIC when upscaling).
//  * Execution providers: TensorRT (FP16 + engine cache) -> CUDA -> CPU,
//    chosen automatically or pinned via Config.
//  * Real-time friendly: reused pre-allocated blob buffer (no per-frame heap
//    churn) and vectorized normalization (no per-pixel lambdas).
//
// Backward compatibility: the original API
//     DepthAnything(modelPath, useCuda, maxBatchSize);
//     cv::Mat  predict(const cv::Mat&);
//     std::vector<cv::Mat> predictBatch(const std::vector<cv::Mat>&);
// is preserved. A richer `DepthAnything(const depth::Config&)` constructor and
// `depth::Config` struct expose the new capabilities.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date:   29.09.2024
// V3 rework: 04.07.2026 - Depth Anything V3 support, correct RGB preprocessing,
//                          TensorRT execution provider, unified engine.
// =============================================================================

#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace depth
{

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// Execution provider preference. `Auto` picks the best available at runtime
/// (TensorRT -> CUDA -> CPU).
enum class Provider
{
    Auto,
    TensorRT,
    CUDA,
    CPU
};

/// Compute precision used by the GPU execution providers.
enum class Precision
{
    FP32,
    FP16
};

/// How a dynamic-shape model maps an arbitrary input image onto the network
/// input grid. Ignored for fixed-shape models (they always use their baked-in
/// resolution).
enum class ResizeMode
{
    /// Preserve aspect ratio, scale so the LONGEST side == process_res, then
    /// round each side to a multiple of the patch size. This mirrors the DA3
    /// default `upper_bound_resize` and is the recommended mode.
    AspectLongest,
    /// Preserve aspect ratio, scale so the SHORTEST side == process_res, then
    /// round to a multiple of the patch size (DA3 `lower_bound_resize`).
    AspectShortest,
    /// Force a square process_res x process_res input (rounded to patch).
    Square,
    /// Legacy V2 behaviour: round the original image dimensions to a multiple
    /// of the patch size (no rescaling to process_res).
    OriginalMultiple
};

/// Model family. `Auto` infers it from the ONNX graph.
enum class Family
{
    Auto,
    V2,
    V3
};

struct Config
{
    std::string modelPath;

    Provider provider = Provider::Auto;
    Precision precision = Precision::FP16; ///< Only affects TensorRT / CUDA.
    Family family = Family::Auto;

    ResizeMode resizeMode = ResizeMode::AspectLongest;
    int process_res = 518; ///< Target longest/shortest/square side (multiple of patch enforced).
    int patch = 0;         ///< 0 = auto (14 for DINOv2 based V2/V3).

    /// OpenCV decodes images as BGR; DA2/DA3 expect RGB. Keep true.
    bool swapRB = true;
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> stddev = {0.229f, 0.224f, 0.225f};

    int maxBatchSize = 8;
    int intraOpThreads = 0; ///< 0 = std::thread::hardware_concurrency().
    int gpuDeviceId = 0;

    /// TensorRT engine cache directory (engines are serialized here so the slow
    /// first-run build is paid only once).
    std::string trtEngineCacheDir = "trt_engine_cache";

    bool verbose = true;
};

} // namespace depth

// -----------------------------------------------------------------------------
// Utility helpers
// -----------------------------------------------------------------------------
namespace DepthUtils
{
/// Round `dimension` UP to the nearest multiple of `factor`.
inline int round_up_to_multiple(int dimension, int factor)
{
    if (factor <= 1)
        return std::max(1, dimension);
    return ((dimension + factor - 1) / factor) * factor;
}

/// Round `dimension` to the NEAREST multiple of `factor` (matches DA3
/// `_make_divisible_by_resize`).
inline int round_nearest_multiple(int dimension, int factor)
{
    if (factor <= 1)
        return std::max(1, dimension);
    int down = (dimension / factor) * factor;
    int up = down + factor;
    int chosen = (std::abs(up - dimension) <= std::abs(dimension - down)) ? up : down;
    return std::max(factor, chosen);
}

/// Aspect-ratio resize that matches DA3: INTER_AREA when shrinking, INTER_CUBIC
/// when enlarging.
inline void resize_matched(const cv::Mat &src, cv::Mat &dst, const cv::Size &size)
{
    const bool upscale = size.width > src.cols || size.height > src.rows;
    cv::resize(src, dst, size, 0, 0, upscale ? cv::INTER_CUBIC : cv::INTER_AREA);
}
} // namespace DepthUtils

// -----------------------------------------------------------------------------
// DepthAnything engine
// -----------------------------------------------------------------------------
class DepthAnything
{
public:
    /// New, fully-configurable constructor.
    explicit DepthAnything(const depth::Config &config);

    /// Backward-compatible constructor. `useCuda` maps to Provider::Auto (which
    /// prefers TensorRT/CUDA when available) or Provider::CPU.
    DepthAnything(const std::string &modelPath, bool useCuda = true, int maxBatchSize = 8);

    /// Estimate depth for a single image. Returns a CV_32FC1 map at the input
    /// image's resolution. For relative models the values are inverse-relative
    /// depth; for metric models they are meters.
    cv::Mat predict(const cv::Mat &image);

    /// Estimate depth for a batch of images.
    std::vector<cv::Mat> predictBatch(const std::vector<cv::Mat> &images);

    void setMaxBatchSize(int batchSize);
    int getMaxBatchSize() const { return config_.maxBatchSize; }

    /// True when the loaded model produces an extra "sky" output (metric DA3).
    bool isMetric() const { return isMetric_; }
    /// Detected model family after construction.
    depth::Family family() const { return detectedFamily_; }
    int patchSize() const { return patch_; }

    ~DepthAnything() = default;

private:
    // --- ONNX Runtime state ------------------------------------------------
    depth::Config config_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "DepthAnything"};
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    std::vector<Ort::AllocatedStringPtr> inputNameHolders_;
    std::vector<const char *> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNameHolders_;
    std::vector<const char *> outputNames_;
    size_t numInputs_ = 0;
    size_t numOutputs_ = 0;
    size_t depthOutputIndex_ = 0; ///< Which output holds the depth map.

    // --- Model geometry ----------------------------------------------------
    bool dynamicShape_ = false;
    cv::Size fixedInputSize_{0, 0}; ///< Valid only when !dynamicShape_.
    int patch_ = 14;
    depth::Family detectedFamily_ = depth::Family::V3;
    bool isMetric_ = false;

    // --- Reusable scratch (avoids per-frame heap churn) --------------------
    std::vector<float> blob_;

    // --- Setup helpers -----------------------------------------------------
    void buildSessionOptions();
    void introspectModel();

    // --- Pipeline ----------------------------------------------------------
    cv::Size computeInputSize(const cv::Mat &image) const;
    cv::Size computeBatchInputSize(const std::vector<cv::Mat> &images) const;
    /// Normalize `image` into the CHW region of `blob_` at offset `dstOffset`.
    void fillBlob(const cv::Mat &image, const cv::Size &netSize, size_t dstOffset);
    std::vector<cv::Mat> runAndDecode(std::vector<int64_t> inputShape,
                                      const std::vector<cv::Size> &originalSizes);
    cv::Mat decodeOne(const float *data, int H, int W, int channelStride,
                      const cv::Size &originalSize) const;
};

// =============================================================================
// Implementation
// =============================================================================

inline DepthAnything::DepthAnything(const depth::Config &config) : config_(config)
{
    if (config_.intraOpThreads <= 0)
        config_.intraOpThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (config_.maxBatchSize <= 0)
        config_.maxBatchSize = 1;

    buildSessionOptions();

#ifdef _WIN32
    std::wstring wModelPath(config_.modelPath.begin(), config_.modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wModelPath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, config_.modelPath.c_str(), sessionOptions_);
#endif

    introspectModel();
}

inline DepthAnything::DepthAnything(const std::string &modelPath, bool useCuda, int maxBatchSize)
    : DepthAnything([&]
                    {
        depth::Config c;
        c.modelPath = modelPath;
        c.provider = useCuda ? depth::Provider::Auto : depth::Provider::CPU;
        c.maxBatchSize = maxBatchSize;
        return c; }())
{
}

inline void DepthAnything::buildSessionOptions()
{
    sessionOptions_.SetIntraOpNumThreads(config_.intraOpThreads);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    const std::vector<std::string> available = Ort::GetAvailableProviders();
    const bool haveTRT =
        std::find(available.begin(), available.end(), "TensorrtExecutionProvider") != available.end();
    const bool haveCUDA =
        std::find(available.begin(), available.end(), "CUDAExecutionProvider") != available.end();

    // Resolve the effective provider.
    depth::Provider prov = config_.provider;
    if (prov == depth::Provider::Auto)
    {
        if (haveTRT)
            prov = depth::Provider::TensorRT;
        else if (haveCUDA)
            prov = depth::Provider::CUDA;
        else
            prov = depth::Provider::CPU;
    }

    auto appendCuda = [&]()
    {
        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.device_id = config_.gpuDeviceId;
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // fast warm-up
        cudaOptions.gpu_mem_limit = 0;      // no explicit limit
        cudaOptions.arena_extend_strategy = 0;
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
    };

    if (prov == depth::Provider::TensorRT && haveTRT)
    {
        // TensorRT EP (configured via string key/value options) with an engine
        // cache and optional FP16. CUDA is appended as a fallback for any node
        // TensorRT cannot handle.
        std::vector<std::string> keys = {
            "device_id",
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
            "trt_max_workspace_size",
        };
        std::vector<std::string> vals = {
            std::to_string(config_.gpuDeviceId),
            "1",
            config_.trtEngineCacheDir,
            "2147483648", // 2 GB
        };
        if (config_.precision == depth::Precision::FP16)
        {
            keys.push_back("trt_fp16_enable");
            vals.push_back("1");
        }

        std::vector<const char *> ckeys, cvals;
        for (auto &k : keys)
            ckeys.push_back(k.c_str());
        for (auto &v : vals)
            cvals.push_back(v.c_str());

        try
        {
            const OrtApi &ortApi = Ort::GetApi();
            OrtTensorRTProviderOptionsV2 *trtOpts = nullptr;
            Ort::ThrowOnError(ortApi.CreateTensorRTProviderOptions(&trtOpts));
            std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(ortApi.ReleaseTensorRTProviderOptions)>
                trtGuard(trtOpts, ortApi.ReleaseTensorRTProviderOptions);
            Ort::ThrowOnError(ortApi.UpdateTensorRTProviderOptions(
                trtOpts, ckeys.data(), cvals.data(), ckeys.size()));
            Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                static_cast<OrtSessionOptions *>(sessionOptions_), trtOpts));
            if (config_.verbose)
                std::cout << "[DepthAnything] Execution provider: TensorRT ("
                          << (config_.precision == depth::Precision::FP16 ? "FP16" : "FP32")
                          << ")" << std::endl;
        }
        catch (const std::exception &e)
        {
            if (config_.verbose)
                std::cout << "[DepthAnything] TensorRT init failed (" << e.what()
                          << "), falling back to CUDA." << std::endl;
        }
        if (haveCUDA)
            appendCuda(); // fallback for unsupported nodes
    }
    else if ((prov == depth::Provider::CUDA || prov == depth::Provider::TensorRT) && haveCUDA)
    {
        appendCuda();
        if (config_.verbose)
            std::cout << "[DepthAnything] Execution provider: CUDA" << std::endl;
    }
    else
    {
        if (config_.provider != depth::Provider::CPU && !haveCUDA && !haveTRT && config_.verbose)
            std::cout << "[DepthAnything] No GPU provider in this ONNXRuntime build; using CPU."
                      << std::endl;
        else if (config_.verbose)
            std::cout << "[DepthAnything] Execution provider: CPU" << std::endl;
    }
}

inline void DepthAnything::introspectModel()
{
    Ort::AllocatorWithDefaultOptions allocator;

    // Inputs -----------------------------------------------------------------
    numInputs_ = session_->GetInputCount();
    for (size_t i = 0; i < numInputs_; ++i)
    {
        inputNameHolders_.push_back(session_->GetInputNameAllocated(i, allocator));
        inputNames_.push_back(inputNameHolders_.back().get());
    }

    Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
    std::vector<int64_t> inShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    if (inShape.size() < 4)
        throw std::runtime_error("DepthAnything: expected a 4D (N,C,H,W) model input.");

    auto isDyn = [](int64_t d) { return d <= 0; };
    dynamicShape_ = isDyn(inShape[0]) || isDyn(inShape[2]) || isDyn(inShape[3]);
    if (!dynamicShape_)
        fixedInputSize_ = cv::Size(static_cast<int>(inShape[3]), static_cast<int>(inShape[2]));

    // Outputs ----------------------------------------------------------------
    numOutputs_ = session_->GetOutputCount();
    depthOutputIndex_ = 0;
    for (size_t i = 0; i < numOutputs_; ++i)
    {
        outputNameHolders_.push_back(session_->GetOutputNameAllocated(i, allocator));
        outputNames_.push_back(outputNameHolders_.back().get());
        std::string name = outputNameHolders_.back().get();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name.find("depth") != std::string::npos)
            depthOutputIndex_ = i;
        if (name.find("sky") != std::string::npos)
            isMetric_ = true;
    }

    // Determine output rank to drive V2 (3D) vs V3 (4D) decoding.
    std::vector<int64_t> outShape =
        session_->GetOutputTypeInfo(depthOutputIndex_).GetTensorTypeAndShapeInfo().GetShape();

    // Patch size: honour config override, else infer. DINOv2 backbones (V2 &
    // V3) use 14; a fixed shape divisible by 14 confirms it.
    patch_ = config_.patch;
    if (patch_ <= 0)
    {
        if (!dynamicShape_)
        {
            const int h = fixedInputSize_.height;
            patch_ = (h % 14 == 0) ? 14 : ((h % 32 == 0) ? 32 : 14);
        }
        else
        {
            patch_ = 14;
        }
    }

    // Family: 4D depth output (N,1,H,W) is the V3 signature; 3D (N,H,W) is V2.
    if (config_.family == depth::Family::Auto)
        detectedFamily_ = (outShape.size() >= 4) ? depth::Family::V3 : depth::Family::V2;
    else
        detectedFamily_ = config_.family;

    if (config_.verbose)
    {
        std::cout << "[DepthAnything] Model: "
                  << (detectedFamily_ == depth::Family::V3 ? "Depth Anything V3" : "Depth Anything V2")
                  << (isMetric_ ? " (metric)" : "") << "\n"
                  << "                inputs=" << numInputs_ << " outputs=" << numOutputs_
                  << " depthOut='" << outputNames_[depthOutputIndex_] << "'\n"
                  << "                shape=" << (dynamicShape_ ? "dynamic" : "fixed")
                  << " patch=" << patch_ << std::endl;
    }
}

inline cv::Size DepthAnything::computeInputSize(const cv::Mat &image) const
{
    if (!dynamicShape_)
        return fixedInputSize_;

    const int w = image.cols, h = image.rows;
    switch (config_.resizeMode)
    {
    case depth::ResizeMode::OriginalMultiple:
        return {DepthUtils::round_up_to_multiple(w, patch_),
                DepthUtils::round_up_to_multiple(h, patch_)};
    case depth::ResizeMode::Square:
    {
        const int s = DepthUtils::round_nearest_multiple(config_.process_res, patch_);
        return {s, s};
    }
    case depth::ResizeMode::AspectShortest:
    case depth::ResizeMode::AspectLongest:
    default:
    {
        const int ref = (config_.resizeMode == depth::ResizeMode::AspectShortest)
                            ? std::min(w, h)
                            : std::max(w, h);
        const double scale = static_cast<double>(config_.process_res) / std::max(1, ref);
        int nw = std::max(1, static_cast<int>(std::round(w * scale)));
        int nh = std::max(1, static_cast<int>(std::round(h * scale)));
        return {DepthUtils::round_nearest_multiple(nw, patch_),
                DepthUtils::round_nearest_multiple(nh, patch_)};
    }
    }
}

inline cv::Size DepthAnything::computeBatchInputSize(const std::vector<cv::Mat> &images) const
{
    if (!dynamicShape_)
        return fixedInputSize_;
    // All images in a batch must share the network input size. Use the largest
    // per-image size so nothing is downscaled below its own target.
    cv::Size best{0, 0};
    for (const auto &img : images)
    {
        cv::Size s = computeInputSize(img);
        best.width = std::max(best.width, s.width);
        best.height = std::max(best.height, s.height);
    }
    return best;
}

inline void DepthAnything::fillBlob(const cv::Mat &image, const cv::Size &netSize, size_t dstOffset)
{
    cv::Mat resized;
    DepthUtils::resize_matched(image, resized, netSize);

    cv::Mat rgb;
    if (config_.swapRB)
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    else
        rgb = resized;

    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // Split into channels and apply (x - mean) / std per channel (vectorized),
    // writing directly into the pre-allocated CHW blob region.
    const int planeSize = netSize.width * netSize.height;
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    for (int c = 0; c < 3; ++c)
    {
        // Destination plane view over blob_ (contiguous, CHW).
        cv::Mat dst(netSize.height, netSize.width, CV_32FC1,
                    blob_.data() + dstOffset + static_cast<size_t>(c) * planeSize);
        cv::Mat normalized = (channels[c] - config_.mean[c]) / config_.stddev[c];
        normalized.copyTo(dst);
    }
}

inline cv::Mat DepthAnything::decodeOne(const float *data, int H, int W, int channelStride,
                                        const cv::Size &originalSize) const
{
    // `channelStride` is 0 for 3D outputs (N,H,W) or H*W for 4D (N,C,H,W) where
    // we always read channel 0 (the depth channel).
    (void)channelStride;
    cv::Mat depth(H, W, CV_32FC1, const_cast<float *>(data));
    cv::Mat out;
    cv::resize(depth, out, originalSize, 0, 0, cv::INTER_CUBIC);
    return out;
}

inline std::vector<cv::Mat>
DepthAnything::runAndDecode(std::vector<int64_t> inputShape,
                            const std::vector<cv::Size> &originalSizes)
{
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, blob_.data(), blob_.size(), inputShape.data(), inputShape.size());

    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, inputNames_.data(), &inputTensor, numInputs_,
                      outputNames_.data(), numOutputs_);

    const Ort::Value &depthOut = outputs[depthOutputIndex_];
    const float *raw = depthOut.GetTensorData<float>();
    std::vector<int64_t> shp = depthOut.GetTensorTypeAndShapeInfo().GetShape();

    int batch = 1, H = 0, W = 0;
    long perImage = 0;
    if (shp.size() == 4)
    {
        // (N, C, H, W) - depth uses channel 0.
        batch = static_cast<int>(shp[0]);
        H = static_cast<int>(shp[2]);
        W = static_cast<int>(shp[3]);
        perImage = static_cast<long>(shp[1]) * H * W; // stride between images
    }
    else if (shp.size() == 3)
    {
        // (N, H, W)
        batch = static_cast<int>(shp[0]);
        H = static_cast<int>(shp[1]);
        W = static_cast<int>(shp[2]);
        perImage = static_cast<long>(H) * W;
    }
    else if (shp.size() == 2)
    {
        // (H, W) - single, no batch dim.
        batch = 1;
        H = static_cast<int>(shp[0]);
        W = static_cast<int>(shp[1]);
        perImage = static_cast<long>(H) * W;
    }
    else
    {
        throw std::runtime_error("DepthAnything: unsupported depth output rank.");
    }

    std::vector<cv::Mat> results;
    results.reserve(batch);
    for (int i = 0; i < batch; ++i)
    {
        const cv::Size orig =
            (static_cast<size_t>(i) < originalSizes.size()) ? originalSizes[i] : cv::Size(W, H);
        results.push_back(decodeOne(raw + static_cast<long>(i) * perImage, H, W, 0, orig));
    }
    return results;
}

inline cv::Mat DepthAnything::predict(const cv::Mat &image)
{
    if (image.empty())
        throw std::runtime_error("DepthAnything::predict: empty input image.");

    const cv::Size netSize = computeInputSize(image);
    blob_.resize(static_cast<size_t>(3) * netSize.width * netSize.height);
    fillBlob(image, netSize, 0);

    std::vector<int64_t> inputShape = {1, 3, netSize.height, netSize.width};
    return runAndDecode(std::move(inputShape), {image.size()})[0];
}

inline std::vector<cv::Mat> DepthAnything::predictBatch(const std::vector<cv::Mat> &images)
{
    if (images.empty())
        return {};

    const int chunk = std::min(static_cast<int>(images.size()), config_.maxBatchSize);
    std::vector<cv::Mat> all;
    all.reserve(images.size());

    for (size_t start = 0; start < images.size(); start += chunk)
    {
        const size_t count = std::min(static_cast<size_t>(chunk), images.size() - start);
        std::vector<cv::Mat> sub(images.begin() + start, images.begin() + start + count);

        const cv::Size netSize = computeBatchInputSize(sub);
        const size_t planeCount = static_cast<size_t>(3) * netSize.width * netSize.height;
        blob_.resize(planeCount * count);

        std::vector<cv::Size> originalSizes;
        originalSizes.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            if (sub[i].empty())
                throw std::runtime_error("DepthAnything::predictBatch: empty image in batch.");
            fillBlob(sub[i], netSize, i * planeCount);
            originalSizes.push_back(sub[i].size());
        }

        std::vector<int64_t> inputShape = {static_cast<int64_t>(count), 3, netSize.height,
                                           netSize.width};
        std::vector<cv::Mat> batchResults = runAndDecode(std::move(inputShape), originalSizes);
        all.insert(all.end(), std::make_move_iterator(batchResults.begin()),
                   std::make_move_iterator(batchResults.end()));
    }
    return all;
}

inline void DepthAnything::setMaxBatchSize(int batchSize)
{
    if (batchSize <= 0)
        throw std::invalid_argument("Batch size must be greater than zero.");
    config_.maxBatchSize = batchSize;
}
