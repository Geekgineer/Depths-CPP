// ================================================================
// Real-Time Camera Depth Estimation with DepthAnything
// ================================================================
//
// This implementation captures frames from a camera, performs batch
// depth estimation using the DepthAnything model (ONNX format), and
// displays real-time visualizations of original and depth-annotated
// frames. It features adaptive batch processing, multi-threaded
// execution (capture, processing, and display), and robust performance
// statistics reporting.
//
// Features:
// - Safe multi-threaded queues for camera frames and depth maps
// - Adaptive frame skipping and dynamic batch size adjustment
// - Support for original, depth-only, or combined display modes
// - MJPEG optimization and real-time FPS overlays
// - Handles frame drops and overloading through backpressure
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 01.04.2025

// Enhanced: Supports smooth combined display with depth overlays
//
// Usage:
//    ./camera_depth_estimation <model.onnx> <camera_id> [display_mode] [skip_frames] [resolution]
// Example:
//    ./camera_depth_estimation depth_model.onnx 0 both 2 256
//
// Display Modes:
//    original - Show original camera stream only
//    depth    - Show only the estimated depth map
//    both     - Display original and depth side-by-side (default)
//
// Dependencies:
//    - OpenCV 4.x
//    - ONNX Runtime (via depth_anything.hpp)
//
// ================================================================


#include "depth_anything.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include <optional>
#include <chrono>

// Thread-safe queue with a maximum size using std::shared_ptr to avoid unnecessary copying
template <typename T>
class SafeQueue
{
public:
    SafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    // Enqueue a new item into the queue. Returns false if the queue is full.
    bool enqueue(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_)
        {
            // Instead of just returning false, replace the oldest item
            if (!queue_.empty())
            {
                queue_.pop(); // Remove oldest frame
            }
        }
        queue_.push(std::move(value));
        cond_var_.notify_one();
        return true;
    }

    // Dequeue an item from the queue. Returns std::nullopt if finished and queue is empty.
    std::optional<T> dequeue()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !finished_)
        {
            cond_var_.wait(lock);
        }
        if (queue_.empty())
            return std::nullopt;
        T result = std::move(queue_.front());
        queue_.pop();
        return result;
    }

    // Non-blocking dequeue that returns the latest item
    std::optional<T> dequeue_latest() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty())
            return std::nullopt;
            
        // Directly get the last item without popping everything
        T result = std::move(queue_.back());
        
        // Clear the queue
        std::queue<T> empty;
        std::swap(queue_, empty);
        
        return result;
    }
    
    // Skip-to-Latest
    void clear_except_latest(size_t keep_count) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (queue_.size() <= keep_count) return;
        
        // Keep only the latest keep_count items
        size_t to_remove = queue_.size() - keep_count;
        for (size_t i = 0; i < to_remove; i++) {
            queue_.pop();
        }
    }

    // Signal that no more items will be enqueued
    void set_finished()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        cond_var_.notify_all();
    }

    // Clear the queue
    void clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }

    // Get current queue size
    size_t size()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    size_t max_size_;
    bool finished_ = false;
};

// Enumeration for display modes
enum class DisplayMode
{
    ORIGINAL,
    DEPTH,
    BOTH
};

// Function to parse display mode from string
bool parse_display_mode(const std::string &mode_str, DisplayMode &mode)
{
    if (mode_str == "original")
    {
        mode = DisplayMode::ORIGINAL;
        return true;
    }
    else if (mode_str == "depth")
    {
        mode = DisplayMode::DEPTH;
        return true;
    }
    else if (mode_str == "both")
    {
        mode = DisplayMode::BOTH;
        return true;
    }
    else
    {
        return false;
    }
}

// Function to report fps
std::string get_fps_string(float fps)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << fps << " FPS";
    return ss.str();
}

// Function to convert fourcc code to string
std::string fourcc_to_string(int fourcc)
{
    std::string result;
    result += static_cast<char>(fourcc & 0xFF);
    result += static_cast<char>((fourcc >> 8) & 0xFF);
    result += static_cast<char>((fourcc >> 16) & 0xFF);
    result += static_cast<char>((fourcc >> 24) & 0xFF);
    return result;
}
int main(int argc, char *argv[])
{
    // Check for proper usage
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <camera_id> [display_mode] [skip_frames]" << std::endl;
        std::cerr << "display_mode options: original, depth, both" << std::endl;
        std::cerr << "skip_frames: Number of frames to skip (default: 2)" << std::endl;
        std::cerr << "Example: " << argv[0] << " depth_model.onnx 0 both 2" << std::endl;
        return -1;
    }

    // Model resolution parameter
    // int modelResolution = 384; // Default resolution
    int modelResolution = 256; // Default resolution
    if (argc == 6) {
        modelResolution = std::atoi(argv[5]);
        if (modelResolution < 128 || modelResolution > 512) {
            std::cerr << "Invalid model resolution: " << argv[5] << ". Valid range is 128-512." << std::endl;
            return -1;
        }
    }

    std::string modelPath = argv[1];
    int cameraID = std::atoi(argv[2]);

    // Default display mode is BOTH
    DisplayMode displayMode = DisplayMode::BOTH;

    if (argc >= 4)
    {
        std::string mode_str = argv[3];
        if (!parse_display_mode(mode_str, displayMode))
        {
            std::cerr << "Invalid display mode: " << mode_str << std::endl;
            std::cerr << "Valid options are: original, depth, both" << std::endl;
            return -1;
        }
    }

    // Frame skipping parameter
    int skip_frames = 1; // Default: process every 16th frame (0, 16, 32, etc.)
    if (argc == 5)
    {
        skip_frames = std::atoi(argv[4]);
        if (skip_frames < 0)
        {
            std::cerr << "Invalid skip_frames value: " << argv[4] << ". It must be non-negative." << std::endl;
            return -1;
        }
    }

    try
    {
        // Initialize the DepthAnything object with CUDA if available
        bool useCuda = true;
        DepthAnything depthEstimator(modelPath, useCuda);

        // Open the camera with optimized settings
        cv::VideoCapture cap;

// Try to open with hardware acceleration
#ifdef _WIN32
        cap.open(cameraID, cv::CAP_DSHOW);
#else
        cap.open(cameraID, cv::CAP_V4L2);
#endif

        if (!cap.isOpened())
        {
            cap.open(cameraID, cv::CAP_ANY);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Could not open the camera with ID " << cameraID << "." << std::endl;
                return -1;
            }
        }

        // Optimize camera settings for performance
        // Lower resolution can dramatically improve processing speed
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 25);

        // Use MJPEG format if supported (significantly reduces CPU load)
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        // Disable auto focus if supported (reduces frame-to-frame variability)
        cap.set(cv::CAP_PROP_AUTOFOCUS, 0);

        // Verify actual camera settings
        double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = cap.get(cv::CAP_PROP_FPS);
        int actual_fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

        std::cout << "Camera Settings:" << std::endl;
        std::cout << "Resolution: " << actual_width << "x" << actual_height << std::endl;
        std::cout << "FPS: " << actual_fps << std::endl;
        std::cout << "Format: " << fourcc_to_string(actual_fourcc) << std::endl;
        std::cout << "Skip frames: " << skip_frames << std::endl;

        // Use smaller queue sizes - we want to process the latest frames, not build up backlogs
        SafeQueue<std::shared_ptr<cv::Mat>> frameQueue(5);        // For processing
        SafeQueue<std::shared_ptr<cv::Mat>> depthQueue(5);        // Depth maps
        SafeQueue<std::shared_ptr<cv::Mat>> displayFrameQueue(5); // For display

        std::atomic<bool> running(true);
        std::atomic<float> fps_capture(0.0f);
        std::atomic<float> fps_processing(0.0f);
        std::atomic<float> fps_display(0.0f);

        // Define batch size - adjust based on your GPU memory and processing capability
        int batchSize = 8; // Start with a smaller batch size
        // Increase for high-end GPUs with plenty of memory
        // or decrease for GPUs with limited memory
                
        
        // Add this adaptive batch size code:
        int adaptiveBatchSize = batchSize; // Start with the default batch size

        // Capture thread: reads frames from the camera and enqueues them
        std::thread captureThread([&]()
                                  {
            int frame_counter = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            int frames_processed = 0;
        

            while (running) {
                std::shared_ptr<cv::Mat> framePtr = std::make_shared<cv::Mat>();
                if (!cap.read(*framePtr)) {
                    std::cerr << "Warning: Failed to read frame from camera." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Brief pause to avoid busy-waiting
                    continue;
                }

                frame_counter++;
                frames_processed++;

                // Update FPS counter every second
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
                if (elapsed > 1000) {
                    fps_capture.store((float)frames_processed * 1000 / elapsed);
                    frames_processed = 0;
                    start_time = current_time;
                }

                // Apply adaptive frame skipping based on queue size
                int dynamic_skip = std::max(skip_frames, static_cast<int>(frameQueue.size() * 2));
                if (dynamic_skip > 0 && (frame_counter % (dynamic_skip + 1)) != 0) {
                    continue;
                }

                // Only keep the latest frames in each queue
                if (frameQueue.size() > 3) {
                    frameQueue.clear_except_latest(2);
                }
                if (displayFrameQueue.size() > 3) {
                    displayFrameQueue.clear_except_latest(2);
                }

                // Enqueue frame for display (always needed)
                displayFrameQueue.enqueue(framePtr);

                // Enqueue frame for processing only if depth estimation is needed
                if (displayMode == DisplayMode::DEPTH || displayMode == DisplayMode::BOTH) {
                    frameQueue.enqueue(framePtr);
                }
            }
            frameQueue.set_finished();
            displayFrameQueue.set_finished(); });

            // Processing thread: dequeues frames, performs batch depth estimation, and enqueues depth maps
            std::thread processingThread([&]()
            {
                if (displayMode != DisplayMode::DEPTH && displayMode != DisplayMode::BOTH) {
                    // No need to process depth if not required
                    return;
                }

                int frames_processed = 0;
                auto start_time = std::chrono::high_resolution_clock::now();
                

                // Vectors for batch processing
                // Pre-allocate memory for batches to avoid repeated allocations
                std::vector<cv::Mat> batchFrames;
                std::vector<std::shared_ptr<cv::Mat>> originalFramePtrs;
                batchFrames.reserve(batchSize);
                originalFramePtrs.reserve(batchSize);

                // Pre-allocate common matrices
                cv::Mat resizedBuffer(modelResolution, modelResolution, CV_8UC3); // Pre-allocate resize buffer

                while (running) {
                    
                    // Update adaptive batch size based on performance metrics
                    float fpsRatio = fps_processing.load() / std::max(1.0f, fps_capture.load());
                    if (fpsRatio < 0.5f) {
                        // We're falling behind - reduce batch size to process faster
                        adaptiveBatchSize = std::max(1, adaptiveBatchSize - 1);
                    } else if (fpsRatio > 0.9f && adaptiveBatchSize < batchSize) {
                        // We're keeping up well - try increasing batch size for efficiency
                        adaptiveBatchSize = std::min(batchSize, adaptiveBatchSize + 1);
                    }
                    
                    
                    // Collect frames for batch processing
                    batchFrames.clear();
                    originalFramePtrs.clear();
                    
                    // Try to collect up to batchSize frames
                    for (int i = 0; i < adaptiveBatchSize && running; ++i) {
                        auto optFrame = frameQueue.dequeue();
                        if (!optFrame.has_value()) {
                            // No more frames available
                            if (i == 0) goto end_processing; // Exit if no frames collected
                            break; // Process the frames we have so far
                        }
                        
                        auto framePtr = std::move(optFrame.value());
                        originalFramePtrs.push_back(framePtr);
                        
                        // Resize to model's optimal input size
                        // cv::Mat resized;
                        // cv::Size optimalSize(384, 384); // Adjust based on your model's optimal size
                        // cv::resize(*framePtr, resized, optimalSize);
                        // batchFrames.push_back(resized);
                        

                        // Resize to model's optimal input size - consider using a smaller resolution
                        cv::Size optimalSize(modelResolution, modelResolution); // Smaller resolution for faster processing
                        cv::resize(*framePtr, resizedBuffer, optimalSize);
                        batchFrames.push_back(resizedBuffer.clone()); // Only clone here when necessary

                        // Don't wait for a full batch if frames are available and we have enough to process
                        if (i >= 2 && frameQueue.size() == 0) {
                            break; // Process what we have instead of waiting for a full batch
                        }
                    }
                    
                    if (batchFrames.empty()) continue;
                    
                    try {
                        // Process batch of frames
                        auto batch_start = std::chrono::high_resolution_clock::now();
                        std::vector<cv::Mat> depthMaps = depthEstimator.predictBatch(batchFrames);
                        auto batch_end = std::chrono::high_resolution_clock::now();
                        auto batch_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
                        
                        if (depthMaps.size() != batchFrames.size()) {
                            std::cerr << "Warning: Batch processing returned different number of results!" << std::endl;
                            continue;
                        }
                        
                        // Check if we're falling behind and need to drop frames
                        if (frameQueue.size() > adaptiveBatchSize * 2) {
                            // Too many frames queued, we're falling behind - skip some
                            std::cout << "Warning: Dropping " << (frameQueue.size() - adaptiveBatchSize) 
                                      << " frames to catch up" << std::endl;
                            frameQueue.clear_except_latest(adaptiveBatchSize);
                        }

                        // Process and enqueue each depth map
                        for (size_t i = 0; i < depthMaps.size(); ++i) {
                            // Post-process the depth map
                            cv::Mat depthVis;
                            cv::normalize(depthMaps[i], depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);
                            
                            // Apply smoothing to reduce noise
                            cv::medianBlur(depthVis, depthVis, 3);
                            
                            // Resize back to original frame size for display
                            if (depthVis.size() != originalFramePtrs[i]->size()) {
                                cv::resize(depthVis, depthVis, originalFramePtrs[i]->size());
                            }

                            // Only keep the latest depth maps
                            if (depthQueue.size() > batchSize) {
                                depthQueue.clear();
                            }

                            // Enqueue depth map for display
                            std::shared_ptr<cv::Mat> depthPtr = std::make_shared<cv::Mat>(depthVis);
                            depthQueue.enqueue(depthPtr);
                            
                            frames_processed++;
                        }
                        
                        // Update FPS counter
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
                        if (elapsed > 1000) {
                            fps_processing.store((float)frames_processed * 1000 / elapsed);
                            frames_processed = 0;
                            start_time = current_time;
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Exception during batch depth estimation: " << e.what() << std::endl;
                    }
                }
                
            end_processing:
                depthQueue.set_finished();
            });

        // Display thread: dequeues frames and depth maps and displays them according to the selected mode
        std::thread displayThread([&]()
                                  {
            int frames_displayed = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Add this as a class member or global variable outside the display thread
            std::shared_ptr<cv::Mat> lastValidDepthPtr = nullptr;

            // Create a fixed-size display window
            cv::namedWindow("Real-time Depth Estimation", cv::WINDOW_NORMAL);
            cv::resizeWindow("Real-time Depth Estimation", 800, 600);
            
            while (running) {
                std::shared_ptr<cv::Mat> framePtr;
                std::shared_ptr<cv::Mat> depthPtr;

                // Get the latest frame for display - non-blocking
                auto optFrame = displayFrameQueue.dequeue_latest();
                if (optFrame.has_value()) {
                    framePtr = std::move(optFrame.value());
                }

                // Get the latest depth map if needed - non-blocking
                if (displayMode == DisplayMode::DEPTH || displayMode == DisplayMode::BOTH) {
                    auto optDepth = depthQueue.dequeue_latest();
                    if (optDepth.has_value()) {
                        depthPtr = std::move(optDepth.value());
                    }
                }

                // Prepare display frame
                bool shouldDisplay = false;
                cv::Mat toDisplay;

                if (displayMode == DisplayMode::ORIGINAL && framePtr) {
                    shouldDisplay = true;
                    toDisplay = framePtr->clone();
                }
                else if (displayMode == DisplayMode::DEPTH && depthPtr) {
                        shouldDisplay = true;
                        cv::applyColorMap(*depthPtr, toDisplay, cv::COLORMAP_TURBO); // TURBO gives better depth visualization
                    }
                    // else if (displayMode == DisplayMode::BOTH && framePtr) {
                    //     shouldDisplay = true;
                    //     toDisplay = framePtr->clone();
                        
                    //     // If we have a depth map, show it
                    //     if (depthPtr) {
                    //         // Apply color map to depth map for better visualization
                    //         cv::Mat depthColor;
                    //         cv::applyColorMap(*depthPtr, depthColor, cv::COLORMAP_TURBO);
    
                    //         // Resize depthColor to match frame size if necessary
                    //         if (depthColor.size() != framePtr->size()) {
                    //             cv::resize(depthColor, depthColor, framePtr->size());
                    //         }
    
                    //         // Create a horizontal layout instead of vertical for better viewing on most displays
                    //         cv::hconcat(*framePtr, depthColor, toDisplay);
                            
                    //         // Optional: Add a blended view as a third panel
                    //         // Uncomment if you want the three-panel view
                    //         // /*
                    //         // Add transparency overlay for depth information over original image
                    //         // cv::Mat blendedView;
                    //         // cv::addWeighted(*framePtr, 0.7, depthColor, 0.3, 0, blendedView);
                            
                    //         // // Create a three-panel view: original, depth, blended
                    //         // cv::Mat threePanel;
                    //         // cv::hconcat(toDisplay, blendedView, threePanel);
                    //         // toDisplay = threePanel;
                    //         // */
                    //     }
                    // }

                    else if (displayMode == DisplayMode::BOTH && framePtr && depthPtr) {
                        shouldDisplay = true;
                        toDisplay = *framePtr; // Avoid unnecessary clone
                    
                        // Apply color map to depth map for better visualization
                        cv::Mat depthColor;
                        cv::applyColorMap(*depthPtr, depthColor, cv::COLORMAP_TURBO);
                    
                        // Resize depthColor to match frame size if necessary (do this once)
                        if (depthColor.size() != framePtr->size()) {
                            cv::resize(depthColor, depthColor, framePtr->size());
                        }
                    
                        // Update the last valid depth pointer with the color-mapped depth
                        lastValidDepthPtr = std::make_shared<cv::Mat>(depthColor.clone());
                    
                        // Use the last valid depth map (if available) to prevent flashing
                        if (lastValidDepthPtr) {
                            // Calculate the midpoint of the frame width and copy the right half
                            int midpoint = framePtr->cols / 2;
                            lastValidDepthPtr->colRange(midpoint, lastValidDepthPtr->cols).copyTo(
                                toDisplay.colRange(midpoint, toDisplay.cols));
                        }
                    }
                    
                    
                    if (shouldDisplay) {
                        // Add FPS information to the display
                        std::string fps_text = "Capture: " + get_fps_string(fps_capture.load());
                        if (displayMode == DisplayMode::DEPTH || displayMode == DisplayMode::BOTH) {
                            fps_text += "  Processing: " + get_fps_string(fps_processing.load());
                        }
                        
                        cv::putText(toDisplay, fps_text, cv::Point(10, 30), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                        
                        // Add instructions
                        cv::putText(toDisplay, "Press 'q' to quit, 'm' to change mode, '+/-' to adjust skip", 
                                    cv::Point(10, toDisplay.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 
                                    0.5, cv::Scalar(255, 255, 255), 1);
                        
                        // Update display frame counter
                        frames_displayed++;
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
                        if (elapsed > 1000) {
                            fps_display.store((float)frames_displayed * 1000 / elapsed);
                            frames_displayed = 0;
                            start_time = current_time;
                        }
                        
                        // Display the resulting frame
                        cv::imshow("Real-time Depth Estimation", toDisplay);
                    }
    
                    // Handle key events with a short delay to allow GUI updates
                    char key = static_cast<char>(cv::waitKey(5)); // Slightly longer wait time for more CPU cycles for processing
                    if (key == 27 || key == 'q' || key == 'Q') { // ESC or 'q' to quit
                        running = false;
                        break;
                    } else if (key == 'm' || key == 'M') { // 'm' to toggle display mode
                        // Cycle through display modes
                        switch (displayMode) {
                            case DisplayMode::ORIGINAL:
                                displayMode = DisplayMode::DEPTH;
                                std::cout << "Display mode: Depth" << std::endl;
                                break;
                            case DisplayMode::DEPTH:
                                displayMode = DisplayMode::BOTH;
                                std::cout << "Display mode: Both" << std::endl;
                                break;
                            case DisplayMode::BOTH:
                                displayMode = DisplayMode::ORIGINAL;
                                std::cout << "Display mode: Original" << std::endl;
                                break;
                        }
                    } else if (key == '+' || key == '=') {
                        // Decrease frame skipping (process more frames)
                        if (skip_frames > 0) {
                            skip_frames--;
                            std::cout << "Decreased frame skipping to: " << skip_frames << std::endl;
                        }
                    } else if (key == '-' || key == '_') {
                        // Increase frame skipping (process fewer frames)
                        skip_frames++;
                        std::cout << "Increased frame skipping to: " << skip_frames << std::endl;
                    }
                    
                    // Brief sleep to prevent hogging CPU
                    if (!shouldDisplay) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                } });

        // Periodic stats reporting thread
        std::thread statsThread([&]()
                                {
                while (running) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    
                    std::cout << "Performance Stats:" << std::endl;
                    std::cout << "  Capture FPS: " << fps_capture.load() << std::endl;
                    std::cout << "  Processing FPS: " << fps_processing.load() << std::endl;
                    std::cout << "  Display FPS: " << fps_display.load() << std::endl;
                    std::cout << "  Frame Skip: " << skip_frames << std::endl;
                    std::cout << "  Adaptive Batch Size: " << adaptiveBatchSize << std::endl;
                    std::cout << "  Queue Sizes - Frame: " << frameQueue.size() 
                              << ", Depth: " << depthQueue.size() 
                              << ", Display: " << displayFrameQueue.size() << std::endl;
                } });

        // Wait for display thread to finish (which will happen when running becomes false)
        displayThread.join();

        // Signal other threads to finish
        running = false;
        frameQueue.set_finished();
        depthQueue.set_finished();
        displayFrameQueue.set_finished();

        // Join all threads
        if (captureThread.joinable())
        {
            captureThread.join();
        }
        if (processingThread.joinable())
        {
            processingThread.join();
        }
        if (statsThread.joinable())
        {
            statsThread.join();
        }

        // Clean up resources
        cap.release();
        cv::destroyAllWindows();

        std::cout << "Application terminated gracefully." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}