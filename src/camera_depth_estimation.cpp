// live_depth_estimation_optimized.cpp

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

// Thread-safe queue with a maximum size using std::shared_ptr to avoid unnecessary copying
template <typename T>
class SafeQueue {
public:
    SafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    // Enqueue a new item into the queue. Returns false if the queue is full.
    bool enqueue(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            return false; // Queue is full, drop the frame
        }
        queue_.push(std::move(value));
        cond_var_.notify_one();
        return true;
    }

    // Dequeue an item from the queue. Returns std::nullopt if finished and queue is empty.
    std::optional<T> dequeue() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !finished_) {
            cond_var_.wait(lock);
        }
        if (queue_.empty()) return std::nullopt;
        T result = std::move(queue_.front());
        queue_.pop();
        return result;
    }

    // Signal that no more items will be enqueued
    void set_finished() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        cond_var_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    size_t max_size_;
    bool finished_ = false;
};

// Enumeration for display modes
enum class DisplayMode {
    ORIGINAL,
    DEPTH,
    BOTH
};

// Function to parse display mode from string
bool parse_display_mode(const std::string& mode_str, DisplayMode& mode) {
    if (mode_str == "original") {
        mode = DisplayMode::ORIGINAL;
        return true;
    }
    else if (mode_str == "depth") {
        mode = DisplayMode::DEPTH;
        return true;
    }
    else if (mode_str == "both") {
        mode = DisplayMode::BOTH;
        return true;
    }
    else {
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc < 3 || argc > 5) { // Adjusted for optional frame skip parameter
        std::cerr << "Usage: " << argv[0] << " <path_to_model.onnx> <camera_id> [display_mode] [skip_frames]" << std::endl;
        std::cerr << "display_mode options: original, depth, both" << std::endl;
        std::cerr << "skip_frames: Number of frames to skip (default: 0)" << std::endl;
        std::cerr << "Example: " << argv[0] << " depth_model.onnx 0 both 2" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    int cameraID = std::atoi(argv[2]); // Convert camera ID to integer

    // Default display mode is BOTH
    DisplayMode displayMode = DisplayMode::BOTH;

    if (argc >= 4) {
        std::string mode_str = argv[3];
        if (!parse_display_mode(mode_str, displayMode)) {
            std::cerr << "Invalid display mode: " << mode_str << std::endl;
            std::cerr << "Valid options are: original, depth, both" << std::endl;
            return -1;
        }
    }

    // Frame skipping parameter
    int skip_frames = 15; // Default: process every frame
    if (argc == 5) {
        skip_frames = std::atoi(argv[4]);
        if (skip_frames < 0) {
            std::cerr << "Invalid skip_frames value: " << argv[4] << ". It must be non-negative." << std::endl;
            return -1;
        }
    }

    try {
        // Initialize the DepthAnything object with the path to your ONNX model
        // Set useCuda to false for CPU-only
        bool useCuda = false; // CPU-only mode
        DepthAnything depthEstimator(modelPath, useCuda);

        // Open the specified camera device with an optimal backend
        cv::VideoCapture cap;

        // Attempt to open with V4L2 backend (commonly efficient on Linux)
        cap.open(cameraID, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            // Fallback to any available backend
            cap.open(cameraID, cv::CAP_ANY);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open the camera with ID " << cameraID << "." << std::endl;
                return -1;
            }
        }

        // Set camera properties to balance quality and performance
        // Lower resolution and frame rate can significantly improve performance
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);   // Adjust as needed (e.g., 640x480)
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);            // Set FPS if supported

        // Verify actual camera settings
        double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "Camera Settings:" << std::endl;
        std::cout << "Resolution: " << actual_width << "x" << actual_height << std::endl;
        std::cout << "FPS: " << actual_fps << std::endl;

        // Initialize thread-safe queues with optimized sizes
        SafeQueue<std::shared_ptr<cv::Mat>> frameQueue(120);         // For processing
        SafeQueue<std::shared_ptr<cv::Mat>> depthQueue(120);         // Depth maps
        SafeQueue<std::shared_ptr<cv::Mat>> displayFrameQueue(120);  // For display

        std::atomic<bool> running(true);

        // Capture thread: reads frames from the camera and enqueues them
        std::thread captureThread([&]() {
            int frame_counter = 0;
            while (running) {
                std::shared_ptr<cv::Mat> framePtr = std::make_shared<cv::Mat>();
                if (!cap.read(*framePtr)) {
                    std::cerr << "Warning: Failed to read frame from camera." << std::endl;
                    continue;
                }

                frame_counter++;

                // Apply frame skipping logic
                if (skip_frames > 0 && (frame_counter % (skip_frames + 1)) != 0) {
                    continue;
                }

                // Enqueue frame for display (always needed)
                if (!displayFrameQueue.enqueue(framePtr)) {
                    // Queue is full; drop the frame to maintain real-time performance
                    std::cerr << "Warning: Display frame queue is full. Dropping frame." << std::endl;
                }

                // Enqueue frame for processing only if depth estimation is needed
                if (displayMode == DisplayMode::DEPTH || displayMode == DisplayMode::BOTH) {
                    if (!frameQueue.enqueue(framePtr)) {
                        // Queue is full; drop the frame to maintain real-time performance
                        std::cerr << "Warning: Processing frame queue is full. Dropping frame." << std::endl;
                    }
                }
            }
            frameQueue.set_finished();
            depthQueue.set_finished(); // Ensure processing thread can finish if not used
            displayFrameQueue.set_finished();
        });

        // Processing thread: dequeues frames, performs depth estimation, and enqueues depth maps
        std::thread processingThread([&]() {
            if (displayMode != DisplayMode::DEPTH && displayMode != DisplayMode::BOTH) {
                // No need to process depth if not required
                return;
            }

            std::shared_ptr<cv::Mat> framePtr;
            while (running) {
                auto optFrame = frameQueue.dequeue();
                if (!optFrame.has_value()) break;
                framePtr = std::move(optFrame.value());

                try {
                    // Perform depth estimation
                    cv::Mat depthMap = depthEstimator.predict(*framePtr);

                    // Normalize depth map for visualization
                    cv::Mat depthVis;
                    cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);

                    // Enqueue depth map for display
                    std::shared_ptr<cv::Mat> depthPtr = std::make_shared<cv::Mat>(depthVis);
                    if (!depthQueue.enqueue(depthPtr)) {
                        // Queue is full; drop the depth map to maintain real-time performance
                        std::cerr << "Warning: Depth map queue is full. Dropping depth map." << std::endl;
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Exception during depth estimation: " << e.what() << std::endl;
                }
            }
            depthQueue.set_finished();
        });

        // Display thread: dequeues frames and depth maps and displays them according to the selected mode
        std::thread displayThread([&]() {
            while (running) {
                std::shared_ptr<cv::Mat> framePtr;
                std::shared_ptr<cv::Mat> depthPtr;

                // Attempt to dequeue a frame for display
                auto optFrame = displayFrameQueue.dequeue();
                if (optFrame.has_value()) {
                    framePtr = std::move(optFrame.value());
                }

                // Conditionally attempt to dequeue a depth map based on display mode
                if (displayMode == DisplayMode::DEPTH || displayMode == DisplayMode::BOTH) {
                    auto optDepth = depthQueue.dequeue();
                    if (optDepth.has_value()) {
                        depthPtr = std::move(optDepth.value());
                    }
                }

                // Determine if there's something to display based on the selected mode
                bool shouldDisplay = false;
                cv::Mat toDisplay;

                if (displayMode == DisplayMode::ORIGINAL && framePtr) {
                    shouldDisplay = true;
                    toDisplay = *framePtr;
                }
                else if (displayMode == DisplayMode::DEPTH && depthPtr) {
                    shouldDisplay = true;
                    toDisplay = *depthPtr;
                }
                else if (displayMode == DisplayMode::BOTH && framePtr && depthPtr) {
                    shouldDisplay = true;
                    // Apply color map to depth map for better visualization
                    cv::Mat depthColor;
                    cv::applyColorMap(*depthPtr, depthColor, cv::COLORMAP_JET);

                    // Ensure both images have the same number of channels
                    if (depthColor.channels() == 1 && framePtr->channels() == 3) {
                        cv::cvtColor(depthColor, depthColor, cv::COLOR_GRAY2BGR);
                    }

                    // Resize depthColor to match frame size if necessary
                    if (depthColor.size() != framePtr->size()) {
                        cv::resize(depthColor, depthColor, framePtr->size());
                    }

                    // Stack original frame and depth map vertically
                    cv::vconcat(*framePtr, depthColor, toDisplay);
                }

                if (shouldDisplay) {
                    // Display the resulting frame
                    cv::imshow("Live Depth Estimation", toDisplay);
                }

                // Handle key events with a short delay to allow GUI updates
                char key = static_cast<char>(cv::waitKey(1));
                if (key == 27 || key == 'q' || key == 'Q') { // ESC or 'q' to quit
                    running = false;
                    break;
                }
            }
        });

        // Wait for display thread to finish (which will happen when running becomes false)
        displayThread.join();

        // Signal other threads to finish
        running = false;
        frameQueue.set_finished();
        depthQueue.set_finished();
        displayFrameQueue.set_finished();

        if (captureThread.joinable()) {
            captureThread.join();
        }
        if (processingThread.joinable()) {
            processingThread.join();
        }

        cap.release();
        cv::destroyAllWindows();

        std::cout << "Application terminated gracefully." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
