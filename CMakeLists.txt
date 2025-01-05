cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

include_directories("include/")

# Add executable for image depth estimation
add_executable(image_depth_estimation
               src/image_depth_estimation.cpp)

# Add executable for camera depth estimation
add_executable(camera_depth_estimation
src/camera_depth_estimation.cpp)

# Add executable for camera depth estimation
add_executable(video_depth_estimation
src/video_depth_estimation.cpp)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set include directories for all executables
target_include_directories(image_depth_estimation PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(camera_depth_estimation PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(video_depth_estimation PRIVATE "${ONNXRUNTIME_DIR}/include")

# Set compile features for all executables
target_compile_features(image_depth_estimation PRIVATE cxx_std_14)
target_compile_features(camera_depth_estimation PRIVATE cxx_std_14)
target_compile_features(video_depth_estimation PRIVATE cxx_std_14)

# Link libraries for all executables
target_link_libraries(image_depth_estimation ${OpenCV_LIBS})
target_link_libraries(camera_depth_estimation ${OpenCV_LIBS})
target_link_libraries(video_depth_estimation ${OpenCV_LIBS})


if(UNIX)
    message(STATUS "We are building on Linux!")
    # Specific Linux build commands or flags
    target_link_libraries(image_depth_estimation "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so") 
    target_link_libraries(camera_depth_estimation "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so") 
    target_link_libraries(video_depth_estimation "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so") 

endif(UNIX)

if(APPLE)
    message(STATUS "We are building on macOS!")
    # Specific macOS build commands or flags
    target_link_libraries(image_depth_estimation "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(camera_depth_estimation "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(video_depth_estimation "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif(APPLE)

if(WIN32)
    message(STATUS "We are building on Windows!")
    # Specific Windows build commands or flags
    target_link_libraries(image_depth_estimation "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(camera_depth_estimation "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(video_depth_estimation "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)
