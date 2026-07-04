cmake_minimum_required(VERSION 3.14)
project(depths_cpp LANGUAGES CXX)

# --- Options -----------------------------------------------------------------
option(ONNXRUNTIME_DIR "Path to the ONNX Runtime directory." "")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_DIR is not set. Pass -DONNXRUNTIME_DIR=... (build.sh does this automatically).")
endif()

# --- Standards / defaults ----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)          # <filesystem>, <optional> are used by the apps
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Dependencies ------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Resolve the ONNX Runtime shared library for the current platform.
if(WIN32)
    set(ORT_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
elseif(APPLE)
    set(ORT_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
else()
    set(ORT_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

# --- Targets -----------------------------------------------------------------
set(DEPTH_APPS
    image_depth_estimation
    video_depth_estimation
    camera_depth_estimation)

foreach(app ${DEPTH_APPS})
    add_executable(${app} src/${app}.cpp)
    target_include_directories(${app} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${ONNXRUNTIME_DIR}/include"
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${app} PRIVATE
        ${OpenCV_LIBS}
        ${ORT_LIB}
        Threads::Threads)
    # Let executables find libonnxruntime.so next to them at runtime.
    set_target_properties(${app} PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
endforeach()

message(STATUS "Depths-CPP configured (C++${CMAKE_CXX_STANDARD}, ${CMAKE_BUILD_TYPE}). "
               "TensorRT/CUDA are used automatically when the GPU ONNX Runtime build "
               "and their runtime libraries are present.")
