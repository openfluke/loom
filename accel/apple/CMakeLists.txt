cmake_minimum_required(VERSION 3.16)
project(loom_accel_apple LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

add_library(loom_accel_apple SHARED
    src/loom_accel_apple.cpp
    src/cpu_reference.cpp
    src/mps_backend.mm
)

# Objective-C++ (.mm) uses ARC; the pure C++ TUs do not.
set_source_files_properties(src/mps_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(loom_accel_apple
    PRIVATE
        src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

target_link_libraries(loom_accel_apple PRIVATE
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Single shipped artifact: libloom_accel_apple.dylib.
# Loom dlopen() and LOOM_ACCEL_APPLE_DYLIB expect that exact filename.
set_target_properties(loom_accel_apple PROPERTIES
    OUTPUT_NAME "loom_accel_apple"
    PREFIX "lib"
    SUFFIX ".dylib"
)

install(TARGETS loom_accel_apple LIBRARY DESTINATION lib)
