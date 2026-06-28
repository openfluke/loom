cmake_minimum_required(VERSION 3.16)
project(loom_accel_intel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{INTEL_OPENVINO_DIR})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{INTEL_OPENVINO_DIR}/runtime/cmake")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

if(DEFINED ENV{INTEL_OPENVINO_DIR})
  set(OV_ROOT "$ENV{INTEL_OPENVINO_DIR}")
else()
  file(GLOB OV_CANDIDATES "${CMAKE_CURRENT_SOURCE_DIR}/deps/openvino_toolkit_*")
  list(SORT OV_CANDIDATES)
  list(GET OV_CANDIDATES -1 OV_ROOT)
endif()

set(LOOM_ACCEL_RPATH
    "${OV_ROOT}/runtime/lib/intel64"
    "${OV_ROOT}/runtime/3rdparty/tbb/lib"
)
if(DEFINED ENV{INTEL_NPU_LIBDIR})
  list(APPEND LOOM_ACCEL_RPATH "$ENV{INTEL_NPU_LIBDIR}")
else()
  set(_NPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/deps/npu-driver/root/usr/lib/x86_64-linux-gnu")
  if(EXISTS "${_NPU_LIB}")
    list(APPEND LOOM_ACCEL_RPATH "${_NPU_LIB}")
  endif()
endif()

add_library(loom_accel_intel SHARED
    src/loom_accel_intel.cpp
    src/layer_models.cpp
)

target_include_directories(loom_accel_intel
    PRIVATE
        src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(loom_accel_intel PRIVATE openvino::runtime)

set_target_properties(loom_accel_intel PROPERTIES
    OUTPUT_NAME "loom_accel_intel"
    VERSION 1.0.0
    SOVERSION 1
    BUILD_RPATH "${LOOM_ACCEL_RPATH}"
    INSTALL_RPATH "${LOOM_ACCEL_RPATH}"
)

install(TARGETS loom_accel_intel LIBRARY DESTINATION lib)
