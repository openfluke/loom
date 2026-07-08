cmake_minimum_required(VERSION 3.16)
project(loom_accel_qualcomm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN_SDK_ROOT points at the QAIRT / QNN AI Engine Direct SDK install
# (e.g. C:/Qualcomm/AIStack/QAIRT/<version>). Provides QNN headers under include/QNN.
if(NOT DEFINED ENV{QNN_SDK_ROOT})
  message(FATAL_ERROR
    "QNN_SDK_ROOT not set. Install the QAIRT SDK and run setup_env.ps1 "
    "(see accel/qualcomm/README.md).")
endif()
set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")

# QNN backends are loaded at runtime (dlopen/LoadLibrary); we only need headers to build.
find_path(QNN_INCLUDE_DIR
  NAMES QnnInterface.h
  PATHS "${QNN_SDK_ROOT}/include/QNN" "${QNN_SDK_ROOT}/include"
  NO_DEFAULT_PATH)
if(NOT QNN_INCLUDE_DIR)
  message(FATAL_ERROR "QnnInterface.h not found under ${QNN_SDK_ROOT}/include")
endif()

add_library(loom_accel_qualcomm SHARED
    src/loom_accel_qualcomm.cpp
    src/qnn_wrapper.cpp
    src/layer_models.cpp
)

target_include_directories(loom_accel_qualcomm
    PRIVATE
        src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${QNN_INCLUDE_DIR}
)

# Backends resolved via LoadLibrary/dlopen at runtime — no import lib to link.
if(WIN32)
    target_link_libraries(loom_accel_qualcomm PRIVATE kernel32)
else()
    target_link_libraries(loom_accel_qualcomm PRIVATE dl)
endif()

# Single shipped artifact matching the name Loom's loader dlopen()s.
set_target_properties(loom_accel_qualcomm PROPERTIES
    OUTPUT_NAME "loom_accel_qualcomm"
    PREFIX ""            # Windows: loom_accel_qualcomm.dll (no "lib")
)

install(TARGETS loom_accel_qualcomm
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib)
