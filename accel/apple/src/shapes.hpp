#pragma once
//
// shapes.hpp — per-tier shapes + layer/dtype tables for the Apple accel plugin.
//
// This mirrors accel/intel/src/layer_models.{hpp,cpp} and bench_manifest.json so the
// Apple bridge is a faithful mirror of the Intel (OpenVINO) and Qualcomm (QNN) plugins:
// same layer names, same small/medium/large tiers, same weight layouts. Header-only.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace loom_apple {

// Per-tier shapes — keep in sync with bench_manifest.json / accel/intel.
struct ShapeSpec {
    std::string size_label;

    int dense_batch = 4;
    int dim = 32;

    int c1_batch = 4;
    int c1_in_c = 2;
    int c1_len = 16;
    int c1_filters = 16;
    int c1_kernel = 3;
    int c1_pad = 1;

    int c2_batch = 4;
    int c2_in_c = 2;
    int c2_h = 4;
    int c2_w = 4;
    int c2_filters = 2;
    int c2_kernel = 3;
    int c2_pad = 1;

    int sp_batch = 4;
    int sp_c = 16;
    int sp_h = 8;
    int sp_w = 8;
    int sp_kernel = 3;
    int sp_pad = 1;
    int sp_pool_ks = 2;
    int sp_pool_stride = 2;
};

inline std::vector<ShapeSpec> size_profiles() {
    ShapeSpec small;
    small.size_label = "small";
    small.dense_batch = 4;
    small.dim = 32;
    small.c1_batch = 4;
    small.c1_in_c = 2;
    small.c1_len = 16;
    small.c1_filters = 16;
    small.c2_batch = 4;
    small.c2_in_c = 2;
    small.c2_h = 4;
    small.c2_w = 4;
    small.c2_filters = 2;
    small.sp_batch = 4;
    small.sp_c = 16;
    small.sp_h = 8;
    small.sp_w = 8;

    ShapeSpec medium;
    medium.size_label = "medium";
    medium.dense_batch = 16;
    medium.dim = 256;
    medium.c1_batch = 8;
    medium.c1_in_c = 32;
    medium.c1_len = 128;
    medium.c1_filters = 32;
    medium.c2_batch = 4;
    medium.c2_in_c = 32;
    medium.c2_h = 28;
    medium.c2_w = 28;
    medium.c2_filters = 32;
    medium.sp_batch = 4;
    medium.sp_c = 64;
    medium.sp_h = 32;
    medium.sp_w = 32;

    ShapeSpec large;
    large.size_label = "large";
    large.dense_batch = 8;
    large.dim = 1024;
    large.c1_batch = 4;
    large.c1_in_c = 128;
    large.c1_len = 512;
    large.c1_filters = 128;
    large.c2_batch = 4;
    large.c2_in_c = 64;
    large.c2_h = 48;
    large.c2_w = 48;
    large.c2_filters = 64;
    large.sp_batch = 4;
    large.sp_c = 128;
    large.sp_h = 48;
    large.sp_w = 48;

    return {small, medium, large};
}

inline bool find_shape(const char* size_label, ShapeSpec* out) {
    if (size_label == nullptr || out == nullptr) {
        return false;
    }
    for (const auto& s : size_profiles()) {
        if (s.size_label == size_label) {
            *out = s;
            return true;
        }
    }
    return false;
}

// The 15 bench layers (matches accel/intel bench_manifest.json).
inline bool known_layer(const std::string& name) {
    static const char* kLayers[] = {
        "MatMul", "Conv1D", "Conv2D", "DepthwiseConv", "AvgPool", "MaxPool",
        "ReLU", "GELU", "Sigmoid", "Softmax", "Add", "Multiply",
        "LayerNorm", "RMSNorm", "MHA-MatMul",
    };
    for (const char* l : kLayers) {
        if (name == l) {
            return true;
        }
    }
    return false;
}

inline bool known_dtype(const std::string& d) {
    return d == "FP32" || d == "FP16" || d == "BF16" ||
           d == "INT8" || d == "INT16" || d == "INT4";
}

// FP16/BF16 I/O is 2 bytes/element; everything else hands over FP32 (4 bytes) —
// mirrors poly/accel_intel.go tensorToAccelBytes (INT8/INT16/INT4 arrive as FP32).
inline size_t io_elem_size(const std::string& dtype_label) {
    return (dtype_label == "FP16" || dtype_label == "BF16") ? 2u : 4u;
}

inline int pool_out(int in, int ks, int stride) {
    return (in - ks) / stride + 1;
}

// Element counts for the input activation of a layer.
inline size_t input_elems(const std::string& name, const ShapeSpec& s) {
    if (name == "Conv1D") {
        return size_t(s.c1_batch) * s.c1_in_c * s.c1_len;
    }
    if (name == "Conv2D") {
        return size_t(s.c2_batch) * s.c2_in_c * s.c2_h * s.c2_w;
    }
    if (name == "DepthwiseConv" || name == "AvgPool" || name == "MaxPool") {
        return size_t(s.sp_batch) * s.sp_c * s.sp_h * s.sp_w;
    }
    // MatMul, MHA-MatMul, ReLU, GELU, Sigmoid, Softmax, Add, Multiply, LayerNorm, RMSNorm
    return size_t(s.dense_batch) * s.dim;
}

// Element counts for the output activation of a layer (flattened, matches intel).
inline size_t output_elems(const std::string& name, const ShapeSpec& s) {
    if (name == "Conv1D") {
        return size_t(s.c1_batch) * s.c1_filters * s.c1_len;  // pad=1,k=3,stride=1 → same len
    }
    if (name == "Conv2D") {
        return size_t(s.c2_batch) * s.c2_filters * s.c2_h * s.c2_w;  // same h,w
    }
    if (name == "DepthwiseConv") {
        return size_t(s.sp_batch) * s.sp_c * s.sp_h * s.sp_w;
    }
    if (name == "AvgPool" || name == "MaxPool") {
        const int oh = pool_out(s.sp_h, s.sp_pool_ks, s.sp_pool_stride);
        const int ow = pool_out(s.sp_w, s.sp_pool_ks, s.sp_pool_stride);
        return size_t(s.sp_batch) * s.sp_c * oh * ow;
    }
    return size_t(s.dense_batch) * s.dim;
}

// Number of float weights Loom uploads for a layer (0 = plugin uses fixed constants,
// matching the OpenVINO layer_models default fills). Mirrors intel weight_float_count.
inline size_t weight_float_count(const std::string& name, const ShapeSpec& s) {
    if (name == "MatMul" || name == "MHA-MatMul") {
        return size_t(s.dim) * s.dim;
    }
    if (name == "Conv1D") {
        return size_t(s.c1_filters) * s.c1_in_c * s.c1_kernel;
    }
    if (name == "Conv2D") {
        return size_t(s.c2_filters) * s.c2_in_c * s.c2_kernel * s.c2_kernel;
    }
    return 0;
}

}  // namespace loom_apple
