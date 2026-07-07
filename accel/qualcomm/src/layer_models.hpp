#pragma once

// Per-tier layer shapes for the Snapdragon (Hexagon HTP) bench.
// Kept in sync with accel/qualcomm/bench_manifest.json and mirrors
// accel/intel/src/layer_models.hpp so both vendors exercise identical shapes.

#include <string>
#include <vector>

namespace layer_models {

struct ShapeSpec {
    std::string size_label;
    std::string shape_note;

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

const std::vector<ShapeSpec>& size_profiles();

// Op kinds the QNN graph builder knows how to emit. Layer names in the manifest
// map onto exactly one of these (see kLayers in loom_accel_qualcomm.cpp).
enum class OpKind {
    MatMul,
    Conv1D,
    Conv2D,
    DepthwiseConv,
    AvgPool,
    MaxPool,
    ReLU,
    GELU,
    Sigmoid,
    Softmax,
    Add,
    Multiply,
    LayerNorm,
    RMSNorm,
};

// I/O + weight tensor dims for a single op instance, resolved from a ShapeSpec.
// QNN uses NHWC/NFC (channel-last) tensor layouts on HTP; conv weights are
// [out_c, kh, kw, in_c]. These fields are laid out channel-last accordingly.
struct GraphShapes {
    OpKind kind;

    std::vector<uint32_t> input_dims;   // activation input
    std::vector<uint32_t> output_dims;  // graph output
    std::vector<uint32_t> weight_dims;  // filter/matmul weights (empty if none)

    // Conv / pool params.
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t pad_h = 0;
    uint32_t pad_w = 0;
    uint32_t filter_h = 1;
    uint32_t filter_w = 1;

    bool has_weights = false;
    // Number of float weight elements the host is expected to upload.
    size_t weight_elems = 0;
};

// Resolve the concrete tensor shapes for a manifest layer name + size tier.
// Returns false if the layer name is unknown.
bool resolve_shapes(const std::string& layer_name, const ShapeSpec& s, GraphShapes* out);

}  // namespace layer_models
