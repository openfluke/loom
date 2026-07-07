#include "layer_models.hpp"

#include <cstdint>

namespace layer_models {
namespace {

ShapeSpec make_spec(
    const char* label,
    const char* note,
    int dense_batch,
    int dim,
    int c1_batch,
    int c1_in_c,
    int c1_len,
    int c1_filters,
    int c2_batch,
    int c2_in_c,
    int c2_h,
    int c2_w,
    int c2_filters,
    int sp_batch,
    int sp_c,
    int sp_h,
    int sp_w) {
    ShapeSpec s;
    s.size_label = label;
    s.shape_note = note;
    s.dense_batch = dense_batch;
    s.dim = dim;
    s.c1_batch = c1_batch;
    s.c1_in_c = c1_in_c;
    s.c1_len = c1_len;
    s.c1_filters = c1_filters;
    s.c1_kernel = 3;
    s.c1_pad = 1;
    s.c2_batch = c2_batch;
    s.c2_in_c = c2_in_c;
    s.c2_h = c2_h;
    s.c2_w = c2_w;
    s.c2_filters = c2_filters;
    s.c2_kernel = 3;
    s.c2_pad = 1;
    s.sp_batch = sp_batch;
    s.sp_c = sp_c;
    s.sp_h = sp_h;
    s.sp_w = sp_w;
    s.sp_kernel = 3;
    s.sp_pad = 1;
    s.sp_pool_ks = 2;
    s.sp_pool_stride = 2;
    return s;
}

uint32_t u(int v) { return static_cast<uint32_t>(v); }

}  // namespace

const std::vector<ShapeSpec>& size_profiles() {
    static const std::vector<ShapeSpec> kProfiles = {
        make_spec("small", "Smoke / latency floor (batch=4, dim=32)",
                  4, 32, 4, 2, 16, 16, 4, 2, 4, 4, 2, 4, 16, 8, 8),
        make_spec("medium", "Mid-size (comfortably inside Hexagon v73 HTP tile budget)",
                  16, 256, 8, 32, 128, 32, 4, 32, 28, 28, 32, 4, 64, 32, 32),
        make_spec("large", "Near Hexagon v73 HTP working-set budget (FP16 activations + weights)",
                  8, 1024, 4, 128, 512, 128, 4, 64, 48, 48, 64, 4, 128, 48, 48),
    };
    return kProfiles;
}

bool resolve_shapes(const std::string& name, const ShapeSpec& s, GraphShapes* out) {
    if (out == nullptr) {
        return false;
    }
    GraphShapes g;

    // Dense-family activation shape [batch, dim] (channel-last: dim is the feature axis).
    const std::vector<uint32_t> dense2d = {u(s.dense_batch), u(s.dim)};

    if (name == "MatMul" || name == "MHA-MatMul") {
        g.kind = OpKind::MatMul;
        g.input_dims = dense2d;
        g.output_dims = dense2d;
        g.weight_dims = {u(s.dim), u(s.dim)};
        g.has_weights = true;
        g.weight_elems = size_t(s.dim) * size_t(s.dim);
    } else if (name == "Conv1D") {
        // QNN Conv on HTP is 2D; a Conv1D of length L is modelled as [N, 1, L, C].
        g.kind = OpKind::Conv1D;
        g.input_dims = {u(s.c1_batch), 1u, u(s.c1_len), u(s.c1_in_c)};
        g.output_dims = {u(s.c1_batch), 1u, u(s.c1_len), u(s.c1_filters)};
        g.weight_dims = {u(s.c1_filters), 1u, u(s.c1_kernel), u(s.c1_in_c)};
        g.stride_h = 1; g.stride_w = 1;
        g.pad_h = 0; g.pad_w = u(s.c1_pad);
        g.filter_h = 1; g.filter_w = u(s.c1_kernel);
        g.has_weights = true;
        g.weight_elems = size_t(s.c1_filters) * size_t(s.c1_in_c) * size_t(s.c1_kernel);
    } else if (name == "Conv2D") {
        g.kind = OpKind::Conv2D;
        g.input_dims = {u(s.c2_batch), u(s.c2_h), u(s.c2_w), u(s.c2_in_c)};
        g.output_dims = {u(s.c2_batch), u(s.c2_h), u(s.c2_w), u(s.c2_filters)};
        g.weight_dims = {u(s.c2_filters), u(s.c2_kernel), u(s.c2_kernel), u(s.c2_in_c)};
        g.stride_h = 1; g.stride_w = 1;
        g.pad_h = u(s.c2_pad); g.pad_w = u(s.c2_pad);
        g.filter_h = u(s.c2_kernel); g.filter_w = u(s.c2_kernel);
        g.has_weights = true;
        g.weight_elems = size_t(s.c2_filters) * size_t(s.c2_in_c) *
                         size_t(s.c2_kernel) * size_t(s.c2_kernel);
    } else if (name == "DepthwiseConv") {
        g.kind = OpKind::DepthwiseConv;
        g.input_dims = {u(s.sp_batch), u(s.sp_h), u(s.sp_w), u(s.sp_c)};
        g.output_dims = {u(s.sp_batch), u(s.sp_h), u(s.sp_w), u(s.sp_c)};
        g.weight_dims = {1u, u(s.sp_kernel), u(s.sp_kernel), u(s.sp_c)};
        g.stride_h = 1; g.stride_w = 1;
        g.pad_h = u(s.sp_pad); g.pad_w = u(s.sp_pad);
        g.filter_h = u(s.sp_kernel); g.filter_w = u(s.sp_kernel);
        g.has_weights = false;  // baked (vendor default weights)
    } else if (name == "AvgPool" || name == "MaxPool") {
        const uint32_t oh = u((s.sp_h - s.sp_pool_ks) / s.sp_pool_stride + 1);
        const uint32_t ow = u((s.sp_w - s.sp_pool_ks) / s.sp_pool_stride + 1);
        g.kind = (name == "AvgPool") ? OpKind::AvgPool : OpKind::MaxPool;
        g.input_dims = {u(s.sp_batch), u(s.sp_h), u(s.sp_w), u(s.sp_c)};
        g.output_dims = {u(s.sp_batch), oh, ow, u(s.sp_c)};
        g.stride_h = u(s.sp_pool_stride); g.stride_w = u(s.sp_pool_stride);
        g.pad_h = 0; g.pad_w = 0;
        g.filter_h = u(s.sp_pool_ks); g.filter_w = u(s.sp_pool_ks);
    } else if (name == "ReLU") {
        g.kind = OpKind::ReLU; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "GELU") {
        g.kind = OpKind::GELU; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "Sigmoid") {
        g.kind = OpKind::Sigmoid; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "Softmax") {
        g.kind = OpKind::Softmax; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "Add") {
        g.kind = OpKind::Add; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "Multiply") {
        g.kind = OpKind::Multiply; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "LayerNorm") {
        g.kind = OpKind::LayerNorm; g.input_dims = dense2d; g.output_dims = dense2d;
    } else if (name == "RMSNorm") {
        g.kind = OpKind::RMSNorm; g.input_dims = dense2d; g.output_dims = dense2d;
    } else {
        return false;
    }

    *out = g;
    return true;
}

}  // namespace layer_models
