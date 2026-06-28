#include "layer_models.hpp"

#include <openvino/opsets/opset13.hpp>

#include <cmath>
#include <stdexcept>

namespace layer_models {
namespace {

using namespace ov;
using namespace ov::op;

std::shared_ptr<v0::Constant> const_f(ov::element::Type dtype, const Shape& shape, float v) {
    if (dtype == element::f32) {
        std::vector<float> data(shape_size(shape), v);
        return v0::Constant::create(element::f32, shape, data);
    }
    if (dtype == element::f16) {
        std::vector<ov::float16> data(shape_size(shape), ov::float16(v));
        return v0::Constant::create(element::f16, shape, data);
    }
    throw std::runtime_error("const_f: unsupported dtype");
}

std::shared_ptr<Model> wrap(const Output<Node>& out, const ParameterVector& params) {
    return std::make_shared<Model>(OutputVector{out}, params);
}

std::shared_ptr<v0::Parameter> param_2d(ov::element::Type dtype, int batch, int dim) {
    return std::make_shared<v0::Parameter>(dtype, Shape{size_t(batch), size_t(dim)});
}

std::shared_ptr<v0::Parameter> param_nchw(ov::element::Type dtype, int batch, int c, int h, int w) {
    return std::make_shared<v0::Parameter>(dtype, Shape{size_t(batch), size_t(c), size_t(h), size_t(w)});
}

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

}  // namespace

const std::vector<ShapeSpec>& size_profiles() {
    static const std::vector<ShapeSpec> kProfiles = {
        make_spec(
            "small",
            "Smoke / latency floor (batch=4, dim=32)",
            4,
            32,
            4,
            2,
            16,
            16,
            4,
            2,
            4,
            4,
            2,
            4,
            16,
            8,
            8),
        make_spec(
            "medium",
            "Mid-size (fits easily in 4 MB NPU SRAM)",
            16,
            256,
            8,
            32,
            128,
            32,
            4,
            32,
            28,
            28,
            32,
            4,
            64,
            32,
            32),
        make_spec(
            "large",
            "Near NPU 3720 4 MB SRAM budget (FP16 activations + weights)",
            8,
            1024,
            4,
            128,
            512,
            128,
            4,
            64,
            48,
            48,
            64,
            4,
            128,
            48,
            48),
    };
    return kProfiles;
}

std::vector<float> fill_ones(size_t n, float v) {
    return std::vector<float>(n, v);
}

std::shared_ptr<Model> matmul_dense(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    auto w = const_f(dtype, Shape{size_t(s.dim), size_t(s.dim)}, 0.02f);
    auto mm = std::make_shared<v0::MatMul>(input, w, false, false);
    return wrap(mm, {input});
}

std::shared_ptr<Model> conv1d_cnn1(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = std::make_shared<v0::Parameter>(
        dtype, Shape{size_t(s.c1_batch), size_t(s.c1_in_c), size_t(s.c1_len)});
    auto w = const_f(dtype, Shape{size_t(s.c1_filters), size_t(s.c1_in_c), size_t(s.c1_kernel)}, 0.02f);
    auto conv = std::make_shared<v1::Convolution>(
        input,
        w,
        Strides{1},
        CoordinateDiff{s.c1_pad},
        CoordinateDiff{s.c1_pad},
        Strides{1});
    auto flat = std::make_shared<v1::Reshape>(
        conv,
        v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{s.c1_batch, -1}),
        false);
    return wrap(flat, {input});
}

std::shared_ptr<Model> conv2d_cnn2(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_nchw(dtype, s.c2_batch, s.c2_in_c, s.c2_h, s.c2_w);
    auto w = const_f(
        dtype,
        Shape{size_t(s.c2_filters), size_t(s.c2_in_c), size_t(s.c2_kernel), size_t(s.c2_kernel)},
        0.02f);
    auto conv = std::make_shared<v1::Convolution>(
        input,
        w,
        Strides{1, 1},
        CoordinateDiff{s.c2_pad, s.c2_pad},
        CoordinateDiff{s.c2_pad, s.c2_pad},
        Strides{1, 1});
    auto flat = std::make_shared<v1::Reshape>(
        conv,
        v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{s.c2_batch, -1}),
        false);
    return wrap(flat, {input});
}

std::shared_ptr<Model> depthwise_conv(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_nchw(dtype, s.sp_batch, s.sp_c, s.sp_h, s.sp_w);
    auto w = const_f(
        dtype,
        Shape{size_t(s.sp_c), 1, 1, size_t(s.sp_kernel), size_t(s.sp_kernel)},
        0.02f);
    auto conv = std::make_shared<v1::GroupConvolution>(
        input,
        w,
        Strides{1, 1},
        CoordinateDiff{s.sp_pad, s.sp_pad},
        CoordinateDiff{s.sp_pad, s.sp_pad},
        Strides{1, 1});
    return wrap(conv, {input});
}

std::shared_ptr<Model> avg_pool(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_nchw(dtype, s.sp_batch, s.sp_c, s.sp_h, s.sp_w);
    auto pool = std::make_shared<v1::AvgPool>(
        input,
        Strides{size_t(s.sp_pool_stride), size_t(s.sp_pool_stride)},
        Shape{0, 0},
        Shape{0, 0},
        Shape{size_t(s.sp_pool_ks), size_t(s.sp_pool_ks)},
        false,
        op::RoundingType::FLOOR);
    return wrap(pool, {input});
}

std::shared_ptr<Model> max_pool(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_nchw(dtype, s.sp_batch, s.sp_c, s.sp_h, s.sp_w);
    auto pool = std::make_shared<v1::MaxPool>(
        input,
        Strides{size_t(s.sp_pool_stride), size_t(s.sp_pool_stride)},
        Shape{0, 0},
        Shape{0, 0},
        Shape{size_t(s.sp_pool_ks), size_t(s.sp_pool_ks)},
        op::RoundingType::FLOOR);
    return wrap(pool, {input});
}

std::shared_ptr<Model> relu_act(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    return wrap(std::make_shared<v0::Relu>(input), {input});
}

std::shared_ptr<Model> gelu_act(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    return wrap(std::make_shared<v7::Gelu>(input), {input});
}

std::shared_ptr<Model> sigmoid_act(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    return wrap(std::make_shared<v0::Sigmoid>(input), {input});
}

std::shared_ptr<Model> softmax(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    return wrap(std::make_shared<v8::Softmax>(input, int64_t{1}), {input});
}

std::shared_ptr<Model> add_residual(ov::element::Type dtype, const ShapeSpec& s) {
    auto a = param_2d(dtype, s.dense_batch, s.dim);
    auto b = const_f(dtype, Shape{size_t(s.dense_batch), size_t(s.dim)}, 0.01f);
    return wrap(std::make_shared<v1::Add>(a, b), {a});
}

std::shared_ptr<Model> mul_gate(ov::element::Type dtype, const ShapeSpec& s) {
    auto a = param_2d(dtype, s.dense_batch, s.dim);
    auto b = const_f(dtype, Shape{size_t(s.dense_batch), size_t(s.dim)}, 0.5f);
    return wrap(std::make_shared<v1::Multiply>(a, b), {a});
}

std::shared_ptr<Model> layer_norm(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    auto axis = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto mvn = std::make_shared<v6::MVN>(input, axis, true, 1e-5f, MVNEpsMode::INSIDE_SQRT);
    auto gamma = const_f(dtype, Shape{size_t(s.dim)}, 1.0f);
    auto beta = const_f(dtype, Shape{size_t(s.dim)}, 0.0f);
    auto scaled = std::make_shared<v1::Multiply>(mvn, gamma);
    auto out = std::make_shared<v1::Add>(scaled, beta);
    return wrap(out, {input});
}

std::shared_ptr<Model> rms_norm(ov::element::Type dtype, const ShapeSpec& s) {
    auto input = param_2d(dtype, s.dense_batch, s.dim);
    auto axis = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto sq = std::make_shared<v1::Multiply>(input, input);
    auto mean = std::make_shared<v1::ReduceMean>(sq, axis, true);
    auto eps = const_f(dtype, Shape{1, 1}, 1e-5f);
    auto denom = std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, eps));
    auto norm = std::make_shared<v1::Divide>(input, denom);
    auto scale = const_f(dtype, Shape{size_t(s.dim)}, 1.0f);
    auto out = std::make_shared<v1::Multiply>(norm, scale);
    return wrap(out, {input});
}

std::shared_ptr<Model> mha_qk_matmul(ov::element::Type dtype, const ShapeSpec& s) {
    auto q = param_2d(dtype, s.dense_batch, s.dim);
    auto k = const_f(dtype, Shape{size_t(s.dim), size_t(s.dim)}, 0.01f);
    auto scores = std::make_shared<v0::MatMul>(q, k, false, false);
    return wrap(scores, {q});
}

}  // namespace layer_models
