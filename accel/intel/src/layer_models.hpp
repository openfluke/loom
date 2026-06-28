#pragma once

#include <openvino/openvino.hpp>

#include <memory>
#include <string>
#include <vector>

namespace layer_models {

// Per-tier shapes — keep in sync with bench_manifest.json.
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

std::vector<float> fill_ones(size_t n, float v = 0.01f);

std::shared_ptr<ov::Model> matmul_dense(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> conv1d_cnn1(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> conv2d_cnn2(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> depthwise_conv(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> avg_pool(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> max_pool(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> relu_act(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> gelu_act(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> sigmoid_act(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> softmax(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> add_residual(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> mul_gate(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> layer_norm(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> rms_norm(ov::element::Type dtype, const ShapeSpec& s);
std::shared_ptr<ov::Model> mha_qk_matmul(ov::element::Type dtype, const ShapeSpec& s);

}  // namespace layer_models
