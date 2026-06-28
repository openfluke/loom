#include "loom_accel.h"

#include "layer_models.hpp"

#include <openvino/opsets/opset13.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct LayerCase {
    const char* name;
    std::shared_ptr<ov::Model> (*build)(ov::element::Type, const layer_models::ShapeSpec&);
};

const std::vector<LayerCase> kLayers = {
    {"MatMul", layer_models::matmul_dense},
    {"Conv1D", layer_models::conv1d_cnn1},
    {"Conv2D", layer_models::conv2d_cnn2},
    {"DepthwiseConv", layer_models::depthwise_conv},
    {"AvgPool", layer_models::avg_pool},
    {"MaxPool", layer_models::max_pool},
    {"ReLU", layer_models::relu_act},
    {"GELU", layer_models::gelu_act},
    {"Sigmoid", layer_models::sigmoid_act},
    {"Softmax", layer_models::softmax},
    {"Add", layer_models::add_residual},
    {"Multiply", layer_models::mul_gate},
    {"LayerNorm", layer_models::layer_norm},
    {"RMSNorm", layer_models::rms_norm},
    {"MHA-MatMul", layer_models::mha_qk_matmul},
};

struct DTypeCase {
    ov::element::Type elem;
    const char* label;
    bool dynamic_quant_on_npu;
};

const std::vector<DTypeCase> kDTypes = {
    {ov::element::f32, "FP32", false},
    {ov::element::f16, "FP16", false},
    {ov::element::f32, "INT8", true},
};

bool copy_err(char* err, size_t err_len, const std::string& msg) {
    if (err == nullptr || err_len == 0) {
        return false;
    }
    std::strncpy(err, msg.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
    return true;
}

const layer_models::ShapeSpec* find_shape(const char* size_label) {
    if (size_label == nullptr) {
        return nullptr;
    }
    for (const auto& s : layer_models::size_profiles()) {
        if (s.size_label == size_label) {
            return &s;
        }
    }
    return nullptr;
}

const LayerCase* find_layer(const char* name) {
    if (name == nullptr) {
        return nullptr;
    }
    for (const auto& layer : kLayers) {
        if (std::string(layer.name) == name) {
            return &layer;
        }
    }
    return nullptr;
}

const DTypeCase* find_dtype(const char* label) {
    if (label == nullptr) {
        return nullptr;
    }
    for (const auto& dt : kDTypes) {
        if (std::string(dt.label) == label) {
            return &dt;
        }
    }
    return nullptr;
}

bool device_available(ov::Core& core, const std::string& device) {
    const auto devices = core.get_available_devices();
    return std::find(devices.begin(), devices.end(), device) != devices.end();
}

ov::AnyMap compile_cfg_for(const DTypeCase& dt, const std::string& device) {
    ov::AnyMap cfg;
    if (dt.elem == ov::element::f16) {
        cfg[ov::hint::inference_precision.name()] = ov::element::f16;
    }
    if (device == "NPU" && dt.dynamic_quant_on_npu) {
        cfg[ov::intel_npu::compiler_dynamic_quantization.name()] = true;
    }
    return cfg;
}

std::shared_ptr<ov::op::v0::Constant> constant_from_floats(
    ov::element::Type dtype,
    const ov::Shape& shape,
    const float* src,
    size_t count) {
    const size_t need = ov::shape_size(shape);
    if (src == nullptr || count < need) {
        return nullptr;
    }
    if (dtype == ov::element::f32) {
        std::vector<float> data(src, src + need);
        return ov::op::v0::Constant::create(dtype, shape, data);
    }
    if (dtype == ov::element::f16) {
        std::vector<ov::float16> data;
        data.reserve(need);
        for (size_t i = 0; i < need; ++i) {
            data.push_back(ov::float16(src[i]));
        }
        return ov::op::v0::Constant::create(dtype, shape, data);
    }
    return nullptr;
}

std::shared_ptr<ov::Model> rebuild_with_weights(
    const std::shared_ptr<ov::Model>& model,
    const char* layer_name,
    ov::element::Type dtype,
    const layer_models::ShapeSpec& shapes,
    const float* weights,
    size_t weight_count) {
    using namespace ov::op;

    if (weights == nullptr || weight_count == 0) {
        return model;
    }

    if (std::string(layer_name) == "MatMul" || std::string(layer_name) == "MHA-MatMul") {
        const ov::Shape wshape{size_t(shapes.dim), size_t(shapes.dim)};
        auto w = constant_from_floats(dtype, wshape, weights, weight_count);
        if (!w) {
            return model;
        }
        auto input = std::make_shared<v0::Parameter>(dtype, ov::Shape{size_t(shapes.dense_batch), size_t(shapes.dim)});
        auto mm = std::make_shared<v0::MatMul>(input, w, false, false);
        return std::make_shared<ov::Model>(ov::OutputVector{mm}, ov::ParameterVector{input});
    }

    if (std::string(layer_name) == "Conv1D") {
        const ov::Shape wshape{
            size_t(shapes.c1_filters),
            size_t(shapes.c1_in_c),
            size_t(shapes.c1_kernel)};
        auto w = constant_from_floats(dtype, wshape, weights, weight_count);
        if (!w) {
            return model;
        }
        auto input = std::make_shared<v0::Parameter>(
            dtype,
            ov::Shape{size_t(shapes.c1_batch), size_t(shapes.c1_in_c), size_t(shapes.c1_len)});
        auto conv = std::make_shared<v1::Convolution>(
            input,
            w,
            ov::Strides{1},
            ov::CoordinateDiff{shapes.c1_pad},
            ov::CoordinateDiff{shapes.c1_pad},
            ov::Strides{1});
        auto flat = std::make_shared<v1::Reshape>(
            conv,
            v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{shapes.c1_batch, -1}),
            false);
        return std::make_shared<ov::Model>(ov::OutputVector{flat}, ov::ParameterVector{input});
    }

    if (std::string(layer_name) == "Conv2D") {
        const ov::Shape wshape{
            size_t(shapes.c2_filters),
            size_t(shapes.c2_in_c),
            size_t(shapes.c2_kernel),
            size_t(shapes.c2_kernel)};
        auto w = constant_from_floats(dtype, wshape, weights, weight_count);
        if (!w) {
            return model;
        }
        auto input = std::make_shared<v0::Parameter>(
            dtype,
            ov::Shape{size_t(shapes.c2_batch), size_t(shapes.c2_in_c), size_t(shapes.c2_h), size_t(shapes.c2_w)});
        auto conv = std::make_shared<v1::Convolution>(
            input,
            w,
            ov::Strides{1, 1},
            ov::CoordinateDiff{shapes.c2_pad, shapes.c2_pad},
            ov::CoordinateDiff{shapes.c2_pad, shapes.c2_pad},
            ov::Strides{1, 1});
        auto flat = std::make_shared<v1::Reshape>(
            conv,
            v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{shapes.c2_batch, -1}),
            false);
        return std::make_shared<ov::Model>(ov::OutputVector{flat}, ov::ParameterVector{input});
    }

    return model;
}

size_t weight_float_count(const char* layer_name, const layer_models::ShapeSpec& shapes) {
    if (layer_name == nullptr) {
        return 0;
    }
    const std::string name(layer_name);
    if (name == "MatMul" || name == "MHA-MatMul") {
        return size_t(shapes.dim) * size_t(shapes.dim);
    }
    if (name == "Conv1D") {
        return size_t(shapes.c1_filters) * size_t(shapes.c1_in_c) * size_t(shapes.c1_kernel);
    }
    if (name == "Conv2D") {
        return size_t(shapes.c2_filters) * size_t(shapes.c2_in_c) * size_t(shapes.c2_kernel) *
               size_t(shapes.c2_kernel);
    }
    return 0;
}

struct CompiledLayer {
    ov::CompiledModel compiled;
    ov::InferRequest request;
    size_t in_bytes = 0;
    size_t out_bytes = 0;
    bool warmed = false;
};

struct Plugin {
    ov::Core core;
    std::string device;
    std::string last_error;
    std::mutex mu;
};

thread_local std::string g_last_global_error;

}  // namespace

struct loom_accel_plugin {
    Plugin impl;
};

struct loom_accel_compiled_layer {
    std::unique_ptr<CompiledLayer> impl;
};

extern "C" {

int loom_accel_api_version(void) {
    return LOOM_ACCEL_API_VERSION;
}

const char* loom_accel_vendor_id(void) {
    return "intel";
}

int loom_accel_npu_available(void) {
    try {
        ov::Core core;
        return device_available(core, "NPU") ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

loom_accel_plugin* loom_accel_plugin_open(const char* device) {
    if (device == nullptr) {
        g_last_global_error = "device is null";
        return nullptr;
    }
    try {
        auto plugin = new loom_accel_plugin();
        plugin->impl.device = device;
        if (!device_available(plugin->impl.core, plugin->impl.device)) {
            plugin->impl.last_error = std::string("device not available: ") + device;
            delete plugin;
            return nullptr;
        }
        return plugin;
    } catch (const std::exception& ex) {
        g_last_global_error = ex.what();
        return nullptr;
    }
}

void loom_accel_plugin_close(loom_accel_plugin* plugin) {
    delete plugin;
}

size_t loom_accel_weight_bytes(const loom_accel_layer_desc* desc) {
    if (desc == nullptr) {
        return 0;
    }
    const auto* shapes = find_shape(desc->size_label);
    if (shapes == nullptr) {
        return 0;
    }
    return weight_float_count(desc->layer_name, *shapes) * sizeof(float);
}

int loom_accel_compile_layer(
    loom_accel_plugin* plugin,
    const loom_accel_layer_desc* desc,
    const float* weights,
    size_t weight_count,
    loom_accel_compiled_layer** out,
    double* compile_ms,
    char* err,
    size_t err_len) {
    if (out == nullptr) {
        copy_err(err, err_len, "out is null");
        return -1;
    }
    *out = nullptr;
    if (plugin == nullptr || desc == nullptr) {
        copy_err(err, err_len, "plugin or desc is null");
        return -1;
    }

    const auto* layer = find_layer(desc->layer_name);
    const auto* shapes = find_shape(desc->size_label);
    const auto* dtype = find_dtype(desc->dtype);
    if (layer == nullptr || shapes == nullptr || dtype == nullptr) {
        copy_err(err, err_len, "unknown layer, size, or dtype");
        return -1;
    }

    try {
        auto model = layer->build(dtype->elem, *shapes);
        model = rebuild_with_weights(model, desc->layer_name, dtype->elem, *shapes, weights, weight_count);

        const auto cfg = compile_cfg_for(*dtype, plugin->impl.device);
        const auto t0 = Clock::now();
        auto compiled = plugin->impl.core.compile_model(model, plugin->impl.device, cfg);
        const auto t1 = Clock::now();

        auto holder = std::make_unique<CompiledLayer>();
        holder->compiled = std::move(compiled);
        holder->request = holder->compiled.create_infer_request();
        holder->in_bytes = holder->request.get_input_tensor(0).get_byte_size();
        holder->out_bytes = holder->request.get_output_tensor(0).get_byte_size();

        auto* layer_out = new loom_accel_compiled_layer();
        layer_out->impl = std::move(holder);
        *out = layer_out;

        if (compile_ms != nullptr) {
            *compile_ms = elapsed_ms(t0, t1);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lock(plugin->impl.mu);
        plugin->impl.last_error = ex.what();
        copy_err(err, err_len, plugin->impl.last_error);
        return -1;
    }
}

void loom_accel_release_layer(loom_accel_compiled_layer* layer) {
    delete layer;
}

int loom_accel_layer_io_bytes(
    loom_accel_compiled_layer* layer,
    size_t* in_bytes,
    size_t* out_bytes,
    char* err,
    size_t err_len) {
    if (layer == nullptr || layer->impl == nullptr) {
        copy_err(err, err_len, "invalid compiled layer");
        return -1;
    }
    if (in_bytes != nullptr) {
        *in_bytes = layer->impl->in_bytes;
    }
    if (out_bytes != nullptr) {
        *out_bytes = layer->impl->out_bytes;
    }
    return 0;
}

int loom_accel_first_infer(
    loom_accel_compiled_layer* layer,
    double* first_infer_ms,
    char* err,
    size_t err_len) {
    if (layer == nullptr || layer->impl == nullptr) {
        copy_err(err, err_len, "invalid compiled layer");
        return -1;
    }
    try {
        const auto t0 = Clock::now();
        layer->impl->request.infer();
        const auto t1 = Clock::now();
        layer->impl->warmed = true;
        if (first_infer_ms != nullptr) {
            *first_infer_ms = elapsed_ms(t0, t1);
        }
        return 0;
    } catch (const std::exception& ex) {
        copy_err(err, err_len, ex.what());
        return -1;
    }
}

int loom_accel_infer(
    loom_accel_compiled_layer* layer,
    const void* in,
    size_t in_bytes,
    void* out,
    size_t out_bytes,
    double* infer_ms,
    char* err,
    size_t err_len) {
    if (layer == nullptr || layer->impl == nullptr) {
        copy_err(err, err_len, "invalid compiled layer");
        return -1;
    }
    if (in == nullptr || out == nullptr) {
        copy_err(err, err_len, "in/out buffer is null");
        return -1;
    }
    if (in_bytes != layer->impl->in_bytes) {
        copy_err(err, err_len, "input byte size mismatch");
        return -1;
    }
    if (out_bytes != layer->impl->out_bytes) {
        copy_err(err, err_len, "output byte size mismatch");
        return -1;
    }

    try {
        auto input = layer->impl->request.get_input_tensor(0);
        std::memcpy(input.data(), in, in_bytes);

        const auto t0 = Clock::now();
        layer->impl->request.infer();
        const auto t1 = Clock::now();

        auto output = layer->impl->request.get_output_tensor(0);
        std::memcpy(out, output.data(), out_bytes);

        if (infer_ms != nullptr) {
            *infer_ms = elapsed_ms(t0, t1);
        }
        return 0;
    } catch (const std::exception& ex) {
        copy_err(err, err_len, ex.what());
        return -1;
    }
}

const char* loom_accel_last_error(loom_accel_plugin* plugin) {
    if (plugin == nullptr) {
        return g_last_global_error.c_str();
    }
    std::lock_guard<std::mutex> lock(plugin->impl.mu);
    return plugin->impl.last_error.c_str();
}

}  // extern "C"
