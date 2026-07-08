#include "loom_accel.h"

#include "layer_models.hpp"
#include "qnn_wrapper.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Mirrors the 15-layer matrix in accel/intel — same names, same order.
const std::vector<const char*> kLayers = {
    "MatMul", "Conv1D", "Conv2D", "DepthwiseConv", "AvgPool", "MaxPool",
    "ReLU", "GELU", "Sigmoid", "Softmax", "Add", "Multiply",
    "LayerNorm", "RMSNorm", "MHA-MatMul",
};

struct DTypeCase {
    qnn::DType elem;
    const char* label;
};

// Hexagon HTP natively supports FP16 + INT4/INT8/INT16 fixed-point; FP32 runs on
// the QnnCpu reference backend (or Adreno GPU). See bench_manifest.json.
const std::vector<DTypeCase> kDTypes = {
    {qnn::DType::FP32,  "FP32"},
    {qnn::DType::FP16,  "FP16"},
    {qnn::DType::INT16, "INT16"},
    {qnn::DType::INT8,  "INT8"},
    {qnn::DType::INT4,  "INT4"},
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

bool known_layer(const char* name) {
    if (name == nullptr) {
        return false;
    }
    for (const auto* l : kLayers) {
        if (std::strcmp(l, name) == 0) {
            return true;
        }
    }
    return false;
}

const DTypeCase* find_dtype(const char* label) {
    if (label == nullptr) {
        return nullptr;
    }
    for (const auto& dt : kDTypes) {
        if (std::strcmp(dt.label, label) == 0) {
            return &dt;
        }
    }
    return nullptr;
}

struct CompiledLayer {
    qnn::CompiledGraph graph;
    size_t in_bytes = 0;
    size_t out_bytes = 0;
    bool warmed = false;
};

struct Plugin {
    qnn::Backend backend;
    std::string device;
    std::string last_error;
    std::mutex mu;
    bool open = false;
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
    return "qualcomm";
}

int loom_accel_npu_available(void) {
    qnn::Backend probe;
    std::string err;
    if (probe.open("NPU", &err)) {
        probe.close();
        return 1;
    }
    return 0;
}

loom_accel_plugin* loom_accel_plugin_open(const char* device) {
    if (device == nullptr) {
        g_last_global_error = "device is null";
        return nullptr;
    }
    auto* plugin = new loom_accel_plugin();
    plugin->impl.device = device;
    std::string err;
    if (!plugin->impl.backend.open(device, &err)) {
        plugin->impl.last_error = err;
        g_last_global_error = err;
        delete plugin;
        return nullptr;
    }
    plugin->impl.open = true;
    return plugin;
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
    layer_models::GraphShapes g;
    if (!layer_models::resolve_shapes(desc->layer_name, *shapes, &g) || !g.has_weights) {
        return 0;
    }
    const auto* dt = find_dtype(desc->dtype);
    const size_t per = dt ? qnn::dtype_host_bytes(dt->elem) : 4;
    return g.weight_elems * per;
}

int loom_accel_compile_layer(
    loom_accel_plugin* plugin,
    const loom_accel_layer_desc* desc,
    const void* weight_bytes,
    size_t weight_byte_len,
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

    const auto* shapes = find_shape(desc->size_label);
    const auto* dt = find_dtype(desc->dtype);
    if (!known_layer(desc->layer_name) || shapes == nullptr || dt == nullptr) {
        copy_err(err, err_len, "unknown layer, size, or dtype");
        return -1;
    }

    layer_models::GraphShapes g;
    if (!layer_models::resolve_shapes(desc->layer_name, *shapes, &g)) {
        copy_err(err, err_len, "resolve_shapes failed");
        return -1;
    }

    auto holder = std::make_unique<CompiledLayer>();
    std::string berr;
    const auto t0 = Clock::now();
    if (!holder->graph.build(plugin->impl.backend, g, dt->elem, weight_bytes, weight_byte_len, &berr)) {
        std::lock_guard<std::mutex> lock(plugin->impl.mu);
        plugin->impl.last_error = berr;
        copy_err(err, err_len, berr);
        return -1;
    }
    const auto t1 = Clock::now();

    holder->in_bytes = holder->graph.in_bytes();
    holder->out_bytes = holder->graph.out_bytes();

    auto* layer_out = new loom_accel_compiled_layer();
    layer_out->impl = std::move(holder);
    *out = layer_out;
    if (compile_ms != nullptr) {
        *compile_ms = elapsed_ms(t0, t1);
    }
    return 0;
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
    std::vector<uint8_t> in(layer->impl->in_bytes, 0);
    std::vector<uint8_t> outbuf(layer->impl->out_bytes, 0);
    std::string eerr;
    const auto t0 = Clock::now();
    if (!layer->impl->graph.execute(in.data(), in.size(), outbuf.data(), outbuf.size(), &eerr)) {
        copy_err(err, err_len, eerr);
        return -1;
    }
    const auto t1 = Clock::now();
    layer->impl->warmed = true;
    if (first_infer_ms != nullptr) {
        *first_infer_ms = elapsed_ms(t0, t1);
    }
    return 0;
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

    std::string eerr;
    const auto t0 = Clock::now();
    if (!layer->impl->graph.execute(in, in_bytes, out, out_bytes, &eerr)) {
        copy_err(err, err_len, eerr);
        return -1;
    }
    const auto t1 = Clock::now();
    if (infer_ms != nullptr) {
        *infer_ms = elapsed_ms(t0, t1);
    }
    return 0;
}

const char* loom_accel_last_error(loom_accel_plugin* plugin) {
    if (plugin == nullptr) {
        return g_last_global_error.c_str();
    }
    std::lock_guard<std::mutex> lock(plugin->impl.mu);
    return plugin->impl.last_error.c_str();
}

}  // extern "C"
