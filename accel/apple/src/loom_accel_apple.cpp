//
// loom_accel_apple.cpp — vendor-neutral C ABI (include/loom_accel.h) for Apple silicon.
//
// Device "GPU" runs the supported ops on Metal via MPSGraph (see mps_backend.mm) and
// falls back to the portable CPU reference for the rest. Device "CPU" always uses the
// CPU reference (the deterministic parity anchor, like Qualcomm's QnnCpu backend).
//
#include "loom_accel.h"

#include "cpu_reference.hpp"
#include "half.hpp"
#include "mps_backend.hpp"
#include "shapes.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace loom_apple;

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void copy_err(char* err, size_t err_len, const std::string& msg) {
    if (err == nullptr || err_len == 0) {
        return;
    }
    std::strncpy(err, msg.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
}

thread_local std::string g_last_global_error;

struct PluginImpl {
    std::string device;  // "CPU" or "GPU"
    std::string last_error;
    std::mutex mu;
};

// Wire format of the bytes crossing the C ABI (fp32 compute internally).
enum class WireFmt { FP32, FP16, BF16 };

WireFmt wire_for(const std::string& dtype_label) {
    if (dtype_label == "FP16") return WireFmt::FP16;
    if (dtype_label == "BF16") return WireFmt::BF16;
    return WireFmt::FP32;  // FP32 / INT8 / INT16 / INT4 hand over fp32 values
}

struct CompiledImpl {
    Prepared prepared;
    WireFmt wire = WireFmt::FP32;
    MpsLayer* mps = nullptr;  // non-null → GPU accelerated
    size_t in_bytes = 0;
    size_t out_bytes = 0;
};

// Unpack C ABI input bytes → fp32 host vector.
void bytes_to_floats(const void* src, size_t n, WireFmt wire, std::vector<float>& dst) {
    dst.resize(n);
    const auto* p = static_cast<const uint8_t*>(src);
    switch (wire) {
        case WireFmt::FP16:
            for (size_t i = 0; i < n; ++i) {
                dst[i] = half_to_float(uint16_t(p[i * 2]) | (uint16_t(p[i * 2 + 1]) << 8));
            }
            break;
        case WireFmt::BF16:
            for (size_t i = 0; i < n; ++i) {
                dst[i] = bfloat16_to_float(uint16_t(p[i * 2]) | (uint16_t(p[i * 2 + 1]) << 8));
            }
            break;
        default:
            std::memcpy(dst.data(), src, n * sizeof(float));
    }
}

// Pack fp32 host vector → C ABI output bytes.
void floats_to_bytes(const std::vector<float>& src, WireFmt wire, void* dst) {
    auto* p = static_cast<uint8_t*>(dst);
    switch (wire) {
        case WireFmt::FP16:
            for (size_t i = 0; i < src.size(); ++i) {
                const uint16_t h = float_to_half(src[i]);
                p[i * 2] = uint8_t(h & 0xff);
                p[i * 2 + 1] = uint8_t((h >> 8) & 0xff);
            }
            break;
        case WireFmt::BF16:
            for (size_t i = 0; i < src.size(); ++i) {
                const uint16_t h = float_to_bfloat16(src[i]);
                p[i * 2] = uint8_t(h & 0xff);
                p[i * 2 + 1] = uint8_t((h >> 8) & 0xff);
            }
            break;
        default:
            std::memcpy(dst, src.data(), src.size() * sizeof(float));
    }
}

bool run_forward(CompiledImpl* c, const std::vector<float>& in_f, std::vector<float>& out_f, std::string* err) {
    out_f.assign(c->prepared.out_elems, 0.0f);
    if (c->mps != nullptr) {
        if (mps_run(c->mps, in_f.data(), in_f.size(), out_f.data(), out_f.size(), err)) {
            return true;
        }
        // GPU failed at runtime — degrade gracefully to CPU reference.
    }
    return cpu_forward(c->prepared, in_f.data(), in_f.size(), out_f.data(), out_f.size(), err);
}

}  // namespace

struct loom_accel_plugin {
    PluginImpl impl;
};

struct loom_accel_compiled_layer {
    std::unique_ptr<CompiledImpl> impl;
};

extern "C" {

int loom_accel_api_version(void) {
    return LOOM_ACCEL_API_VERSION;
}

const char* loom_accel_vendor_id(void) {
    return "apple";
}

int loom_accel_npu_available(void) {
    // "NPU" in the vendor-neutral ABI == the accelerated device. On Apple that is the
    // Metal GPU (MPSGraph); ANE is reached indirectly through it / Core ML.
    return mps_device_available() ? 1 : 0;
}

loom_accel_plugin* loom_accel_plugin_open(const char* device) {
    if (device == nullptr) {
        g_last_global_error = "device is null";
        return nullptr;
    }
    const std::string dev(device);
    if (dev != "CPU" && dev != "GPU") {
        g_last_global_error = std::string("unknown device: ") + dev + " (expected CPU or GPU)";
        return nullptr;
    }
    if (dev == "GPU" && !mps_device_available()) {
        g_last_global_error = "no Metal GPU device available";
        return nullptr;
    }
    auto* plugin = new loom_accel_plugin();
    plugin->impl.device = dev;
    return plugin;
}

void loom_accel_plugin_close(loom_accel_plugin* plugin) {
    delete plugin;
}

size_t loom_accel_weight_bytes(const loom_accel_layer_desc* desc) {
    if (desc == nullptr) {
        return 0;
    }
    ShapeSpec s;
    if (!find_shape(desc->size_label, &s)) {
        return 0;
    }
    const size_t count = weight_float_count(desc->layer_name ? desc->layer_name : "", s);
    return count * io_elem_size(desc->dtype ? desc->dtype : "FP32");
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

    const std::string name = desc->layer_name ? desc->layer_name : "";
    const std::string dtype = desc->dtype ? desc->dtype : "";
    ShapeSpec s;
    if (!known_layer(name) || !known_dtype(dtype) || !find_shape(desc->size_label, &s)) {
        copy_err(err, err_len, "unknown layer, dtype, or size");
        return -1;
    }

    const auto t0 = Clock::now();

    auto impl = std::make_unique<CompiledImpl>();
    impl->wire = wire_for(dtype);
    impl->prepared.name = name;
    impl->prepared.dtype_label = dtype;
    impl->prepared.spec = s;
    impl->prepared.in_elems = input_elems(name, s);
    impl->prepared.out_elems = output_elems(name, s);
    impl->prepared.weights = resolve_weights(name, s, weight_bytes, weight_byte_len, dtype);

    const size_t esize = io_elem_size(dtype);
    impl->in_bytes = impl->prepared.in_elems * esize;
    impl->out_bytes = impl->prepared.out_elems * esize;

    if (plugin->impl.device == "GPU") {
        std::string mps_err;
        MpsLayer* mps = mps_build(impl->prepared, &mps_err);
        impl->mps = mps;  // may stay null → CPU fallback for this op
    }

    const auto t1 = Clock::now();

    auto* layer = new loom_accel_compiled_layer();
    layer->impl = std::move(impl);
    *out = layer;
    if (compile_ms != nullptr) {
        *compile_ms = elapsed_ms(t0, t1);
    }
    return 0;
}

void loom_accel_release_layer(loom_accel_compiled_layer* layer) {
    if (layer != nullptr && layer->impl && layer->impl->mps != nullptr) {
        mps_release(layer->impl->mps);
        layer->impl->mps = nullptr;
    }
    delete layer;
}

int loom_accel_layer_io_bytes(
    loom_accel_compiled_layer* layer,
    size_t* in_bytes,
    size_t* out_bytes,
    char* err,
    size_t err_len) {
    if (layer == nullptr || !layer->impl) {
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
    if (layer == nullptr || !layer->impl) {
        copy_err(err, err_len, "invalid compiled layer");
        return -1;
    }
    CompiledImpl* c = layer->impl.get();
    std::vector<float> in_f(c->prepared.in_elems, 0.0f);
    std::vector<float> out_f;
    std::string ferr;
    const auto t0 = Clock::now();
    const bool ok = run_forward(c, in_f, out_f, &ferr);
    const auto t1 = Clock::now();
    if (!ok) {
        copy_err(err, err_len, ferr);
        return -1;
    }
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
    if (layer == nullptr || !layer->impl) {
        copy_err(err, err_len, "invalid compiled layer");
        return -1;
    }
    if (in == nullptr || out == nullptr) {
        copy_err(err, err_len, "in/out buffer is null");
        return -1;
    }
    CompiledImpl* c = layer->impl.get();
    if (in_bytes != c->in_bytes) {
        copy_err(err, err_len, "input byte size mismatch");
        return -1;
    }
    if (out_bytes != c->out_bytes) {
        copy_err(err, err_len, "output byte size mismatch");
        return -1;
    }

    std::vector<float> in_f;
    bytes_to_floats(in, c->prepared.in_elems, c->wire, in_f);

    std::vector<float> out_f;
    std::string ferr;
    const auto t0 = Clock::now();
    const bool ok = run_forward(c, in_f, out_f, &ferr);
    const auto t1 = Clock::now();
    if (!ok) {
        copy_err(err, err_len, ferr);
        return -1;
    }

    floats_to_bytes(out_f, c->wire, out);
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
