#pragma once

// Thin RAII wrapper over the Qualcomm QNN AI Engine Direct C API.
//
// This is the QNN analog of the OpenVINO ov::Core / ov::Model / ov::CompiledModel
// usage in accel/intel. It:
//   * loads a QNN backend shared library (QnnHtp for NPU, QnnCpu for reference),
//   * resolves the QNN_INTERFACE_VER_TYPE function table via QnnInterface_getProviders,
//   * creates backend + context,
//   * builds a single-op graph from layer_models::GraphShapes,
//   * finalizes and executes it (compile once, infer many).
//
// The QAIRT SDK (QNN_SDK_ROOT) supplies the headers below. See install_qairt.ps1.

#include "layer_models.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnCommon.h"

namespace qnn {

// dtype selector mirrors bench_manifest dtypes. Per Qualcomm HTP support:
//   FP32  — CPU/GPU only (HTP falls back to QnnCpu reference backend)
//   FP16  — native HTP (16-bit float, no quant)
//   INT16 — native HTP fixed-point: 8-bit weights + 16-bit activations
//   INT8  — native HTP fixed-point: 8-bit weights + 8-bit activations
//   INT4  — native HTP fixed-point: 4-bit weights + 8-bit activations
enum class DType { FP32, FP16, INT16, INT8, INT4 };

const char* dtype_label(DType d);
size_t dtype_host_bytes(DType d);  // bytes per element in the host activation buffer
bool dtype_is_quantized(DType d);  // integer fixed-point mode (weights requantized in-plugin)

// Loaded backend + resolved interface table. One per device ("CPU" / "NPU").
class Backend {
public:
    Backend() = default;
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // device is "CPU" (QnnCpu) or "NPU" (QnnHtp). Returns false + sets err on failure.
    bool open(const std::string& device, std::string* err);
    void close();

    bool is_htp() const { return is_htp_; }
    const QNN_INTERFACE_VER_TYPE& iface() const { return iface_; }
    Qnn_BackendHandle_t backend() const { return backend_; }
    Qnn_ContextHandle_t context() const { return context_; }
    Qnn_DeviceHandle_t device() const { return device_handle_; }

    // QNN has no per-graph free API; graphs live until the context is destroyed.
    // Benchmark code releases layers one at a time, so resetting the context on
    // release keeps memory flat and avoids HTP prepare slowdowns / crashes.
    bool reset_context(std::string* err);

private:
    void* lib_ = nullptr;                 // backend .dll/.so handle
    void* system_lib_ = nullptr;          // QnnSystem handle (HTP)
    Qnn_LogHandle_t logger_ = nullptr;
    QNN_INTERFACE_VER_TYPE iface_{};
    Qnn_BackendHandle_t backend_ = nullptr;
    Qnn_ContextHandle_t context_ = nullptr;
    Qnn_DeviceHandle_t device_handle_ = nullptr;
    bool is_htp_ = false;
};

// One compiled single-op graph: finalized, with I/O tensors ready for execute().
class CompiledGraph {
public:
    CompiledGraph() = default;
    ~CompiledGraph();

    CompiledGraph(const CompiledGraph&) = delete;
    CompiledGraph& operator=(const CompiledGraph&) = delete;

    // Build + finalize a graph for `shapes` at precision `dtype`. weight_bytes may be
    // null (baked defaults). Returns false + sets err on failure.
    bool build(
        Backend& be,
        const layer_models::GraphShapes& shapes,
        DType dtype,
        const void* weight_bytes,
        size_t weight_byte_len,
        std::string* err);

    // Run one forward. in/out buffers are host-side, sizes must match io bytes.
    bool execute(const void* in, size_t in_bytes, void* out, size_t out_bytes, std::string* err);

    size_t in_bytes() const { return in_bytes_; }
    size_t out_bytes() const { return out_bytes_; }

private:
    Backend* be_ = nullptr;
    Qnn_GraphHandle_t graph_ = nullptr;
    std::string graph_name_;

    // Persisted tensor descriptors + backing storage kept alive for execute().
    std::vector<Qnn_Tensor_t> input_tensors_;
    std::vector<Qnn_Tensor_t> output_tensors_;
    std::vector<std::vector<uint8_t>> io_storage_;

    DType dtype_ = DType::FP32;
    size_t in_bytes_ = 0;
    size_t out_bytes_ = 0;
};

}  // namespace qnn
