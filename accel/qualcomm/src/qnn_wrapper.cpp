#include "qnn_wrapper.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qnn {
namespace {

// ---- platform dynamic loading ------------------------------------------------

void* load_library(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* load_symbol(void* lib, const char* sym) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), sym));
#else
    return dlsym(lib, sym);
#endif
}

void close_library(void* lib) {
    if (!lib) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(lib));
#else
    dlclose(lib);
#endif
}

const char* backend_lib_for(const std::string& device, bool* is_htp) {
    // Loom's C ABI device strings map to QNN backend libs:
    //   NPU/HTP/DSP → Hexagon Tensor Processor   GPU → Adreno   CPU → Kryo reference
    *is_htp = false;
    if (device == "NPU" || device == "HTP" || device == "DSP") {
        *is_htp = true;
#if defined(_WIN32)
        return "QnnHtp.dll";
#else
        return "libQnnHtp.so";
#endif
    }
    if (device == "GPU" || device == "ADRENO") {
#if defined(_WIN32)
        return "QnnGpu.dll";
#else
        return "libQnnGpu.so";
#endif
    }
#if defined(_WIN32)
    return "QnnCpu.dll";
#else
    return "libQnnCpu.so";
#endif
}

// External activation tensor precision (what the host uploads/reads).
// Quantized modes keep float I/O and requantize weights internally, matching the
// FP32 host-byte contract in poly/accel_intel.go (INT* activations arrive as FP32).
Qnn_DataType_t qnn_activation_dtype(DType d) {
    if (d == DType::FP16) return QNN_DATATYPE_FLOAT_16;
    return QNN_DATATYPE_FLOAT_32;
}

// Static weight tensor precision per the Qualcomm HTP datatype table:
//   INT8/INT16 → 8-bit weights, INT4 → 4-bit weights, FP16 → half, FP32 → float.
Qnn_DataType_t qnn_weight_dtype(DType d) {
    switch (d) {
        case DType::FP16:  return QNN_DATATYPE_FLOAT_16;
        case DType::INT16: return QNN_DATATYPE_SFIXED_POINT_8;
        case DType::INT8:  return QNN_DATATYPE_SFIXED_POINT_8;
        case DType::INT4:  return QNN_DATATYPE_SFIXED_POINT_4;
        default:           return QNN_DATATYPE_FLOAT_32;
    }
}

int weight_quant_bits(DType d) {
    switch (d) {
        case DType::INT4: return 4;
        case DType::INT8:
        case DType::INT16: return 8;  // 8-bit weights (INT16 uses 16-bit activations)
        default: return 0;
    }
}

// Build a v1 tensor descriptor. `dims` must outlive the tensor (points at caller storage).
Qnn_Tensor_t make_tensor(
    const char* name,
    Qnn_TensorType_t type,
    Qnn_DataType_t dtype,
    uint32_t* dims,
    uint32_t rank,
    void* client_data,
    uint32_t client_bytes,
    const Qnn_QuantizeParams_t* quant) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.id = 0;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dtype;
    Qnn_QuantizeParams_t default_quant = QNN_QUANTIZE_PARAMS_INIT;
    t.v1.quantizeParams = quant ? *quant : default_quant;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = client_data;
    t.v1.clientBuf.dataSize = client_bytes;
    return t;
}

size_t elem_count(const std::vector<uint32_t>& dims) {
    size_t n = 1;
    for (uint32_t d : dims) n *= d;
    return n;
}

// Per-tensor symmetric fixed-point quant params.
Qnn_QuantizeParams_t make_quant(float scale) {
    Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
    q.encodingDefinition = QNN_DEFINITION_DEFINED;
    q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    q.scaleOffsetEncoding.scale = scale;
    q.scaleOffsetEncoding.offset = 0;
    return q;
}

// Symmetric per-tensor quantize of float32 weights into `bits`-wide fixed point.
// Returns the packed bytes (INT4 packs two nibbles per byte) and the scale used.
std::vector<uint8_t> quantize_weights(const float* src, size_t n, int bits, float* scale_out) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = src[i] < 0 ? -src[i] : src[i];
        if (a > amax) amax = a;
    }
    const int qmax = (1 << (bits - 1)) - 1;  // e.g. 127 for int8, 7 for int4
    const float scale = (amax > 0.0f) ? amax / static_cast<float>(qmax) : 1.0f;
    if (scale_out) *scale_out = scale;

    auto q = [&](float v) -> int {
        int r = static_cast<int>(v / scale + (v >= 0 ? 0.5f : -0.5f));
        if (r > qmax) r = qmax;
        if (r < -qmax - 1) r = -qmax - 1;
        return r;
    };

    if (bits <= 4) {
        std::vector<uint8_t> out((n + 1) / 2, 0);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t nib = static_cast<uint8_t>(q(src[i]) & 0x0F);
            if ((i & 1) == 0) out[i / 2] = nib;
            else out[i / 2] |= static_cast<uint8_t>(nib << 4);
        }
        return out;
    }
    std::vector<uint8_t> out(n);  // 8-bit
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>(static_cast<int8_t>(q(src[i])));
    }
    return out;
}

const char* op_type_name(layer_models::OpKind k) {
    using layer_models::OpKind;
    switch (k) {
        case OpKind::MatMul:        return "MatMul";
        case OpKind::Conv1D:        return "Conv2d";   // modelled as [N,1,L,C] 2D conv
        case OpKind::Conv2D:        return "Conv2d";
        case OpKind::DepthwiseConv: return "DepthWiseConv2d";
        case OpKind::AvgPool:       return "PoolAvg2d";
        case OpKind::MaxPool:       return "PoolMax2d";
        case OpKind::ReLU:          return "Relu";
        case OpKind::GELU:          return "Gelu";
        case OpKind::Sigmoid:       return "Sigmoid";
        case OpKind::Softmax:       return "Softmax";
        case OpKind::Add:           return "ElementWiseAdd";
        case OpKind::Multiply:      return "ElementWiseMultiply";
        case OpKind::LayerNorm:     return "LayerNorm";
        case OpKind::RMSNorm:       return "RmsNorm";
    }
    return "Relu";
}

}  // namespace

const char* dtype_label(DType d) {
    switch (d) {
        case DType::FP16:  return "FP16";
        case DType::INT16: return "INT16";
        case DType::INT8:  return "INT8";
        case DType::INT4:  return "INT4";
        default:           return "FP32";
    }
}

size_t dtype_host_bytes(DType d) {
    // Host activation/weight upload element size. FP16 uses IEEE half; every other
    // mode (incl. INT4/INT8/INT16) uploads dequantized float32 — the plugin
    // requantizes to the target fixed-point precision.
    if (d == DType::FP16) return 2;
    return 4;
}

bool dtype_is_quantized(DType d) {
    return d == DType::INT16 || d == DType::INT8 || d == DType::INT4;
}

// ---- Backend -----------------------------------------------------------------

Backend::~Backend() { close(); }

bool Backend::open(const std::string& device, std::string* err) {
    const char* libname = backend_lib_for(device, &is_htp_);
    lib_ = load_library(libname);
    if (!lib_) {
        if (err) *err = std::string("failed to load QNN backend: ") + libname +
                        " (is QNN_SDK_ROOT/lib/aarch64-windows-msvc on PATH?)";
        return false;
    }

    // QAIRT does not ship a typedef for this entry point; declare it to match
    // QnnInterface.h: Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t***, uint32_t*).
    typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t***, uint32_t*);
    auto get_providers = reinterpret_cast<GetProvidersFn>(
        load_symbol(lib_, "QnnInterface_getProviders"));
    if (!get_providers) {
        if (err) *err = "QnnInterface_getProviders not found in backend";
        return false;
    }

    const QnnInterface_t** providers = nullptr;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
        if (err) *err = "QnnInterface_getProviders returned no providers";
        return false;
    }

    bool picked = false;
    for (uint32_t i = 0; i < num_providers; ++i) {
        if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
            iface_ = providers[i]->QNN_INTERFACE_VER_NAME;
            picked = true;
            break;
        }
    }
    if (!picked) {
        if (err) *err = "no QNN provider matches this SDK's core API version";
        return false;
    }

    // Default QNN logging is extremely verbose on HTP (DSP transport warnings,
    // graph-prepare stage spam). Clamp to errors unless the host sets
    // LOOM_QNN_VERBOSE=1 in the environment.
    if (iface_.logCreate != nullptr) {
        const char* verbose = std::getenv("LOOM_QNN_VERBOSE");
        const auto level = (verbose != nullptr && verbose[0] != '\0' && verbose[0] != '0')
                               ? QNN_LOG_LEVEL_WARN
                               : QNN_LOG_LEVEL_ERROR;
        iface_.logCreate(nullptr, level, &logger_);
    }

    if (iface_.backendCreate(logger_, nullptr, &backend_) != QNN_SUCCESS) {
        if (err) *err = "QnnBackend_create failed";
        return false;
    }

    if (is_htp_) {
        // HTP wants an explicit device handle; CPU backend works without one.
        if (iface_.deviceCreate != nullptr) {
            iface_.deviceCreate(nullptr, nullptr, &device_handle_);
        }
    }

    if (iface_.contextCreate(backend_, device_handle_, nullptr, &context_) != QNN_SUCCESS) {
        if (err) *err = "QnnContext_create failed";
        return false;
    }
    return true;
}

bool Backend::reset_context(std::string* err) {
    if (!backend_) {
        if (err) *err = "backend not open";
        return false;
    }
    if (context_ && iface_.contextFree) {
        iface_.contextFree(context_, nullptr);
        context_ = nullptr;
    }
    if (iface_.contextCreate(backend_, device_handle_, nullptr, &context_) != QNN_SUCCESS) {
        if (err) *err = "QnnContext_create failed after reset";
        return false;
    }
    return true;
}

void Backend::close() {
    if (context_ && iface_.contextFree) {
        iface_.contextFree(context_, nullptr);
        context_ = nullptr;
    }
    if (device_handle_ && iface_.deviceFree) {
        iface_.deviceFree(device_handle_);
        device_handle_ = nullptr;
    }
    if (backend_ && iface_.backendFree) {
        iface_.backendFree(backend_);
        backend_ = nullptr;
    }
    if (logger_ && iface_.logFree) {
        iface_.logFree(logger_);
        logger_ = nullptr;
    }
    close_library(lib_);
    lib_ = nullptr;
    close_library(system_lib_);
    system_lib_ = nullptr;
}

// ---- CompiledGraph -----------------------------------------------------------

CompiledGraph::~CompiledGraph() {
    // Drop our graph from the shared context so the next compile starts clean.
    if (be_ != nullptr) {
        be_->reset_context(nullptr);
        graph_ = nullptr;
    }
    input_tensors_.clear();
    output_tensors_.clear();
    io_storage_.clear();
}

bool CompiledGraph::build(
    Backend& be,
    const layer_models::GraphShapes& shapes,
    DType dtype,
    const void* weight_bytes,
    size_t weight_byte_len,
    std::string* err) {
    be_ = &be;
    dtype_ = dtype;
    // Every compiled graph shares the plugin's single QNN context, so the graph
    // name must be unique per build — QnnGraph_create rejects a duplicate name,
    // which previously made every dtype after the first (FP32) fail.
    static std::atomic<uint64_t> g_graph_seq{0};
    graph_name_ = std::string("loom_") + op_type_name(shapes.kind) + "_" +
                  dtype_label(dtype) + "_" +
                  std::to_string(g_graph_seq.fetch_add(1));

    const auto& iface = be.iface();
    if (iface.graphCreate(be.context(), graph_name_.c_str(), nullptr, &graph_) != QNN_SUCCESS) {
        if (err) *err = "QnnGraph_create failed";
        return false;
    }

    // External activations stay float (FP32, or FP16 for the FP16 mode); only the
    // static weight tensor is quantized. This matches poly/accel_intel.go, which
    // hands INT4/INT8/INT16 activations to the plugin as FP32 values.
    const Qnn_DataType_t act_dt = qnn_activation_dtype(dtype);
    const Qnn_DataType_t w_dt = qnn_weight_dtype(dtype);

    // Persist dimension arrays for the lifetime of the tensors.
    static thread_local std::vector<std::vector<uint32_t>> dim_store;
    dim_store.clear();
    auto dims_ptr = [&](const std::vector<uint32_t>& d) -> uint32_t* {
        dim_store.push_back(d);
        return dim_store.back().data();
    };

    // Input activation tensor (APP_WRITE).
    std::vector<uint32_t> in_dims = shapes.input_dims;
    Qnn_Tensor_t in_t = make_tensor(
        "input", QNN_TENSOR_TYPE_APP_WRITE, act_dt, dims_ptr(in_dims),
        static_cast<uint32_t>(in_dims.size()), nullptr, 0, nullptr);
    if (iface.tensorCreateGraphTensor(graph_, &in_t) != QNN_SUCCESS) {
        if (err) *err = "create input tensor failed";
        return false;
    }

    std::vector<Qnn_Tensor_t> node_inputs;
    node_inputs.push_back(in_t);

    // Static weights (uploaded from Loom's WeightStore as FP32) for MatMul / Conv.
    std::vector<uint8_t> weight_copy;
    if (shapes.has_weights && weight_bytes != nullptr && weight_byte_len > 0) {
        Qnn_QuantizeParams_t qw = QNN_QUANTIZE_PARAMS_INIT;
        const int bits = weight_quant_bits(dtype);
        if (bits > 0) {
            // FP32 in → fixed-point weights (per-tensor symmetric scale).
            const size_t n = weight_byte_len / sizeof(float);
            float scale = 1.0f;
            weight_copy = quantize_weights(
                reinterpret_cast<const float*>(weight_bytes), n, bits, &scale);
            qw = make_quant(scale);
        } else {
            // FP32 / FP16 weights pass through as-is.
            weight_copy.assign(
                static_cast<const uint8_t*>(weight_bytes),
                static_cast<const uint8_t*>(weight_bytes) + weight_byte_len);
        }
        std::vector<uint32_t> w_dims = shapes.weight_dims;
        Qnn_Tensor_t w_t = make_tensor(
            "weights", QNN_TENSOR_TYPE_STATIC, w_dt, dims_ptr(w_dims),
            static_cast<uint32_t>(w_dims.size()), weight_copy.data(),
            static_cast<uint32_t>(weight_copy.size()), bits > 0 ? &qw : nullptr);
        if (iface.tensorCreateGraphTensor(graph_, &w_t) != QNN_SUCCESS) {
            if (err) *err = "create weight tensor failed";
            return false;
        }
        node_inputs.push_back(w_t);
        io_storage_.push_back(std::move(weight_copy));
    }

    // Output tensor (APP_READ) — float, matching the host read-back contract.
    std::vector<uint32_t> out_dims = shapes.output_dims;
    Qnn_Tensor_t out_t = make_tensor(
        "output", QNN_TENSOR_TYPE_APP_READ, act_dt, dims_ptr(out_dims),
        static_cast<uint32_t>(out_dims.size()), nullptr, 0, nullptr);
    if (iface.tensorCreateGraphTensor(graph_, &out_t) != QNN_SUCCESS) {
        if (err) *err = "create output tensor failed";
        return false;
    }

    // Node params (conv/pool strides + padding, softmax axis, …).
    static thread_local std::vector<uint32_t> p_stride;   // [h, w]
    static thread_local std::vector<uint32_t> p_pad;      // [[top,bottom],[left,right]]
    static thread_local std::vector<uint32_t> p_filter;   // [h, w]
    static thread_local std::vector<uint32_t> p_stride_dims;
    static thread_local std::vector<uint32_t> p_pad_dims;
    static thread_local std::vector<uint32_t> p_filter_dims;
    std::vector<Qnn_Param_t> params;

    auto add_tensor_param = [&](const char* pname, std::vector<uint32_t>& data,
                                std::vector<uint32_t>& pdims) {
        Qnn_Tensor_t pt = make_tensor(
            pname, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, pdims.data(),
            static_cast<uint32_t>(pdims.size()), data.data(),
            static_cast<uint32_t>(data.size() * sizeof(uint32_t)), nullptr);
        Qnn_Param_t p = QNN_PARAM_INIT;
        p.paramType = QNN_PARAMTYPE_TENSOR;
        p.name = pname;
        p.tensorParam = pt;
        params.push_back(p);
    };

    using layer_models::OpKind;
    const bool is_conv = shapes.kind == OpKind::Conv1D || shapes.kind == OpKind::Conv2D ||
                         shapes.kind == OpKind::DepthwiseConv;
    const bool is_pool = shapes.kind == OpKind::AvgPool || shapes.kind == OpKind::MaxPool;

    if (is_conv) {
        p_stride = {shapes.stride_h, shapes.stride_w};
        p_stride_dims = {2u};
        add_tensor_param("stride", p_stride, p_stride_dims);
        p_pad = {shapes.pad_h, shapes.pad_h, shapes.pad_w, shapes.pad_w};
        p_pad_dims = {2u, 2u};
        add_tensor_param("pad_amount", p_pad, p_pad_dims);
    } else if (is_pool) {
        p_filter = {shapes.filter_h, shapes.filter_w};
        p_filter_dims = {2u};
        add_tensor_param("filter_size", p_filter, p_filter_dims);
        p_stride = {shapes.stride_h, shapes.stride_w};
        p_stride_dims = {2u};
        add_tensor_param("stride", p_stride, p_stride_dims);
        p_pad = {0u, 0u, 0u, 0u};
        p_pad_dims = {2u, 2u};
        add_tensor_param("pad_amount", p_pad, p_pad_dims);
    } else if (shapes.kind == OpKind::Softmax) {
        Qnn_Param_t axis = QNN_PARAM_INIT;
        axis.paramType = QNN_PARAMTYPE_SCALAR;
        axis.name = "axis";
        axis.scalarParam.dataType = QNN_DATATYPE_UINT_32;
        axis.scalarParam.uint32Value = static_cast<uint32_t>(shapes.input_dims.size() - 1);
        params.push_back(axis);
    }

    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = graph_name_.c_str();
    op.v1.packageName = "qti.aisw";
    op.v1.typeName = op_type_name(shapes.kind);
    op.v1.numOfParams = static_cast<uint32_t>(params.size());
    op.v1.params = params.empty() ? nullptr : params.data();
    op.v1.numOfInputs = static_cast<uint32_t>(node_inputs.size());
    op.v1.inputTensors = node_inputs.data();
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &out_t;

    if (iface.graphAddNode(graph_, op) != QNN_SUCCESS) {
        if (err) *err = std::string("graphAddNode failed for ") + op_type_name(shapes.kind);
        return false;
    }

    if (iface.graphFinalize(graph_, nullptr, nullptr) != QNN_SUCCESS) {
        if (err) *err = "QnnGraph_finalize failed";
        return false;
    }

    // Keep tensor handles for execute(); allocate host I/O backing.
    in_bytes_ = elem_count(shapes.input_dims) * dtype_host_bytes(dtype);
    out_bytes_ = elem_count(shapes.output_dims) * dtype_host_bytes(dtype);
    input_tensors_ = {in_t};
    output_tensors_ = {out_t};
    return true;
}

bool CompiledGraph::execute(const void* in, size_t in_bytes, void* out, size_t out_bytes,
                            std::string* err) {
    if (be_ == nullptr || graph_ == nullptr) {
        if (err) *err = "graph not built";
        return false;
    }
    if (in_bytes != in_bytes_ || out_bytes != out_bytes_) {
        if (err) *err = "io byte size mismatch";
        return false;
    }

    // Point the input/output client buffers at the caller's host memory.
    input_tensors_[0].v1.clientBuf.data = const_cast<void*>(in);
    input_tensors_[0].v1.clientBuf.dataSize = static_cast<uint32_t>(in_bytes);
    output_tensors_[0].v1.clientBuf.data = out;
    output_tensors_[0].v1.clientBuf.dataSize = static_cast<uint32_t>(out_bytes);

    const auto& iface = be_->iface();
    const Qnn_ErrorHandle_t rc = iface.graphExecute(
        graph_,
        input_tensors_.data(), static_cast<uint32_t>(input_tensors_.size()),
        output_tensors_.data(), static_cast<uint32_t>(output_tensors_.size()),
        nullptr, nullptr);
    if (rc != QNN_SUCCESS) {
        if (err) *err = "QnnGraph_execute failed";
        return false;
    }
    return true;
}

}  // namespace qnn
