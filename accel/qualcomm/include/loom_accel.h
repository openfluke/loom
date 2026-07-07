/* loom_accel.h — vendor-neutral C ABI for Loom accelerator plugins.
 *
 * Each vendor ships libloom_accel_<vendor>.{so,dll} implementing these symbols.
 * Loom loads the library at runtime (dlopen / LoadLibrary); QNN and other SDKs
 * stay private to the plugin.
 *
 * This file is kept byte-for-byte in sync with:
 *   accel/intel/include/loom_accel.h
 *   poly/accel/include/loom_accel.h
 */
#ifndef LOOM_ACCEL_H
#define LOOM_ACCEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_ACCEL_API_VERSION 2

typedef struct loom_accel_plugin loom_accel_plugin;
typedef struct loom_accel_compiled_layer loom_accel_compiled_layer;

typedef struct loom_accel_layer_desc {
    const char* layer_name; /* MatMul, Conv1D, … — see bench_manifest.json */
    const char* dtype;      /* FP32, FP16, INT8 */
    const char* size_label; /* small, medium, large */
} loom_accel_layer_desc;

/* Plugin lifecycle — one plugin instance per device (CPU or NPU/HTP). */
int loom_accel_api_version(void);

const char* loom_accel_vendor_id(void);

/* Returns 1 if the NPU device is visible to the plugin, else 0. */
int loom_accel_npu_available(void);

loom_accel_plugin* loom_accel_plugin_open(const char* device);
void loom_accel_plugin_close(loom_accel_plugin* plugin);

/* Expected weight blob size in bytes (native layout per desc->dtype). 0 if baked defaults. */
size_t loom_accel_weight_bytes(const loom_accel_layer_desc* desc);

/* Compile once at network init. weight_bytes may be NULL (vendor defaults).
 * FP32/INT8: little-endian float32 per element (INT8 uses dequantized values).
 * FP16: little-endian IEEE half per element. */
int loom_accel_compile_layer(
    loom_accel_plugin* plugin,
    const loom_accel_layer_desc* desc,
    const void* weight_bytes,
    size_t weight_byte_len,
    loom_accel_compiled_layer** out,
    double* compile_ms,
    char* err,
    size_t err_len);

void loom_accel_release_layer(loom_accel_compiled_layer* layer);

int loom_accel_layer_io_bytes(
    loom_accel_compiled_layer* layer,
    size_t* in_bytes,
    size_t* out_bytes,
    char* err,
    size_t err_len);

/* First infer after compile (graph warm-up). Optional; infer() also works. */
int loom_accel_first_infer(
    loom_accel_compiled_layer* layer,
    double* first_infer_ms,
    char* err,
    size_t err_len);

/* Steady-state infer — reuses compiled graph + infer request (no compile). */
int loom_accel_infer(
    loom_accel_compiled_layer* layer,
    const void* in,
    size_t in_bytes,
    void* out,
    size_t out_bytes,
    double* infer_ms,
    char* err,
    size_t err_len);

const char* loom_accel_last_error(loom_accel_plugin* plugin);

#ifdef __cplusplus
}
#endif

#endif /* LOOM_ACCEL_H */
