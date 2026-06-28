/* Vendor-neutral C ABI — keep in sync with accel/intel/include/loom_accel.h */
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
    const char* layer_name;
    const char* dtype;
    const char* size_label;
} loom_accel_layer_desc;

int loom_accel_api_version(void);
const char* loom_accel_vendor_id(void);
int loom_accel_npu_available(void);

loom_accel_plugin* loom_accel_plugin_open(const char* device);
void loom_accel_plugin_close(loom_accel_plugin* plugin);

size_t loom_accel_weight_bytes(const loom_accel_layer_desc* desc);

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

int loom_accel_first_infer(
    loom_accel_compiled_layer* layer,
    double* first_infer_ms,
    char* err,
    size_t err_len);

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

#endif
