//go:build linux && cgo

package accel

/*
#cgo linux LDFLAGS: -ldl
#cgo CFLAGS: -I${SRCDIR}/include

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "loom_accel.h"

typedef int (*fn_api_version)(void);
typedef const char* (*fn_vendor_id)(void);
typedef int (*fn_npu_available)(void);
typedef loom_accel_plugin* (*fn_plugin_open)(const char*);
typedef void (*fn_plugin_close)(loom_accel_plugin*);
typedef size_t (*fn_weight_bytes)(const loom_accel_layer_desc*);
typedef int (*fn_compile_layer)(
    loom_accel_plugin*, const loom_accel_layer_desc*, const float*, size_t,
    loom_accel_compiled_layer**, double*, char*, size_t);
typedef void (*fn_release_layer)(loom_accel_compiled_layer*);
typedef int (*fn_io_bytes)(loom_accel_compiled_layer*, size_t*, size_t*, char*, size_t);
typedef int (*fn_first_infer)(loom_accel_compiled_layer*, double*, char*, size_t);
typedef int (*fn_infer)(loom_accel_compiled_layer*, const void*, size_t, void*, size_t, double*, char*, size_t);
typedef const char* (*fn_last_error)(loom_accel_plugin*);

typedef struct {
    void* handle;
    fn_api_version api_version;
    fn_vendor_id vendor_id;
    fn_npu_available npu_available;
    fn_plugin_open plugin_open;
    fn_plugin_close plugin_close;
    fn_weight_bytes weight_bytes;
    fn_compile_layer compile_layer;
    fn_release_layer release_layer;
    fn_io_bytes io_bytes;
    fn_first_infer first_infer;
    fn_infer infer;
    fn_last_error last_error;
} loom_intel_api;

static int load_intel_api(const char* path, loom_intel_api* api, char* err, size_t err_len) {
    memset(api, 0, sizeof(*api));
    api->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api->handle) {
        if (err && err_len) {
            const char* msg = dlerror();
            strncpy(err, msg ? msg : "dlopen failed", err_len - 1);
            err[err_len - 1] = '\0';
        }
        return -1;
    }
#define LOAD(name) \
    api->name = (fn_##name)dlsym(api->handle, "loom_accel_" #name); \
    if (!api->name) { \
        if (err && err_len) { \
            const char* msg = dlerror(); \
            strncpy(err, msg ? msg : "dlsym failed", err_len - 1); \
            err[err_len - 1] = '\0'; \
        } \
        dlclose(api->handle); \
        memset(api, 0, sizeof(*api)); \
        return -1; \
    }
    LOAD(api_version);
    LOAD(vendor_id);
    LOAD(npu_available);
    LOAD(plugin_open);
    LOAD(plugin_close);
    LOAD(weight_bytes);
    LOAD(compile_layer);
    LOAD(release_layer);
    api->io_bytes = (fn_io_bytes)dlsym(api->handle, "loom_accel_layer_io_bytes");
    if (!api->io_bytes) {
        if (err && err_len) {
            const char* msg = dlerror();
            strncpy(err, msg ? msg : "dlsym failed: layer_io_bytes", err_len - 1);
            err[err_len - 1] = '\0';
        }
        dlclose(api->handle);
        memset(api, 0, sizeof(*api));
        return -1;
    }
    LOAD(first_infer);
    LOAD(infer);
    LOAD(last_error);
#undef LOAD
    return 0;
}

static void unload_intel_api(loom_intel_api* api) {
    if (api && api->handle) {
        dlclose(api->handle);
    }
    memset(api, 0, sizeof(*api));
}

static int intel_api_version(loom_intel_api* api) { return api->api_version(); }
static const char* intel_vendor_id(loom_intel_api* api) { return api->vendor_id(); }
static int intel_npu_available(loom_intel_api* api) { return api->npu_available(); }
static loom_accel_plugin* intel_plugin_open(loom_intel_api* api, const char* dev) { return api->plugin_open(dev); }
static void intel_plugin_close(loom_intel_api* api, loom_accel_plugin* p) { api->plugin_close(p); }
static size_t intel_weight_bytes(loom_intel_api* api, const loom_accel_layer_desc* d) { return api->weight_bytes(d); }
static int intel_compile_layer(loom_intel_api* api, loom_accel_plugin* p, const loom_accel_layer_desc* d,
    const float* w, size_t wc, loom_accel_compiled_layer** out, double* cms, char* err, size_t el) {
    return api->compile_layer(p, d, w, wc, out, cms, err, el);
}
static void intel_release_layer(loom_intel_api* api, loom_accel_compiled_layer* l) { api->release_layer(l); }
static int intel_io_bytes(loom_intel_api* api, loom_accel_compiled_layer* l, size_t* inb, size_t* outb, char* err, size_t el) {
    return api->io_bytes(l, inb, outb, err, el);
}
static int intel_first_infer(loom_intel_api* api, loom_accel_compiled_layer* l, double* ms, char* err, size_t el) {
    return api->first_infer(l, ms, err, el);
}
static int intel_infer(loom_intel_api* api, loom_accel_compiled_layer* l,
    const void* in, size_t ib, void* out, size_t ob, double* ms, char* err, size_t el) {
    return api->infer(l, in, ib, out, ob, ms, err, el);
}
static const char* intel_last_error(loom_intel_api* api, loom_accel_plugin* p) { return api->last_error(p); }

static int preload_shared_lib(const char* path) {
    if (!path || !path[0]) {
        return 0;
    }
    void* h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        return -1;
    }
    return 0;
}
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"
)

type intelPlugin struct {
	api    C.loom_intel_api
	plugin *C.loom_accel_plugin
	device string
}

type intelCompiled struct {
	api    *C.loom_intel_api
	plugin *C.loom_accel_plugin
	layer  *C.loom_accel_compiled_layer
	inB    uintptr
	outB   uintptr
}

func defaultIntelPath() string {
	if v := os.Getenv("LOOM_ACCEL_INTEL_SO"); v != "" {
		return v
	}
	if root := os.Getenv("CHAOSGLUE_ROOT"); root != "" {
		return filepath.Join(root, "npu/intel/cabi/build/libloom_accel_intel.so")
	}
	for _, rel := range []string{
		filepath.Join("..", "..", "npu", "intel", "cabi", "build", "libloom_accel_intel.so"), // lucy/
		filepath.Join("..", "npu", "intel", "cabi", "build", "libloom_accel_intel.so"),       // loom/
	} {
		if abs, err := filepath.Abs(rel); err == nil {
			if _, err := os.Stat(abs); err == nil {
				return abs
			}
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "git/chaosglue/npu/intel/cabi/build/libloom_accel_intel.so")
}

func intelNPUAvailable(path string) bool {
	if err := PrepareRuntime(); err != nil {
		return false
	}
	if path == "" {
		path = defaultIntelPath()
	}
	var api C.loom_intel_api
	if C.load_intel_api(C.CString(path), &api, nil, 0) != 0 {
		return false
	}
	defer C.unload_intel_api(&api)
	return C.intel_npu_available(&api) != 0
}

func preloadSharedLib(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.preload_shared_lib(cpath) != 0 {
		return fmt.Errorf("%s", C.GoString(C.dlerror()))
	}
	return nil
}

func openIntelPlugin(path, device string) (Plugin, error) {
	if err := PrepareRuntime(); err != nil {
		return nil, err
	}
	if path == "" {
		path = defaultIntelPath()
	}
	var api C.loom_intel_api
	errBuf := make([]byte, 512)
	if C.load_intel_api(C.CString(path), &api, (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) != 0 {
		err := fmt.Errorf("accel: load %s: %s", path, cstr(errBuf))
		if hint := runtimeHint(err); hint != "" {
			err = fmt.Errorf("%w%s", err, hint)
		}
		return nil, err
	}
	if C.intel_api_version(&api) != C.LOOM_ACCEL_API_VERSION {
		C.unload_intel_api(&api)
		return nil, fmt.Errorf("accel: API version mismatch")
	}
	cDev := C.CString(device)
	defer C.free(unsafe.Pointer(cDev))
	plugin := C.intel_plugin_open(&api, cDev)
	if plugin == nil {
		msg := C.GoString(C.intel_last_error(&api, nil))
		C.unload_intel_api(&api)
		return nil, fmt.Errorf("accel: open device %s: %s", device, msg)
	}
	return &intelPlugin{api: api, plugin: plugin, device: device}, nil
}

func (p *intelPlugin) VendorID() string {
	return C.GoString(C.intel_vendor_id(&p.api))
}

func (p *intelPlugin) Device() string {
	return p.device
}

func (p *intelPlugin) WeightBytes(desc LayerDesc) (uintptr, error) {
	cdesc := cLayerDesc(desc)
	defer freeLayerDesc(cdesc)
	n := C.intel_weight_bytes(&p.api, cdesc)
	return uintptr(n), nil
}

func (p *intelPlugin) CompileLayer(desc LayerDesc, weights []float32) (*CompileResult, error) {
	cdesc := cLayerDesc(desc)
	defer freeLayerDesc(cdesc)

	var wptr *C.float
	var wcount C.size_t
	if len(weights) > 0 {
		wptr = (*C.float)(unsafe.Pointer(&weights[0]))
		wcount = C.size_t(len(weights))
	}

	var out *C.loom_accel_compiled_layer
	var compileMs C.double
	errBuf := make([]byte, 512)
	rc := C.intel_compile_layer(
		&p.api,
		p.plugin, cdesc, wptr, wcount, &out, &compileMs,
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("accel: compile %s/%s/%s: %s", desc.LayerName, desc.DType, desc.SizeLabel, cstr(errBuf))
	}

	var inB, outB C.size_t
	if C.intel_io_bytes(&p.api, out, &inB, &outB, nil, 0) != 0 {
		C.intel_release_layer(&p.api, out)
		return nil, fmt.Errorf("accel: io_bytes failed")
	}

	var firstMs C.double
	_ = C.intel_first_infer(&p.api, out, &firstMs, nil, 0)

	compiled := &intelCompiled{
		api:    &p.api,
		plugin: p.plugin,
		layer:  out,
		inB:    uintptr(inB),
		outB:   uintptr(outB),
	}

	return &CompileResult{
		Layer:        compiled,
		CompileMs:    float64(compileMs),
		FirstInferMs: float64(firstMs),
		InBytes:      uintptr(inB),
		OutBytes:     uintptr(outB),
	}, nil
}

func (p *intelPlugin) Close() {
	if p.plugin != nil {
		C.intel_plugin_close(&p.api, p.plugin)
		p.plugin = nil
	}
	C.unload_intel_api(&p.api)
}

func (c *intelCompiled) InBytes() uintptr  { return c.inB }
func (c *intelCompiled) OutBytes() uintptr { return c.outB }

func (c *intelCompiled) Infer(in, out []byte) (InferResult, error) {
	if uintptr(len(in)) != c.inB || uintptr(len(out)) != c.outB {
		return InferResult{}, fmt.Errorf("accel: buffer size mismatch in=%d want=%d out=%d want=%d", len(in), c.inB, len(out), c.outB)
	}
	var inferMs C.double
	errBuf := make([]byte, 512)
	rc := C.intel_infer(
		c.api,
		c.layer,
		unsafe.Pointer(&in[0]), C.size_t(len(in)),
		unsafe.Pointer(&out[0]), C.size_t(len(out)),
		&inferMs,
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)),
	)
	if rc != 0 {
		return InferResult{}, fmt.Errorf("accel: infer: %s", cstr(errBuf))
	}
	return InferResult{InferMs: float64(inferMs)}, nil
}

func (c *intelCompiled) Release() {
	if c.layer != nil && c.api != nil {
		C.intel_release_layer(c.api, c.layer)
		c.layer = nil
	}
}

func cLayerDesc(desc LayerDesc) *C.loom_accel_layer_desc {
	d := (*C.loom_accel_layer_desc)(C.malloc(C.size_t(unsafe.Sizeof(C.loom_accel_layer_desc{}))))
	d.layer_name = C.CString(desc.LayerName)
	d.dtype = C.CString(desc.DType)
	d.size_label = C.CString(desc.SizeLabel)
	return d
}

func freeLayerDesc(d *C.loom_accel_layer_desc) {
	if d == nil {
		return
	}
	C.free(unsafe.Pointer(d.layer_name))
	C.free(unsafe.Pointer(d.dtype))
	C.free(unsafe.Pointer(d.size_label))
	C.free(unsafe.Pointer(d))
}

func cstr(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
