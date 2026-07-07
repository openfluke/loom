//go:build darwin && cgo

package accel

/*
#cgo darwin LDFLAGS: -ldl
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
    loom_accel_plugin*, const loom_accel_layer_desc*, const void*, size_t,
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
} loom_apple_api;

static int load_apple_api(const char* path, loom_apple_api* api, char* err, size_t err_len) {
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
#define LOAD(field, sym) \
    api->field = (fn_##field)dlsym(api->handle, sym); \
    if (!api->field) { \
        if (err && err_len) { \
            const char* msg = dlerror(); \
            strncpy(err, msg ? msg : "dlsym failed", err_len - 1); \
            err[err_len - 1] = '\0'; \
        } \
        dlclose(api->handle); \
        memset(api, 0, sizeof(*api)); \
        return -1; \
    }
    LOAD(api_version, "loom_accel_api_version");
    LOAD(vendor_id, "loom_accel_vendor_id");
    LOAD(npu_available, "loom_accel_npu_available");
    LOAD(plugin_open, "loom_accel_plugin_open");
    LOAD(plugin_close, "loom_accel_plugin_close");
    LOAD(weight_bytes, "loom_accel_weight_bytes");
    LOAD(compile_layer, "loom_accel_compile_layer");
    LOAD(release_layer, "loom_accel_release_layer");
    LOAD(io_bytes, "loom_accel_layer_io_bytes");
    LOAD(first_infer, "loom_accel_first_infer");
    LOAD(infer, "loom_accel_infer");
    LOAD(last_error, "loom_accel_last_error");
#undef LOAD
    return 0;
}

static void unload_apple_api(loom_apple_api* api) {
    if (api && api->handle) {
        dlclose(api->handle);
    }
    memset(api, 0, sizeof(*api));
}

static int apple_api_version(loom_apple_api* api) { return api->api_version(); }
static const char* apple_vendor_id(loom_apple_api* api) { return api->vendor_id(); }
static int apple_gpu_available(loom_apple_api* api) { return api->npu_available(); }
static loom_accel_plugin* apple_plugin_open(loom_apple_api* api, const char* dev) { return api->plugin_open(dev); }
static void apple_plugin_close(loom_apple_api* api, loom_accel_plugin* p) { api->plugin_close(p); }
static size_t apple_weight_bytes(loom_apple_api* api, const loom_accel_layer_desc* d) { return api->weight_bytes(d); }
static int apple_compile_layer(loom_apple_api* api, loom_accel_plugin* p, const loom_accel_layer_desc* d,
    const void* w, size_t wlen, loom_accel_compiled_layer** out, double* cms, char* err, size_t el) {
    return api->compile_layer(p, d, w, wlen, out, cms, err, el);
}
static void apple_release_layer(loom_apple_api* api, loom_accel_compiled_layer* l) { api->release_layer(l); }
static int apple_io_bytes(loom_apple_api* api, loom_accel_compiled_layer* l, size_t* inb, size_t* outb, char* err, size_t el) {
    return api->io_bytes(l, inb, outb, err, el);
}
static int apple_first_infer(loom_apple_api* api, loom_accel_compiled_layer* l, double* ms, char* err, size_t el) {
    return api->first_infer(l, ms, err, el);
}
static int apple_infer(loom_apple_api* api, loom_accel_compiled_layer* l,
    const void* in, size_t ib, void* out, size_t ob, double* ms, char* err, size_t el) {
    return api->infer(l, in, ib, out, ob, ms, err, el);
}
static const char* apple_last_error(loom_apple_api* api, loom_accel_plugin* p) { return api->last_error(p); }
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"
)

const applePluginName = "libloom_accel_apple.dylib"

var appleGPUCache struct {
	once sync.Once
	ok   bool
	path string
}

type applePlugin struct {
	api    C.loom_apple_api
	plugin *C.loom_accel_plugin
	device string
}

type appleCompiled struct {
	api   *C.loom_apple_api
	layer *C.loom_accel_compiled_layer
	inB   uintptr
	outB  uintptr
}

// PrepareAppleRuntime is a no-op — Metal / Accelerate ship with macOS, no extra
// runtime search path is required (unlike OpenVINO or QNN).
func PrepareAppleRuntime() error { return nil }

func defaultApplePath() string {
	if v := os.Getenv("LOOM_ACCEL_APPLE_DYLIB"); v != "" {
		return v
	}
	if root := os.Getenv("LOOM_ROOT"); root != "" {
		if p := filepath.Join(root, "accel", "apple", "build", applePluginName); appleFileExists(p) {
			return p
		}
	}
	for _, sub := range []string{
		filepath.Join("accel", "apple", "build", applePluginName),
		filepath.Join("accel", "apple", "build", "Release", applePluginName),
	} {
		if p := appleFindUpwardFile(sub); p != "" {
			return p
		}
	}
	return ""
}

func appleGPUAvailable(path string) bool {
	appleGPUCache.once.Do(func() {
		appleGPUCache.path = path
		appleGPUCache.ok = probeAppleGPU(path)
	})
	if path != "" && path != appleGPUCache.path {
		return probeAppleGPU(path)
	}
	return appleGPUCache.ok
}

func probeAppleGPU(path string) bool {
	if path == "" {
		path = defaultApplePath()
	}
	if path == "" {
		return false
	}
	var api C.loom_apple_api
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.load_apple_api(cpath, &api, nil, 0) != 0 {
		return false
	}
	defer C.unload_apple_api(&api)
	return C.apple_gpu_available(&api) != 0
}

func openApplePlugin(path, device string) (Plugin, error) {
	if path == "" {
		path = defaultApplePath()
	}
	if path == "" {
		return nil, fmt.Errorf("%w: libloom_accel_apple.dylib not found (build accel/apple or set LOOM_ACCEL_APPLE_DYLIB)", ErrUnavailable)
	}
	var api C.loom_apple_api
	errBuf := make([]byte, 512)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.load_apple_api(cpath, &api, (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) != 0 {
		return nil, fmt.Errorf("accel: load %s: %s", path, cstr(errBuf))
	}
	if C.apple_api_version(&api) != C.LOOM_ACCEL_API_VERSION {
		C.unload_apple_api(&api)
		return nil, fmt.Errorf("accel: API version mismatch")
	}
	cDev := C.CString(device)
	defer C.free(unsafe.Pointer(cDev))
	plugin := C.apple_plugin_open(&api, cDev)
	if plugin == nil {
		msg := C.GoString(C.apple_last_error(&api, nil))
		C.unload_apple_api(&api)
		return nil, fmt.Errorf("accel: open device %s: %s", device, msg)
	}
	return &applePlugin{api: api, plugin: plugin, device: device}, nil
}

func (p *applePlugin) VendorID() string { return C.GoString(C.apple_vendor_id(&p.api)) }
func (p *applePlugin) Device() string   { return p.device }

func (p *applePlugin) WeightBytes(desc LayerDesc) (uintptr, error) {
	cdesc := cLayerDesc(desc)
	defer freeLayerDesc(cdesc)
	n := C.apple_weight_bytes(&p.api, cdesc)
	return uintptr(n), nil
}

func (p *applePlugin) CompileLayer(desc LayerDesc, weightBytes []byte) (*CompileResult, error) {
	cdesc := cLayerDesc(desc)
	defer freeLayerDesc(cdesc)

	var wptr unsafe.Pointer
	var wlen C.size_t
	if len(weightBytes) > 0 {
		wptr = unsafe.Pointer(&weightBytes[0])
		wlen = C.size_t(len(weightBytes))
	}

	var out *C.loom_accel_compiled_layer
	var compileMs C.double
	errBuf := make([]byte, 512)
	rc := C.apple_compile_layer(
		&p.api,
		p.plugin, cdesc, wptr, wlen, &out, &compileMs,
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("accel: compile %s/%s/%s: %s", desc.LayerName, desc.DType, desc.SizeLabel, cstr(errBuf))
	}

	var inB, outB C.size_t
	if C.apple_io_bytes(&p.api, out, &inB, &outB, nil, 0) != 0 {
		C.apple_release_layer(&p.api, out)
		return nil, fmt.Errorf("accel: io_bytes failed")
	}

	var firstMs C.double
	_ = C.apple_first_infer(&p.api, out, &firstMs, nil, 0)

	compiled := &appleCompiled{
		api:   &p.api,
		layer: out,
		inB:   uintptr(inB),
		outB:  uintptr(outB),
	}
	return &CompileResult{
		Layer:        compiled,
		CompileMs:    float64(compileMs),
		FirstInferMs: float64(firstMs),
		InBytes:      uintptr(inB),
		OutBytes:     uintptr(outB),
	}, nil
}

func (p *applePlugin) Close() {
	if p.plugin != nil {
		C.apple_plugin_close(&p.api, p.plugin)
		p.plugin = nil
	}
	C.unload_apple_api(&p.api)
}

func (c *appleCompiled) InBytes() uintptr  { return c.inB }
func (c *appleCompiled) OutBytes() uintptr { return c.outB }

func (c *appleCompiled) Infer(in, out []byte) (InferResult, error) {
	if uintptr(len(in)) != c.inB || uintptr(len(out)) != c.outB {
		return InferResult{}, fmt.Errorf("accel: buffer size mismatch in=%d want=%d out=%d want=%d", len(in), c.inB, len(out), c.outB)
	}
	var inferMs C.double
	errBuf := make([]byte, 512)
	rc := C.apple_infer(
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

func (c *appleCompiled) Release() {
	if c.layer != nil && c.api != nil {
		C.apple_release_layer(c.api, c.layer)
		c.layer = nil
	}
}

// --- path helpers (self-contained; do not collide with intel/qualcomm loader symbols) ---

func appleFileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func appleFindUpwardFile(rel string) string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for dir := cwd; ; dir = filepath.Dir(dir) {
		candidate := filepath.Join(dir, rel)
		if appleFileExists(candidate) {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return ""
}

// --- C ABI marshalling helpers (darwin build only; linux/windows have their own copies) ---

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
