//go:build windows && cgo

package accel

/*
#cgo CFLAGS: -I${SRCDIR}/include

#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
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
    HMODULE handle;
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
} loom_qcom_api;

static int load_qcom_api(const char* path, loom_qcom_api* api, char* err, size_t err_len) {
    memset(api, 0, sizeof(*api));
    // LOAD_WITH_ALTERED_SEARCH_PATH lets the plugin's own dir resolve co-located deps.
    api->handle = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!api->handle) {
        if (err && err_len) {
            snprintf(err, err_len - 1, "LoadLibrary failed (err=%lu): %s", (unsigned long)GetLastError(), path);
            err[err_len - 1] = '\0';
        }
        return -1;
    }
#define LOAD(field, sym) \
    api->field = (fn_##field)GetProcAddress(api->handle, sym); \
    if (!api->field) { \
        if (err && err_len) { \
            snprintf(err, err_len - 1, "GetProcAddress failed: %s", sym); \
            err[err_len - 1] = '\0'; \
        } \
        FreeLibrary(api->handle); \
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

static void unload_qcom_api(loom_qcom_api* api) {
    if (api && api->handle) {
        FreeLibrary(api->handle);
    }
    memset(api, 0, sizeof(*api));
}

static int qcom_api_version(loom_qcom_api* api) { return api->api_version(); }
static const char* qcom_vendor_id(loom_qcom_api* api) { return api->vendor_id(); }
static int qcom_npu_available(loom_qcom_api* api) { return api->npu_available(); }
static loom_accel_plugin* qcom_plugin_open(loom_qcom_api* api, const char* dev) { return api->plugin_open(dev); }
static void qcom_plugin_close(loom_qcom_api* api, loom_accel_plugin* p) { api->plugin_close(p); }
static size_t qcom_weight_bytes(loom_qcom_api* api, const loom_accel_layer_desc* d) { return api->weight_bytes(d); }
static int qcom_compile_layer(loom_qcom_api* api, loom_accel_plugin* p, const loom_accel_layer_desc* d,
    const void* w, size_t wlen, loom_accel_compiled_layer** out, double* cms, char* err, size_t el) {
    return api->compile_layer(p, d, w, wlen, out, cms, err, el);
}
static void qcom_release_layer(loom_qcom_api* api, loom_accel_compiled_layer* l) { api->release_layer(l); }
static int qcom_io_bytes(loom_qcom_api* api, loom_accel_compiled_layer* l, size_t* inb, size_t* outb, char* err, size_t el) {
    return api->io_bytes(l, inb, outb, err, el);
}
static int qcom_first_infer(loom_qcom_api* api, loom_accel_compiled_layer* l, double* ms, char* err, size_t el) {
    return api->first_infer(l, ms, err, el);
}
static int qcom_infer(loom_qcom_api* api, loom_accel_compiled_layer* l,
    const void* in, size_t ib, void* out, size_t ob, double* ms, char* err, size_t el) {
    return api->infer(l, in, ib, out, ob, ms, err, el);
}
static const char* qcom_last_error(loom_qcom_api* api, loom_accel_plugin* p) { return api->last_error(p); }
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"
)

const qualcommPluginName = "loom_accel_qualcomm.dll"

var qualcommNPUCache struct {
	once sync.Once
	ok   bool
	path string
}

type qualcommPlugin struct {
	api    C.loom_qcom_api
	plugin *C.loom_accel_plugin
	device string
}

type qualcommCompiled struct {
	api    *C.loom_qcom_api
	plugin *C.loom_accel_plugin
	layer  *C.loom_accel_compiled_layer
	inB    uintptr
	outB   uintptr
}

// PrepareQualcommRuntime puts the QNN backend dirs on the process DLL search path
// so the plugin's LoadLibrary("QnnHtp.dll"/"QnnCpu.dll") resolves.
func PrepareQualcommRuntime() error {
	dirs := qualcommRuntimeDirs()
	if len(dirs) == 0 {
		return nil
	}
	cur := os.Getenv("PATH")
	prefix := ""
	for _, d := range dirs {
		prefix += d + string(os.PathListSeparator)
	}
	if cur == "" || len(prefix) == 0 {
		return nil
	}
	return os.Setenv("PATH", prefix+cur)
}

func qualcommRuntimeDirs() []string {
	var dirs []string
	if v := os.Getenv("LOOM_QUALCOMM_RUNTIME"); v != "" {
		dirs = append(dirs, v)
	}
	if v := os.Getenv("QNN_SDK_ROOT"); v != "" {
		dirs = append(dirs, filepath.Join(v, "lib", "aarch64-windows-msvc"))
	}
	if root := qualcommFindUpward("accel", "qualcomm", "deps", "qairt-runtime", "aarch64-windows-msvc"); root != "" {
		dirs = append(dirs, root)
	}
	return dirs
}

func defaultQualcommPath() string {
	if v := os.Getenv("LOOM_ACCEL_QUALCOMM_DLL"); v != "" {
		return v
	}
	if root := os.Getenv("LOOM_ROOT"); root != "" {
		if p := filepath.Join(root, "accel", "qualcomm", "build", qualcommPluginName); qualcommFileExists(p) {
			return p
		}
	}
	// CMake (multi-config) may drop the DLL under build/Release/.
	for _, sub := range []string{
		filepath.Join("accel", "qualcomm", "build", qualcommPluginName),
		filepath.Join("accel", "qualcomm", "build", "Release", qualcommPluginName),
	} {
		if p := qualcommFindUpwardFile(sub); p != "" {
			return p
		}
	}
	return ""
}

func qualcommNPUAvailable(path string) bool {
	qualcommNPUCache.once.Do(func() {
		qualcommNPUCache.path = path
		qualcommNPUCache.ok = probeQualcommNPU(path)
	})
	if path != "" && path != qualcommNPUCache.path {
		return probeQualcommNPU(path)
	}
	return qualcommNPUCache.ok
}

func probeQualcommNPU(path string) bool {
	if err := PrepareQualcommRuntime(); err != nil {
		return false
	}
	if path == "" {
		path = defaultQualcommPath()
	}
	if path == "" {
		return false
	}
	var api C.loom_qcom_api
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.load_qcom_api(cpath, &api, nil, 0) != 0 {
		return false
	}
	defer C.unload_qcom_api(&api)
	return C.qcom_npu_available(&api) != 0
}

func openQualcommPlugin(path, device string) (Plugin, error) {
	if err := PrepareQualcommRuntime(); err != nil {
		return nil, err
	}
	if path == "" {
		path = defaultQualcommPath()
	}
	if path == "" {
		return nil, fmt.Errorf("%w: loom_accel_qualcomm.dll not found (build accel/qualcomm or set LOOM_ACCEL_QUALCOMM_DLL)", ErrUnavailable)
	}
	var api C.loom_qcom_api
	errBuf := make([]byte, 512)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.load_qcom_api(cpath, &api, (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) != 0 {
		return nil, fmt.Errorf("accel: load %s: %s", path, cstr(errBuf))
	}
	if C.qcom_api_version(&api) != C.LOOM_ACCEL_API_VERSION {
		C.unload_qcom_api(&api)
		return nil, fmt.Errorf("accel: API version mismatch")
	}
	cDev := C.CString(device)
	defer C.free(unsafe.Pointer(cDev))
	plugin := C.qcom_plugin_open(&api, cDev)
	if plugin == nil {
		msg := C.GoString(C.qcom_last_error(&api, nil))
		C.unload_qcom_api(&api)
		return nil, fmt.Errorf("accel: open device %s: %s", device, msg)
	}
	return &qualcommPlugin{api: api, plugin: plugin, device: device}, nil
}

func (p *qualcommPlugin) VendorID() string { return C.GoString(C.qcom_vendor_id(&p.api)) }
func (p *qualcommPlugin) Device() string   { return p.device }

func (p *qualcommPlugin) WeightBytes(desc LayerDesc) (uintptr, error) {
	cdesc := cLayerDesc(desc)
	defer freeLayerDesc(cdesc)
	n := C.qcom_weight_bytes(&p.api, cdesc)
	return uintptr(n), nil
}

func (p *qualcommPlugin) CompileLayer(desc LayerDesc, weightBytes []byte) (*CompileResult, error) {
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
	rc := C.qcom_compile_layer(
		&p.api,
		p.plugin, cdesc, wptr, wlen, &out, &compileMs,
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("accel: compile %s/%s/%s: %s", desc.LayerName, desc.DType, desc.SizeLabel, cstr(errBuf))
	}

	var inB, outB C.size_t
	if C.qcom_io_bytes(&p.api, out, &inB, &outB, nil, 0) != 0 {
		C.qcom_release_layer(&p.api, out)
		return nil, fmt.Errorf("accel: io_bytes failed")
	}

	var firstMs C.double
	_ = C.qcom_first_infer(&p.api, out, &firstMs, nil, 0)

	compiled := &qualcommCompiled{
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

func (p *qualcommPlugin) Close() {
	if p.plugin != nil {
		C.qcom_plugin_close(&p.api, p.plugin)
		p.plugin = nil
	}
	C.unload_qcom_api(&p.api)
}

func (c *qualcommCompiled) InBytes() uintptr  { return c.inB }
func (c *qualcommCompiled) OutBytes() uintptr { return c.outB }

func (c *qualcommCompiled) Infer(in, out []byte) (InferResult, error) {
	if uintptr(len(in)) != c.inB || uintptr(len(out)) != c.outB {
		return InferResult{}, fmt.Errorf("accel: buffer size mismatch in=%d want=%d out=%d want=%d", len(in), c.inB, len(out), c.outB)
	}
	var inferMs C.double
	errBuf := make([]byte, 512)
	rc := C.qcom_infer(
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

func (c *qualcommCompiled) Release() {
	if c.layer != nil && c.api != nil {
		C.qcom_release_layer(c.api, c.layer)
		c.layer = nil
	}
}

// --- path helpers (self-contained; do not collide with intel loader symbols) ---

func qualcommFileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func qualcommFindUpward(parts ...string) string {
	return qualcommFindUpwardFile(filepath.Join(parts...))
}

func qualcommFindUpwardFile(rel string) string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for dir := cwd; ; dir = filepath.Dir(dir) {
		candidate := filepath.Join(dir, rel)
		if qualcommFileExists(candidate) {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return ""
}

// --- C ABI marshalling helpers (Windows build only; Linux copies live in plugin_linux.go) ---

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
