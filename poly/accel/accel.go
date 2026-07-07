// Package accel loads vendor accelerator plugins (libloom_accel_*.so) at runtime.
// OpenVINO and other SDKs stay outside Loom; Loom only knows the C ABI in include/loom_accel.h.
// Plugin sources and build: loom/accel/intel/
package accel

import "errors"

// ErrUnavailable is returned when CGO/dlopen is disabled or the plugin is missing.
var ErrUnavailable = errors.New("accel: plugin unavailable (build with CGO_ENABLED=1)")

// LayerDesc identifies a single compiled layer (matches bench_manifest.json).
type LayerDesc struct {
	LayerName string
	DType     string // FP32, FP16, INT8
	SizeLabel string // small, medium, large
}

// CompileResult holds a compiled layer handle and one-time init timings.
type CompileResult struct {
	Layer        CompiledLayer
	CompileMs    float64
	FirstInferMs float64
	InBytes      uintptr
	OutBytes     uintptr
}

// InferResult is one forward pass through a compiled layer.
type InferResult struct {
	InferMs float64
}

// Plugin is an opened vendor device (CPU or NPU).
type Plugin interface {
	VendorID() string
	Device() string
	WeightBytes(desc LayerDesc) (uintptr, error)
	CompileLayer(desc LayerDesc, weightBytes []byte) (*CompileResult, error)
	Close()
}

// CompiledLayer is a compiled graph + infer request (init once, infer many).
type CompiledLayer interface {
	Infer(in, out []byte) (InferResult, error)
	Release()
	InBytes() uintptr
	OutBytes() uintptr
}

// OpenIntel loads libloom_accel_intel.so from path (see DefaultIntelPath).
func OpenIntel(path, device string) (Plugin, error) {
	if err := PrepareRuntime(); err != nil {
		return nil, err
	}
	return openIntelPlugin(path, device)
}

// NPUDAvailable reports whether the Intel plugin sees an NPU device.
func NPUAvailable(path string) bool {
	return intelNPUAvailable(path)
}

// DefaultIntelPath resolves LOOM_ACCEL_INTEL_SO or searches accel/intel/build/ from cwd.
func DefaultIntelPath() string {
	return defaultIntelPath()
}

// OpenQualcomm loads loom_accel_qualcomm.dll from path (see DefaultQualcommPath).
// device is "CPU" (QnnCpu reference backend) or "NPU" (Hexagon HTP).
func OpenQualcomm(path, device string) (Plugin, error) {
	if err := PrepareQualcommRuntime(); err != nil {
		return nil, err
	}
	return openQualcommPlugin(path, device)
}

// QualcommNPUAvailable reports whether the Qualcomm plugin sees a Hexagon NPU.
func QualcommNPUAvailable(path string) bool {
	return qualcommNPUAvailable(path)
}

// DefaultQualcommPath resolves LOOM_ACCEL_QUALCOMM_DLL or searches accel/qualcomm/build/.
func DefaultQualcommPath() string {
	return defaultQualcommPath()
}

// RuntimeLDLibraryPath is OpenVINO + NPU dirs for LD_LIBRARY_PATH (Linux/cgo).
func RuntimeLDLibraryPath() string {
	return runtimeLDLibraryPath()
}
