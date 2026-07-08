package accel

import "fmt"

// AccelConfig locates vendor plugins.
//   IntelSO    — libloom_accel_intel.so       (LOOM_ACCEL_INTEL_SO / accel/intel/build)
//   QualcommSO — loom_accel_qualcomm.dll      (LOOM_ACCEL_QUALCOMM_DLL / accel/qualcomm/build)
//   AppleSO    — libloom_accel_apple.dylib    (LOOM_ACCEL_APPLE_DYLIB / accel/apple/build)
type AccelConfig struct {
	IntelSO    string
	QualcommSO string
	AppleSO    string
}

// Registry holds opened vendor plugins for a session.
type Registry struct {
	IntelPath string
	IntelCPU  Plugin
	IntelNPU  Plugin

	QualcommPath string
	QualcommCPU  Plugin
	QualcommNPU  Plugin

	ApplePath string
	AppleCPU  Plugin
	AppleGPU  Plugin
}

// Discover opens Intel CPU (+ NPU when available).
func Discover(cfg AccelConfig) (*Registry, error) {
	path := cfg.IntelSO
	if path == "" {
		path = DefaultIntelPath()
	}
	cpu, err := OpenIntel(path, "CPU")
	if err != nil {
		return nil, fmt.Errorf("accel discover intel CPU: %w", err)
	}
	var npu Plugin
	if NPUAvailable(path) {
		npu, _ = OpenIntel(path, "NPU")
	}
	return &Registry{IntelPath: path, IntelCPU: cpu, IntelNPU: npu}, nil
}

// DiscoverQualcomm opens the Qualcomm QNN plugin: CPU reference backend (parity
// anchor) plus the Hexagon NPU/HTP backend when available.
func DiscoverQualcomm(cfg AccelConfig) (*Registry, error) {
	path := cfg.QualcommSO
	if path == "" {
		path = DefaultQualcommPath()
	}
	cpu, err := OpenQualcomm(path, "CPU")
	if err != nil {
		return nil, fmt.Errorf("accel discover qualcomm CPU: %w", err)
	}
	var npu Plugin
	if QualcommNPUAvailable(path) {
		npu, _ = OpenQualcomm(path, "NPU")
	}
	return &Registry{QualcommPath: path, QualcommCPU: cpu, QualcommNPU: npu}, nil
}

// DiscoverApple opens the Apple Metal/Accelerate plugin: a portable CPU reference
// backend (parity anchor) plus the Metal GPU (MPSGraph) backend when a Metal device
// is available. Apple's Neural Engine (ANE) is reached indirectly through Metal/Core ML.
func DiscoverApple(cfg AccelConfig) (*Registry, error) {
	path := cfg.AppleSO
	if path == "" {
		path = DefaultApplePath()
	}
	cpu, err := OpenApple(path, "CPU")
	if err != nil {
		return nil, fmt.Errorf("accel discover apple CPU: %w", err)
	}
	var gpu Plugin
	if AppleGPUAvailable(path) {
		gpu, _ = OpenApple(path, "GPU")
	}
	return &Registry{ApplePath: path, AppleCPU: cpu, AppleGPU: gpu}, nil
}

// PluginFor returns the plugin for an execution target.
func (r *Registry) PluginFor(t ExecTarget) Plugin {
	if r == nil {
		return nil
	}
	switch t {
	case ExecQualcommNPU:
		if r.QualcommNPU != nil {
			return r.QualcommNPU
		}
		return r.QualcommCPU
	case ExecQualcommCPU:
		return r.QualcommCPU
	case ExecIntelNPU:
		if r.IntelNPU != nil {
			return r.IntelNPU
		}
		return r.IntelCPU
	case ExecAppleGPU:
		if r.AppleGPU != nil {
			return r.AppleGPU
		}
		return r.AppleCPU
	case ExecAppleCPU:
		return r.AppleCPU
	default:
		return r.IntelCPU
	}
}

// Close releases all plugins.
func (r *Registry) Close() {
	if r == nil {
		return
	}
	for _, p := range []*Plugin{
		&r.IntelCPU, &r.IntelNPU,
		&r.QualcommCPU, &r.QualcommNPU,
		&r.AppleCPU, &r.AppleGPU,
	} {
		if *p != nil {
			(*p).Close()
			*p = nil
		}
	}
}
