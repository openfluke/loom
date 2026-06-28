package accel

import "fmt"

// AccelConfig locates vendor plugins (LOOM_ACCEL_INTEL_SO or accel/intel/build/).
type AccelConfig struct {
	IntelSO string
}

// Registry holds opened vendor plugins for a session.
type Registry struct {
	IntelPath string
	IntelCPU  Plugin
	IntelNPU  Plugin
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

// PluginFor returns the plugin for an execution target.
func (r *Registry) PluginFor(t ExecTarget) Plugin {
	if r == nil {
		return nil
	}
	if t == ExecIntelNPU && r.IntelNPU != nil {
		return r.IntelNPU
	}
	return r.IntelCPU
}

// Close releases all plugins.
func (r *Registry) Close() {
	if r == nil {
		return
	}
	if r.IntelCPU != nil {
		r.IntelCPU.Close()
		r.IntelCPU = nil
	}
	if r.IntelNPU != nil {
		r.IntelNPU.Close()
		r.IntelNPU = nil
	}
}
