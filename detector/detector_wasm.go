//go:build js && wasm
// +build js,wasm

package detector

import "encoding/json"

// Report stub for WASM (types defined but not populated)
type Report struct {
	Key         string            `json:"key"`
	WhenISO     string            `json:"when_iso"`
	Runtime     string            `json:"runtime"`
	Backend     string            `json:"backend"`
	AdapterType string            `json:"adapter_type"`
	VendorID    string            `json:"vendor_id_hex"`
	DeviceID    string            `json:"device_id_hex"`
	Name        string            `json:"name"`
	Driver      string            `json:"driver"`
	Recommended Recommendations   `json:"recommended"`
	Limits      Limits            `json:"limits"`
	Features    []string          `json:"features"`
	Env         map[string]string `json:"env,omitempty"`
}

type Limits struct {
	MaxComputeInvocationsPerWorkgroup uint32 `json:"max_compute_invocations_per_workgroup"`
	MaxComputeWorkgroupSizeX          uint32 `json:"max_compute_workgroup_size_x"`
	MaxComputeWorkgroupSizeY          uint32 `json:"max_compute_workgroup_size_y"`
	MaxComputeWorkgroupSizeZ          uint32 `json:"max_compute_workgroup_size_z"`
	MaxComputeWorkgroupsPerDimension  uint32 `json:"max_compute_workgroups_per_dimension"`
	MaxComputeWorkgroupStorageSize    uint32 `json:"max_compute_workgroup_storage_size"`
	MaxStorageBufferBindingSize       uint64 `json:"max_storage_buffer_binding_size"`
	MaxBufferSize                     uint64 `json:"max_buffer_size"`
}

type Recommendations struct {
	WorkgroupX  uint32 `json:"workgroup_x"`
	WorkgroupY  uint32 `json:"workgroup_y"`
	WorkgroupZ  uint32 `json:"workgroup_z"`
	TileX       uint32 `json:"tile_x"`
	TileY       uint32 `json:"tile_y"`
	BudgetBytes uint64 `json:"budget_bytes"`
}

// DetectAllJSON returns empty array for WASM builds (no GPU detection available)
func DetectAllJSON() (string, error) {
	return "[]", nil
}

// DetectAll returns empty slice for WASM builds
func DetectAll() ([]Report, error) {
	return []Report{}, nil
}

// DetectBest returns nil for WASM builds
func DetectBest() (*Report, error) {
	return nil, nil
}

// DetectBestJSON returns empty object for WASM builds
func DetectBestJSON() (string, error) {
	empty := map[string]interface{}{
		"error": "GPU detection not available in WASM",
	}
	data, _ := json.Marshal(empty)
	return string(data), nil
}

// Detect returns nil for WASM builds (alias for DetectBest)
func Detect() (*Report, error) {
	return nil, nil
}
