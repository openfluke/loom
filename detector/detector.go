package detector

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

/* ---------- public API ---------- */

// Report is a portable summary of the current adapter/device caps.
type Report struct {
	WhenISO     string            `json:"when_iso"`
	Runtime     string            `json:"runtime"` // "native" or "wasm" (best-effort)
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
	// Conservative 1D workgroup that should run everywhere.
	WorkgroupX uint32 `json:"workgroup_x"`
	WorkgroupY uint32 `json:"workgroup_y"`
	WorkgroupZ uint32 `json:"workgroup_z"`

	// Tiling hints for big ops — refine per-op later.
	TileX uint32 `json:"tile_x"`
	TileY uint32 `json:"tile_y"`

	// Soft VRAM/heap budget in bytes for staging + temps.
	BudgetBytes uint64 `json:"budget_bytes"`
}

// DetectJSON runs a probe and returns the JSON string.
func DetectJSON() (string, error) {
	rep, err := Detect()
	if err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Detect probes the default adapter/device and synthesizes a report.
func Detect() (*Report, error) {
	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return nil, fmt.Errorf("wgpu.CreateInstance returned nil")
	}
	defer inst.Release()

	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		return nil, fmt.Errorf("request adapter: %w", err)
	}
	if adapter == nil {
		return nil, fmt.Errorf("no adapter")
	}
	defer adapter.Release()

	// ✅ Most forks expose GetInfo(), not GetAdapterInfo()/GetProperties()
	info := adapter.GetInfo()
	limits := adapter.GetLimits()

	// Enumerate features (adapter-level).
	var feats []string
	for _, f := range adapter.EnumerateFeatures() {
		feats = append(feats, featureName(f))
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredFeatures: nil,
	})
	if err != nil {
		return nil, fmt.Errorf("request device: %w", err)
	}
	defer device.Release()

	wgX, wgY, wgZ := chooseWorkgroup(limits)
	tileX, tileY := chooseTile(limits, wgX, wgY, wgZ)

	budget := uint64(128 * 1024 * 1024)
	if mbStr := os.Getenv("LOOM_BUDGET_MB"); mbStr != "" {
		if mb, err := strconv.Atoi(mbStr); err == nil && mb > 0 {
			budget = uint64(mb) * 1024 * 1024
		}
	}

	rep := &Report{
		WhenISO:     time.Now().UTC().Format(time.RFC3339),
		Runtime:     detectRuntime(),
		Backend:     backendName(info.BackendType),
		AdapterType: adapterTypeName(info.AdapterType),
		VendorID:    fmt.Sprintf("0x%04x", info.VendorId),
		DeviceID:    fmt.Sprintf("0x%04x", info.DeviceId),
		Name:        strings.TrimSpace(info.Name),
		Driver:      strings.TrimSpace(info.DriverDescription),
		Limits: Limits{
			MaxComputeInvocationsPerWorkgroup: limits.Limits.MaxComputeInvocationsPerWorkgroup,
			MaxComputeWorkgroupSizeX:          limits.Limits.MaxComputeWorkgroupSizeX,
			MaxComputeWorkgroupSizeY:          limits.Limits.MaxComputeWorkgroupSizeY,
			MaxComputeWorkgroupSizeZ:          limits.Limits.MaxComputeWorkgroupSizeZ,
			MaxComputeWorkgroupsPerDimension:  limits.Limits.MaxComputeWorkgroupsPerDimension,
			MaxComputeWorkgroupStorageSize:    limits.Limits.MaxComputeWorkgroupStorageSize,
			MaxStorageBufferBindingSize:       limits.Limits.MaxStorageBufferBindingSize,
			MaxBufferSize:                     limits.Limits.MaxBufferSize,
		},
		Features: feats,
		Recommended: Recommendations{
			WorkgroupX: wgX, WorkgroupY: wgY, WorkgroupZ: wgZ,
			TileX: tileX, TileY: tileY,
			BudgetBytes: budget,
		},
		Env: pickEnv([]string{"LOOM_BUDGET_MB"}),
	}

	_ = device
	return rep, nil
}

/* ---------- helpers ---------- */

func chooseWorkgroup(l wgpu.SupportedLimits) (uint32, uint32, uint32) {
	maxX := l.Limits.MaxComputeWorkgroupSizeX
	maxTot := l.Limits.MaxComputeInvocationsPerWorkgroup

	candidates := []uint32{256, 128, 64, 32, 16, 8, 4, 1}
	for _, c := range candidates {
		if c <= maxX && c <= maxTot {
			return c, 1, 1
		}
	}
	// absolute portability fallback
	return 1, 1, 1
}

func chooseTile(l wgpu.SupportedLimits, wgX, wgY, wgZ uint32) (uint32, uint32) {
	// Keep tile ~ a few workgroups worth, capped by per-dimension dispatch limits.
	tx := wgX * 8
	if tx < 1 {
		tx = 1
	}
	if tx > l.Limits.MaxComputeWorkgroupsPerDimension {
		tx = l.Limits.MaxComputeWorkgroupsPerDimension
	}

	ty := uint32(1)
	if wgY > 1 {
		ty = wgY * 8
		if ty > l.Limits.MaxComputeWorkgroupsPerDimension {
			ty = l.Limits.MaxComputeWorkgroupsPerDimension
		}
	}
	return tx, ty
}

func featureName(f wgpu.FeatureName) string     { return f.String() }
func backendName(b wgpu.BackendType) string     { return b.String() }
func adapterTypeName(t wgpu.AdapterType) string { return t.String() }

func detectRuntime() string {
	// Simple heuristic; use build tags later if you like.
	if runtime.GOOS == "js" {
		return "wasm"
	}
	return "native"
}

func pickEnv(keys []string) map[string]string {
	out := map[string]string{}
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
