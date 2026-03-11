//go:build !wasm
// +build !wasm

package detector

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

/* ---------- public types ---------- */

type Report struct {
	Key         string            `json:"key"` // stable selection key: backend/vendor/device/name
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
	WorkgroupX  uint32 `json:"workgroup_x"`
	WorkgroupY  uint32 `json:"workgroup_y"`
	WorkgroupZ  uint32 `json:"workgroup_z"`
	TileX       uint32 `json:"tile_x"`
	TileY       uint32 `json:"tile_y"`
	BudgetBytes uint64 `json:"budget_bytes"`
}

/* ---------- top-level API ---------- */

// Detect returns one adapter report according to LOOM_GPU policy (default: discrete/auto).
func Detect() (*Report, error) {
	policy := strings.ToLower(strings.TrimSpace(os.Getenv("LOOM_GPU")))
	if policy == "scan" {
		all, err := DetectAll()
		if err != nil {
			return nil, err
		}
		if len(all) == 0 {
			return nil, fmt.Errorf("no adapters found")
		}
		// default pick: prefer discrete if present
		idx := pickIndex(all, "discrete")
		return &all[idx], nil
	}
	// single-pick by policy
	return detectByPolicy(policy)
}

func DetectJSON() (string, error) {
	r, err := Detect()
	if err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// DetectAll scans HighPerformance, LowPower (and optional fallback) and returns unique adapters.
func DetectAll() ([]Report, error) {
	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return nil, fmt.Errorf("wgpu.CreateInstance returned nil")
	}
	defer inst.Release()

	type pref struct {
		pp  wgpu.PowerPreference
		fb  bool // ForceFallbackAdapter
		tag string
	}
	prefs := []pref{
		{wgpu.PowerPreferenceHighPerformance, false, "discrete"},
		{wgpu.PowerPreferenceLowPower, false, "integrated"},
	}
	if os.Getenv("LOOM_INCLUDE_FALLBACK") == "1" {
		// software/fallback where available (e.g., D3D12 WARP)
		prefs = append(prefs, pref{wgpu.PowerPreferenceLowPower, true, "fallback"})
	}

	seen := map[string]bool{}
	var out []Report

	for _, p := range prefs {
		ad, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
			PowerPreference:      p.pp,
			ForceFallbackAdapter: p.fb,
		})
		if err != nil || ad == nil {
			continue
		}

		rep, key := buildReportForAdapter(ad)
		ad.Release()

		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, rep)
	}

	// Stable order: discrete first, then integrated, then fallback; tie-break by name.
	sort.SliceStable(out, func(i, j int) bool {
		rank := func(t string) int {
			switch t {
			case "discrete-gpu":
				return 0
			case "integrated-gpu":
				return 1
			default:
				return 2
			}
		}
		ri, rj := rank(out[i].AdapterType), rank(out[j].AdapterType)
		if ri != rj {
			return ri < rj
		}
		return out[i].Name < out[j].Name
	})

	return out, nil
}

func DetectAllJSON() (string, error) {
	reps, err := DetectAll()
	if err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(reps, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

/* ---------- selection helpers ---------- */

// SelectAdapterByKey re-requests an adapter matching a Report.Key.
func SelectAdapterByKey(key string) (*wgpu.Adapter, error) {
	all, err := DetectAll()
	if err != nil {
		return nil, err
	}
	var match *Report
	for i := range all {
		if strings.EqualFold(all[i].Key, key) {
			match = &all[i]
			break
		}
	}
	if match == nil {
		return nil, fmt.Errorf("no adapter matches key: %s", key)
	}
	// Map key back to a power preference guess; if ambiguous, prefer discrete.
	pp := wgpu.PowerPreferenceHighPerformance
	if strings.Contains(match.AdapterType, "integrated") {
		pp = wgpu.PowerPreferenceLowPower
	}
	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return nil, fmt.Errorf("wgpu.CreateInstance returned nil")
	}
	// We can only *hint*; platform may still return same adapter.
	ad, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: pp})
	if err != nil || ad == nil {
		inst.Release()
		return nil, fmt.Errorf("failed to re-request adapter for key %q", key)
	}
	// Caller must Release both adapter and instance when done.
	return ad, nil
}

func detectByPolicy(policy string) (*Report, error) {
	inst := wgpu.CreateInstance(nil)
	if inst == nil {
		return nil, fmt.Errorf("wgpu.CreateInstance returned nil")
	}
	defer inst.Release()

	pp := wgpu.PowerPreferenceHighPerformance // default: discrete/auto
	switch strings.ToLower(strings.TrimSpace(policy)) {
	case "integrated", "lowpower", "igpu":
		pp = wgpu.PowerPreferenceLowPower
	}

	ad, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: pp})
	if err != nil || ad == nil {
		return nil, fmt.Errorf("request adapter failed")
	}
	defer ad.Release()

	rep, _ := buildReportForAdapter(ad)
	return &rep, nil
}

/* ---------- internal: build a report ---------- */

func buildReportForAdapter(adapter *wgpu.Adapter) (Report, string) {
	info := adapter.GetInfo()
	limits := adapter.GetLimits()

	// Enumerate features.
	var feats []string
	for _, f := range adapter.EnumerateFeatures() {
		feats = append(feats, f.String())
	}

	// Make a minimal device (optional; here just sanity).
	device, _ := adapter.RequestDevice(&wgpu.DeviceDescriptor{})
	if device != nil {
		defer device.Release()
	}

	wgX, wgY, wgZ := chooseWorkgroup(limits)
	tileX, tileY := chooseTile(limits, wgX, wgY, wgZ)

	budget := uint64(128 * 1024 * 1024)
	if mbStr := os.Getenv("LOOM_BUDGET_MB"); mbStr != "" {
		if mb, err := strconv.Atoi(mbStr); err == nil && mb > 0 {
			budget = uint64(mb) * 1024 * 1024
		}
	}

	key := makeKey(info)

	rep := Report{
		Key:         key,
		WhenISO:     time.Now().UTC().Format(time.RFC3339),
		Runtime:     detectRuntime(),
		Backend:     info.BackendType.String(),
		AdapterType: info.AdapterType.String(),
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
			WorkgroupX:  wgX,
			WorkgroupY:  wgY,
			WorkgroupZ:  wgZ,
			TileX:       tileX,
			TileY:       tileY,
			BudgetBytes: budget,
		},
		Env: pickEnv([]string{"LOOM_GPU", "LOOM_BUDGET_MB", "LOOM_INCLUDE_FALLBACK"}),
	}
	return rep, key
}

func makeKey(info wgpu.AdapterInfo) string {
	// Backend/vendor/device/name is plenty stable for selection within a run.
	// Example: "vulkan/0x10de/0x25a2/NVIDIA GeForce RTX 3050 Laptop GPU"
	parts := []string{
		strings.ToLower(info.BackendType.String()),
		fmt.Sprintf("0x%04x", info.VendorId),
		fmt.Sprintf("0x%04x", info.DeviceId),
		strings.ReplaceAll(strings.TrimSpace(info.Name), "/", "_"),
	}
	return strings.Join(parts, "/")
}

/* ---------- misc helpers ---------- */

func chooseWorkgroup(l wgpu.SupportedLimits) (uint32, uint32, uint32) {
	maxX := l.Limits.MaxComputeWorkgroupSizeX
	maxTot := l.Limits.MaxComputeInvocationsPerWorkgroup
	for _, c := range []uint32{256, 128, 64, 32, 16, 8, 4, 1} {
		if c <= maxX && c <= maxTot {
			return c, 1, 1
		}
	}
	return 1, 1, 1
}

func chooseTile(l wgpu.SupportedLimits, wgX, wgY, wgZ uint32) (uint32, uint32) {
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

func detectRuntime() string {
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

func pickIndex(rs []Report, pref string) int {
	pref = strings.ToLower(pref)
	best := 0
	for i := range rs {
		if pref == "discrete" && rs[i].AdapterType == "discrete-gpu" {
			return i
		}
		if pref == "integrated" && rs[i].AdapterType == "integrated-gpu" {
			return i
		}
	}
	return best
}
