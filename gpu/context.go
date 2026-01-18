package gpu

import (
	"fmt"
	"strings"
	"sync"

	"github.com/openfluke/webgpu/wgpu"
)

// Context holds the single WebGPU context for the application
type Context struct {
	Instance *wgpu.Instance
	Adapter  *wgpu.Adapter
	Device   *wgpu.Device
	Queue    *wgpu.Queue
	once     sync.Once
}

var ctx Context
var preferredAdapter string

// Debug enables verbose logging for GPU operations
var Debug bool = false

// SetDebug enables or disables verbose logging
func SetDebug(enabled bool) {
	Debug = enabled
	Log("GPU Debug Mode: %v", enabled)
}

// Log prints a debug message if Debug is true
func Log(format string, args ...interface{}) {
	if Debug {
		fmt.Printf("[GPU] "+format+"\n", args...)
	}
}

// SetAdapterPreference sets a substring to look for in adapter names
func SetAdapterPreference(name string) {
	preferredAdapter = strings.ToLower(name)
}

// GetContext returns the singleton GPU context, initializing it if necessary
func GetContext() (*Context, error) {
	var initErr error
	ctx.once.Do(func() {
		ctx.Instance = wgpu.CreateInstance(nil)
		if ctx.Instance == nil {
			initErr = fmt.Errorf("failed to create WebGPU instance")
			return
		}

		// 0. Explicit Selection Logic
		adapters := ctx.Instance.EnumerateAdapters(nil)
		var bestAdapter *wgpu.Adapter
		bestScore := -1

		for _, a := range adapters {
			info := a.GetInfo()
			if Debug {
				fmt.Printf("Displaying Adapter: 0x%X (Vendor: 0x%X, DeviceID: 0x%X, Type: %d) Name: %s\n", info.DeviceId, info.VendorId, info.DeviceId, info.AdapterType, info.Name)
			}

			score := 0
			name := strings.ToLower(info.Name)
			// info.AdapterType: 0=Other, 1=Integrated, 2=Discrete, 3=Virtual, 4=CPU

			// 1. Preference override
			if preferredAdapter != "" && strings.Contains(name, preferredAdapter) {
				score += 10000 // Massive boost for manual selection
			}

			// 2. Hardware Tiers
			isDiscrete := info.AdapterType == wgpu.AdapterTypeDiscreteGPU
			isIntegrated := info.AdapterType == wgpu.AdapterTypeIntegratedGPU

			// Detect "Fake" Discrete GPU (llvmpipe / Mesa software rasterizer)
			// Vendor 0x10005 is Mesa. "llvmpipe" usually appears in name.
			isFake := info.VendorId == 0x10005 || strings.Contains(name, "llvmpipe") || strings.Contains(name, "soft")

			if isDiscrete && !isFake {
				score += 1000 // Real Discrete GPU (Nvidia, AMD)
			}
			if isIntegrated {
				score += 500 // Integrated GPU (Intel, etc.) - Better than software!
			}

			// Vendor specifics
			if strings.Contains(name, "nvidia") || info.VendorId == 0x10DE {
				score += 200
			}

			// Penalize Fake/Software
			if isFake {
				score -= 1000
			}

			if score > bestScore {
				bestScore = score
				bestAdapter = a
			}
		}

		if bestAdapter != nil {
			info := bestAdapter.GetInfo()
			if Debug {
				fmt.Printf("--> Selected Best Adapter: %s (Type: %d, Score: %d)\n", info.Name, info.AdapterType, bestScore)
			}
			ctx.Adapter = bestAdapter
		}

		// Helper to try init with an adapter option
		tryInit := func(opts *wgpu.RequestAdapterOptions) error {
			if ctx.Adapter != nil {
				return nil // Already found
			}
			var err error
			ctx.Adapter, err = ctx.Instance.RequestAdapter(opts)
			return err
		}

		// 1. Try High Performance (if not found above)
		if ctx.Adapter == nil {
			initErr = tryInit(&wgpu.RequestAdapterOptions{
				PowerPreference: wgpu.PowerPreferenceHighPerformance,
			})
		}

		if initErr != nil && ctx.Adapter == nil {
			fmt.Printf("High performance adapter failed: %v. Falling back...\n", initErr)
			// 2. Try Low Power / Default
			initErr = tryInit(&wgpu.RequestAdapterOptions{
				PowerPreference: wgpu.PowerPreferenceLowPower,
			})
		}

		if initErr != nil && ctx.Adapter == nil {
			fmt.Printf("Low power adapter failed: %v. Trying default...\n", initErr)
			initErr = tryInit(nil)
		}

		if ctx.Adapter == nil {
			initErr = fmt.Errorf("all adapter attempts failed: %v", initErr)
			return
		}

		// Initialize Device
		info := ctx.Adapter.GetInfo()
		if Debug {
			fmt.Printf("Using GPU Adapter: %s (Vendor: %s)\n", info.Name, info.VendorName)
		}

		var err error
		// Request device with increased limits if needed
		ctx.Device, err = ctx.Adapter.RequestDevice(nil)
		if err != nil {
			initErr = err
			return
		}

		ctx.Queue = ctx.Device.GetQueue()
	})

	if initErr != nil {
		return nil, initErr
	}
	if ctx.Device == nil || ctx.Queue == nil {
		return nil, fmt.Errorf("WebGPU device or queue not initialized")
	}

	return &ctx, nil
}
