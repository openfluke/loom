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

func adapterTypeString(t wgpu.AdapterType) string {
	switch t {
	case wgpu.AdapterTypeDiscreteGPU:
		return "DiscreteGPU"
	case wgpu.AdapterTypeIntegratedGPU:
		return "IntegratedGPU"
	case wgpu.AdapterTypeCPU:
		return "CPU"
	case wgpu.AdapterTypeUnknown:
		return "Unknown"
	default:
		return "Other"
	}
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

			if preferredAdapter != "" && strings.Contains(name, preferredAdapter) {
				score += 10000
			}

			isDiscrete := info.AdapterType == wgpu.AdapterTypeDiscreteGPU
			isIntegrated := info.AdapterType == wgpu.AdapterTypeIntegratedGPU

			isFake := info.VendorId == 0x10005 ||
				info.VendorId == 0x1414 ||
				strings.Contains(name, "llvmpipe") ||
				strings.Contains(name, "soft") ||
				strings.Contains(name, "warp") ||
				strings.Contains(name, "basic render")

			isApple := strings.Contains(name, "apple") ||
				(isIntegrated && info.VendorId == 0 && info.DeviceId == 0)

			if isDiscrete && !isFake {
				score += 1000
			}
			if isApple {
				score += 700
			} else if isIntegrated {
				score += 500
			}

			if strings.Contains(name, "nvidia") || info.VendorId == 0x10DE {
				score += 200
			}
			if strings.Contains(name, "amd") || info.VendorId == 0x1002 {
				score += 100
			}

			if isFake {
				score -= 1000
			}
			if info.AdapterType == wgpu.AdapterTypeCPU {
				score -= 5000
			}

			if score > bestScore {
				bestScore = score
				bestAdapter = a
			}
		}

		var selectedIsApple bool
		if bestAdapter != nil {
			info := bestAdapter.GetInfo()
			name := strings.ToLower(info.Name)
			selectedIsApple = strings.Contains(name, "apple") ||
				(info.AdapterType == wgpu.AdapterTypeIntegratedGPU && info.VendorId == 0 && info.DeviceId == 0)
			fmt.Printf("Selected WebGPU adapter: %s (vendor=0x%X type=%s score=%d)\n",
				info.Name, info.VendorId, adapterTypeString(info.AdapterType), bestScore)
			ctx.Adapter = bestAdapter
		}

		tryInit := func(opts *wgpu.RequestAdapterOptions) error {
			if ctx.Adapter != nil {
				return nil
			}
			var err error
			ctx.Adapter, err = ctx.Instance.RequestAdapter(opts)
			return err
		}

		if ctx.Adapter == nil {
			initErr = tryInit(&wgpu.RequestAdapterOptions{
				PowerPreference: wgpu.PowerPreferenceHighPerformance,
			})
		}

		if initErr != nil && ctx.Adapter == nil {
			fmt.Printf("High performance adapter failed: %v. Falling back...\n", initErr)
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
		fmt.Printf("Using WebGPU adapter: %s (vendor=%s/0x%X type=%s)\n",
			info.Name, info.VendorName, info.VendorId, adapterTypeString(info.AdapterType))

		var err error
		// ─────────────────────────────────────────────────────────────────────────────
		// LIMIT REQUEST STRATEGY
		// ─────────────────────────────────────────────────────────────────────────────
		// Large models like Qwen2.5-0.5B need >512MB for their LM head.
		// Blindly cloning all limits can fail if the driver is pickier than it reports.
		// Instead, we specifically target the limits we know we need.

		adapterLimits := ctx.Adapter.GetLimits().Limits
		Log("Adapter Max Storage Buffer Binding: %d MB", adapterLimits.MaxStorageBufferBindingSize/(1024*1024))

		// 1. Build a baseline from adapter limits
		limits := adapterLimits

		// 2. Cap buffer limits at 1GB unless adapter is smaller (safety measure)
		// Many NVIDIA drivers report 4GB+, but sometimes requesting the absolute max fails.
		if limits.MaxStorageBufferBindingSize > 1024*1024*1024 {
			limits.MaxStorageBufferBindingSize = 1024 * 1024 * 1024
		}
		if limits.MaxBufferSize > 1024*1024*1024 {
			limits.MaxBufferSize = 1024 * 1024 * 1024
		}

		deviceDesc := &wgpu.DeviceDescriptor{
			RequiredLimits: &wgpu.RequiredLimits{
				Limits: limits,
			},
		}

		if selectedIsApple {
			fmt.Println("🍎 Apple Silicon GPU detected — using Metal optimised device descriptor")
			deviceDesc.Label = "loom-apple-silicon"
		}

		// Try with high limits
		ctx.Device, err = ctx.Adapter.RequestDevice(deviceDesc)
		if err != nil {
			fmt.Printf("⚠️  High GPU limits request failed: %v. Falling back to default limits (this may break large models)\n", err)
			// Try with default limits (nil descriptor)
			ctx.Device, err = ctx.Adapter.RequestDevice(nil)
			if err != nil {
				initErr = err
				return
			}
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

// CreateBuffer creates a buffer on the device and tracks its VRAM usage
func (c *Context) CreateBuffer(desc *wgpu.BufferDescriptor) (*wgpu.Buffer, error) {
	b, err := c.Device.CreateBuffer(desc)
	if err == nil && desc != nil {
		TrackVRAM(desc.Label, desc.Size)
	}
	return b, err
}

// CreateBufferInit creates a buffer with data and tracks its VRAM usage
func (c *Context) CreateBufferInit(desc *wgpu.BufferInitDescriptor) (*wgpu.Buffer, error) {
	b, err := c.Device.CreateBufferInit(desc)
	if err == nil && desc != nil {
		TrackVRAM(desc.Label, uint64(len(desc.Contents)))
	}
	return b, err
}
