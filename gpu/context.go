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

// GetContext returns the singleton GPU context, initializing it if necessary
func GetContext() (*Context, error) {
	var initErr error
	ctx.once.Do(func() {
		ctx.Instance = wgpu.CreateInstance(nil)
		if ctx.Instance == nil {
			initErr = fmt.Errorf("failed to create WebGPU instance")
			return
		}

		// 0. Try to find NVIDIA explicitly via EnumerateAdapters
		adapters := ctx.Instance.EnumerateAdapters(nil)
		for _, a := range adapters {
			info := a.GetInfo()
			fmt.Printf("Displaying Adapter: %s (Vendor: %s, DeviceID: 0x%X, VendorID: 0x%X, Type: %d)\n", info.Name, info.VendorName, info.DeviceId, info.VendorId, info.AdapterType)
			isNvidia := false
			if strings.Contains(strings.ToLower(info.Name), "nvidia") {
				isNvidia = true
			}
			if strings.Contains(strings.ToLower(info.VendorName), "nvidia") {
				isNvidia = true
			}

			if isNvidia {
				fmt.Printf("--> Force Selecting NVIDIA Adapter: %s\n", info.Name)
				ctx.Adapter = a
				break
			}
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
		fmt.Printf("Using GPU Adapter: %s (Vendor: %s)\n", info.Name, info.VendorName)

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
