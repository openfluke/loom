package poly

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUContext manages the GPU device and queue for acceleration.
type WGPUContext struct {
	Instance       *wgpu.Instance
	Adapter        *wgpu.Adapter
	Device         *wgpu.Device
	Queue          *wgpu.Queue
	PipelineCache  map[string]*wgpu.ComputePipeline
	ActivationPool map[string]*wgpu.Buffer
	// GPUTileSize is the auto-detected optimal tile size for this GPU.
	// Can be overridden by the caller after init.
	GPUTileSize    int
}

// InitWGPU initializes the WebGPU context for the network.
func (n *VolumetricNetwork) InitWGPU() error {
	if n.GPUContext != nil {
		return nil
	}

	instance := wgpu.CreateInstance(nil)
	if instance == nil {
		return fmt.Errorf("failed to create WGPU instance")
	}

	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		instance.Release()
		return fmt.Errorf("failed to request adapter: %v", err)
	}

	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		instance.Release()
		return fmt.Errorf("failed to request device: %v", err)
	}

	n.GPUContext = &WGPUContext{
		Instance:       instance,
		Adapter:        adapter,
		Device:         device,
		Queue:          device.GetQueue(),
		PipelineCache:  make(map[string]*wgpu.ComputePipeline),
		ActivationPool: make(map[string]*wgpu.Buffer),
	}

	// Auto-detect optimal GPU tile size from this adapter's limits
	limits := adapter.GetLimits()
	n.GPUContext.GPUTileSize = CalculateOptimalGPUTileSizeFromLimits(
		limits.Limits.MaxComputeWorkgroupStorageSize,
		limits.Limits.MaxComputeInvocationsPerWorkgroup,
		64, // default headDim; caller can override via GPUTileSize after init
	)

	n.UseGPU = true
	return nil
}

// Release releases all WebGPU resources.
func (c *WGPUContext) Release() {
	if c.Device != nil {
		c.Device.Release()
	}
	if c.Adapter != nil {
		c.Adapter.Release()
	}
	if c.Instance != nil {
		c.Instance.Release()
	}
}

// CreatePersistentBuffer creates a storage buffer that stays in VRAM.
func (c *WGPUContext) CreatePersistentBuffer(data []float32, label string) (*wgpu.Buffer, error) {
	buf, err := c.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label,
		Contents: wgpu.ToBytes(data),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	return buf, err
}
