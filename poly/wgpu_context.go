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
	// ActiveEncoder, when non-nil, is used by all Dispatch* calls instead of
	// creating their own encoder. This lets the entire forward pass be recorded
	// into a single command buffer and submitted once, reducing GPU overhead.
	ActiveEncoder   *wgpu.CommandEncoder
	// PendingDestroys holds temporary uniform buffers that must not be destroyed
	// until after FlushFrame() submits the active encoder. When not batching,
	// buffers are destroyed immediately instead of queued here.
	PendingDestroys []*wgpu.Buffer
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

// BeginFrame creates a shared CommandEncoder that all subsequent Dispatch* calls
// will record into until FlushFrame is called.
func (c *WGPUContext) BeginFrame() error {
	enc, err := c.Device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	c.ActiveEncoder = enc
	c.PendingDestroys = c.PendingDestroys[:0] // reset slice, keep backing array
	return nil
}

// FlushFrame finishes and submits the shared CommandEncoder, then destroys any
// temporary uniform buffers that were kept alive for the duration of recording.
func (c *WGPUContext) FlushFrame() {
	if c.ActiveEncoder == nil {
		return
	}
	cmd, _ := c.ActiveEncoder.Finish(nil)
	c.Queue.Submit(cmd)
	c.ActiveEncoder = nil
	// Now safe to destroy temp uniform buffers — GPU has consumed the commands.
	for _, buf := range c.PendingDestroys {
		buf.Destroy()
	}
	c.PendingDestroys = c.PendingDestroys[:0]
}

// deferOrDestroy either queues buf for destruction after FlushFrame (when
// batching) or destroys it immediately (standalone dispatch).
func (c *WGPUContext) deferOrDestroy(buf *wgpu.Buffer) {
	if c.ActiveEncoder != nil {
		c.PendingDestroys = append(c.PendingDestroys, buf)
	} else {
		buf.Destroy()
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
