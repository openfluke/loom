package poly

import (
	"fmt"
	"unsafe"

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
	GPUTileSize int
	// ActiveEncoder, when non-nil, is used by all Dispatch* calls instead of
	// creating their own encoder. This lets the entire forward pass be recorded
	// into a single command buffer and submitted once, reducing GPU overhead.
	ActiveEncoder *wgpu.CommandEncoder
	// PendingDestroys holds temporary uniform buffers that must not be destroyed
	// until after FlushFrame() submits the active encoder. When not batching,
	// buffers are destroyed immediately instead of queued here.
	PendingDestroys []*wgpu.Buffer

	// --- Performance Optimization Caches ---
	LayoutCache    map[string]*wgpu.BindGroupLayout
	BindGroupCache map[uint64]*wgpu.BindGroup

	// Uniform Pool
	UniformPool []*wgpu.Buffer
	UniformIdx  int

	// Negotiated limits
	Limits wgpu.Limits
}

// BindGroupKey is used for the BindGroupCache
type BindGroupKey struct {
	Pipeline *wgpu.ComputePipeline
	Buffers  []*wgpu.Buffer
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

	// 1. Request a temporary default device to get a safe baseline for all limits
	defaultDevice, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		instance.Release()
		return fmt.Errorf("failed to request default device for limits: %v", err)
	}
	limits := defaultDevice.GetLimits().Limits
	defaultDevice.Release()

	// 2. Boost the storage and buffer size limits to allow for large embeddings and weights (1GB storage, 2GB total)
	if limits.MaxStorageBufferBindingSize < 1024*1024*1024 {
		limits.MaxStorageBufferBindingSize = 1024 * 1024 * 1024
	}
	if limits.MaxBufferSize < 2*1024*1024*1024 {
		limits.MaxBufferSize = 2 * 1024 * 1024 * 1024
	}

	deviceDesc := &wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: limits,
		},
	}

	// Request the device with explicitly supported baseline limits + necessary boosts
	fmt.Printf("⚠️  DEBUG: Requesting Device %s (backend %v) with MaxStorage=%d MB, MaxBuffer=%d MB, WorkgroupStorage=%d\n",
		adapter.GetInfo().Name, adapter.GetInfo().BackendType,
		limits.MaxStorageBufferBindingSize/(1024*1024),
		limits.MaxBufferSize/(1024*1024),
		limits.MaxComputeWorkgroupStorageSize)

	device, err := adapter.RequestDevice(deviceDesc)

	if err != nil {
		fmt.Printf("⚠️  High GPU limits request failed: %v. Falling back to default limits (large models will likely crash)\n", err)
		device, err = adapter.RequestDevice(nil)
	}

	if err != nil {
		fmt.Printf("⚠️  WebGPU device request failed completely: %v\n", err)
		adapter.Release()
		instance.Release()
		return fmt.Errorf("failed to request device: %v", err)
	}

	finalLimits := device.GetLimits()

	n.GPUContext = &WGPUContext{
		Instance:       instance,
		Adapter:        adapter,
		Device:         device,
		Queue:          device.GetQueue(),
		Limits:         finalLimits.Limits,
		PipelineCache:  make(map[string]*wgpu.ComputePipeline),
		ActivationPool: make(map[string]*wgpu.Buffer),
		LayoutCache:    make(map[string]*wgpu.BindGroupLayout),
		BindGroupCache: make(map[uint64]*wgpu.BindGroup),
		UniformPool:    make([]*wgpu.Buffer, 0, 1024),
	}

	// Auto-detect optimal GPU tile size from these actual device limits
	n.GPUContext.GPUTileSize = CalculateOptimalGPUTileSizeFromLimits(
		finalLimits.Limits.MaxComputeWorkgroupStorageSize,
		finalLimits.Limits.MaxComputeInvocationsPerWorkgroup,
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
	c.UniformIdx = 0
}

// ResetCache clears all BindGroups and Pipelines.
// Should be called when model architecture or precision changes.
func (c *WGPUContext) ResetCache() {
	for k, bg := range c.BindGroupCache {
		bg.Release()
		delete(c.BindGroupCache, k)
	}
	// We keep the Pipelines and Layouts as they are often reusable
	// based on the shader source hash, but we could clear them too if needed.
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

// GetUniformBuffer provides a pre-allocated uniform buffer from the pool.
func (c *WGPUContext) GetUniformBuffer(size uint64) *wgpu.Buffer {
	// WebGPU uniform buffer bindings often require 16-byte alignment.
	if size < 16 {
		size = 16
	}
	if size%16 != 0 {
		size = (size + 15) &^ 15
	}

	for i := c.UniformIdx; i < len(c.UniformPool); i++ {
		if c.UniformPool[i].GetSize() >= size {
			c.UniformIdx = i + 1
			return c.UniformPool[i]
		}
	}

	// Not found, create a new one
	buf, _ := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  size,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	c.UniformPool = append(c.UniformPool, buf)
	c.UniformIdx = len(c.UniformPool)
	return buf
}

// BindGroupKeyHash generates a stable hash for a set of buffers and a pipeline.
func BindGroupKeyHash(pipeline *wgpu.ComputePipeline, buffers ...*wgpu.Buffer) uint64 {
	h := uint64(uintptr(unsafe.Pointer(pipeline)))
	for _, b := range buffers {
		h ^= uint64(uintptr(unsafe.Pointer(b))) + 0x9e3779b9 + (h << 6) + (h >> 2)
	}
	return h
}

// GetBindGroup retrieves or creates a BindGroup for the given pipeline and buffers.
func (c *WGPUContext) GetBindGroup(pipeline *wgpu.ComputePipeline, buffers ...*wgpu.Buffer) (*wgpu.BindGroup, error) {
	key := BindGroupKeyHash(pipeline, buffers...)
	if bg, ok := c.BindGroupCache[key]; ok {
		return bg, nil
	}

	entries := make([]wgpu.BindGroupEntry, len(buffers))
	for i, b := range buffers {
		if b == nil {
			return nil, fmt.Errorf("binding %d is nil", i)
		}

		size := b.GetSize()

		entries[i] = wgpu.BindGroupEntry{
			Binding: uint32(i),
			Buffer:  b,
			Size:    size,
		}
	}

	bg, err := c.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  pipeline.GetBindGroupLayout(0),
		Entries: entries,
	})
	if err != nil {
		return nil, err
	}
	c.BindGroupCache[key] = bg
	return bg, nil
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

// ReadBuffer reads data from a GPU buffer back to a float32 slice.
func (c *WGPUContext) ReadBuffer(buf *wgpu.Buffer) ([]float32, error) {
	size := buf.GetSize()
	stagingBuf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "Read Staging",
		Size:  size,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil { return nil, err }
	defer stagingBuf.Destroy()

	enc, err := c.Device.CreateCommandEncoder(nil)
	if err != nil { return nil, err }
	enc.CopyBufferToBuffer(buf, 0, stagingBuf, 0, size)
	cmd, _ := enc.Finish(nil)
	c.Queue.Submit(cmd)

	done := make(chan struct{})
	err = stagingBuf.MapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	})
	if err != nil { return nil, err }

	for {
		c.Device.Poll(false, nil)
		select {
		case <-done:
			goto Finished
		default:
			// yield?
		}
	}

Finished:
	data := stagingBuf.GetMappedRange(0, uint(size))
	defer stagingBuf.Unmap()

	res := make([]float32, size/4)
	copy(wgpu.ToBytes(res), data)
	return res, nil
}
