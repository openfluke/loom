package poly

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)
 
var (
	sharedInstance *wgpu.Instance
	sharedAdapter  *wgpu.Adapter
	sharedDevice   *wgpu.Device
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

	// BlankBuffer is a small zeroed buffer for optional bindings (e.g. bias)
	BlankBuffer *wgpu.Buffer
}

// WGPUBufferBinding represents a slice of a GPU buffer for binding.
type WGPUBufferBinding struct {
	Buffer *wgpu.Buffer
	Offset uint64
	Size   uint64
}

func (c *WGPUContext) GetSubBuffer(buf *wgpu.Buffer, offset, size uint64) *WGPUBufferBinding {
	return &WGPUBufferBinding{Buffer: buf, Offset: offset, Size: size}
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

	if sharedInstance == nil {
		sharedInstance = wgpu.CreateInstance(nil)
		if sharedInstance == nil {
			return fmt.Errorf("failed to create WGPU instance")
		}
	}
 
	if sharedAdapter == nil {
		adapter, err := sharedInstance.RequestAdapter(&wgpu.RequestAdapterOptions{
			PowerPreference: wgpu.PowerPreferenceHighPerformance,
		})
		if err != nil {
			return fmt.Errorf("failed to request adapter: %v", err)
		}
		sharedAdapter = adapter
	}
 
	adapter := sharedAdapter

	// 1. Request a temporary default device to get a safe baseline for all limits
	defaultDevice, err := adapter.RequestDevice(nil)
	if err != nil {
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

	if sharedDevice == nil {
		// Request the device with explicitly supported baseline limits + necessary boosts
		device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
			RequiredLimits: &wgpu.RequiredLimits{
				Limits: limits,
			},
			RequiredFeatures: []wgpu.FeatureName{
				wgpu.FeatureNameShaderF16,
			},
		})
		if err != nil {
			// If requesting with boosted limits and F16 fails, try again with default limits only.
			device, err = adapter.RequestDevice(nil)
			if err != nil {
				return fmt.Errorf("failed to request device even with default limits: %v", err)
			}
		}
		sharedDevice = device
	}
 
	device := sharedDevice
	finalLimits := device.GetLimits()

	n.GPUContext = &WGPUContext{
		Instance:       sharedInstance,
		Adapter:        sharedAdapter,
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

	// Initialize BlankBuffer
	n.GPUContext.BlankBuffer, _ = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "Blank Buffer",
		Size:  64,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})

	n.UseGPU = true
	return nil
}

// Release releases pooled resources. Device is shared and persists.
func (c *WGPUContext) Release() {
	c.Cleanup()
}

// Cleanup explicitly releases all cached/pooled resources (buffers, pipelines, bind groups).
func (c *WGPUContext) Cleanup() {
	if c == nil { return }
	
	// 1. Release Activation Pool
	for name, buf := range c.ActivationPool {
		if buf != nil {
			buf.Release()
		}
		delete(c.ActivationPool, name)
	}

	// 2. Release Uniform Pool
	for _, buf := range c.UniformPool {
		if buf != nil {
			buf.Release()
		}
	}
	c.UniformPool = nil
	c.UniformIdx = 0

	// 3. Release Bind Groups
	for k, bg := range c.BindGroupCache {
		if bg != nil {
			bg.Release()
		}
		delete(c.BindGroupCache, k)
	}

	// 4. Release Pipelines
	for k, p := range c.PipelineCache {
		if p != nil {
			p.Release()
		}
		delete(c.PipelineCache, k)
	}

	// 5. Release Layouts
	for k, l := range c.LayoutCache {
		if l != nil {
			l.Release()
		}
		delete(c.LayoutCache, k)
	}

	// 6. Release any pending destroys
	for _, buf := range c.PendingDestroys {
		if buf != nil {
			buf.Release()
		}
	}
	c.PendingDestroys = nil
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

// BindGroupKeyHash generates a stable hash for a set of buffers (or bindings) and a pipeline.
func BindGroupKeyHash(pipeline *wgpu.ComputePipeline, bindings ...any) uint64 {
	h := uint64(uintptr(unsafe.Pointer(pipeline)))
	for _, b := range bindings {
		if b == nil { continue }
		switch v := b.(type) {
		case *wgpu.Buffer:
			h ^= uint64(uintptr(unsafe.Pointer(v))) + 0x9e3779b9 + (h << 6) + (h >> 2)
		case *WGPUBufferBinding:
			h ^= uint64(uintptr(unsafe.Pointer(v.Buffer))) + 0x9e3779b9 + (h << 6) + (h >> 2)
			h ^= v.Offset + 0x9e3779b9 + (h << 6) + (h >> 2)
			h ^= v.Size + 0x9e3779b9 + (h << 6) + (h >> 2)
		}
	}
	return h
}

// GetBindGroup retrieves or creates a BindGroup for the given pipeline and buffers/bindings.
func (c *WGPUContext) GetBindGroup(pipeline *wgpu.ComputePipeline, bindings ...any) (*wgpu.BindGroup, error) {
	key := BindGroupKeyHash(pipeline, bindings...)
	if bg, ok := c.BindGroupCache[key]; ok {
		return bg, nil
	}

	entries := make([]wgpu.BindGroupEntry, len(bindings))
	for i, b := range bindings {
		if b == nil {
			return nil, fmt.Errorf("binding %d is nil", i)
		}

		var bBuf *wgpu.Buffer
		var offset, size uint64

		switch v := b.(type) {
		case *wgpu.Buffer:
			if v == nil {
				return nil, fmt.Errorf("binding %d is nil *wgpu.Buffer", i)
			}
			bBuf = v
			offset = 0
			size = v.GetSize()
		case *WGPUBufferBinding:
			if v == nil || v.Buffer == nil {
				return nil, fmt.Errorf("binding %d is nil or has nil buffer in *WGPUBufferBinding", i)
			}
			bBuf = v.Buffer
			offset = v.Offset
			size = v.Size
		default:
			return nil, fmt.Errorf("binding %d has invalid type %T", i, b)
		}

		entries[i] = wgpu.BindGroupEntry{
			Binding: uint32(i),
			Buffer:  bBuf,
			Offset:  offset,
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
