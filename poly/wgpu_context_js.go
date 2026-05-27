//go:build js

package poly

import (
	"fmt"
	"sync"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

var (
	sharedInstance *wgpu.Instance
	sharedAdapter  *wgpu.Adapter
	sharedDevice   *wgpu.Device
	mu             sync.Mutex // Protects global initialization
)

// InitWGPU initializes WebGPU using navigator.gpu / setupWebGPU() in the host page.
func (n *VolumetricNetwork) InitWGPU() error {
	if n.GPUContext != nil {
		return nil
	}

	mu.Lock()
	defer mu.Unlock()

	if sharedDevice == nil {
		if sharedInstance == nil {
			sharedInstance = wgpu.CreateInstance(nil)
			if sharedInstance == nil {
				return fmt.Errorf("failed to create WGPU instance (is navigator.gpu available?)")
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
		device, err := sharedAdapter.RequestDevice(nil)
		if err != nil {
			return fmt.Errorf("device request failed (call setupWebGPU() first): %v", err)
		}
		sharedDevice = device
	}

	// 2. Map Shared State to Network Context
	device := sharedDevice
	finalLimits := device.GetLimits()

	n.GPUContext = &WGPUContext{
		Instance:          sharedInstance,
		Adapter:           sharedAdapter,
		Device:            device,
		Queue:             device.GetQueue(),
		Limits:            finalLimits.Limits,
		PipelineCache:     make(map[string]*wgpu.ComputePipeline),
		ActivationPool:    make(map[string]*wgpu.Buffer),
		LayoutCache:       make(map[string]*wgpu.BindGroupLayout),
		BindGroupCache:    make(map[uint64]*wgpu.BindGroup),
		UniformPool:       make([]*wgpu.Buffer, 0, 1024),
		HasTimestampQuery: false, // browser wgpu bindings lack timestamp-query helpers
	}

	n.GPUContext.GPUTileSize = CalculateOptimalGPUTileSizeFromLimits(
		finalLimits.Limits.MaxComputeWorkgroupStorageSize,
		finalLimits.Limits.MaxComputeInvocationsPerWorkgroup,
		64,
	)

	n.GPUContext.BlankBuffer, _ = device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "Blank Buffer",
		Size:  64,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})

	n.UseGPU = true
	return nil
}

func min64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

// Release releases pooled resources. Device is shared and persists.
func (c *WGPUContext) Release() {
	c.Cleanup()
}

// Cleanup explicitly releases all cached/pooled resources (buffers, pipelines, bind groups).
func (c *WGPUContext) Cleanup() {
	if c == nil {
		return
	}

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
	c.FrameCount++
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

// GetActivationBuffer retrieves or creates a persistent activation buffer.
func (c *WGPUContext) GetActivationBuffer(name string, size uint64, usage wgpu.BufferUsage) *wgpu.Buffer {
	// WebGPU/Vulkan/D3D12 consistency: Enforce 16-byte alignment and 64-byte minimum.
	if size < 64 {
		size = 64
	}
	if size%16 != 0 {
		size = (size + 15) &^ 15
	}

	// Augment usage for common operations
	actualUsage := usage
	if (usage & wgpu.BufferUsageMapRead) != 0 {
		actualUsage |= wgpu.BufferUsageCopyDst
	} else if (usage & wgpu.BufferUsageMapWrite) != 0 {
		actualUsage |= wgpu.BufferUsageCopySrc
	} else {
		// Generic activation buffer for storage/compute
		actualUsage |= wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst
	}

	// Check if we can reuse an existing buffer
	if buf, ok := c.ActivationPool[name]; ok && buf != nil {
		// We reuse if it's large enough AND has all requested usage bits.
		if buf.GetSize() >= size && (getBufferUsage(buf)&actualUsage == actualUsage) {
			return buf
		}
		// Size mismatch or missing usage bits; must recreate.
		buf.Destroy()
	}

	buf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: name,
		Size:  size,
		Usage: actualUsage,
	})
	if err != nil || buf == nil {
		fmt.Printf("❌ ERROR: Failed to create GPU activation buffer '%s' (size %d): %v\n", name, size, err)
		return nil
	}
	c.ActivationPool[name] = buf
	return buf
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

// CreatePersistentBufferUint32 creates a storage buffer for uint32 data (e.g. masks).
func (c *WGPUContext) CreatePersistentBufferUint32(data []uint32, label string) (*wgpu.Buffer, error) {
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
	if err != nil {
		return nil, err
	}
	defer stagingBuf.Destroy()

	enc, err := c.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	enc.CopyBufferToBuffer(buf, 0, stagingBuf, 0, size)
	cmd, _ := enc.Finish(nil)
	c.Queue.Submit(cmd)

	done := make(chan struct{})
	err = stagingBuf.MapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	})
	if err != nil {
		return nil, err
	}

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

// ReadUint64Buffer reads a GPU buffer back as uint64 values.
func (c *WGPUContext) ReadUint64Buffer(buf *wgpu.Buffer, count int) ([]uint64, error) {
	if count <= 0 {
		return nil, nil
	}
	size := uint64(count * 8)
	stagingBuf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "Read U64 Staging",
		Size:  size,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	defer stagingBuf.Destroy()

	enc, err := c.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	enc.CopyBufferToBuffer(buf, 0, stagingBuf, 0, size)
	cmd, _ := enc.Finish(nil)
	c.Queue.Submit(cmd)
	c.Device.Poll(true, nil)

	done := make(chan struct{})
	if err := stagingBuf.MapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	}); err != nil {
		return nil, err
	}

	for {
		c.Device.Poll(false, nil)
		select {
		case <-done:
			data := stagingBuf.GetMappedRange(0, uint(size))
			defer stagingBuf.Unmap()
			out := make([]uint64, count)
			copy(wgpu.ToBytes(out), data[:size])
			return out, nil
		default:
		}
	}
}

// TimeCommands is not supported in the browser/WebAssembly wgpu bindings.
func (c *WGPUContext) TimeCommands(_ string, _ func() error) (time.Duration, error) {
	return 0, fmt.Errorf("timestamp queries unavailable in wasm/js build")
}
