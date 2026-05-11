package poly

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

var (
	sharedInstance *wgpu.Instance
	sharedAdapter  *wgpu.Adapter
	sharedDevice   *wgpu.Device
	mu             sync.Mutex // Protects global initialization
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

	// HasTimestampQuery reports whether GPU timestamp queries were enabled.
	HasTimestampQuery bool

	// FrameCount increments with every training iteration or forward pass
	// used for hardware-native randomized seeds.
	FrameCount uint32
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

	mu.Lock()
	defer mu.Unlock()

	// 1. One-time Global Initialization
	if sharedDevice == nil {
		if sharedInstance == nil {
			// Force Vulkan at instance level — newer wgpu-native ignores
			// RequestAdapterOptions.BackendType; the correct API is InstanceExtras.backends.
			var instanceDesc *wgpu.InstanceDescriptor
			if runtime.GOOS == "android" || runtime.GOOS == "windows" {
				Alog("Forcing Vulkan via InstanceDescriptor.Backends...")
				instanceDesc = &wgpu.InstanceDescriptor{
					Backends: wgpu.InstanceBackendVulkan,
				}
			}
			sharedInstance = wgpu.CreateInstance(instanceDesc)
			if sharedInstance == nil {
				// Fallback: no backend constraint
				Alog("⚠️  Vulkan instance failed, retrying with all backends...")
				sharedInstance = wgpu.CreateInstance(nil)
			}
			if sharedInstance == nil {
				return fmt.Errorf("failed to create WGPU instance")
			}
		}

		if sharedAdapter == nil {
			opts := &wgpu.RequestAdapterOptions{
				PowerPreference: wgpu.PowerPreferenceHighPerformance,
			}

			adapter, err := sharedInstance.RequestAdapter(opts)
			if err != nil {
				return fmt.Errorf("failed to request adapter: %v", err)
			}
			sharedAdapter = adapter
			info := adapter.GetInfo()
			fmt.Println("=========================================================")
			fmt.Printf("Adapter: %s [%v]\n", info.Name, info.BackendType)
			fmt.Println("=========================================================")
		}

		adapter := sharedAdapter
		adapterLimits := adapter.GetLimits().Limits

		Alog(fmt.Sprintf("Adapter Hardware Limits: MaxStorage=%d MB, MaxBuffer=%d MB",
			adapterLimits.MaxStorageBufferBindingSize/(1024*1024),
			adapterLimits.MaxBufferSize/(1024*1024)))

		// Build required limits directly from adapter caps — no dummy device probe.
		// The dummy probe + Release() invalidates adapter state in newer wgpu-native.
		requiredLimits := adapterLimits

		// Override only storage/buffer sizes; all other fields stay at adapter max.
		requiredLimits.MaxStorageBufferBindingSize = min64(1024*1024*1024, adapterLimits.MaxStorageBufferBindingSize)
		requiredLimits.MaxBufferSize = min64(2048*1024*1024, adapterLimits.MaxBufferSize)

		var requiredFeatures []wgpu.FeatureName
		if adapter.HasFeature(wgpu.FeatureNameShaderF16) {
			Alog("Hardware Feature: shader-f16 supported.")
			requiredFeatures = append(requiredFeatures, wgpu.FeatureNameShaderF16)
		}
		if adapter.HasFeature(wgpu.FeatureNameTimestampQuery) {
			Alog("Hardware Feature: timestamp-query supported.")
			requiredFeatures = append(requiredFeatures, wgpu.FeatureNameTimestampQuery)
		}

		storageAttempts := []uint64{
			requiredLimits.MaxStorageBufferBindingSize,
			min64(512*1024*1024, adapterLimits.MaxStorageBufferBindingSize),
			min64(256*1024*1024, adapterLimits.MaxStorageBufferBindingSize),
		}
		bufferAttempts := []uint64{
			requiredLimits.MaxBufferSize,
			min64(512*1024*1024, adapterLimits.MaxBufferSize),
			min64(256*1024*1024, adapterLimits.MaxBufferSize),
		}

		var device *wgpu.Device
		var err error
		succeeded := false
		for i, storage := range storageAttempts {
			requiredLimits.MaxStorageBufferBindingSize = storage
			requiredLimits.MaxBufferSize = bufferAttempts[i]
			Alog(fmt.Sprintf("Requesting Device [Storage:%dMB, Buffer:%dMB] Backend: %s",
				storage/(1024*1024), bufferAttempts[i]/(1024*1024),
				adapter.GetInfo().BackendType))
			device, err = adapter.RequestDevice(&wgpu.DeviceDescriptor{
				RequiredLimits:   &wgpu.RequiredLimits{Limits: requiredLimits},
				RequiredFeatures: requiredFeatures,
			})
			if err == nil {
				Alog(fmt.Sprintf("✅ Device acquired at %dMB storage tier.", storage/(1024*1024)))
				succeeded = true
				break
			}
			Alog(fmt.Sprintf("⚠️  Tier %d failed: %v", i+1, err))
		}
		if !succeeded {
			Alog("⚠️  All limit tiers failed. Falling back to adapter defaults.")
			device, err = adapter.RequestDevice(nil)
			if err != nil {
				return fmt.Errorf("FATAL: device request failed: %v", err)
			}
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
		HasTimestampQuery: device.HasFeature(wgpu.FeatureNameTimestampQuery),
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

// BindGroupKeyHash generates a stable hash for a set of buffers (or bindings) and a pipeline.
func BindGroupKeyHash(pipeline *wgpu.ComputePipeline, bindings ...any) uint64 {
	h := uint64(uintptr(unsafe.Pointer(pipeline)))
	for _, b := range bindings {
		if b == nil {
			continue
		}
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
	submission := c.Queue.Submit(cmd)
	c.Device.Poll(true, &wgpu.WrappedSubmissionIndex{Queue: c.Queue, SubmissionIndex: submission})

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

// TimeCommands records compute work into a single command buffer, wraps it in
// timestamp queries, and returns the measured on-device duration.
func (c *WGPUContext) TimeCommands(label string, record func() error) (time.Duration, error) {
	if c == nil || !c.HasTimestampQuery {
		return 0, fmt.Errorf("timestamp queries unavailable")
	}
	if c.ActiveEncoder != nil {
		return 0, fmt.Errorf("timestamp timing requires idle encoder")
	}

	querySet, err := c.Device.CreateQuerySet(&wgpu.QuerySetDescriptor{
		Label: label + "_timestamps",
		Type:  wgpu.QueryTypeTimestamp,
		Count: 2,
	})
	if err != nil {
		return 0, err
	}
	defer querySet.Release()

	resolveSize := uint64(wgpu.QueryResolveBufferAlignment)
	resolveBuf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_resolve",
		Size:  resolveSize,
		Usage: wgpu.BufferUsageQueryResolve | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return 0, err
	}
	defer resolveBuf.Destroy()

	stagingBuf, err := c.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label + "_staging",
		Size:  resolveSize,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return 0, err
	}
	defer stagingBuf.Destroy()

	if err := c.BeginFrame(); err != nil {
		return 0, err
	}
	if err := c.ActiveEncoder.WriteTimestamp(querySet, 0); err != nil {
		c.ActiveEncoder = nil
		return 0, err
	}
	if err := record(); err != nil {
		c.ActiveEncoder = nil
		return 0, err
	}
	if err := c.ActiveEncoder.WriteTimestamp(querySet, 1); err != nil {
		c.ActiveEncoder = nil
		return 0, err
	}
	if err := c.ActiveEncoder.ResolveQuerySet(querySet, 0, 2, resolveBuf, 0); err != nil {
		c.ActiveEncoder = nil
		return 0, err
	}
	c.ActiveEncoder.CopyBufferToBuffer(resolveBuf, 0, stagingBuf, 0, 16)

	cmd, _ := c.ActiveEncoder.Finish(nil)
	submission := c.Queue.Submit(cmd)
	c.ActiveEncoder = nil
	for _, buf := range c.PendingDestroys {
		buf.Destroy()
	}
	c.PendingDestroys = c.PendingDestroys[:0]
	c.UniformIdx = 0

	c.Device.Poll(true, &wgpu.WrappedSubmissionIndex{Queue: c.Queue, SubmissionIndex: submission})

	done := make(chan struct{})
	if err := stagingBuf.MapAsync(wgpu.MapModeRead, 0, 16, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	}); err != nil {
		return 0, err
	}

	for {
		c.Device.Poll(false, nil)
		select {
		case <-done:
			data := stagingBuf.GetMappedRange(0, 16)
			defer stagingBuf.Unmap()
			stamps := make([]uint64, 2)
			copy(wgpu.ToBytes(stamps), data[:16])
			if stamps[1] < stamps[0] {
				return 0, fmt.Errorf("timestamp query underflow")
			}
			return time.Duration(stamps[1] - stamps[0]), nil
		default:
		}
	}
}
