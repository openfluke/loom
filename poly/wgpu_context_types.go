package poly

import (
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// WGPUContext manages the GPU device and queue for acceleration.
type WGPUContext struct {
	Instance          *wgpu.Instance
	Adapter           *wgpu.Adapter
	Device            *wgpu.Device
	Queue             *wgpu.Queue
	PipelineCache     map[string]*wgpu.ComputePipeline
	ActivationPool    map[string]*wgpu.Buffer
	GPUTileSize       int
	ActiveEncoder     *wgpu.CommandEncoder
	ActivePass        *wgpu.ComputePassEncoder // fused compute pass while ActiveEncoder is open
	PendingDestroys   []*wgpu.Buffer
	LayoutCache       map[string]*wgpu.BindGroupLayout
	BindGroupCache    map[uint64]*wgpu.BindGroup
	UniformPool       []*wgpu.Buffer
	UniformIdx        int
	// UniformSticky reuses the same uniform buffer (and thus BindGroup cache entries)
	// for identical params within one BeginFrame — critical for chunked decode.
	UniformSticky     map[string]*wgpu.Buffer
	Limits            wgpu.Limits
	BlankBuffer       *wgpu.Buffer
	HasTimestampQuery bool
	FrameCount        uint32
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

// BindGroupKey is used for the BindGroupCache.
type BindGroupKey struct {
	Pipeline *wgpu.ComputePipeline
	Buffers  []*wgpu.Buffer
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
