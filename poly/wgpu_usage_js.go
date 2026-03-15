//go:build js

package poly

import "github.com/openfluke/webgpu/wgpu"

func getBufferUsage(buf *wgpu.Buffer) wgpu.BufferUsage {
	// wgpu.Buffer in the JS backend does not expose GetUsage().
	// Return 0 so the buffer pool check falls through to size-only matching.
	return 0
}
