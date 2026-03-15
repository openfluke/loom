//go:build !js

package poly

import "github.com/openfluke/webgpu/wgpu"

func getBufferUsage(buf *wgpu.Buffer) wgpu.BufferUsage {
	return buf.GetUsage()
}
