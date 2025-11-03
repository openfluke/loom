//go:build gpu

package pods

import (
	"github.com/openfluke/webgpu/wgpu"
)

// WGSL-backed implementation shell. You’ll fill these in when your kernels are ready.
type WGPU struct {
	Device *wgpu.Device
	Queue  *wgpu.Queue
	// pipelines, bind group layouts, etc.
}

func (g *WGPU) DispatchScanU32(inCount int, inclusive bool) error { /* TODO: record+submit */
	return nil
}
func (g *WGPU) DispatchReduceF32(n int, kind string) (float32, error) { /* TODO */ return 0, nil }
func (g *WGPU) DispatchGEMM(m, n, k int, a, b, c uintptr, alpha, beta float32) error { /* TODO */
	return nil
}
func (g *WGPU) DispatchSoftmaxF32(n int, logitsPtr uintptr) error { /* TODO */ return nil }
