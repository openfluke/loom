package pods

// GPUHooks describes the optional GPU backend. Keep it slice-based so CPU fallback is easy.
type GPUHooks interface {
	DispatchScanU32(in []uint32, inclusive bool) ([]uint32, error)
	DispatchReduceF32(in []float32, kind string) (float32, error) // kind: "sum"|"max"|"mean"
	DispatchSoftmaxF32(in []float32) ([]float32, error)
}

// Default to a no-op GPU so everything builds/runs without tags.
var GPU GPUHooks = noopGPU{}

type noopGPU struct{}

func (noopGPU) DispatchScanU32([]uint32, bool) ([]uint32, error)     { return nil, ErrNoGPU }
func (noopGPU) DispatchReduceF32([]float32, string) (float32, error) { return 0, ErrNoGPU }
func (noopGPU) DispatchSoftmaxF32([]float32) ([]float32, error)      { return nil, ErrNoGPU }

// (Later) provide a gpu_wgpu.go with `//go:build gpu` that sets `GPU = realBackend{...}`
