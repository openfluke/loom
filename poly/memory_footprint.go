package poly

import "fmt"

// MemoryFootprint reports Poly-accounted host model weights and GPU weights + KV only.
// It does not include Go runtime stats, CPU-resident KV tensors, activation pools, or driver overhead.
type MemoryFootprint struct {
	HostWeightsMB float64 // CPU layer weights + embeddings + LM head + final norm (see calculateHostModelBytes)
	GPUWeightsMB  float64 // GPU weights/scales/embeddings (see GetVRAMWeightsBytes)
	GPUKVMB       float64 // GPU KV buffers (see GetVRAMKVCacheBytes)
}

// NewMemoryFootprintFromTransformer captures host weights and GPU weights + KV for tr.
func NewMemoryFootprintFromTransformer[T Numeric](tr *Transformer[T]) MemoryFootprint {
	var out MemoryFootprint
	if tr == nil {
		return out
	}
	out.HostWeightsMB = float64(tr.calculateHostModelBytes()) / (1024 * 1024)
	if tr.Network != nil {
		out.GPUWeightsMB = float64(tr.Network.GetVRAMWeightsBytes()) / (1024 * 1024)
		out.GPUKVMB = float64(tr.Network.GetVRAMKVCacheBytes()) / (1024 * 1024)
	}
	return out
}

// FormatOneLine returns a compact single-line summary for metrics footers.
func (m MemoryFootprint) FormatOneLine() string {
	return fmt.Sprintf(
		"host weights %.2f MB | GPU weights %.2f MB | GPU KV %.2f MB",
		m.HostWeightsMB,
		m.GPUWeightsMB,
		m.GPUKVMB,
	)
}

// FormatDetailed returns a multi-line block for logs / post-load banners.
func (m MemoryFootprint) FormatDetailed() string {
	return fmt.Sprintf(
		"📊 Memory: host weights %8.2f MB | GPU weights %8.2f MB | GPU KV %8.2f MB\n"+
			"   (Extra GPU VRAM may be used for activation/uniform pools; not listed here.)\n",
		m.HostWeightsMB,
		m.GPUWeightsMB,
		m.GPUKVMB,
	)
}

// PrintTransformerMemoryFootprint prints FormatDetailed. The includeHeapDetail argument is
// deprecated and ignored (kept for call-site compatibility).
func PrintTransformerMemoryFootprint[T Numeric](tr *Transformer[T], includeHeapDetail bool) {
	_ = includeHeapDetail
	m := NewMemoryFootprintFromTransformer(tr)
	fmt.Print(m.FormatDetailed())
}
