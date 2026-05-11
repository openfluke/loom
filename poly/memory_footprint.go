package poly

import (
	"fmt"
	"runtime"
)

// MemoryFootprint summarizes host and GPU memory for a loaded Transformer + network.
//
// HostProcessMB uses runtime.MemStats.Sys (Go runtime virtual memory obtained from OS;
// this is NOT true resident set size / RSS).
// HeapInuseMB / HeapIdleMB break down the Go heap: idle is retained empty spans — not model weights.
// HostModelMB sums accounted CPU weight slices (often ~0 after ReleaseInferenceHostWeights).
// VRAMMB is from VolumetricNetwork.GetVRAMUsage (weights, KV, pools, embeddings on GPU).
type MemoryFootprint struct {
	HostProcessMB float64
	HeapAllocMB   float64
	HeapInuseMB   float64
	HeapIdleMB    float64
	HeapReleasedMB float64
	HostModelMB   float64
	VRAMMB        float64
}

// NewMemoryFootprintFromTransformer captures current host + GPU usage for tr.
func NewMemoryFootprintFromTransformer[T Numeric](tr *Transformer[T]) MemoryFootprint {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	var modelMB, vramMB float64
	if tr != nil && tr.Network != nil {
		modelMB = float64(tr.calculateHostModelBytes()) / (1024 * 1024)
		vramMB = float64(tr.Network.GetVRAMUsage()) / (1024 * 1024)
	}

	return MemoryFootprint{
		HostProcessMB:  float64(ms.Sys) / (1024 * 1024),
		HeapAllocMB:    float64(ms.HeapAlloc) / (1024 * 1024),
		HeapInuseMB:    float64(ms.HeapInuse) / (1024 * 1024),
		HeapIdleMB:     float64(ms.HeapIdle) / (1024 * 1024),
		HeapReleasedMB: float64(ms.HeapReleased) / (1024 * 1024),
		HostModelMB:    modelMB,
		VRAMMB:         vramMB,
	}
}

// CombinedHostPlusVRAMMB is RSS plus Poly-reported VRAM. On discrete GPUs this approximates
// cross-device footprint; on unified memory (e.g. Apple Silicon) physical RAM may be shared
// between CPU and GPU so this sum can over-estimate unique DRAM use.
func (m MemoryFootprint) CombinedHostPlusVRAMMB() float64 {
	return m.HostProcessMB + m.VRAMMB
}

// FormatOneLine returns a compact single-line summary for metrics footers.
func (m MemoryFootprint) FormatOneLine() string {
	return fmt.Sprintf(
		"RSS %.2f MB (Go heap in use %.2f MB) | GPU %.2f MB | Poly CPU weights %.2f MB",
		m.HostProcessMB,
		m.HeapInuseMB,
		m.VRAMMB,
		m.HostModelMB,
	)
}

// FormatDetailed returns a multi-line block for logs / post-load banners.
func (m MemoryFootprint) FormatDetailed() string {
	return fmt.Sprintf(
		"📊 Memory footprint\n"+
			"   Go runtime Sys:          %8.2f MB  (not OS RSS)\n"+
			"     └ Go heap in use:      %8.2f MB  (live objects)\n"+
			"     └ Go heap idle:        %8.2f MB  (retained spans — not duplicate weights)\n"+
			"   Poly CPU weights:        %8.2f MB  (usually 0 after GPU-resident load)\n"+
			"   GPU (Poly VRAM report):  %8.2f MB\n"+
			"   ─────────────────────────────────────\n"+
			"   Sys + VRAM (info only):  %8.2f MB  (not physical-RSS sum)\n"+
			"\n"+
			"   Note: Same model ⇒ similar VRAM; block-wise load mainly cuts peak *decode* RAM\n"+
			"   (full safetensors map) and CPU weight mirrors — not Go’s idle heap or drivers.\n",
		m.HostProcessMB,
		m.HeapInuseMB,
		m.HeapIdleMB,
		m.HostModelMB,
		m.VRAMMB,
		m.CombinedHostPlusVRAMMB(),
	)
}

// PrintTransformerMemoryFootprint prints FormatDetailed plus optional raw heap stats.
func PrintTransformerMemoryFootprint[T Numeric](tr *Transformer[T], includeHeapDetail bool) {
	m := NewMemoryFootprintFromTransformer(tr)
	fmt.Print(m.FormatDetailed())
	if includeHeapDetail {
		fmt.Printf(
			"   Go heap detail: alloc=%.2f MB | inuse=%.2f MB | idle=%.2f MB | released=%.2f MB\n",
			m.HeapAllocMB, m.HeapInuseMB, m.HeapIdleMB, m.HeapReleasedMB,
		)
	}
}
