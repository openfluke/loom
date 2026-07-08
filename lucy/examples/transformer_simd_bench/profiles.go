package tfsimdbench

import "github.com/openfluke/loom/poly"

// BenchProfile is one CPU execution mode. All are CPU-only (UseGPU is never set);
// the matrix crosses scalar/SIMD forward with single-/multi-core tiling.
type BenchProfile struct {
	Name      string // short label used in tables / filenames
	MultiCore bool   // EnableMultiCoreTiling (parallel across cores)
	Simd      bool   // Plan 9 SIMD forward (SetSimdForwardRecursive)
}

// BenchProfiles is the full 2×2 CPU matrix. cpu_sc is the scalar single-core
// baseline every other cell is compared against for speedup + output parity.
var BenchProfiles = []BenchProfile{
	{Name: "cpu_sc", MultiCore: false, Simd: false},
	{Name: "cpu_mc", MultiCore: true, Simd: false},
	{Name: "cpu_simd_sc", MultiCore: false, Simd: true},
	{Name: "cpu_simd_mc", MultiCore: true, Simd: true},
}

// applyProfile mutates an already-loaded CPU transformer in place to match prof.
// The model stays resident across profiles (CPU host weights are not released),
// so we only flip tiling / SIMD flags, refresh tile sizes, and clear the KV cache.
func applyProfile(tr *poly.Transformer[float32], prof BenchProfile) {
	net := tr.Network
	net.UseGPU = false
	tr.ForwardMode = poly.TransformerForwardNormal

	net.EnableMultiCoreTiling = prof.MultiCore
	tr.EnableTiling(-1) // cache-block all layers; MC flag below controls parallelism
	for i := range net.Layers {
		l := &net.Layers[i]
		l.UseTiling = true
		l.EnableMultiCoreTiling = prof.MultiCore
	}
	net.SetSimdForwardRecursive(prof.Simd)
	net.RefreshRuntimeTileSizes()
	tr.Reset()
}
