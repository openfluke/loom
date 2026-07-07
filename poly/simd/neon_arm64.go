//go:build arm64

package simd

import "unsafe"

func simdEnabled() bool { return true }

// dotTileSimd computes a float64-accumulated dot product with a 4-wide
// unrolled loop. Go's arm64 assembler lacks the float32->float64 widening
// (VFCVTL) and vector-double add (VFADD) NEON forms, so we express the kernel
// in Go and let the backend emit NEON FMLA/FADD. Accumulation is done in
// float64 to match the amd64 AVX2 kernel and the C++ f32AccF64 reference.
func dotTileSimd(x, w *float32, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	xs := unsafe.Slice(x, n)
	ws := unsafe.Slice(w, n)
	sum := prev
	i := 0
	for i+4 <= n {
		sum += float64(xs[i])*float64(ws[i]) +
			float64(xs[i+1])*float64(ws[i+1]) +
			float64(xs[i+2])*float64(ws[i+2]) +
			float64(xs[i+3])*float64(ws[i+3])
		i += 4
	}
	for i < n {
		sum += float64(xs[i]) * float64(ws[i])
		i++
	}
	return sum
}
