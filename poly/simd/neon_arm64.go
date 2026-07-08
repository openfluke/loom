//go:build arm64

package simd

import "unsafe"

func simdEnabled() bool { return true }

//go:noescape
func dotSimdAccum8(x, w *float32, blocks int, out *float64)

// dotTileSimd computes a float64-accumulated dot product using a real NEON
// kernel: dotSimdAccum8 runs the hot 8-wide FMA loop (FCVTL widen + VFMLA), and
// the horizontal reduction + prev add + scalar tail below mirror the amd64 AVX2
// kernel's exact operation order. Because float32*float32 products are exact in
// float64, this arm64 path is bit-identical to dotF32AccF64Avx2 on amd64.
func dotTileSimd(x, w *float32, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}

	var p [8]float64
	blocks := n >> 3 // n / 8
	if blocks > 0 {
		dotSimdAccum8(x, w, blocks, &p[0])
	}

	// Horizontal reduce, matching AVX2: VADDPD Y1,Y0 → cross-128 add → pairwise.
	s0 := p[0] + p[4]
	s1 := p[1] + p[5]
	s2 := p[2] + p[6]
	s3 := p[3] + p[7]
	t0 := s0 + s2
	t1 := s1 + s3
	acc := t0 + t1
	acc += prev

	// Scalar tail for the remaining < 8 elements (AVX2 adds these after prev).
	done := blocks << 3
	if done < n {
		xs := unsafe.Slice(x, n)
		ws := unsafe.Slice(w, n)
		for i := done; i < n; i++ {
			acc += float64(xs[i]) * float64(ws[i])
		}
	}
	return acc
}
