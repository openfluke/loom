package simd

// DotTile computes sum(x[i]*w[i] for i in [i0,i1)) + prev with float64 accumulation.
func DotTile(x, w []float32, i0, i1 int, prev float64) float64 {
	if i0 >= i1 {
		return prev
	}
	n := i1 - i0
	if simdEnabled() {
		return dotTileSimd(&x[i0], &w[i0], n, prev)
	}
	return dotTileGo(x, w, i0, i1, prev)
}

func dotTileGo(x, w []float32, i0, i1 int, prev float64) float64 {
	sum := prev
	for i := i0; i < i1; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

// SimdEnabled reports whether vector kernels are linked for this GOARCH.
func SimdEnabled() bool {
	return simdEnabled()
}
