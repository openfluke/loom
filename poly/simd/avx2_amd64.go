//go:build amd64

package simd

func simdEnabled() bool { return true }

//go:noescape
func dotF32AccF64Avx2(x, w *float32, n int, prev float64) float64

func dotTileSimd(x, w *float32, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	return dotF32AccF64Avx2(x, w, n, prev)
}
