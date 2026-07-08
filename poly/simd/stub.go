//go:build !amd64 && !arm64

package simd

func simdEnabled() bool { return false }

func dotTileSimd(x, w *float32, n int, prev float64) float64 {
	_ = x
	_ = w
	_ = n
	return prev
}
