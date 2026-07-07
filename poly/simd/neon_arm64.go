//go:build arm64

package simd

func simdEnabled() bool { return true }

func dotTileSimd(x, w *float32, n int, prev float64) float64 {
	sum := prev
	i := 0
	for i+4 <= n {
		sum += float64(x[i])*float64(w[i]) +
			float64(x[i+1])*float64(w[i+1]) +
			float64(x[i+2])*float64(w[i+2]) +
			float64(x[i+3])*float64(w[i+3])
		i += 4
	}
	for i < n {
		sum += float64(x[i]) * float64(w[i])
		i++
	}
	return sum
}
