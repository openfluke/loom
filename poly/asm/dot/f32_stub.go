//go:build !amd64 && !arm64

package dot

func f32(x, w []float32, n int) float32 {
	return F32Go(x, w, n)
}
