//go:build !amd64 && !arm64

package dot

func f32AccF64(x, w []float32, n int) float64 {
	return F32TileAccF64Go(x, w, n)
}
