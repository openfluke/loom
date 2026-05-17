//go:build !amd64 && !arm64

package dot

func f64(x, w []float64, n int) float64 {
	return F64Go(x, w, n)
}
