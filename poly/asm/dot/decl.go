//go:build amd64 || arm64

package dot

import "github.com/openfluke/loom/poly/asm"

//go:noescape
func dotF32(x, w *float32, n int) float32

//go:noescape
func dotF64(x, w *float64, n int) float64

func f32(x, w []float32, n int) float32 {
	if asm.Enabled() {
		return dotF32(&x[0], &w[0], n)
	}
	return F32Go(x, w, n)
}

func f64(x, w []float64, n int) float64 {
	if asm.Enabled() {
		return dotF64(&x[0], &w[0], n)
	}
	return F64Go(x, w, n)
}
