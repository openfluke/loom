//go:build amd64 || arm64

package dot

import "github.com/openfluke/loom/poly/asm"

//go:noescape
func dotF32AccF64(x, w *float32, n int) float64

func f32AccF64(x, w []float32, n int) float64 {
	if asm.Enabled() {
		return dotF32AccF64(&x[0], &w[0], n)
	}
	return F32TileAccF64Go(x, w, n)
}
