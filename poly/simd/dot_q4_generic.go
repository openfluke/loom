//go:build !amd64 && !arm64

package simd

func dotQ4_0RowSimd(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	return dotQ4_0RowFusedTile(in, scales, packed, baseW, 0, n, prev)
}

func dotQ4_0Rows4Simd(in []float32, scales []float32, packed []uint32, baseW, n int, out []float32) {
	for r := 0; r < 4; r++ {
		out[r] = float32(dotQ4_0RowFusedTile(in, scales, packed, baseW+r*n, 0, n, 0))
	}
}
