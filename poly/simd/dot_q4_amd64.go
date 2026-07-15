//go:build amd64

package simd

//go:noescape
func q4BlockDot32Avx2(in *float32, packed4 *uint32, scale float32) float64

func dotQ4_0RowSimd(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	if baseW%32 == 0 {
		for i+32 <= n {
			block := (baseW + i) / 32
			sum += q4BlockDot32Avx2(&in[i], &packed[(baseW+i)/8], scales[block])
			i += 32
		}
	}
	if i < n {
		sum = dotQ4_0RowFusedTile(in, scales, packed, baseW, i, n, sum)
	}
	return sum
}

// dotQ4_0Rows4Simd walks columns once; for each 32-block applies AVX2 to 4 weight rows.
func dotQ4_0Rows4Simd(in []float32, scales []float32, packed []uint32, baseW, n int, out []float32) {
	var acc [4]float64
	for i := 0; i < n; i += 32 {
		inBlk := &in[i]
		for r := 0; r < 4; r++ {
			bw := baseW + r*n + i
			acc[r] += q4BlockDot32Avx2(inBlk, &packed[bw/8], scales[bw/32])
		}
	}
	out[0] = float32(acc[0])
	out[1] = float32(acc[1])
	out[2] = float32(acc[2])
	out[3] = float32(acc[3])
}
