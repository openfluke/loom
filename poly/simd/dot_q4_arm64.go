//go:build arm64

package simd

// dotQ4_0RowSimd — ARM64 fused Q4: unpack one 32-weight block, then NEON DotTile.
// amd64 has in-register AVX2 nibble math; here we keep the Go unpack (correct vs
// PackQ4_0GPU) and ride the existing arm64 DotTile FMA path over 32 floats —
// much better than the old 8-wide fused-tile stub.
func dotQ4_0RowSimd(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	if baseW%32 == 0 {
		var qs [32]float32
		for i+32 <= n {
			block := (baseW + i) / 32
			q4UnpackBlock32(packed[(baseW+i)/8:], scales[block], qs[:])
			sum = DotTile(in[i:i+32], qs[:], 0, 32, sum)
			i += 32
		}
	}
	if i < n {
		sum = dotQ4_0RowFusedTile(in, scales, packed, baseW, i, n, sum)
	}
	return sum
}

func dotQ4_0Rows4Simd(in []float32, scales []float32, packed []uint32, baseW, n int, out []float32) {
	for r := 0; r < 4; r++ {
		out[r] = float32(dotQ4_0RowSimd(in, scales, packed, baseW+r*n, n, 0))
	}
}

// q4UnpackBlock32 expands 4 packed uint32 words (32 signed nibbles) * scale into dst[0:32].
func q4UnpackBlock32(packed4 []uint32, scale float32, dst []float32) {
	if len(packed4) < 4 || len(dst) < 32 {
		return
	}
	o := 0
	for w := 0; w < 4; w++ {
		word := packed4[w]
		for nib := 0; nib < 8; nib++ {
			q := int32((word >> (uint(nib) * 4)) & 0xF)
			if q > 7 {
				q -= 16
			}
			dst[o] = float32(q) * scale
			o++
		}
	}
}
