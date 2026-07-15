package simd

// DotQ4_0Row computes prev + sum_{i=0..n-1} in[i] * (q[i] * scale_block(baseW+i))
// where q/scale come from PackQ4_0GPU layout (32-weight blocks, 8 nibbles per uint32).
// Fused: does not expand a full FP32 weight row.
func DotQ4_0Row(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	if n <= 0 || len(in) < n {
		return prev
	}
	if baseW < 0 || (baseW+n+7)/8 > len(packed) {
		return prev
	}
	lastScale := (baseW + n - 1) / 32
	if lastScale >= len(scales) {
		return prev
	}
	if simdEnabled() {
		return dotQ4_0RowSimd(in, scales, packed, baseW, n, prev)
	}
	return dotQ4_0RowGo(in, scales, packed, baseW, n, prev)
}

// DotQ4_0Rows4 writes 4 consecutive matrix-row dots into out[0:4].
// baseW is the flat index of row 0; row r uses baseW + r*n. Keeps activations hot across rows.
func DotQ4_0Rows4(in []float32, scales []float32, packed []uint32, baseW, n int, out []float32) {
	if len(out) < 4 || n <= 0 || len(in) < n {
		return
	}
	if simdEnabled() && baseW%32 == 0 && n%32 == 0 {
		dotQ4_0Rows4Simd(in, scales, packed, baseW, n, out)
		return
	}
	for r := 0; r < 4; r++ {
		out[r] = float32(DotQ4_0Row(in, scales, packed, baseW+r*n, n, 0))
	}
}

// dotQ4_0RowGo is the portable fused scalar kernel (sum(in*q)*scale per 8-nibble word).
func dotQ4_0RowGo(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	limit := n / 8
	for k := 0; k < limit; k++ {
		globalIdx := baseW + i
		scale := float64(scales[globalIdx/32])
		w := packed[globalIdx/8]
		acc := 0.0
		for nib := 0; nib < 8; nib++ {
			q := int32((w >> (uint(nib) * 4)) & 0xF)
			if q > 7 {
				q -= 16
			}
			acc += float64(in[i+nib]) * float64(q)
		}
		sum += acc * scale
		i += 8
	}
	for ; i < n; i++ {
		globalIdx := baseW + i
		scale := float64(scales[globalIdx/32])
		nibble := (globalIdx % 8) * 4
		q := int32((packed[globalIdx/8] >> uint(nibble)) & 0xF)
		if q > 7 {
			q -= 16
		}
		sum += float64(in[i]) * float64(q) * scale
	}
	return sum
}

// q4UnpackWord8 writes 8 signed-nibble weights * scale into dst[0:8].
func q4UnpackWord8(word uint32, scale float32, dst []float32) {
	for nib := 0; nib < 8; nib++ {
		q := int32((word >> (uint(nib) * 4)) & 0xF)
		if q > 7 {
			q -= 16
		}
		dst[nib] = float32(q) * scale
	}
}
