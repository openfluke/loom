package simd

// dotQ4_0RowFusedTile unpacks 8 weights at a time into a stack scratch and uses DotTile.
// Still no full-row FP32 Master — only 8 floats of scratch.
func dotQ4_0RowFusedTile(in []float32, scales []float32, packed []uint32, baseW, i0, i1 int, prev float64) float64 {
	sum := prev
	var qs [8]float32
	i := i0
	for i+8 <= i1 {
		globalIdx := baseW + i
		q4UnpackWord8(packed[globalIdx/8], scales[globalIdx/32], qs[:])
		sum = DotTile(in[i:i+8], qs[:], 0, 8, sum)
		i += 8
	}
	for ; i < i1; i++ {
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
