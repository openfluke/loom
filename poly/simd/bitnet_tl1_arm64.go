//go:build arm64

package simd

//go:noescape
func bitNetTL1RowDotAccum(nibbles *uint8, qlut *int16, pairCount int, tailCode uint8, tailAct int8) int32

//go:noescape
func bitNetTL1PairBatch16(indices *uint8, lut *int16, acc *int32, n int)

const bitNetTL1BatchRows = 16

// BitNetTL1Available reports whether the TL1 LUT matvec kernel exists on this GOARCH.
func BitNetTL1Available() bool { return true }

// BitNetTL1RowDot returns the int32 dot product for one weight row via TL1 lookup.
func BitNetTL1RowDot(nibbles []uint8, qlut []int16, pairCount int, tailCode uint8, tailAct int8) int32 {
	return BitNetTL1RowDotGo(nibbles, qlut, pairCount, tailCode, tailAct)
}

// BitNetTL1MatVecBatched computes out[0:rows] = matrix * xq using TL1 LUT batching
// along the output-row dimension. nibbles is row-major with rowStride bytes/row.
func BitNetTL1MatVecBatched(nibbles []uint8, rowStride, rows, cols int, qlut []int16, tails []uint8, tailAct int8, out []float64, outputScale float64) {
	if rows <= 0 || cols <= 0 {
		return
	}
	fullPairs := cols / 2
	var idxBuf [bitNetTL1BatchRows]uint8

	for r0 := 0; r0 < rows; r0 += bitNetTL1BatchRows {
		rn := bitNetTL1BatchRows
		if r0+rn > rows {
			rn = rows - r0
		}
		var acc [bitNetTL1BatchRows]int32
		for p := 0; p < fullPairs; p++ {
			lut := qlut[p*16:]
			for ri := 0; ri < rn; ri++ {
				idxBuf[ri] = nibbleAt(nibbles[(r0+ri)*rowStride:], p)
			}
			bitNetTL1PairBatch16(&idxBuf[0], &lut[0], &acc[0], rn)
		}
		if cols&1 == 1 {
			for ri := 0; ri < rn; ri++ {
				tc := tails[r0+ri]
				if tc <= 2 && tc != 1 {
					w := int32(tc) - 1
					acc[ri] += w * int32(tailAct)
				}
			}
		}
		for ri := 0; ri < rn; ri++ {
			out[r0+ri] = float64(acc[ri]) * outputScale
		}
	}
}
