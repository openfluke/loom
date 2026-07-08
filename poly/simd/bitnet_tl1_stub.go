//go:build !arm64

package simd

// BitNetTL1Available reports whether the TL1 LUT matvec kernel exists on this GOARCH.
func BitNetTL1Available() bool { return false }

// BitNetTL1RowDot is unused when BitNetTL1Available is false.
func BitNetTL1RowDot(nibbles []uint8, qlut []int16, pairCount int, tailCode uint8, tailAct int8) int32 {
	return 0
}

// BitNetTL1MatVecBatched is unused on non-arm64 builds.
func BitNetTL1MatVecBatched(nibbles []uint8, rowStride, rows, cols int, qlut []int16, tails []uint8, tailAct int8, out []float64, outputScale float64) {
}
