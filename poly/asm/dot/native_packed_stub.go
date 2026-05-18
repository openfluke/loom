//go:build !amd64

package dot

func nibblePackedRowNativeI64(x []uint8, packed []uint32, rowOff, n int) int64 {
	return NibblePackedRowNativeI64Go(x, packed, rowOff, n)
}

func twoBitPackedRowNativeI64(x []uint8, packed []uint32, rowOff, n int, ternary bool) int64 {
	return TwoBitPackedRowNativeI64Go(x, packed, rowOff, n, ternary)
}

func binaryPackedRowNativeI64(x []uint8, packed []uint32, rowOff, n int) int64 {
	return BinaryPackedRowNativeI64Go(x, packed, rowOff, n)
}
