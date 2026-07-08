//go:build !arm64

package simd

// BitNetPackedAvailable reports whether the packed-2-bit ternary kernel exists for
// this GOARCH. Only arm64 has it; amd64 keeps the byte-code AVX2 (VPMADDUBSW) path.
func BitNetPackedAvailable() bool { return false }

// BitNetTernaryPackedRowDot is never called when BitNetPackedAvailable is false;
// it exists so the poly dispatch compiles on every arch.
func BitNetTernaryPackedRowDot(packed []uint8, acts []int8, blocks int) int32 {
	return 0
}
