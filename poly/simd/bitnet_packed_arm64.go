//go:build arm64

package simd

//go:noescape
func bitNetTernaryPackedDotAccum(packed *uint8, acts *int8, blocks int, out *int32)

// BitNetPackedAvailable reports whether the packed-2-bit ternary kernel exists
// for this GOARCH (arm64 NEON only for now; amd64 uses the byte-code AVX2 path).
func BitNetPackedAvailable() bool { return true }

// BitNetTernaryPackedRowDot returns sum(code*act) for one weight row read directly
// from the 2-bit packed layout. packed must hold blocks*16 bytes (4 codes/byte),
// acts must hold blocks*64 int8 in column order, zero-padded past the real column
// count. The result is exact integer, so it is bit-identical to the byte-code and
// scalar paths.
func BitNetTernaryPackedRowDot(packed []uint8, acts []int8, blocks int) int32 {
	if blocks <= 0 || len(packed) < blocks*16 || len(acts) < blocks*64 {
		return 0
	}
	var p [4]int32
	bitNetTernaryPackedDotAccum(&packed[0], &acts[0], blocks, &p[0])
	return p[0] + p[1] + p[2] + p[3]
}
