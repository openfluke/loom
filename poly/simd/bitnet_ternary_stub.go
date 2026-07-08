//go:build !amd64 && !arm64

package simd

import "unsafe"

// bitNetTernaryCodeRowDotSimd falls back to the scalar Go dot on architectures
// without a ternary asm kernel (amd64 has AVX2, arm64 has NEON).
func bitNetTernaryCodeRowDotSimd(codes *uint8, acts *int8, nBytes int) int32 {
	if codes == nil || acts == nil || nBytes <= 0 {
		return 0
	}
	c := unsafe.Slice(codes, nBytes)
	a := unsafe.Slice(acts, nBytes)
	return bitNetTernaryCodeRowDotGo(c, a, nBytes)
}
