//go:build !amd64

package simd

import "unsafe"

// bitNetTernaryCodeRowDotSimd falls back to the scalar Go dot on non-amd64
// (e.g. arm64, where simdEnabled() is true but no ternary asm exists yet).
func bitNetTernaryCodeRowDotSimd(codes *uint8, acts *int8, nBytes int) int32 {
	if codes == nil || acts == nil || nBytes <= 0 {
		return 0
	}
	c := unsafe.Slice(codes, nBytes)
	a := unsafe.Slice(acts, nBytes)
	return bitNetTernaryCodeRowDotGo(c, a, nBytes)
}
