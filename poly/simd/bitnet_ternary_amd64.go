//go:build amd64

package simd

//go:noescape
func bitNetTernaryCodeRowDotSimd(codes *uint8, acts *int8, nBytes int) int32
