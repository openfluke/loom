//go:build arm64

package simd

import "unsafe"

//go:noescape
func bitNetTernaryDotAccum(codes *uint8, acts *int8, chunks int, out *int32)

// bitNetTernaryCodeRowDotSimd is the arm64 baseline-NEON BitNet MAD kernel.
// bitNetTernaryDotAccum (bitnet_ternary_arm64.s) runs the SMULL/SMULL2/SADALP
// inner loop over 16-byte chunks into eight int32 lane partials; the reduction
// and scalar tail below run in Go. Because the products are exact integers, the
// summation order is irrelevant — this is bit-identical to the amd64 AVX2 kernel
// and to bitNetTernaryCodeRowDotGo.
func bitNetTernaryCodeRowDotSimd(codes *uint8, acts *int8, nBytes int) int32 {
	if codes == nil || acts == nil || nBytes <= 0 {
		return 0
	}

	var p [8]int32
	chunks := nBytes >> 4 // nBytes / 16
	if chunks > 0 {
		bitNetTernaryDotAccum(codes, acts, chunks, &p[0])
	}
	sum := p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7]

	// Scalar tail for a final < 16-byte remainder (nBytes is a multiple of 32
	// in the BitNet path, so this normally does nothing).
	done := chunks << 4
	if done < nBytes {
		c := unsafe.Slice(codes, nBytes)
		a := unsafe.Slice(acts, nBytes)
		for i := done; i < nBytes; i++ {
			sum += int32(c[i]) * int32(a[i])
		}
	}
	return sum
}
