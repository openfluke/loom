//go:build arm64

package simd

import "unsafe"

func dotI8AccSimd(q, k *int8, n int, prev int32) int32 {
	if q == nil || k == nil || n <= 0 {
		return prev
	}

	var p [8]int32
	chunks := n >> 4
	if chunks > 0 {
		// SMULL/SADALP kernel is identical for signed int8×int8 products.
		bitNetTernaryDotAccum((*uint8)(unsafe.Pointer(q)), k, chunks, &p[0])
	}
	sum := prev + p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7]

	done := chunks << 4
	if done < n {
		qs := unsafe.Slice(q, n)
		ks := unsafe.Slice(k, n)
		for i := done; i < n; i++ {
			sum += int32(qs[i]) * int32(ks[i])
		}
	}
	return sum
}
