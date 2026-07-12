//go:build arm64

package simd

import "unsafe"

func dotU8AccSimd(q, k *uint8, n int, prev int32) int32 {
	if q == nil || k == nil || n <= 0 {
		return prev
	}
	qs := unsafe.Slice(q, n)
	ks := unsafe.Slice(k, n)
	sum := prev
	for i := 0; i < n; i++ {
		sum += int32(qs[i]) * int32(ks[i])
	}
	return sum
}
