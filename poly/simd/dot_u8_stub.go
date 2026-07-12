//go:build !amd64 && !arm64

package simd

func dotU8AccSimd(q, k *uint8, n int, prev int32) int32 {
	if q == nil || k == nil || n <= 0 {
		return prev
	}
	sum := prev
	for i := 0; i < n; i++ {
		sum += int32((*q)[i]) * int32((*k)[i])
	}
	return sum
}
