//go:build !amd64 && !arm64

package simd

func dotI8AccSimd(q, k *int8, n int, prev int32) int32 {
	_ = q
	_ = k
	return prev
}
