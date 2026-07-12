//go:build amd64

package simd

//go:noescape
func dotI8AccI32Avx2(q, k *int8, n int, prev int32) int32

func dotI8AccSimd(q, k *int8, n int, prev int32) int32 {
	if n <= 0 {
		return prev
	}
	return dotI8AccI32Avx2(q, k, n, prev)
}
