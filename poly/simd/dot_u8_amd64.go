//go:build amd64

package simd

//go:noescape
func dotU8AccI32Avx2(q, k *uint8, n int, prev int32) int32

func dotU8AccSimd(q, k *uint8, n int, prev int32) int32 {
	if n <= 0 {
		return prev
	}
	return dotU8AccI32Avx2(q, k, n, prev)
}
