//go:build arm64

package simd

//go:noescape
func saxpyF32AccF64Neon(acc *float64, alpha float64, x *float32, n int)

func saxpyF32AccF64Simd(acc *float64, alpha float64, x *float32, n int) {
	if n <= 0 {
		return
	}
	saxpyF32AccF64Neon(acc, alpha, x, n)
}
