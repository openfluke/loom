//go:build arm64

package simd

import "unsafe"

func saxpyF32AccF64Simd(acc *float64, alpha float64, x *float32, n int) {
	if n <= 0 {
		return
	}
	saxpyF32AccF64Go(unsafe.Slice(acc, n), alpha, unsafe.Slice(x, n), n)
}
