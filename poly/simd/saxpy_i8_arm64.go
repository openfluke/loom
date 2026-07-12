//go:build arm64

package simd

import "unsafe"

func saxpyI8ScaleI32AccSimd(gradW *int32, input *int8, scale int32, n int) {
	gw := unsafe.Slice(gradW, n)
	in := unsafe.Slice(input, n)
	for i := 0; i < n; i++ {
		gw[i] += int32(in[i]) * scale
	}
}
