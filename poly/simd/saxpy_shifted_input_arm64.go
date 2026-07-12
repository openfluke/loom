//go:build arm64

package simd

import "unsafe"

func saxpyI8ShiftedInputGradAccSimd(gradIn *int32, weights *int8, gradOut int32, n int) {
	gi := unsafe.Slice(gradIn, n)
	w := unsafe.Slice(weights, n)
	for i := 0; i < n; i++ {
		gi[i] += (int32(w[i]) * gradOut) >> 8
	}
}

func saxpyU8ScaleI32AccSimd(gradW *int32, input *uint8, scale int32, n int) {
	gw := unsafe.Slice(gradW, n)
	in := unsafe.Slice(input, n)
	for i := 0; i < n; i++ {
		gw[i] += int32(in[i]) * scale
	}
}

func saxpyU8ShiftedInputGradAccSimd(gradIn *int32, weights *uint8, gradOut int32, n int) {
	gi := unsafe.Slice(gradIn, n)
	w := unsafe.Slice(weights, n)
	for i := 0; i < n; i++ {
		gi[i] += (int32(w[i]) * gradOut) >> 8
	}
}
