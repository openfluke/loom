//go:build !amd64 && !arm64

package simd

func saxpyI8ScaleI32AccSimd(gradW *int32, input *int8, scale int32, n int) {
	_ = gradW
	_ = input
	_ = scale
	_ = n
}
