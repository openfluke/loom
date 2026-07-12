//go:build amd64

package simd

//go:noescape
func saxpyI8ScaleI32AccAvx2(gradW *int32, input *int8, scale int32, n int)

func saxpyI8ScaleI32AccSimd(gradW *int32, input *int8, scale int32, n int) {
	if n <= 0 {
		return
	}
	saxpyI8ScaleI32AccAvx2(gradW, input, scale, n)
}
