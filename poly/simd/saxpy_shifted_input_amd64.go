//go:build amd64

package simd

//go:noescape
func saxpyI8ShiftedInputGradAccAvx2(gradIn *int32, weights *int8, gradOut int32, n int)

//go:noescape
func saxpyU8ScaleI32AccAvx2(gradW *int32, input *uint8, scale int32, n int)

//go:noescape
func saxpyU8ShiftedInputGradAccAvx2(gradIn *int32, weights *uint8, gradOut int32, n int)

func saxpyI8ShiftedInputGradAccSimd(gradIn *int32, weights *int8, gradOut int32, n int) {
	if n <= 0 {
		return
	}
	saxpyI8ShiftedInputGradAccAvx2(gradIn, weights, gradOut, n)
}

func saxpyU8ScaleI32AccSimd(gradW *int32, input *uint8, scale int32, n int) {
	if n <= 0 {
		return
	}
	saxpyU8ScaleI32AccAvx2(gradW, input, scale, n)
}

func saxpyU8ShiftedInputGradAccSimd(gradIn *int32, weights *uint8, gradOut int32, n int) {
	if n <= 0 {
		return
	}
	saxpyU8ShiftedInputGradAccAvx2(gradIn, weights, gradOut, n)
}
