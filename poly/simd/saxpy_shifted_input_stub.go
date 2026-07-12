//go:build !amd64 && !arm64

package simd

func saxpyI8ShiftedInputGradAccSimd(gradIn *int32, weights *int8, gradOut int32, n int) {
	if gradIn == nil || weights == nil || n <= 0 {
		return
	}
	gi := make([]int32, n)
	copy(gi, (*gradIn)[:n:n])
	w := (*weights)[:n:n]
	saxpyI8ShiftedInputGradAccGo(gi, w, 0, gradOut, n)
	copy((*gradIn)[:n], gi)
}

func saxpyU8ScaleI32AccSimd(gradW *int32, input *uint8, scale int32, n int) {
	if gradW == nil || input == nil || n <= 0 {
		return
	}
	gw := (*gradW)[:n:n]
	in := (*input)[:n:n]
	saxpyU8ScaleI32AccGo(gw, 0, in, scale, n)
}

func saxpyU8ShiftedInputGradAccSimd(gradIn *int32, weights *uint8, gradOut int32, n int) {
	if gradIn == nil || weights == nil || n <= 0 {
		return
	}
	gi := make([]int32, n)
	copy(gi, (*gradIn)[:n:n])
	w := (*weights)[:n:n]
	saxpyU8ShiftedInputGradAccGo(gi, w, 0, gradOut, n)
	copy((*gradIn)[:n], gi)
}
