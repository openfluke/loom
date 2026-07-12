package simd

// SaxpyI8ShiftedInputGradAcc computes gradIn[i] += (int32(weights[i]) * gradOut) >> 8.
func SaxpyI8ShiftedInputGradAcc(gradIn []int32, weights []int8, rowOff int, gradOut int32, n int) {
	if n <= 0 || rowOff < 0 || rowOff+n > len(weights) || n > len(gradIn) {
		return
	}
	if int8DotSimdEnabled() && n >= 8 {
		saxpyI8ShiftedInputGradAccSimd(&gradIn[0], &weights[rowOff], gradOut, n)
		return
	}
	saxpyI8ShiftedInputGradAccGo(gradIn, weights, rowOff, gradOut, n)
}

func saxpyI8ShiftedInputGradAccGo(gradIn []int32, weights []int8, rowOff int, gradOut int32, n int) {
	for i := 0; i < n; i++ {
		gradIn[i] += (int32(weights[rowOff+i]) * gradOut) >> 8
	}
}

// SaxpyU8ScaleI32Acc computes gradW[rowOff+i] += int32(input[i]) * scale.
func SaxpyU8ScaleI32Acc(gradW []int32, rowOff int, input []uint8, scale int32, n int) {
	if n <= 0 || rowOff < 0 || rowOff+n > len(gradW) || n > len(input) {
		return
	}
	if int8DotSimdEnabled() && n >= 8 {
		saxpyU8ScaleI32AccSimd(&gradW[rowOff], &input[0], scale, n)
		return
	}
	saxpyU8ScaleI32AccGo(gradW, rowOff, input, scale, n)
}

func saxpyU8ScaleI32AccGo(gradW []int32, rowOff int, input []uint8, scale int32, n int) {
	for i := 0; i < n; i++ {
		gradW[rowOff+i] += int32(input[i]) * scale
	}
}

// SaxpyU8ShiftedInputGradAcc computes gradIn[i] += (int32(weights[i]) * gradOut) >> 8 for unsigned weights.
func SaxpyU8ShiftedInputGradAcc(gradIn []int32, weights []uint8, rowOff int, gradOut int32, n int) {
	if n <= 0 || rowOff < 0 || rowOff+n > len(weights) || n > len(gradIn) {
		return
	}
	if int8DotSimdEnabled() && n >= 8 {
		saxpyU8ShiftedInputGradAccSimd(&gradIn[0], &weights[rowOff], gradOut, n)
		return
	}
	saxpyU8ShiftedInputGradAccGo(gradIn, weights, rowOff, gradOut, n)
}

func saxpyU8ShiftedInputGradAccGo(gradIn []int32, weights []uint8, rowOff int, gradOut int32, n int) {
	for i := 0; i < n; i++ {
		gradIn[i] += (int32(weights[rowOff+i]) * gradOut) >> 8
	}
}
