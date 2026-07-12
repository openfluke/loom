package simd

// SaxpyI8ScaleI32Acc computes gradW[rowOff+i] += int32(input[i]) * scale for i in [0,n).
func SaxpyI8ScaleI32Acc(gradW []int32, rowOff int, input []int8, scale int32, n int) {
	if n <= 0 || rowOff < 0 || rowOff+n > len(gradW) || n > len(input) {
		return
	}
	if int8DotSimdEnabled() && n >= 8 {
		saxpyI8ScaleI32AccSimd(&gradW[rowOff], &input[0], scale, n)
		return
	}
	saxpyI8ScaleI32AccGo(gradW, rowOff, input, scale, n)
}

func saxpyI8ScaleI32AccGo(gradW []int32, rowOff int, input []int8, scale int32, n int) {
	for i := 0; i < n; i++ {
		gradW[rowOff+i] += int32(input[i]) * scale
	}
}
