package simd

// DotI8Tile computes prev + sum(a[aOff+i]*b[bOff+i]) for i in [0,n) with int32 accumulation.
// Callers apply fixed-point scaling (e.g. >>8 for int8 MAC). Uses vector kernels on amd64/arm64.
func DotI8Tile(a, b []int8, aOff, bOff, n int, prev int32) int32 {
	if n <= 0 {
		return prev
	}
	if aOff < 0 || bOff < 0 || aOff+n > len(a) || bOff+n > len(b) {
		return dotI8Go(a, b, aOff, bOff, n, prev)
	}
	if simdEnabled() && int8DotSimdEnabled() && n >= 8 {
		return dotI8AccSimd(&a[aOff], &b[bOff], n, prev)
	}
	return dotI8Go(a, b, aOff, bOff, n, prev)
}

func dotI8Go(a, b []int8, aOff, bOff, n int, prev int32) int32 {
	sum := prev
	for i := 0; i < n; i++ {
		sum += int32(a[aOff+i]) * int32(b[bOff+i])
	}
	return sum
}
