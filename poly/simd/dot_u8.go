package simd

// DotU8Tile computes prev + sum(a[aOff+i]*b[bOff+i]) for unsigned bytes with int32 accumulation.
func DotU8Tile(a, b []uint8, aOff, bOff, n int, prev int32) int32 {
	if n <= 0 {
		return prev
	}
	if aOff < 0 || bOff < 0 || aOff+n > len(a) || bOff+n > len(b) {
		return dotU8Go(a, b, aOff, bOff, n, prev)
	}
	if simdEnabled() && int8DotSimdEnabled() && n >= 8 {
		return dotU8AccSimd(&a[aOff], &b[bOff], n, prev)
	}
	return dotU8Go(a, b, aOff, bOff, n, prev)
}

func dotU8Go(a, b []uint8, aOff, bOff, n int, prev int32) int32 {
	sum := prev
	for i := 0; i < n; i++ {
		sum += int32(a[aOff+i]) * int32(b[bOff+i])
	}
	return sum
}
