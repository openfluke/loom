package simd

// BitNetTernaryCodeRowDot returns sum(codes_i * acts_i) for i in [0, nBytes),
// where each code is an unsigned 2-bit value {0,1,2} (the BitNet ternary weight is
// code-1). nBytes must be a multiple of 32; codes and acts must both be at least
// nBytes long and zero-padded past the real column count. This is the BitNet MAD
// kernel: the caller subtracts sum(acts) once to recover sum((code-1)*act).
func BitNetTernaryCodeRowDot(codes []uint8, acts []int8, nBytes int) int32 {
	if nBytes <= 0 {
		return 0
	}
	if ternarySimdEnabled() {
		return bitNetTernaryCodeRowDotSimd(&codes[0], &acts[0], nBytes)
	}
	return bitNetTernaryCodeRowDotGo(codes, acts, nBytes)
}

func bitNetTernaryCodeRowDotGo(codes []uint8, acts []int8, nBytes int) int32 {
	var sum int32
	for i := 0; i < nBytes; i++ {
		sum += int32(codes[i]) * int32(acts[i])
	}
	return sum
}
