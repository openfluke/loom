package simd

// bitNetTernarySimdForward is set when the network enables Plan 9 SIMD forward
// (SetSimdForward / SetSimdForwardRecursive). BitNet packed matvec uses the AVX2
// ternary kernel only when this flag and hardware SIMD are both on.
var bitNetTernarySimdForward bool

// SetBitNetTernarySimdForward toggles the BitNet packed-ternary AVX2 matvec path.
func SetBitNetTernarySimdForward(enabled bool) {
	bitNetTernarySimdForward = enabled
}

func ternarySimdEnabled() bool {
	return simdEnabled() && bitNetTernarySimdForward
}

// BitNetTernarySimdActive reports whether the AVX2 packed-ternary matvec path is on.
func BitNetTernarySimdActive() bool {
	return ternarySimdEnabled()
}
