package poly

import "github.com/openfluke/loom/poly/simd"

// SetBitNetTernarySimdForward toggles the packed-ternary AVX2 matvec path (BitNet).
func SetBitNetTernarySimdForward(enabled bool) {
	simd.SetBitNetTernarySimdForward(enabled)
}

// BitNetTernarySimdActive reports whether packed BitNet matvec uses AVX2.
func BitNetTernarySimdActive() bool {
	return simd.BitNetTernarySimdActive()
}
