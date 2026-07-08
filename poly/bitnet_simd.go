package poly

import "github.com/openfluke/loom/poly/simd"

// SetBitNetTernarySimdForward toggles the packed-ternary SIMD matvec path (BitNet).
func SetBitNetTernarySimdForward(enabled bool) {
	simd.SetBitNetTernarySimdForward(enabled)
}

// BitNetTernarySimdActive reports whether packed BitNet matvec uses AVX2.
func BitNetTernarySimdActive() bool {
	return simd.BitNetTernarySimdActive()
}

// SetBitNetTL1Forward enables microsoft/BitNet TL1 LUT matvec on arm64 (also needs SIMD on).
func SetBitNetTL1Forward(enabled bool) {
	simd.SetBitNetTL1Forward(enabled)
}
