package simd

var bitNetTL1Forward bool

// SetBitNetTL1Forward enables the microsoft/BitNet TL1 LUT matvec path on arm64.
// Requires SetBitNetTernarySimdForward(true) as well. When off, packed-2-bit MAD is used.
func SetBitNetTL1Forward(enabled bool) {
	bitNetTL1Forward = enabled
}

// BitNetTL1Active reports whether TL1 LUT matvec is enabled.
func BitNetTL1Active() bool {
	return bitNetTL1Forward && bitNetTernarySimdForward
}
