//go:build arm64

#include "textflag.h"

// func dotSimdAccum8(x, w *float32, blocks int, out *float64)
//
// Hot inner loop for the arm64 float64-accumulated dot product. Processes
// `blocks` groups of 8 float32 pairs and writes 8 float64 lane partials to
// out[0..7], where out[j] = sum_k x[8k+j]*w[8k+j].
//
// The lane layout mirrors the amd64 AVX2 kernel (dotF32AccF64Avx2): two
// 4-lane accumulators (here split as V24/V25 for elems 0..3 and V26/V27 for
// elems 4..7). The Go caller (neon_arm64.go) performs the identical horizontal
// reduction + prev add + scalar tail, so arm64 is bit-identical to amd64.
//
// float32->float64 widening (VFCVTL/VFCVTL2) and vector-double FMA (VFMLA) are
// encoded via WORD where Go's arm64 assembler accepts only the disassembler
// form. Each WORD was verified with `go tool objdump` (see comments).
//
// Products of two exactly-widened float32 values fit in the float64 mantissa,
// so the fused VFMLA matches a separate multiply+add bit-for-bit — matching the
// AVX2 VFMADD231PD path.
TEXT ·dotSimdAccum8(SB), NOSPLIT, $0-32
	MOVD x+0(FP), R0
	MOVD w+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD out+24(FP), R3

	VEOR V24.B16, V24.B16, V24.B16
	VEOR V25.B16, V25.B16, V25.B16
	VEOR V26.B16, V26.B16, V26.B16
	VEOR V27.B16, V27.B16, V27.B16

	CBZ R2, store
loop:
	VLD1.P 32(R0), [V0.S4, V1.S4]
	VLD1.P 32(R1), [V2.S4, V3.S4]
	WORD $0x0E617810 // VFCVTL  V0.S2, V16.D2
	WORD $0x4E617811 // VFCVTL2 V0.S4, V17.D2
	WORD $0x0E617832 // VFCVTL  V1.S2, V18.D2
	WORD $0x4E617833 // VFCVTL2 V1.S4, V19.D2
	WORD $0x0E617854 // VFCVTL  V2.S2, V20.D2
	WORD $0x4E617855 // VFCVTL2 V2.S4, V21.D2
	WORD $0x0E617876 // VFCVTL  V3.S2, V22.D2
	WORD $0x4E617877 // VFCVTL2 V3.S4, V23.D2
	VFMLA V20.D2, V16.D2, V24.D2
	VFMLA V21.D2, V17.D2, V25.D2
	VFMLA V22.D2, V18.D2, V26.D2
	VFMLA V23.D2, V19.D2, V27.D2
	SUB  $1, R2
	CBNZ R2, loop
store:
	VST1 [V24.D2, V25.D2, V26.D2, V27.D2], (R3)
	RET
