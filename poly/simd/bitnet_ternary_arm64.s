//go:build arm64

#include "textflag.h"

// func bitNetTernaryDotAccum(codes *uint8, acts *int8, chunks int, out *int32)
//
// Baseline-NEON BitNet MAD kernel. Processes `chunks` groups of 16 bytes and
// writes eight int32 lane partials to out[0..7]; the Go caller (bitnet_ternary_arm64.go)
// sums them. codes are unsigned {0,1,2} and acts are signed int8, both interpreted
// as signed int8 by SMULL — values 0..2 are unaffected by the sign bit. Each product
// (|code*act| <= 254) fits in int16; SADALP pairwise-widens adjacent int16 products
// and accumulates them into two int32x4 lanes, so there is no overflow for realistic
// column counts. Integer arithmetic is exact and order-independent, so this matches
// the amd64 VPMADDUBSW/VPMADDWD kernel bit-for-bit.
//
// SMULL / SMULL2 / SADALP are WORD-encoded because Go's arm64 assembler lacks the
// mnemonics; each encoding was verified with clang (otool -t). All three are
// baseline ARMv8.0 NEON, so there is no dotprod/ASIMDDP dependency.
TEXT ·bitNetTernaryDotAccum(SB), NOSPLIT, $0-32
	MOVD codes+0(FP), R0
	MOVD acts+8(FP), R1
	MOVD chunks+16(FP), R2
	MOVD out+24(FP), R3

	VEOR V16.B16, V16.B16, V16.B16 // int32x4 accumulator A
	VEOR V17.B16, V17.B16, V17.B16 // int32x4 accumulator B

	CBZ R2, store
loop:
	VLD1.P 16(R0), [V1.B16] // 16 codes (unsigned bytes)
	VLD1.P 16(R1), [V2.B16] // 16 acts  (signed bytes)
	WORD $0x0E22C023        // SMULL  V3.8H,  V1.8B,  V2.8B   (low 8 products, i16)
	WORD $0x4E22C024        // SMULL2 V4.8H,  V1.16B, V2.16B  (high 8 products, i16)
	WORD $0x4E606870        // SADALP V16.4S, V3.8H           (widen+pair-add, accum)
	WORD $0x4E606891        // SADALP V17.4S, V4.8H
	SUB $1, R2
	CBNZ R2, loop
store:
	VST1 [V16.S4, V17.S4], (R3)
	RET
