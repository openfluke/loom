//go:build arm64

#include "textflag.h"

// func bitNetTernaryPackedDotAccum(packed *uint8, acts *int8, blocks int, out *int32)
//
// Packed-2-bit BitNet MAD kernel. Reads the weights as 2-bit codes straight from
// the packed layout (4 codes/byte, 16 bytes = 64 codes per block) — 4x less memory
// traffic than the unpacked byte-code kernel, which is the whole point on
// bandwidth-bound decode. Writes four int32 lane partials to out[0..3]; the Go
// caller sums them.
//
// Per block: load 16 packed bytes (64 codes), unpack the four 2-bit fields into
// C0..C3 (columns 4i+0..4i+3), load 64 activations de-interleaved with LD4 so
// A0..A3 line up with C0..C3, then SMULL/SMULL2 + SADALP accumulate. codes are
// unsigned {0,1,2} and acts signed int8; SMULL treats codes as signed but 0..2 is
// unaffected. The caller subtracts sum(acts) once (weight = code-1). Integer math
// is exact and order-independent, so this is bit-identical to the byte-code AVX2
// kernel and the scalar Go reference. Activations must be zero-padded to blocks*64
// so padding columns contribute nothing.
//
// MOVI/AND/USHR/LD4/SMULL/SMULL2/SADALP/ADD are WORD-encoded (Go's arm64 assembler
// lacks the mnemonics); every encoding was verified with clang. All are baseline
// ARMv8.0 NEON — no dotprod dependency.
TEXT ·bitNetTernaryPackedDotAccum(SB), NOSPLIT, $0-32
	MOVD packed+0(FP), R0
	MOVD acts+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD out+24(FP), R3

	WORD $0x4f00e47f               // MOVI V31.16B, #3   (2-bit field mask)
	VEOR V12.B16, V12.B16, V12.B16 // 4 independent int32x4 accumulators (one per
	VEOR V13.B16, V13.B16, V13.B16 // sub-field) to break the SADALP dependency
	VEOR V14.B16, V14.B16, V14.B16 // chain — decode is compute-bound, so ILP here
	VEOR V15.B16, V15.B16, V15.B16 // matters more than the memory saving.

	CBZ R2, store
loop:
	VLD1.P 16(R0), [V0.B16] // 16 packed bytes = 64 ternary codes

	WORD $0x4e3f1c01 // AND   V1.16B, V0.16B, V31.16B      C0 = codes (col 4i+0)
	WORD $0x6f0e0402 // USHR  V2.16B, V0.16B, #2
	WORD $0x4e3f1c42 // AND   V2.16B, V2.16B, V31.16B      C1 = (col 4i+1)
	WORD $0x6f0c0403 // USHR  V3.16B, V0.16B, #4
	WORD $0x4e3f1c63 // AND   V3.16B, V3.16B, V31.16B      C2 = (col 4i+2)
	WORD $0x6f0a0404 // USHR  V4.16B, V0.16B, #6           C3 = (col 4i+3), only 2 bits left

	WORD $0x4cdf0028 // LD4 {V8.16B,V9.16B,V10.16B,V11.16B}, [R1], #64  A0..A3

	WORD $0x0e28c034 // SMULL  V20.8H, V1.8B,  V8.8B
	WORD $0x4e28c035 // SMULL2 V21.8H, V1.16B, V8.16B
	WORD $0x4e606a8c // SADALP V12.4S, V20.8H
	WORD $0x4e606aac // SADALP V12.4S, V21.8H

	WORD $0x0e29c056 // SMULL  V22.8H, V2.8B,  V9.8B
	WORD $0x4e29c057 // SMULL2 V23.8H, V2.16B, V9.16B
	WORD $0x4e606acd // SADALP V13.4S, V22.8H
	WORD $0x4e606aed // SADALP V13.4S, V23.8H

	WORD $0x0e2ac078 // SMULL  V24.8H, V3.8B,  V10.8B
	WORD $0x4e2ac079 // SMULL2 V25.8H, V3.16B, V10.16B
	WORD $0x4e606b0e // SADALP V14.4S, V24.8H
	WORD $0x4e606b2e // SADALP V14.4S, V25.8H

	WORD $0x0e2bc09a // SMULL  V26.8H, V4.8B,  V11.8B
	WORD $0x4e2bc09b // SMULL2 V27.8H, V4.16B, V11.16B
	WORD $0x4e606b4f // SADALP V15.4S, V26.8H
	WORD $0x4e606b6f // SADALP V15.4S, V27.8H

	SUB  $1, R2
	CBNZ R2, loop
store:
	WORD $0x4ead858c // ADD V12.4S, V12.4S, V13.4S
	WORD $0x4eaf85ce // ADD V14.4S, V14.4S, V15.4S
	WORD $0x4eae858c // ADD V12.4S, V12.4S, V14.4S
	VST1 [V12.S4], (R3)
	RET
