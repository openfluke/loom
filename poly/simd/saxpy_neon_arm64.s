//go:build arm64

#include "textflag.h"

// func saxpyF32AccF64Neon(acc *float64, alpha float64, x *float32, n int)
//
// acc[i] += alpha * float64(x[i]). Four-wide NEON loop mirrors dot_neon_arm64.s
// widen+FMA pattern; scalar tail matches saxpy_avx2_amd64.s operation order.
TEXT ·saxpyF32AccF64Neon(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), R0
	FMOVD alpha+8(FP), F0
	MOVD x+16(FP), R1
	MOVD n+24(FP), R2

	CMP $4, R2
	BLT tail

	FMOVD alpha+8(FP), F31
	WORD $0x4e0807ff // DUP V31.2D, V31.D[0]

loop4:
	VLD1.P 16(R1), [V0.S4]
	VLD1 (R0), [V1.D2, V2.D2]
	WORD $0x0E617810 // VFCVTL  V0.S2, V16.D2
	WORD $0x4E617811 // VFCVTL2 V0.S4, V17.D2
	VFMLA V16.D2, V31.D2, V1.D2
	VFMLA V17.D2, V31.D2, V2.D2
	VST1 [V1.D2, V2.D2], (R0)
	ADD $32, R0
	SUB $4, R2
	CMP $4, R2
	BGE loop4

tail:
	CBZ R2, done
tail1:
	FMOVS (R1), F4
	FCVTSD F4, F2
	FMULD F0, F2
	FMOVD (R0), F3
	FADDD F3, F2
	FMOVD F2, (R0)
	ADD $4, R1
	ADD $8, R0
	SUB $1, R2
	CBNZ R2, tail1

done:
	RET
