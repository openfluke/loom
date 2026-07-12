//go:build arm64

#include "textflag.h"

// func saxpyF32AccF64Neon(acc *float64, alpha float64, x *float32, n int)
TEXT ·saxpyF32AccF64Neon(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), R0
	MOVSD alpha+8(FP), F0
	MOVD x+16(FP), R1
	MOVD n+24(FP), R2

	CMP $4, R2
	BLT tail

	FDUP V31.D2, F0

loop4:
	VLD1.P 16(R1), [V0.S4]
	VLD1.P 32(R0), [V1.D2, V2.D2]
	WORD $0x0E617800 // VFCVTL  V3.D2, V0.S2
	WORD $0x4E617801 // VFCVTL2 V4.D2, V0.S4
	VFMLA V3.D2, V31.D2, V1.D2
	VFMLA V4.D2, V31.D2, V2.D2
	VST1.P [V1.D2, V2.D2], 32(R0)
	SUB $4, R2
	CMP $4, R2
	BGE loop4

tail:
	CBZ R2, done
tail1:
	LDRSW (R1), R3
	FCVTSD F2, R3
	FMULD F2, F0, F2
	LDRD (R0), F3
	FADDD F2, F3, F2
	STRD F2, (R0)
	ADD $4, R1
	ADD $8, R0
	SUB $1, R2
	CBNZ R2, tail1

done:
	RET
