//go:build arm64

#include "textflag.h"

// func bitNetTL1RowDotAccum(nibbles *uint8, qlut *int16, pairCount int, tailCode uint8, tailAct int8) int32
//
// TL1 row dot using 32-bit accumulators throughout. See bitnet_tl1.go for semantics.
TEXT ·bitNetTL1RowDotAccum(SB), NOSPLIT, $0-40
	MOVD nibbles+0(FP), R0
	MOVD qlut+8(FP), R1
	MOVD pairCount+16(FP), R2
	MOVB tailCode+24(FP), R3
	MOVB tailAct+32(FP), R4

	MOVW $0, R5

	CBZ R2, tail
	MOVW $0, R6

pairLoop:
	MOVD R6, R7
	LSR  $1, R7, R7
	ADD  R0, R7, R8
	MOVBU (R8), R9
	TST  $1, R6
	BEQ  highNib
	ANDW $0x0f, R9, R9
	B    gotIdx
highNib:
	LSRW $4, R9, R9
gotIdx:
	MOVD R6, R10
	LSL  $5, R10, R10
	ADD  R1, R10, R10
	MOVD R9, R11
	LSL  $1, R11, R11
	ADD  R10, R11, R10
	MOVH (R10), R11
	ADDW R11, R5, R5

	ADD  $1, R6
	CMP  R6, R2
	BLT  pairLoop

tail:
	CMP  $1, R3
	BEQ  done
	CMP  $2, R3
	BHI  done
	SUBW $1, R3, R7
	MOVW R7, R8
	MOVW R4, R9
	MULW R9, R8
	ADDW R8, R5, R5
done:
	MOVD R5, R0
	RET
