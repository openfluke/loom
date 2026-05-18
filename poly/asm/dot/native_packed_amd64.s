#include "textflag.h"

// int64 dotNibblePackedRowNativeI64(x *uint8, packed *uint32, rowOff int, n int) int64
TEXT ·dotNibblePackedRowNativeI64(SB), NOSPLIT, $0-40
	MOVQ	x+0(FP), AX
	MOVQ	packed+8(FP), BX
	MOVQ	rowOff+16(FP), SI
	MOVQ	n+24(FP), DX
	XORQ	R10, R10
	XORQ	R11, R11

loop:
	CMPQ	R11, DX
	JGE	done
	MOVQ	SI, R12
	ADDQ	R11, R12
	MOVQ	R12, R13
	SHRQ	$3, R13
	MOVQ	R12, R14
	ANDQ	$7, R14
	SHLQ	$2, R14
	MOVL	0(BX)(R13*4), R15
	MOVQ	R15, R8
	MOVQ	R14, CX
	SHRQ	CL, R8
	ANDQ	$15, R8
	CMPQ	R8, $8
	JL	skipSext
	SUBQ	$16, R8
skipSext:
	MOVBQSX	(AX)(R11*1), R9
	IMULQ	R8, R9
	ADDQ	R9, R10
	INCQ	R11
	JMP	loop

done:
	MOVQ	R10, ret+32(FP)
	RET
