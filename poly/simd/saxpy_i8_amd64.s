#include "textflag.h"

// void saxpyI8ScaleI32AccAvx2(int32 *gradW, int8 *input, int32 scale, int n)
// gradW[i] += int32(input[i]) * scale
TEXT ·saxpyI8ScaleI32AccAvx2(SB), NOSPLIT, $0-40
	MOVQ	gradW+0(FP), DI
	MOVQ	input+8(FP), SI
	MOVL	scale+16(FP), R8
	MOVQ	n+24(FP), CX

	CMPQ	CX, $0
	JE	done

loop:
	MOVBQSX	(SI), R9
	IMULL	R8, R9
	MOVL	(DI), R10
	ADDL	R9, R10
	MOVL	R10, (DI)
	ADDQ	$1, SI
	ADDQ	$4, DI
	DECQ	CX
	JNZ	loop

done:
	RET
