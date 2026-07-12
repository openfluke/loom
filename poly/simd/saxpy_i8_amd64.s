#include "textflag.h"

// void saxpyI8ScaleI32AccAvx2(int32 *gradW, int8 *input, int32 scale, int n)
TEXT ·saxpyI8ScaleI32AccAvx2(SB), NOSPLIT, $0-40
	MOVQ	gradW+0(FP), DI
	MOVQ	input+8(FP), SI
	MOVL	scale+16(FP), R8
	MOVQ	n+24(FP), CX

	VMOVD	R8, X8
	VPBROADCASTD	X8, Y7

	CMPQ	CX, $8
	JL	tail

loop8:
	MOVQ	(SI), X1
	VPMOVSXBW	X1, X2
	VPMOVSXWD	X2, Y3
	VPMULLD	Y3, Y7, Y4
	VMOVDQU	(DI), Y5
	VPADDD	Y4, Y5, Y6
	VMOVDQU	Y6, (DI)
	ADDQ	$8, SI
	ADDQ	$32, DI
	SUBQ	$8, CX
	CMPQ	CX, $8
	JGE	loop8

tail:
	TESTQ	CX, CX
	JE	done

tailLoop:
	MOVBQSX	(SI), R9
	IMULL	R8, R9
	MOVL	(DI), R10
	ADDL	R9, R10
	MOVL	R10, (DI)
	ADDQ	$1, SI
	ADDQ	$4, DI
	DECQ	CX
	JNZ	tailLoop

done:
	VZEROUPPER
	RET
