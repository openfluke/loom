#include "textflag.h"

// void saxpyI8ShiftedInputGradAccAvx2(int32 *gradIn, int8 *weights, int32 gradOut, int n)
TEXT ·saxpyI8ShiftedInputGradAccAvx2(SB), NOSPLIT, $0-36
	MOVQ	gradIn+0(FP), DI
	MOVQ	weights+8(FP), SI
	MOVL	gradOut+16(FP), R8
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
	VPSRAD	$8, Y4, Y4
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
	SARL	$8, R9
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

// void saxpyU8ScaleI32AccAvx2(int32 *gradW, uint8 *input, int32 scale, int n)
TEXT ·saxpyU8ScaleI32AccAvx2(SB), NOSPLIT, $0-36
	MOVQ	gradW+0(FP), DI
	MOVQ	input+8(FP), SI
	MOVL	scale+16(FP), R8
	MOVQ	n+24(FP), CX

	VMOVD	R8, X8
	VPBROADCASTD	X8, Y7

	CMPQ	CX, $8
	JL	tail2

loop82:
	MOVQ	(SI), X1
	VPMOVZXBW	X1, X2
	VPMOVZXWD	X2, Y3
	VPMULLD	Y3, Y7, Y4
	VMOVDQU	(DI), Y5
	VPADDD	Y4, Y5, Y6
	VMOVDQU	Y6, (DI)
	ADDQ	$8, SI
	ADDQ	$32, DI
	SUBQ	$8, CX
	CMPQ	CX, $8
	JGE	loop82

tail2:
	TESTQ	CX, CX
	JE	done2

tailLoop2:
	MOVBQZX	(SI), R9
	IMULL	R8, R9
	MOVL	(DI), R10
	ADDL	R9, R10
	MOVL	R10, (DI)
	ADDQ	$1, SI
	ADDQ	$4, DI
	DECQ	CX
	JNZ	tailLoop2

done2:
	VZEROUPPER
	RET

// void saxpyU8ShiftedInputGradAccAvx2(int32 *gradIn, uint8 *weights, int32 gradOut, int n)
TEXT ·saxpyU8ShiftedInputGradAccAvx2(SB), NOSPLIT, $0-36
	MOVQ	gradIn+0(FP), DI
	MOVQ	weights+8(FP), SI
	MOVL	gradOut+16(FP), R8
	MOVQ	n+24(FP), CX

	VMOVD	R8, X8
	VPBROADCASTD	X8, Y7

	CMPQ	CX, $8
	JL	tail3

loop83:
	MOVQ	(SI), X1
	VPMOVZXBW	X1, X2
	VPMOVZXWD	X2, Y3
	VPMULLD	Y3, Y7, Y4
	VPSRAD	$8, Y4, Y4
	VMOVDQU	(DI), Y5
	VPADDD	Y4, Y5, Y6
	VMOVDQU	Y6, (DI)
	ADDQ	$8, SI
	ADDQ	$32, DI
	SUBQ	$8, CX
	CMPQ	CX, $8
	JGE	loop83

tail3:
	TESTQ	CX, CX
	JE	done3

tailLoop3:
	MOVBQZX	(SI), R9
	IMULL	R8, R9
	SARL	$8, R9
	MOVL	(DI), R10
	ADDL	R9, R10
	MOVL	R10, (DI)
	ADDQ	$1, SI
	ADDQ	$4, DI
	DECQ	CX
	JNZ	tailLoop3

done3:
	VZEROUPPER
	RET
