#include "textflag.h"

// float32 dotF32(x *float32, w *float32, n int) float32
TEXT ·dotF32(SB), NOSPLIT, $0-24
	MOVQ	x+0(FP), AX
	MOVQ	w+8(FP), BX
	MOVQ	n+16(FP), CX

	XORPS	X0, X0
	CMPQ	$4, CX
	JL	tail

loop4:
	MOVSS	(AX), X1
	MOVSS	4(AX), X2
	MOVSS	8(AX), X3
	MOVSS	12(AX), X4
	MOVSS	(BX), X5
	MOVSS	4(BX), X6
	MOVSS	8(BX), X7
	MOVSS	12(BX), X8
	MULSS	X5, X1
	MULSS	X6, X2
	MULSS	X7, X3
	MULSS	X8, X4
	ADDSS	X1, X0
	ADDSS	X2, X0
	ADDSS	X3, X0
	ADDSS	X4, X0
	ADDQ	$16, AX
	ADDQ	$16, BX
	SUBQ	$4, CX
	CMPQ	$4, CX
	JGE	loop4

tail:
	CMPQ	$0, CX
	JE	done
	MOVSS	(AX), X1
	MOVSS	(BX), X2
	MULSS	X2, X1
	ADDSS	X1, X0
	ADDQ	$4, AX
	ADDQ	$4, BX
	SUBQ	$1, CX
	JMP	tail

done:
	MOVSS	X0, ret+24(FP)
	RET
