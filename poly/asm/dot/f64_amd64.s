#include "textflag.h"

// float64 dotF64(x *float64, w *float64, n int) float64
TEXT ·dotF64(SB), NOSPLIT, $0-32
	MOVQ	x+0(FP), AX
	MOVQ	w+8(FP), BX
	MOVQ	n+16(FP), CX

	XORPD	X0, X0
	CMPQ	CX, $4
	JL	tail

loop4:
	MOVSD	(AX), X1
	MOVSD	8(AX), X2
	MOVSD	16(AX), X3
	MOVSD	24(AX), X4
	MOVSD	(BX), X5
	MOVSD	8(BX), X6
	MOVSD	16(BX), X7
	MOVSD	24(BX), X8
	MULSD	X5, X1
	MULSD	X6, X2
	MULSD	X7, X3
	MULSD	X8, X4
	ADDSD	X1, X0
	ADDSD	X2, X0
	ADDSD	X3, X0
	ADDSD	X4, X0
	ADDQ	$32, AX
	ADDQ	$32, BX
	SUBQ	$4, CX
	CMPQ	CX, $4
	JGE	loop4

tail:
	CMPQ	CX, $0
	JE	done
	MOVSD	(AX), X1
	MOVSD	(BX), X2
	MULSD	X2, X1
	ADDSD	X1, X0
	ADDQ	$8, AX
	ADDQ	$8, BX
	SUBQ	$1, CX
	JMP	tail

done:
	MOVSD	X0, ret+24(FP)
	RET
