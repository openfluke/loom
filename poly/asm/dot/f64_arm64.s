#include "textflag.h"

// float64 dotF64(x *float64, w *float64, n int) float64
TEXT ·dotF64(SB), NOSPLIT, $0-32
	MOVD	x+0(FP), R0
	MOVD	w+8(FP), R1
	MOVD	n+16(FP), R2

	FMOVD	$0.0, F0
	CMP	$4, R2
	BLT	tail

loop4:
	FMOVD	(R0), F1
	FMOVD	8(R0), F2
	FMOVD	16(R0), F3
	FMOVD	24(R0), F4
	FMOVD	(R1), F5
	FMOVD	8(R1), F6
	FMOVD	16(R1), F7
	FMOVD	24(R1), F8
	FMULD	F5, F1, F1
	FMULD	F6, F2, F2
	FMULD	F7, F3, F3
	FMULD	F8, F4, F4
	FADDD	F1, F0, F0
	FADDD	F2, F0, F0
	FADDD	F3, F0, F0
	FADDD	F4, F0, F0
	ADD	$32, R0
	ADD	$32, R1
	SUB	$4, R2
	CMP	$4, R2
	BGE	loop4

tail:
	CMP	$0, R2
	BEQ	done
	FMOVD	(R0), F1
	FMOVD	(R1), F2
	FMULD	F2, F1, F1
	FADDD	F1, F0, F0
	ADD	$8, R0
	ADD	$8, R1
	SUB	$1, R2
	B	tail

done:
	FMOVD	F0, ret+24(FP)
	RET
