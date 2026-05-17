#include "textflag.h"

// float32 dotF32(x *float32, w *float32, n int) float32
TEXT ·dotF32(SB), NOSPLIT, $0-24
	MOVD	x+0(FP), R0
	MOVD	w+8(FP), R1
	MOVD	n+16(FP), R2

	FMOVS	$0.0, F0
	CMP	$4, R2
	BLT	tail

loop4:
	FMOVS	(R0), F1
	FMOVS	4(R0), F2
	FMOVS	8(R0), F3
	FMOVS	12(R0), F4
	FMOVS	(R1), F5
	FMOVS	4(R1), F6
	FMOVS	8(R1), F7
	FMOVS	12(R1), F8
	FMULS	F5, F1, F1
	FMULS	F6, F2, F2
	FMULS	F7, F3, F3
	FMULS	F8, F4, F4
	FADDS	F1, F0, F0
	FADDS	F2, F0, F0
	FADDS	F3, F0, F0
	FADDS	F4, F0, F0
	ADD	$16, R0
	ADD	$16, R1
	SUB	$4, R2
	CMP	$4, R2
	BGE	loop4

tail:
	CMP	$0, R2
	BEQ	done
	FMOVS	(R0), F1
	FMOVS	(R1), F2
	FMULS	F2, F1, F1
	FADDS	F1, F0, F0
	ADD	$4, R0
	ADD	$4, R1
	SUB	$1, R2
	B	tail

done:
	FMOVS	F0, ret+24(FP)
	RET
