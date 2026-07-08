//go:build arm64

#include "textflag.h"

// func bitNetTL1PairBatch16(indices *uint8, lut *int16, acc *int32, n int)
//
// Add lut[indices[i]] to acc[i] for i in [0,n), n <= 16. One K-pair position,
// batched across output rows (Microsoft TL1 M-dimension batching).
TEXT ·bitNetTL1PairBatch16(SB), NOSPLIT, $0-32
	MOVD indices+0(FP), R0
	MOVD lut+8(FP), R1
	MOVD acc+16(FP), R2
	MOVD n+24(FP), R3

	MOVW $0, R4
batchLoop:
	CMP  R3, R4
	BGE  done
	MOVBU (R0)(R4), R5
	MOVD R5, R6
	LSL  $1, R6, R6
	MOVH (R1)(R6), R6
	MOVD R4, R7
	LSL  $2, R7, R7
	ADD  R2, R7, R7
	MOVW (R7), R8
	ADDW R6, R8, R8
	MOVW R8, (R7)
	ADD  $1, R4
	B    batchLoop
done:
	RET
