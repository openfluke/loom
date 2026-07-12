#include "textflag.h"

// void saxpyF32AccF64Avx2(double *acc, double alpha, float *x, int n)
// acc[i] += alpha * float64(x[i])
TEXT ·saxpyF32AccF64Avx2(SB), NOSPLIT, $0-32
	MOVQ    acc+0(FP), DI
	MOVSD   alpha+8(FP), X12
	VBROADCASTSD X12, Y15
	MOVQ    x+16(FP), SI
	MOVQ    n+24(FP), CX

	CMPQ    CX, $8
	JL      tail

loop8:
	VMOVUPS    (SI), Y0
	VEXTRACTF128 $1, Y0, X1
	VCVTPS2PD  X0, Y4
	VCVTPS2PD  X1, Y5
	VMOVUPD    (DI), Y0
	VMOVUPD    32(DI), Y2
	VFMADD231PD Y4, Y15, Y0
	VFMADD231PD Y5, Y15, Y2
	VMOVUPD    Y0, (DI)
	VMOVUPD    Y2, 32(DI)
	ADDQ    $32, SI
	ADDQ    $64, DI
	SUBQ    $8, CX
	CMPQ    CX, $8
	JGE     loop8

tail:
	CMPQ    CX, $0
	JE      done

tailLoop:
	MOVSS   (SI), X4
	VCVTSS2SD X4, X4, X4
	MULSD   X12, X4
	ADDSD   (DI), X4
	MOVSD   X4, (DI)
	ADDQ    $4, SI
	ADDQ    $8, DI
	DECQ    CX
	JNZ     tailLoop

done:
	VZEROUPPER
	RET
