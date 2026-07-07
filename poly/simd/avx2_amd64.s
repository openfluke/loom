#include "textflag.h"

// double dotF32AccF64Avx2(float *x, float *w, int n, double prev)
// AVX2+FMA — matches cpp/include/dot_tile.hpp f32AccF64Simd on x86_64.
TEXT ·dotF32AccF64Avx2(SB), NOSPLIT, $0-36
	MOVQ    x+0(FP), AX
	MOVQ    w+8(FP), BX
	MOVQ    n+16(FP), CX
	MOVSD   prev+24(FP), X2

	VXORPD  Y0, Y0, Y0
	VXORPD  Y1, Y1, Y1

	CMPQ    CX, $8
	JL      reduce

loop8:
	VMOVUPS (AX), Y3
	VMOVUPS (BX), Y4

	VEXTRACTF128 $0, Y3, X5
	VEXTRACTF128 $1, Y3, X6
	VEXTRACTF128 $0, Y4, X7
	VEXTRACTF128 $1, Y4, X8

	VCVTPS2PD X5, Y5
	VCVTPS2PD X6, Y6
	VCVTPS2PD X7, Y7
	VCVTPS2PD X8, Y8

	VFMADD231PD Y5, Y7, Y0
	VFMADD231PD Y6, Y8, Y1

	ADDQ    $32, AX
	ADDQ    $32, BX
	SUBQ    $8, CX
	CMPQ    CX, $8
	JGE     loop8

reduce:
	VADDPD  Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPD  X0, X1, X0
	VPERMILPD $1, X0, X1
	VADDSD  X1, X0, X0
	VADDSD  X2, X0, X0

	CMPQ    CX, $0
	JE      done

tail:
	MOVSS   (AX), X3
	MOVSS   (BX), X4
	VCVTSS2SD X3, X3, X3
	VCVTSS2SD X4, X4, X4
	VMULSD  X4, X3, X3
	VADDSD  X3, X0, X0
	ADDQ    $4, AX
	ADDQ    $4, BX
	DECQ    CX
	JNZ     tail

done:
	VZEROUPPER
	MOVSD   X0, ret+32(FP)
	RET
