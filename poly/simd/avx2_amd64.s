#include "textflag.h"

// double dotF32AccF64Avx2(float *x, float *w, int n, double prev)
// AVX2+FMA — matches cpp/include/dot_tile.hpp f32AccF64Simd on x86_64.
//
// This mirrors the arm64 NEON kernel (dot_neon_arm64.s dotSimdAccum8): four
// independent 2-lane float64 accumulators (X0=elems 0,1 · X1=2,3 · X2=4,5 ·
// X3=6,7) break the VFMADD231PD latency chain so the FMA units stay fed, where
// the previous 2-accumulator (Y0/Y1) loop was latency-bound. The horizontal
// reduction below is the same tree used by neon_arm64.go and the old AVX2
// kernel, and IEEE add is commutative, so the result is bit-identical to both.
// float32*float32 products are exact in float64, so the fused FMAs match a
// separate multiply+add bit-for-bit.
TEXT ·dotF32AccF64Avx2(SB), NOSPLIT, $0-40
	MOVQ    x+0(FP), AX
	MOVQ    w+8(FP), BX
	MOVQ    n+16(FP), CX
	MOVSD   prev+24(FP), X12

	VXORPD  X0, X0, X0
	VXORPD  X1, X1, X1
	VXORPD  X2, X2, X2
	VXORPD  X3, X3, X3

	CMPQ    CX, $8
	JL      reduce

loop8:
	// Widen 8 float32 from x and w into four 2-lane float64 registers each,
	// reading straight from memory (VCVTPS2PD fuses the load + convert).
	VCVTPS2PD (AX), X4       // x[0,1]
	VCVTPS2PD 8(AX), X5      // x[2,3]
	VCVTPS2PD 16(AX), X6     // x[4,5]
	VCVTPS2PD 24(AX), X7     // x[6,7]
	VCVTPS2PD (BX), X8       // w[0,1]
	VCVTPS2PD 8(BX), X9      // w[2,3]
	VCVTPS2PD 16(BX), X10    // w[4,5]
	VCVTPS2PD 24(BX), X11    // w[6,7]

	VFMADD231PD X8, X4, X0   // X0 += x[0,1]*w[0,1]
	VFMADD231PD X9, X5, X1   // X1 += x[2,3]*w[2,3]
	VFMADD231PD X10, X6, X2  // X2 += x[4,5]*w[4,5]
	VFMADD231PD X11, X7, X3  // X3 += x[6,7]*w[6,7]

	ADDQ    $32, AX
	ADDQ    $32, BX
	SUBQ    $8, CX
	CMPQ    CX, $8
	JGE     loop8

reduce:
	// [p0+p4, p1+p5] and [p2+p6, p3+p7], then (t0)+(t1)+prev — the exact tree
	// from neon_arm64.go, matching the previous Y0/Y1 AVX2 reduction bit-for-bit.
	VADDPD  X2, X0, X0       // X0 = [p0+p4, p1+p5]
	VADDPD  X3, X1, X1       // X1 = [p2+p6, p3+p7]
	VADDPD  X1, X0, X0       // X0 = [t0, t1]
	VPERMILPD $1, X0, X1     // X1.lo = t1
	VADDSD  X1, X0, X0       // t0 + t1
	VADDSD  X12, X0, X0      // + prev

	CMPQ    CX, $0
	JE      done

tail:
	MOVSS   (AX), X4
	MOVSS   (BX), X5
	VCVTSS2SD X4, X4, X4
	VCVTSS2SD X5, X5, X5
	VMULSD  X5, X4, X4
	VADDSD  X4, X0, X0
	ADDQ    $4, AX
	ADDQ    $4, BX
	DECQ    CX
	JNZ     tail

done:
	VZEROUPPER
	MOVSD   X0, ret+32(FP)
	RET
