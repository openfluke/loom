#include "textflag.h"

DATA q4nibbleMask<>+0(SB)/4, $0x0f0f0f0f
DATA q4nibbleMask<>+4(SB)/4, $0x0f0f0f0f
DATA q4nibbleMask<>+8(SB)/4, $0x0f0f0f0f
DATA q4nibbleMask<>+12(SB)/4, $0x0f0f0f0f
GLOBL q4nibbleMask<>(SB), RODATA|NOPTR, $16

DATA q4seven<>+0(SB)/4, $0x07070707
DATA q4seven<>+4(SB)/4, $0x07070707
DATA q4seven<>+8(SB)/4, $0x07070707
DATA q4seven<>+12(SB)/4, $0x07070707
GLOBL q4seven<>(SB), RODATA|NOPTR, $16

DATA q4sixteen<>+0(SB)/4, $0x10101010
DATA q4sixteen<>+4(SB)/4, $0x10101010
DATA q4sixteen<>+8(SB)/4, $0x10101010
DATA q4sixteen<>+12(SB)/4, $0x10101010
GLOBL q4sixteen<>(SB), RODATA|NOPTR, $16

// float64 q4BlockDot32Avx2(float32 *in, uint32 *packed4, float32 scale)
//
// Vector nibble unpack (16 bytes → 32 signed int8) then FMA with activations.
TEXT ·q4BlockDot32Avx2(SB), NOSPLIT, $128-32
	MOVQ    in+0(FP), SI
	MOVQ    packed4+8(FP), DI
	MOVSS   scale+16(FP), X15
	VCVTSS2SD X15, X15, X15

	VXORPD  X0, X0, X0
	VXORPD  X1, X1, X1
	VXORPD  X2, X2, X2
	VXORPD  X3, X3, X3

	VMOVDQU (DI), X4
	VMOVDQU q4nibbleMask<>(SB), X5
	VMOVDQU q4seven<>(SB), X14
	VMOVDQU q4sixteen<>(SB), X13

	VPAND   X5, X4, X6
	VPSRLW  $4, X4, X7
	VPAND   X5, X7, X7

	// Sign-extend 4-bit values in byte lanes: q>7 → q-16.
	VMOVDQA X6, X8
	VPCMPGTB X14, X8, X8
	VPAND   X13, X8, X8
	VPSUBB  X8, X6, X6

	VMOVDQA X7, X8
	VPCMPGTB X14, X8, X8
	VPAND   X13, X8, X8
	VPSUBB  X8, X7, X7

	// Interleave → chronological q0..q31 as signed bytes.
	VPUNPCKLBW X7, X6, X4
	VPUNPCKHBW X7, X6, X5
	VMOVDQU X4, 0(SP)                // bytes 0..15
	VMOVDQU X5, 16(SP)               // bytes 16..31

	// Process 8 weights at a time: VPMOVSXBD 4 bytes → i32 → f32 → f64 FMA (×2).
	MOVQ    $0, AX                   // byte offset into qs
	MOVQ    $4, CX                   // 4 × 8 weights

chunk8:
	// First 4 qs
	VPMOVSXBD 0(SP)(AX*1), X8
	VCVTDQ2PS X8, X8
	VCVTPS2PD 0(SI), X9
	VCVTPS2PD 8(SI), X10
	VSHUFPS $0x0E, X8, X8, X11
	VCVTPS2PD X8, X8
	VCVTPS2PD X11, X11
	VFMADD231PD X8, X9, X0
	VFMADD231PD X11, X10, X1

	// Next 4 qs
	VPMOVSXBD 4(SP)(AX*1), X8
	VCVTDQ2PS X8, X8
	VCVTPS2PD 16(SI), X9
	VCVTPS2PD 24(SI), X10
	VSHUFPS $0x0E, X8, X8, X11
	VCVTPS2PD X8, X8
	VCVTPS2PD X11, X11
	VFMADD231PD X8, X9, X2
	VFMADD231PD X11, X10, X3

	ADDQ    $8, AX
	ADDQ    $32, SI
	DECQ    CX
	JNZ     chunk8

	VADDPD  X2, X0, X0
	VADDPD  X3, X1, X1
	VADDPD  X1, X0, X0
	VPERMILPD $1, X0, X1
	VADDSD  X1, X0, X0
	VMULSD  X15, X0, X0

	VZEROUPPER
	MOVSD   X0, ret+24(FP)
	RET
