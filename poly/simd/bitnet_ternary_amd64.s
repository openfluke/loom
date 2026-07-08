#include "textflag.h"

// int32 bitNetTernaryCodeRowDotSimd(uint8 *codes, int8 *acts, int nBytes)
// AVX2 BitNet MAD kernel: returns sum(codes_i * acts_i) over nBytes (mult of 32).
// codes are unsigned {0,1,2}; acts are signed int8. Uses VPMADDUBSW (u8*s8 -> i16
// pair sums) then VPMADDWD with a vector of 1s to widen/accumulate into i32 lanes.
TEXT ·bitNetTernaryCodeRowDotSimd(SB), NOSPLIT, $0-28
	MOVQ	codes+0(FP), DI
	MOVQ	acts+8(FP), SI
	MOVQ	nBytes+16(FP), CX

	VPXOR		Y0, Y0, Y0     // i32 accumulator
	VPCMPEQW	Y5, Y5, Y5     // all-ones bits => -1 per i16 lane
	VPABSW		Y5, Y5         // => +1 per i16 lane

	SHRQ	$5, CX             // chunk count = nBytes / 32
	TESTQ	CX, CX
	JZ	reduce

loop32:
	VMOVDQU		(DI), Y1       // 32 codes (unsigned bytes)
	VMOVDQU		(SI), Y2       // 32 acts  (signed bytes)
	VPMADDUBSW	Y2, Y1, Y3     // i16[16] = code*act adjacent pair sums
	VPMADDWD	Y5, Y3, Y4     // i32[8]  = widen + sum adjacent pairs
	VPADDD		Y4, Y0, Y0
	ADDQ	$32, DI
	ADDQ	$32, SI
	DECQ	CX
	JNZ	loop32

reduce:
	VEXTRACTI128	$1, Y0, X1
	VPADDD	X1, X0, X0
	VPHADDD	X0, X0, X0
	VPHADDD	X0, X0, X0
	MOVD	X0, AX
	MOVL	AX, ret+24(FP)
	VZEROUPPER
	RET
