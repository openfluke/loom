// Package asm hosts Plan 9 assembly CPU kernels for poly layers.
//
// Layout:
//
//	asm/dot/     — shared dot products (f32/f64), used by every layer
//	asm/matmul/  — shared output-parallel GEMV helpers
//	asm/dense/   — dense layer forward
//	asm/swiglu/  — (future)
//	asm/mha/     — (future)
package asm
