// Package asm hosts Plan 9 assembly CPU kernels for poly layers.
//
// See README.md in this directory for layout, compute paths, codegen, Lucy
// benchmarks, and the shipped/TODO checklist.
//
// Layout:
//
//	asm/dot/     — shared dot products (f32/f64, native int, packed row)
//	asm/matmul/  — shared output-parallel tiled forward
//	asm/dense/   — dense layer forward
//	asm/swiglu/  — (future)
//	asm/mha/     — (future)
package asm
