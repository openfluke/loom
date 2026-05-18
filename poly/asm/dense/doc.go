// Package dense provides CPU forward kernels for LayerDense.
//
// Use from poly via VolumetricLayer.UseAsmForward.
//
// Float DTypes: tiled F32/F64 matmul via asm/dense.Forward (float accumulate in asm).
//
// Low-bit / integer DTypes (Int8, Int4, Ternary, Binary, …): poly routes to
// denseForwardAsmNative in poly/dense_asm_native.go — quantize I/O at the boundary
// (scale only), morphed []uint8 weights, integer×integer dots in asm (int64 acc,
// no FP inside the kernel). See poly/asm/README.md.
//
// FP8 and Float16 still use the float asm path until native FP8/FP16 kernels land.
package dense
