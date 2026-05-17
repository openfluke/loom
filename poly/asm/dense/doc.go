// Package dense provides CPU forward kernels for LayerDense.
//
// Use from poly via VolumetricLayer.UseAsmForward. All poly.Numeric element types
// are supported: float32/float64 use asm/dot; integers use matmul helpers.
package dense
