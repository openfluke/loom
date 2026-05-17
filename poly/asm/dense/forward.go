package dense

import (
	"github.com/openfluke/loom/poly/asm/matmul"
)

// Forward runs dense forward: preAct = input @ weights^T.
// mc enables multi-core output tiling; tileSize applies when mc is true.
func Forward[T any](preAct, input, weights []T, batch, inDim, outDim int, mc bool, tileSize int) {
	switch any(*new(T)).(type) {
	case float32:
		matmul.ForwardGEMVF32(
			any(preAct).([]float32),
			any(input).([]float32),
			any(weights).([]float32),
			batch, inDim, outDim, mc, tileSize,
		)
	case float64:
		matmul.ForwardGEMVF64(
			any(preAct).([]float64),
			any(input).([]float64),
			any(weights).([]float64),
			batch, inDim, outDim, mc, tileSize,
		)
	default:
		matmul.ForwardNumeric(preAct, input, weights, batch, inDim, outDim, mc, tileSize)
	}
}
