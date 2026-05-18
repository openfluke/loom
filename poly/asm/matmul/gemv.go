package matmul

// ForwardGEMVF32 computes preAct = input @ weights^T with float32 elements.
// weights are row-major: weights[o*inDim+i].
func ForwardGEMVF32(preAct, input, weights []float32, batch, inDim, outDim int, mc bool, tileSize int) {
	ForwardTiledF32(preAct, input, weights, batch, inDim, outDim, mc, tileSize)
}

// ForwardGEMVF64 computes preAct = input @ weights^T with float64 elements.
func ForwardGEMVF64(preAct, input, weights []float64, batch, inDim, outDim int, mc bool, tileSize int) {
	ForwardTiledF64(preAct, input, weights, batch, inDim, outDim, mc, tileSize)
}
