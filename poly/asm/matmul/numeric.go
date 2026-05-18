package matmul

// Numeric matches poly.Numeric (local copy to avoid importing poly).
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// ForwardSC computes preAct = input @ weights^T for integer element types (float64 accumulate).
func ForwardSC[T Numeric](preAct, input, weights []T, batch, inDim, outDim int) {
	ForwardTiledNumeric(preAct, input, weights, batch, inDim, outDim, false, 32)
}

// ForwardMC is the multi-core variant of ForwardSC.
func ForwardMC[T Numeric](preAct, input, weights []T, batch, inDim, outDim, tileSize int) {
	ForwardTiledNumeric(preAct, input, weights, batch, inDim, outDim, true, tileSize)
}
