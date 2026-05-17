package matmul

// Numeric matches poly.Numeric (local copy to avoid importing poly).
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

func dotNumeric[T Numeric](x, w []T, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(x[i]) * float64(w[i])
	}
	return sum
}

// ForwardSC computes preAct = input @ weights^T for integer element types (float64 accumulate).
func ForwardSC[T Numeric](preAct, input, weights []T, batch, inDim, outDim int) {
	for b := 0; b < batch; b++ {
		inRow := input[b*inDim : (b+1)*inDim]
		outRow := preAct[b*outDim : (b+1)*outDim]
		for o := 0; o < outDim; o++ {
			outRow[o] = T(dotNumeric(inRow, weights[o*inDim:(o+1)*inDim], inDim))
		}
	}
}

// ForwardMC is the multi-core variant of ForwardSC.
func ForwardMC[T Numeric](preAct, input, weights []T, batch, inDim, outDim, tileSize int) {
	OverOutputTiles(outDim, tileSize, func(o0, o1 int) {
		for b := 0; b < batch; b++ {
			inRow := input[b*inDim : (b+1)*inDim]
			outRow := preAct[b*outDim : (b+1)*outDim]
			for o := o0; o < o1; o++ {
				outRow[o] = T(dotNumeric(inRow, weights[o*inDim:(o+1)*inDim], inDim))
			}
		}
	})
}
