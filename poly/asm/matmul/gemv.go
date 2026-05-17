package matmul

import "github.com/openfluke/loom/poly/asm/dot"

// ForwardGEMVF32 computes preAct = input @ weights^T with float32 elements.
// weights are row-major: weights[o*inDim+i].
func ForwardGEMVF32(preAct, input, weights []float32, batch, inDim, outDim int, mc bool, tileSize int) {
	run := func(o0, o1 int) {
		for b := 0; b < batch; b++ {
			inRow := input[b*inDim : (b+1)*inDim]
			outRow := preAct[b*outDim : (b+1)*outDim]
			for o := o0; o < o1; o++ {
				outRow[o] = dot.F32(inRow, weights[o*inDim:(o+1)*inDim], inDim)
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

// ForwardGEMVF64 computes preAct = input @ weights^T with float64 elements.
func ForwardGEMVF64(preAct, input, weights []float64, batch, inDim, outDim int, mc bool, tileSize int) {
	run := func(o0, o1 int) {
		for b := 0; b < batch; b++ {
			inRow := input[b*inDim : (b+1)*inDim]
			outRow := preAct[b*outDim : (b+1)*outDim]
			for o := o0; o < o1; o++ {
				outRow[o] = dot.F64(inRow, weights[o*inDim:(o+1)*inDim], inDim)
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}
