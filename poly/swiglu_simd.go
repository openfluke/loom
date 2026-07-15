package poly

import (
	"github.com/openfluke/loom/poly/simd"
)

func trySwiGLUForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := swigluForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func swigluLayerSimdViable(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	minDim := DenseSimdMinDim()
	in := layer.InputHeight
	inter := layer.OutputHeight
	if in < minDim && inter < minDim {
		return false
	}
	return true
}

func swigluForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	if usePackedTernaryCPU(layer) {
		return SwiGLUForwardPackedTernaryCPU(layer, input)
	}
	if usePackedQ4CPU(layer) {
		return SwiGLUForwardPackedQ4CPU(layer, input)
	}
	if !swigluLayerSimdViable(layer) {
		return swigluForwardTiledParallel(layer, input)
	}

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)
	return swigluForwardSimdF32WithWeights(layer, input, wData)
}

func swigluForwardSimdF32WithWeights(layer *VolumetricLayer, input *Tensor[float32], wData []float32) (preAct, postAct *Tensor[float32]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capSwigluTileToLayer(tileSize, inputSize, intermediateSize)

	wSize := inputSize * intermediateSize
	gateWStart := 0
	upWStart := wSize
	downWStart := 2 * wSize
	gateBStart := 3 * wSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	preAct = NewTensor[float32](seqLen, intermediateSize)
	postAct = NewTensor[float32](seqLen, inputSize)

	for s := 0; s < seqLen; s++ {
		inRow := input.Data[s*inputSize : (s+1)*inputSize]
		preRow := preAct.Data[s*intermediateSize : (s+1)*intermediateSize]
		swigluSimdProjectGateUp(inRow, wData, gateWStart, upWStart, gateBStart, upBStart, preRow, inputSize, intermediateSize, tileSize, layer.Activation)
	}
	for s := 0; s < seqLen; s++ {
		preRow := preAct.Data[s*intermediateSize : (s+1)*intermediateSize]
		outRow := postAct.Data[s*inputSize : (s+1)*inputSize]
		swigluSimdProjectDown(preRow, wData, downWStart, downBStart, outRow, intermediateSize, inputSize, tileSize)
	}

	return preAct, postAct
}

// swigluSimdProjectGateUp is a GEMV (per token: input · gateW/upW). The input
// vector fits in L1 and the weight rows are streamed exactly once, so we call
// the NEON/AVX2 DotTile kernel ONCE over the full inner dimension per output
// row — giving the kernel a long contiguous run and a single float64 reduction
// instead of one reduction per tiny inner tile (which erased the SIMD win).
// tileSize is unused for the inner dim now; the full-range float64 reduction is
// bit-identical across arm64/amd64 (see poly/simd/dot.go).
func swigluSimdProjectGateUp(input, wData []float32, gWStart, uWStart, gBStart, uBStart int, output []float32, inDim, outDim, tileSize int, activation ActivationType) {
	gW := wData[gWStart:]
	uW := wData[uWStart:]

	for o := 0; o < outDim; o++ {
		rowOff := o * inDim
		row := gW[rowOff : rowOff+inDim]
		sG := simd.DotTile(input, row, 0, inDim, float64(wData[gBStart+o]))
		row = uW[rowOff : rowOff+inDim]
		sU := simd.DotTile(input, row, 0, inDim, float64(wData[uBStart+o]))
		output[o] = float32(swigluGateProduct(sG, sU, activation))
	}
}

func swigluSimdProjectDown(input, wData []float32, dWStart, dBStart int, output []float32, inDim, outDim, tileSize int) {
	dW := wData[dWStart:]

	for o := 0; o < outDim; o++ {
		rowOff := o * inDim
		row := dW[rowOff : rowOff+inDim]
		output[o] = float32(simd.DotTile(input, row, 0, inDim, float64(wData[dBStart+o])))
	}
}
