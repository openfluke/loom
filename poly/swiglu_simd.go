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
	if !swigluLayerSimdViable(layer) {
		return swigluForwardTiledParallel(layer, input)
	}

	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capSwigluTileToLayer(tileSize, inputSize, intermediateSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

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

func swigluSimdProjectGateUp(input, wData []float32, gWStart, uWStart, gBStart, uBStart int, output []float32, inDim, outDim, tileSize int, activation ActivationType) {
	gW := wData[gWStart:]
	uW := wData[uWStart:]

	sumG := make([]float64, outDim)
	sumU := make([]float64, outDim)

	for oTile := 0; oTile < outDim; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outDim {
			oEnd = outDim
		}
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for o := oTile; o < oEnd; o++ {
				sG := float64(wData[gBStart+o])
				sU := float64(wData[uBStart+o])
				if iTile > 0 {
					sG = sumG[o]
					sU = sumU[o]
				}
				rowOff := o * inDim
				sG = simd.DotTile(input, gW[rowOff:rowOff+inDim], iTile, iEnd, sG)
				sU = simd.DotTile(input, uW[rowOff:rowOff+inDim], iTile, iEnd, sU)
				sumG[o] = sG
				sumU[o] = sU
			}
		}
	}

	for o := 0; o < outDim; o++ {
		output[o] = float32(swigluGateProduct(sumG[o], sumU[o], activation))
	}
}

func swigluSimdProjectDown(input, wData []float32, dWStart, dBStart int, output []float32, inDim, outDim, tileSize int) {
	dW := wData[dWStart:]

	for oTile := 0; oTile < outDim; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outDim {
			oEnd = outDim
		}
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for o := oTile; o < oEnd; o++ {
				sum := float64(wData[dBStart+o])
				if iTile > 0 {
					sum = float64(output[o])
				}
				rowOff := o * inDim
				sum = simd.DotTile(input, dW[rowOff:rowOff+inDim], iTile, iEnd, sum)
				output[o] = float32(sum)
			}
		}
	}
}
