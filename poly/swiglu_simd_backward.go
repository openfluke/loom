package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func trySwiGLUBackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	if !swigluLayerSimdViable(layer) {
		return nil, nil, false
	}
	giF, gwF := swigluBackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func swigluBackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capSwigluTileToLayer(tileSize, inputSize, intermediateSize)

	gradInput = NewTensor[float32](input.Shape...)
	wCount := layer.WeightStore.WeightCount(layer.DType)
	gradWeights = NewTensor[float32](wCount)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	gateBStart := 2*inputSize*intermediateSize + intermediateSize*inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	gradInter := make([]float64, seqLen*intermediateSize)
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	localGradWs := make([][]float64, seqLen)
	for s := 0; s < seqLen; s++ {
		localGradWs[s] = make([]float64, wCount)
	}

	// Phase 1: down projection backward (per sequence position, same as tiled).
	for s := 0; s < seqLen; s++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(s int) {
			defer func() { <-sem; wg.Done() }()
			swigluSimdProjectDownBackward(
				gradInter[s*intermediateSize:(s+1)*intermediateSize],
				gradOutput.Data[s*inputSize:(s+1)*inputSize],
				preAct.Data[s*intermediateSize:(s+1)*intermediateSize],
				wData, localGradWs[s],
				intermediateSize, inputSize, downWStart, downBStart, tileSize,
			)
		}(s)
	}
	wg.Wait()

	localGI := make([][]float64, seqLen)
	for s := 0; s < seqLen; s++ {
		localGI[s] = make([]float64, inputSize)
		sem <- struct{}{}
		wg.Add(1)
		go func(s int) {
			defer func() { <-sem; wg.Done() }()
			swigluSimdProjectGateUpBackward(
				gradInter[s*intermediateSize:(s+1)*intermediateSize],
				input.Data[s*inputSize:(s+1)*inputSize],
				wData, gateWStart, upWStart, gateBStart, upBStart,
				localGI[s], localGradWs[s],
				inputSize, intermediateSize, tileSize,
			)
		}(s)
	}
	wg.Wait()

	for s := 0; s < seqLen; s++ {
		for i, v := range localGI[s] {
			gi64[s*inputSize+i] += v
		}
		for i, v := range localGradWs[s] {
			gw64[i] += v
		}
	}

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
	return gradInput, gradWeights
}

// swigluSimdProjectDownBackward: ∂L/∂(down) with saxpy on dW columns and dX from strided weight columns.
func swigluSimdProjectDownBackward(
	gradInter []float64,
	gradOutput, preAct, wData []float32,
	gw64 []float64,
	inDim, outDim, dWStart, dBStart, tileSize int,
) {
	dW := wData[dWStart:]

	for oTile := 0; oTile < outDim; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outDim {
			oEnd = outDim
		}
		localColGW := make([]float64, (oEnd-oTile)*inDim)

		for o := oTile; o < oEnd; o++ {
			dy := float64(gradOutput[o])
			gw64[dBStart+o] += dy
			lo := (o - oTile) * inDim
			simd.SaxpyF32AccF64(localColGW[lo:lo+inDim], dy, preAct, inDim)
			simd.SaxpyF32AccF64InStride(gradInter, dy, dW[o:], outDim, inDim)
		}

		for o := oTile; o < oEnd; o++ {
			lo := (o - oTile) * inDim
			for i := 0; i < inDim; i++ {
				gw64[dWStart+i*outDim+o] += localColGW[lo+i]
			}
		}
	}
}

// swigluSimdProjectGateUpBackward: recompute gate/up linear forms via DotTile; dW/dX via saxpy.
func swigluSimdProjectGateUpBackward(
	gradInter []float64,
	input, wData []float32,
	gateWStart, upWStart, gateBStart, upBStart int,
	gradInput, gw64 []float64,
	inDim, outDim, tileSize int,
) {
	gW := wData[gateWStart:]
	uW := wData[upWStart:]

	sumSigG := make([]float64, outDim)
	sumUp := make([]float64, outDim)
	for o := 0; o < outDim; o++ {
		rowOff := o * inDim
		sumSigG[o] = simd.DotTile(input, gW[rowOff:rowOff+inDim], 0, inDim, float64(wData[gateBStart+o]))
		sumUp[o] = simd.DotTile(input, uW[rowOff:rowOff+inDim], 0, inDim, float64(wData[upBStart+o]))
	}

	for oTile := 0; oTile < outDim; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outDim {
			oEnd = outDim
		}
		for o := oTile; o < oEnd; o++ {
			gi := gradInter[o]
			x := sumSigG[o]
			sig := 1.0 / (1.0 + math.Exp(-x))
			silu := x * sig
			dSilu := sig * (1.0 + x*(1.0-sig))

			dUp := gi * silu
			dGate := gi * sumUp[o] * dSilu

			gw64[upBStart+o] += dUp
			gw64[gateBStart+o] += dGate

			rowOff := o * inDim
			simd.SaxpyF32AccF64(gw64[upWStart+rowOff:upWStart+rowOff+inDim], dUp, input, inDim)
			simd.SaxpyF32AccF64(gw64[gateWStart+rowOff:gateWStart+rowOff+inDim], dGate, input, inDim)
			simd.SaxpyF32AccF64(gradInput, dUp, uW[rowOff:rowOff+inDim], inDim)
			simd.SaxpyF32AccF64(gradInput, dGate, gW[rowOff:rowOff+inDim], inDim)
		}
	}
}
