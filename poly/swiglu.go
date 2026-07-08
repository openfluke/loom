package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// SwiGLUForwardPolymorphic performs a forward pass through a SwiGLU layer.
func SwiGLUForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseGPU {
		return SwiGLUForwardWGPU(layer, input)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := trySwiGLUForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return SwiGLUForwardTiled(layer, input)
}

// SwiGLUBackwardPolymorphic calculates gradients for the SwiGLU layer.
func SwiGLUBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.UseGPU {
		return SwiGLUBackwardWGPU(layer, gradOutput, input, preAct)
	}
	return SwiGLUBackwardTiled(layer, gradOutput, input, preAct)
}

// SwiGLUForwardTiled performs an optimized, tiled forward pass for SwiGLU (multi-core).
func SwiGLUForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return swigluForwardTiledParallel(layer, input)
}

func swigluGateProduct(g, u float64, activation ActivationType) float64 {
	if activation == ActivationReLU2 {
		if g < 0 {
			g = 0
		}
		return (g * g) * u
	}
	sig := 1.0 / (1.0 + math.Exp(-g))
	return (g * sig) * u
}

func swigluTiledProjectGateUp[TIn Numeric, TOut Numeric, TW Numeric](input []TIn, wData []TW, gWStart, uWStart, gBStart, uBStart int, output []TOut, inDim, outDim, seqLen int, tileSize int) {
	gW := wData[gWStart:]
	uW := wData[uWStart:]
	gB := wData[gBStart:]
	uB := wData[uBStart:]

	sumG := make([]float64, outDim)
	sumU := make([]float64, outDim)

	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			sumG[o] = float64(gB[o])
			sumU[o] = float64(uB[o])
		}

		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}

			for oTile := 0; oTile < outDim; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outDim {
					oEnd = outDim
				}

				for o := oTile; o < oEnd; o++ {
					sG := sumG[o]
					sU := sumU[o]
					for i := iTile; i < iEnd; i++ {
						inVal := float64(input[s*inDim+i])
						sG += inVal * float64(gW[o*inDim+i])
						sU += inVal * float64(uW[o*inDim+i])
					}
					sumG[o] = sG
					sumU[o] = sU
				}
			}
		}

		for o := 0; o < outDim; o++ {
			output[s*outDim+o] = TOut(swigluGateProduct(sumG[o], sumU[o], ActivationSilu))
		}
	}
}

func swigluTiledProjectDown[TIn Numeric, TOut Numeric, TW Numeric](input []TIn, wData []TW, dWStart, dBStart int, output []TOut, inDim, outDim, seqLen int, tileSize int) {
	dW := wData[dWStart:]
	dB := wData[dBStart:]

	for s := 0; s < seqLen; s++ {
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
					sum := float64(0)
					if iTile == 0 {
						sum = float64(dB[o])
					} else {
						sum = float64(output[s*outDim+o])
					}
					for i := iTile; i < iEnd; i++ {
						sum += float64(input[s*inDim+i]) * float64(dW[o*inDim+i])
					}
					output[s*outDim+o] = TOut(sum)
				}
			}
		}
	}
}

// SwiGLUBackwardTiled calculates gradients for SwiGLU using a tiled multi-core approach.
func SwiGLUBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return swigluBackwardTiledParallel(layer, gradOutput, input, preAct)
}

func swigluTiledProjectDownBackward[T Numeric](gradInter []float64, gradOutput []T, preAct []T, wData []T, gradWeights []float64, inDim, outDim, seqLen, dWStart, dBStart, tileSize int) {
	dW := wData[dWStart:]

	for s := 0; s < seqLen; s++ {
		for oTile := 0; oTile < outDim; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outDim {
				oEnd = outDim
			}

			for o := oTile; o < oEnd; o++ {
				dy := float64(gradOutput[s*outDim+o])
				gradWeights[dBStart+o] += dy

				for iTile := 0; iTile < inDim; iTile += tileSize {
					iEnd := iTile + tileSize
					if iEnd > inDim {
						iEnd = inDim
					}

					for i := iTile; i < iEnd; i++ {
						act := float64(preAct[s*inDim+i])
						gradWeights[dWStart+i*outDim+o] += act * dy
						gradInter[s*inDim+i] += dy * float64(dW[i*outDim+o])
					}
				}
			}
		}
	}
}

func swigluTiledProjectGateUpBackward[T Numeric](gradInter []float64, input []T, wData []T, gateWStart, upWStart, gateBStart, upBStart int, gradInput []float64, gradWeights []float64, inDim, outDim, seqLen, tileSize int) {
	gW := wData[gateWStart:]
	uW := wData[upWStart:]
	gB := wData[gateBStart:]
	uB := wData[upBStart:]

	sumSigG := make([]float64, outDim)
	sumUp := make([]float64, outDim)

	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			sumSigG[o] = float64(gB[o])
			sumUp[o] = float64(uB[o])
		}
		for i := 0; i < inDim; i++ {
			inVal := float64(input[s*inDim+i])
			for o := 0; o < outDim; o++ {
				sumSigG[o] += inVal * float64(gW[o*inDim+i])
				sumUp[o] += inVal * float64(uW[o*inDim+i])
			}
		}

		for oTile := 0; oTile < outDim; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outDim {
				oEnd = outDim
			}

			for o := oTile; o < oEnd; o++ {
				gi := gradInter[s*outDim+o]

				x := sumSigG[o]
				sig := 1.0 / (1.0 + math.Exp(-x))
				silu := x * sig
				dSilu := sig * (1.0 + x*(1.0-sig))

				dUp := gi * silu
				dGate := gi * sumUp[o] * dSilu

				gradWeights[upBStart+o] += dUp
				gradWeights[gateBStart+o] += dGate

				for iTile := 0; iTile < inDim; iTile += tileSize {
					iEnd := iTile + tileSize
					if iEnd > inDim {
						iEnd = inDim
					}

					for i := iTile; i < iEnd; i++ {
						inVal := float64(input[s*inDim+i])
						gradWeights[upWStart+o*inDim+i] += inVal * dUp
						gradWeights[gateWStart+o*inDim+i] += inVal * dGate

						gradInput[s*inDim+i] += dUp*float64(uW[o*inDim+i]) + dGate*float64(gW[o*inDim+i])
					}
				}
			}
		}
	}
}

// ── SwiGLU CPU forward (sequential across sequence positions) ────────────────────

func swigluForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	layer.EnsureRuntimeTileSizes()

	if usePackedTernaryCPU(layer) {
		return SwiGLUForwardPackedTernaryCPU(layer, input)
	}

	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}
	tileSize = capSwigluTileToLayer(tileSize, inputSize, intermediateSize)

	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	gateBStart := 2*inputSize*intermediateSize + intermediateSize*inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	// Sequential across seq: per-token goroutines matched bad CPU outputs on some small LMs; inner loops stay tiled.
	for s := 0; s < seqLen; s++ {
		swigluTiledProjectGateUp(input.Data[s*inputSize:(s+1)*inputSize], wData, gateWStart, upWStart, gateBStart, upBStart, preAct.Data[s*intermediateSize:(s+1)*intermediateSize], inputSize, intermediateSize, 1, tileSize)
	}
	for s := 0; s < seqLen; s++ {
		swigluTiledProjectDown(preAct.Data[s*intermediateSize:(s+1)*intermediateSize], wData, downWStart, downBStart, postAct.Data[s*inputSize:(s+1)*inputSize], intermediateSize, inputSize, 1, tileSize)
	}

	return preAct, postAct
}

func SwiGLUForwardPackedTernaryCPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	wSize := inputSize * intermediateSize

	gateWStart := 0
	upWStart := wSize
	downWStart := 2 * wSize
	gateBStart := 3 * wSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	gateW, okG := layer.WeightStore.GetBitNetTernaryMatrix(gateWStart, intermediateSize, inputSize)
	upW, okU := layer.WeightStore.GetBitNetTernaryMatrix(upWStart, intermediateSize, inputSize)
	downW, okD := layer.WeightStore.GetBitNetTernaryMatrix(downWStart, inputSize, intermediateSize)
	if !okG || !okU || !okD {
		exact := layer.Network.UseExactDType
		layer.Network.UseExactDType = false
		pre, post := swigluForwardTiledParallel(layer, input)
		layer.Network.UseExactDType = exact
		return pre, post
	}

	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)
	gate := make([]float64, intermediateSize)
	up := make([]float64, intermediateSize)
	down := make([]float64, inputSize)
	inputQ := make([]int8, inputSize)
	preQ := make([]int8, intermediateSize)

	for s := 0; s < seqLen; s++ {
		row := input.Data[s*inputSize : (s+1)*inputSize]
		inputQ, activationMax := bitNetQuantizeActivationNumeric(row, inputQ)
		bitNetTernaryMatVecQuantized(gateW, inputQ, activationMax, gate)
		bitNetTernaryMatVecQuantized(upW, inputQ, activationMax, up)
		for o := 0; o < intermediateSize; o++ {
			g := gate[o] + bitNetTernaryBias(layer.WeightStore, gateBStart+o)
			u := up[o] + bitNetTernaryBias(layer.WeightStore, upBStart+o)
			preAct.Data[s*intermediateSize+o] = T(swigluGateProduct(g, u, layer.Activation))
		}

		preRow := preAct.Data[s*intermediateSize : (s+1)*intermediateSize]
		bitNetRMSNormTensorRowWeighted(preRow, layer.InnerNormWeight, layer.RMSNormEps)
		preQ, activationMax = bitNetQuantizeActivationNumeric(preRow, preQ)
		bitNetTernaryMatVecQuantized(downW, preQ, activationMax, down)
		for i := 0; i < inputSize; i++ {
			postAct.Data[s*inputSize+i] = T(down[i] + bitNetTernaryBias(layer.WeightStore, downBStart+i))
		}
	}

	return preAct, postAct
}

func swigluBackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[T](input.Shape...)
	wCount := layer.WeightStore.WeightCount(layer.DType)
	gradWeights = NewTensor[T](wCount)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

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

	for s := 0; s < seqLen; s++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(s int) {
			defer func() { <-sem; wg.Done() }()
			swigluTiledProjectDownBackward(gradInter[s*intermediateSize:(s+1)*intermediateSize], gradOutput.Data[s*inputSize:(s+1)*inputSize], preAct.Data[s*intermediateSize:(s+1)*intermediateSize], wData, localGradWs[s], intermediateSize, inputSize, 1, downWStart, downBStart, tileSize)
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
			swigluTiledProjectGateUpBackward(gradInter[s*intermediateSize:(s+1)*intermediateSize], input.Data[s*inputSize:(s+1)*inputSize], wData, gateWStart, upWStart, gateBStart, upBStart, localGI[s], localGradWs[s], inputSize, intermediateSize, 1, tileSize)
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
		gradInput.Data[i] = T(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gw64[i])
	}

	return gradInput, gradWeights
}
