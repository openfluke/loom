package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// LSTMForwardPolymorphic performs a forward pass through a polymorphic LSTM layer.
// preAct stores [iSum, fSum, gSum, oSum, cCurr] (5 * hiddenSize)
func LSTMForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if useLSTMNativeExact(layer) {
		return LSTMForwardNativeExact(layer, input)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryLSTMForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return LSTMForwardTiled(layer, input)
}

// LSTMBackwardPolymorphic calculates gradients for the LSTM layer using BPTT.
func LSTMBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if useLSTMNativeExact(layer) {
		return LSTMBackwardNativeExact(layer, gradOutput, input, preAct)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if gi, gw, ok := tryLSTMBackwardSimd(layer, gradOutput, input, preAct); ok {
			return gi, gw
		}
	}
	return LSTMBackwardTiled(layer, gradOutput, input, preAct)
}

// LSTMForwardTiled implements a tiled (blocked) LSTM forward pass for cache efficiency.
func LSTMForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return lstmForwardTiledParallel(layer, input)
}

func LSTMBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return lstmBackwardTiledParallel(layer, gradOutput, input, preAct)
}

func lstmForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	preAct = NewTensor[T](batchSize, seqLength, 5*hiddenSize)
	postAct = NewTensor[T](batchSize, seqLength, hiddenSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	ihSize, hhSize, bSize := hiddenSize*inputSize, hiddenSize*hiddenSize, hiddenSize
	gateSize := ihSize + hhSize + bSize

	wI, wF, wG, wO := wData[0:gateSize], wData[gateSize:2*gateSize], wData[2*gateSize:3*gateSize], wData[3*gateSize:4*gateSize]

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			hPrev := make([]float64, hiddenSize)
			cPrev := make([]float64, hiddenSize)
			for t := 0; t < seqLength; t++ {
				itBase := b*seqLength*inputSize + t*inputSize
				pIdxBase := b*seqLength*5*hiddenSize + t*5*hiddenSize

				for hTile := 0; hTile < hiddenSize; hTile += tileSize {
					hEnd := hTile + tileSize
					if hEnd > hiddenSize {
						hEnd = hiddenSize
					}
					for h := hTile; h < hEnd; h++ {
						preAct.Data[pIdxBase+h] = T(wI[ihSize+hhSize+h])
						preAct.Data[pIdxBase+hiddenSize+h] = T(wF[ihSize+hhSize+h])
						preAct.Data[pIdxBase+2*hiddenSize+h] = T(wG[ihSize+hhSize+h])
						preAct.Data[pIdxBase+3*hiddenSize+h] = T(wO[ihSize+hhSize+h])
					}
					for iTile := 0; iTile < inputSize; iTile += tileSize {
						iEnd := iTile + tileSize
						if iEnd > inputSize {
							iEnd = inputSize
						}
						for h := hTile; h < hEnd; h++ {
							hOff := h * inputSize
							iSum, fSum, gSum, oSum := float64(0), float64(0), float64(0), float64(0)
							for i := iTile; i < iEnd; i++ {
								x := float64(input.Data[itBase+i])
								iSum += x * float64(wI[hOff+i])
								fSum += x * float64(wF[hOff+i])
								gSum += x * float64(wG[hOff+i])
								oSum += x * float64(wO[hOff+i])
							}
							preAct.Data[pIdxBase+h] += T(iSum)
							preAct.Data[pIdxBase+hiddenSize+h] += T(fSum)
							preAct.Data[pIdxBase+2*hiddenSize+h] += T(gSum)
							preAct.Data[pIdxBase+3*hiddenSize+h] += T(oSum)
						}
					}
					for hpTile := 0; hpTile < hiddenSize; hpTile += tileSize {
						hpEnd := hpTile + tileSize
						if hpEnd > hiddenSize {
							hpEnd = hiddenSize
						}
						for h := hTile; h < hEnd; h++ {
							hOff := ihSize + h*hiddenSize
							iSum, fSum, gSum, oSum := float64(0), float64(0), float64(0), float64(0)
							for hp := hpTile; hp < hpEnd; hp++ {
								hv := hPrev[hp]
								iSum += hv * float64(wI[hOff+hp])
								fSum += hv * float64(wF[hOff+hp])
								gSum += hv * float64(wG[hOff+hp])
								oSum += hv * float64(wO[hOff+hp])
							}
							preAct.Data[pIdxBase+h] += T(iSum)
							preAct.Data[pIdxBase+hiddenSize+h] += T(fSum)
							preAct.Data[pIdxBase+2*hiddenSize+h] += T(gSum)
							preAct.Data[pIdxBase+3*hiddenSize+h] += T(oSum)
						}
					}
				}

				for h := 0; h < hiddenSize; h++ {
					iS := float64(preAct.Data[pIdxBase+h])
					fS := float64(preAct.Data[pIdxBase+hiddenSize+h])
					gS := float64(preAct.Data[pIdxBase+2*hiddenSize+h])
					oS := float64(preAct.Data[pIdxBase+3*hiddenSize+h])

					iG := 1.0 / (1.0 + math.Exp(-iS))
					fG := 1.0 / (1.0 + math.Exp(-fS))
					gG := math.Tanh(gS)
					oG := 1.0 / (1.0 + math.Exp(-oS))

					cC := fG*cPrev[h] + iG*gG
					hC := oG * math.Tanh(cC)

					postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hC)
					preAct.Data[pIdxBase+4*hiddenSize+h] = T(cC)
					hPrev[h], cPrev[h] = hC, cC
				}
			}
		}(b)
	}
	wg.Wait()
	return preAct, postAct
}

func lstmBackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength

	gradInput = NewTensor[T](batchSize, seqLength, inputSize)
	wCount := layer.WeightStore.WeightCount(layer.DType)
	gradWeights = NewTensor[T](wCount)

	ihSize, hhSize, bSize := hiddenSize*inputSize, hiddenSize*hiddenSize, hiddenSize
	gateSize := ihSize + hhSize + bSize

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)
	wI, wF, wG, wO := wData[0:gateSize], wData[gateSize:2*gateSize], wData[2*gateSize:3*gateSize], wData[3*gateSize:4*gateSize]

	// High-precision buffers for bit-exact parity
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			
			localGI := make([]float64, seqLength*inputSize)
			localGW := make([]float64, wCount)
			gradH, gradC := make([]float64, hiddenSize), make([]float64, hiddenSize)

			for t := seqLength - 1; t >= 0; t-- {
				nextGradH, nextGradC := make([]float64, hiddenSize), make([]float64, hiddenSize)
				pIdx := b*seqLength*5*hiddenSize + t*5*hiddenSize
				itConv := t * inputSize

				deltas := make([]float64, 4*hiddenSize)
				for h := 0; h < hiddenSize; h++ {
					dh := gradH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])
					iS, fS, gS, oS, cC := float64(preAct.Data[pIdx+h]), float64(preAct.Data[pIdx+hiddenSize+h]), float64(preAct.Data[pIdx+2*hiddenSize+h]), float64(preAct.Data[pIdx+3*hiddenSize+h]), float64(preAct.Data[pIdx+4*hiddenSize+h])
					iG, fG, oG := 1.0/(1.0+math.Exp(-iS)), 1.0/(1.0+math.Exp(-fS)), 1.0/(1.0+math.Exp(-oS))
					gG, cT := math.Tanh(gS), math.Tanh(cC)

					cP := float64(0)
					if t > 0 {
						cP = float64(preAct.Data[pIdx-5*hiddenSize+4*hiddenSize+h])
					}
					dc := gradC[h] + dh*oG*(1.0-cT*cT)

					deltas[h] = dc * gG * iG * (1.0 - iG)         // i
					deltas[hiddenSize+h] = dc * cP * fG * (1.0 - fG)  // f
					deltas[2*hiddenSize+h] = dc * iG * (1.0 - gG*gG)  // g
					deltas[3*hiddenSize+h] = dh * cT * oG * (1.0 - oG) // o

					nextGradC[h] = dc * fG
					
					// Bias updates
					localGW[ihSize+hhSize+h] += deltas[h]
					localGW[gateSize+ihSize+hhSize+h] += deltas[hiddenSize+h]
					localGW[2*gateSize+ihSize+hhSize+h] += deltas[2*hiddenSize+h]
					localGW[3*gateSize+ihSize+hhSize+h] += deltas[3*hiddenSize+h]

					// Weight updates (Input-to-Hidden)
					itIn := b*seqLength*inputSize + itConv
					for i := 0; i < inputSize; i++ {
						x := float64(input.Data[itIn+i])
						localGW[h*inputSize+i] += deltas[h] * x
						localGW[gateSize+h*inputSize+i] += deltas[hiddenSize+h] * x
						localGW[2*gateSize+h*inputSize+i] += deltas[2*hiddenSize+h] * x
						localGW[3*gateSize+h*inputSize+i] += deltas[3*hiddenSize+h] * x

						wSum := float64(wI[h*inputSize+i])*deltas[h] + float64(wF[h*inputSize+i])*deltas[hiddenSize+h] + float64(wG[h*inputSize+i])*deltas[2*hiddenSize+h] + float64(wO[h*inputSize+i])*deltas[3*hiddenSize+h]
						localGI[itConv+i] += wSum
					}

					// Weight updates (Hidden-to-Hidden)
					for hp := 0; hp < hiddenSize; hp++ {
						hvP := 0.0
						if t > 0 {
							pP := pIdx - 5*hiddenSize
							oGP := 1.0 / (1.0 + math.Exp(-float64(preAct.Data[pP+3*hiddenSize+hp])))
							hvP = oGP * math.Tanh(float64(preAct.Data[pP+4*hiddenSize+hp]))
						}
						localGW[ihSize+h*hiddenSize+hp] += deltas[h] * hvP
						localGW[gateSize+ihSize+h*hiddenSize+hp] += deltas[hiddenSize+h] * hvP
						localGW[2*gateSize+ihSize+h*hiddenSize+hp] += deltas[2*hiddenSize+h] * hvP
						localGW[3*gateSize+ihSize+h*hiddenSize+hp] += deltas[3*hiddenSize+h] * hvP

						wSumHH := float64(wI[ihSize+h*hiddenSize+hp])*deltas[h] + float64(wF[ihSize+h*hiddenSize+hp])*deltas[hiddenSize+h] + float64(wG[ihSize+h*hiddenSize+hp])*deltas[2*hiddenSize+h] + float64(wO[ihSize+h*hiddenSize+hp])*deltas[3*hiddenSize+h]
						nextGradH[hp] += wSumHH
					}
				}
				gradH, gradC = nextGradH, nextGradC
			}
			
			mu.Lock()
			for i := range localGI { gi64[b*seqLength*inputSize+i] += localGI[i] }
			for i := range localGW { gw64[i] += localGW[i] }
			mu.Unlock()
		}(b)
	}
	wg.Wait()

	for i := range gradInput.Data { gradInput.Data[i] = T(gi64[i]) }
	for i := range gradWeights.Data { gradWeights.Data[i] = T(gw64[i]) }
	return gradInput, gradWeights
}
