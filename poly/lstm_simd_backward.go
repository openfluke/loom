package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryLSTMBackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	giF, gwF := lstmBackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func lstmBackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	hiddenSize := layer.OutputHeight
	seqLength := layer.SeqLength

	gradInput = NewTensor[float32](batchSize, seqLength, inputSize)
	wCount := layer.WeightStore.WeightCount(layer.DType)
	gradWeights = NewTensor[float32](wCount)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	ihSize := hiddenSize * inputSize
	hhSize := hiddenSize * hiddenSize
	gateSize := ihSize + hhSize + hiddenSize
	wI := wData[0:gateSize]
	wF := wData[gateSize : 2*gateSize]
	wG := wData[2*gateSize : 3*gateSize]
	wO := wData[3*gateSize : 4*gateSize]

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	lstmSimdBackwardBPTTParallel(
		gi64, gw64,
		gradOutput.Data, input.Data, preAct.Data,
		wI, wF, wG, wO,
		batchSize, inputSize, hiddenSize, seqLength, ihSize, hhSize, gateSize, wCount,
	)

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
	return gradInput, gradWeights
}

// lstmSimdBackwardBPTTParallel mirrors lstmBackwardTiledParallel (parallel over batch, BPTT over time).
func lstmSimdBackwardBPTTParallel(
	gi64, gw64 []float64,
	gradOutput, input, preAct []float32,
	wI, wF, wG, wO []float32,
	batchSize, inputSize, hiddenSize, seqLength, ihSize, hhSize, gateSize, wCount int,
) {
	var mu sync.Mutex
	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(b int) {
			defer func() { <-sem; wg.Done() }()

			localGI := make([]float64, seqLength*inputSize)
			localGW := make([]float64, wCount)
			gradH := make([]float64, hiddenSize)
			gradC := make([]float64, hiddenSize)

			inBatch := b * seqLength * inputSize
			outBatch := b * seqLength * hiddenSize
			preBatch := b * seqLength * 5 * hiddenSize

			for t := seqLength - 1; t >= 0; t-- {
				nextGradH := make([]float64, hiddenSize)
				nextGradC := make([]float64, hiddenSize)
				pIdx := preBatch + t*5*hiddenSize
				itConv := t * inputSize
				inputRow := input[inBatch+itConv : inBatch+itConv+inputSize]
				giRow := localGI[itConv : itConv+inputSize]

				for h := 0; h < hiddenSize; h++ {
					dh := gradH[h] + float64(gradOutput[outBatch+t*hiddenSize+h])
					iS := float64(preAct[pIdx+h])
					fS := float64(preAct[pIdx+hiddenSize+h])
					gS := float64(preAct[pIdx+2*hiddenSize+h])
					oS := float64(preAct[pIdx+3*hiddenSize+h])
					cC := float64(preAct[pIdx+4*hiddenSize+h])

					iG := 1.0 / (1.0 + math.Exp(-iS))
					fG := 1.0 / (1.0 + math.Exp(-fS))
					oG := 1.0 / (1.0 + math.Exp(-oS))
					gG := math.Tanh(gS)
					cT := math.Tanh(cC)

					cP := 0.0
					if t > 0 {
						cP = float64(preAct[pIdx-5*hiddenSize+4*hiddenSize+h])
					}
					dc := gradC[h] + dh*oG*(1.0-cT*cT)

					dI := dc * gG * iG * (1.0 - iG)
					dF := dc * cP * fG * (1.0 - fG)
					dG := dc * iG * (1.0 - gG*gG)
					dO := dh * cT * oG * (1.0 - oG)

					nextGradC[h] = dc * fG

					localGW[ihSize+hhSize+h] += dI
					localGW[gateSize+ihSize+hhSize+h] += dF
					localGW[2*gateSize+ihSize+hhSize+h] += dG
					localGW[3*gateSize+ihSize+hhSize+h] += dO

					wIHBase := h * inputSize
					simd.SaxpyF32AccF64(localGW[wIHBase:wIHBase+inputSize], dI, inputRow, inputSize)
					simd.SaxpyF32AccF64(localGW[gateSize+wIHBase:gateSize+wIHBase+inputSize], dF, inputRow, inputSize)
					simd.SaxpyF32AccF64(localGW[2*gateSize+wIHBase:2*gateSize+wIHBase+inputSize], dG, inputRow, inputSize)
					simd.SaxpyF32AccF64(localGW[3*gateSize+wIHBase:3*gateSize+wIHBase+inputSize], dO, inputRow, inputSize)

					simd.SaxpyF32AccF64(giRow, dI, wI[wIHBase:wIHBase+inputSize], inputSize)
					simd.SaxpyF32AccF64(giRow, dF, wF[wIHBase:wIHBase+inputSize], inputSize)
					simd.SaxpyF32AccF64(giRow, dG, wG[wIHBase:wIHBase+inputSize], inputSize)
					simd.SaxpyF32AccF64(giRow, dO, wO[wIHBase:wIHBase+inputSize], inputSize)

					for hp := 0; hp < hiddenSize; hp++ {
						hvP := 0.0
						if t > 0 {
							pP := pIdx - 5*hiddenSize
							oGP := 1.0 / (1.0 + math.Exp(-float64(preAct[pP+3*hiddenSize+hp])))
							hvP = oGP * math.Tanh(float64(preAct[pP+4*hiddenSize+hp]))
						}
						hhOff := ihSize + h*hiddenSize + hp
						localGW[hhOff] += dI * hvP
						localGW[gateSize+hhOff] += dF * hvP
						localGW[2*gateSize+hhOff] += dG * hvP
						localGW[3*gateSize+hhOff] += dO * hvP

						wSumHH := float64(wI[hhOff])*dI + float64(wF[hhOff])*dF + float64(wG[hhOff])*dG + float64(wO[hhOff])*dO
						nextGradH[hp] += wSumHH
					}
				}
				gradH = nextGradH
				gradC = nextGradC
			}

			mu.Lock()
			giStart := b * seqLength * inputSize
			for i := 0; i < len(localGI); i++ {
				gi64[giStart+i] += localGI[i]
			}
			for i := 0; i < len(localGW); i++ {
				gw64[i] += localGW[i]
			}
			mu.Unlock()
		}(b)
	}
	wg.Wait()
}
