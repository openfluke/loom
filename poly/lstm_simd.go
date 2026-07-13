package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryLSTMForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := lstmForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

// lstmGateSum computes bias + dot(x_t, wGate_IH[h,:]) + dot(h_{t-1}, wGate_HH[h,:]) for one gate.
func lstmGateSum(wGate []float32, h int, xT, hPrev []float32, inputSize, hiddenSize, ihSize, hhSize, tileSize int) float64 {
	sum := float64(wGate[ihSize+hhSize+h])
	ihRow := wGate[h*inputSize : (h+1)*inputSize]
	for iTile := 0; iTile < inputSize; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > inputSize {
			iEnd = inputSize
		}
		sum = simd.DotTile(xT, ihRow, iTile, iEnd, sum)
	}
	hhRow := wGate[ihSize+h*hiddenSize : ihSize+(h+1)*hiddenSize]
	for pTile := 0; pTile < hiddenSize; pTile += tileSize {
		pEnd := pTile + tileSize
		if pEnd > hiddenSize {
			pEnd = hiddenSize
		}
		sum = simd.DotTile(hPrev, hhRow, pTile, pEnd, sum)
	}
	return sum
}

func lstmForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)
	return lstmForwardSimdF32WithWeights(layer, input, wData)
}

func lstmForwardSimdF32WithWeights(layer *VolumetricLayer, input *Tensor[float32], wData []float32) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	hiddenSize := layer.OutputHeight
	seqLength := layer.SeqLength
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capRNNTileToLayer(tileSize, inputSize, hiddenSize)

	preAct = NewTensor[float32](batchSize, seqLength, 5*hiddenSize)
	postAct = NewTensor[float32](batchSize, seqLength, hiddenSize)

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
			hPrev := make([]float32, hiddenSize) // float32 for DotTile; recurrent hidden state
			cPrev := make([]float64, hiddenSize) // full-precision cell state
			for t := 0; t < seqLength; t++ {
				itBase := b*seqLength*inputSize + t*inputSize
				xT := input.Data[itBase : itBase+inputSize]
				pIdxBase := b*seqLength*5*hiddenSize + t*5*hiddenSize
				hNext := make([]float32, hiddenSize)
				for h := 0; h < hiddenSize; h++ {
					iS := lstmGateSum(wI, h, xT, hPrev, inputSize, hiddenSize, ihSize, hhSize, tileSize)
					fS := lstmGateSum(wF, h, xT, hPrev, inputSize, hiddenSize, ihSize, hhSize, tileSize)
					gS := lstmGateSum(wG, h, xT, hPrev, inputSize, hiddenSize, ihSize, hhSize, tileSize)
					oS := lstmGateSum(wO, h, xT, hPrev, inputSize, hiddenSize, ihSize, hhSize, tileSize)

					// Round gate pre-activations to float32 (matches tiled storage) before gating.
					preAct.Data[pIdxBase+h] = float32(iS)
					preAct.Data[pIdxBase+hiddenSize+h] = float32(fS)
					preAct.Data[pIdxBase+2*hiddenSize+h] = float32(gS)
					preAct.Data[pIdxBase+3*hiddenSize+h] = float32(oS)

					iSr := float64(preAct.Data[pIdxBase+h])
					fSr := float64(preAct.Data[pIdxBase+hiddenSize+h])
					gSr := float64(preAct.Data[pIdxBase+2*hiddenSize+h])
					oSr := float64(preAct.Data[pIdxBase+3*hiddenSize+h])

					iG := 1.0 / (1.0 + math.Exp(-iSr))
					fG := 1.0 / (1.0 + math.Exp(-fSr))
					gG := math.Tanh(gSr)
					oG := 1.0 / (1.0 + math.Exp(-oSr))

					cC := fG*cPrev[h] + iG*gG
					hC := oG * math.Tanh(cC)

					preAct.Data[pIdxBase+4*hiddenSize+h] = float32(cC)
					postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = float32(hC)
					cPrev[h] = cC
					hNext[h] = float32(hC)
				}
				hPrev = hNext
			}
		}(b)
	}
	wg.Wait()
	return preAct, postAct
}
