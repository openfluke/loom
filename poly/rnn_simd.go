package poly

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func tryRNNForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := rnnForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func rnnForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	hiddenSize := layer.OutputHeight
	seqLength := layer.SeqLength
	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	tileSize = capRNNTileToLayer(tileSize, inputSize, hiddenSize)

	preAct = NewTensor[float32](batchSize, seqLength, hiddenSize)
	postAct = NewTensor[float32](batchSize, seqLength, hiddenSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	ihSize, hhSize := hiddenSize*inputSize, hiddenSize*hiddenSize
	wIH, wHH, bH := wData[0:ihSize], wData[ihSize:ihSize+hhSize], wData[ihSize+hhSize:]

	rnnSimdForwardBatchParallel(input, preAct, postAct, wIH, wHH, bH, batchSize, inputSize, hiddenSize, seqLength, tileSize)
	return preAct, postAct
}

func rnnSimdHiddenSum(b float32, xT, hPrev, wIHRow, wHHRow []float32, inputSize, hiddenSize, tileSize int) float64 {
	sum := float64(b)
	for iTile := 0; iTile < inputSize; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > inputSize {
			iEnd = inputSize
		}
		sum = simd.DotTile(xT, wIHRow, iTile, iEnd, sum)
	}
	for pTile := 0; pTile < hiddenSize; pTile += tileSize {
		pEnd := pTile + tileSize
		if pEnd > hiddenSize {
			pEnd = hiddenSize
		}
		sum = simd.DotTile(hPrev, wHHRow, pTile, pEnd, sum)
	}
	return sum
}

func rnnSimdForwardBatchParallel(
	input, preAct, postAct *Tensor[float32],
	wIH, wHH, bH []float32,
	batchSize, inputSize, hiddenSize, seqLength, tileSize int,
) {
	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for b := 0; b < batchSize; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			hPrev := make([]float32, hiddenSize)
			for t := 0; t < seqLength; t++ {
				xT := input.Data[b*seqLength*inputSize+t*inputSize : b*seqLength*inputSize+(t+1)*inputSize]
				outBase := b*seqLength*hiddenSize + t*hiddenSize
				for h := 0; h < hiddenSize; h++ {
					wIHRow := wIH[h*inputSize : (h+1)*inputSize]
					wHHRow := wHH[h*hiddenSize : (h+1)*hiddenSize]
					sum := rnnSimdHiddenSum(bH[h], xT, hPrev, wIHRow, wHHRow, inputSize, hiddenSize, tileSize)
					preAct.Data[outBase+h] = float32(sum)
					postAct.Data[outBase+h] = float32(math.Tanh(sum))
				}
				copy(hPrev, postAct.Data[outBase:outBase+hiddenSize])
			}
		}(b)
	}
	wg.Wait()
}
