package poly

import (
	"runtime"
	"sync"
)

// embedding_simd.go — parallel embedding lookup / scatter with explicit weight buffer.

func embeddingForwardSimdF32WithWeights(layer *VolumetricLayer, input *Tensor[float32], wData []float32) (preAct, postAct *Tensor[float32]) {
	vocabSize := layer.VocabSize
	embeddingDim := embeddingDim(layer)
	seqLen := embeddingTokenCount(input)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embeddingDim)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for iTile := 0; iTile < seqLen; iTile += tileSize {
		sem <- struct{}{}
		wg.Add(1)
		go func(iTile int) {
			defer func() { <-sem; wg.Done() }()
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embeddingDim
				outBase := i * embeddingDim
				for j := 0; j < embeddingDim; j++ {
					if rowBase+j >= len(wData) {
						break
					}
					preAct.Data[outBase+j] = wData[rowBase+j]
				}
			}
		}(iTile)
	}
	wg.Wait()

	for i := range preAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

func embeddingBackwardSimdF32WithWeights(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	vocabSize := layer.VocabSize
	embeddingDim := embeddingDim(layer)
	gradStride := embeddingGradStride(gradOutput, embeddingDim)
	seqLen := embeddingTokenCount(input)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](vocabSize, embeddingDim)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	numTiles := (seqLen + tileSize - 1) / tileSize
	localGradWSlices := make([][]float32, numTiles)
	for k := 0; k < numTiles; k++ {
		localGradWSlices[k] = make([]float32, vocabSize*embeddingDim)
	}

	for k := 0; k < numTiles; k++ {
		iTile := k * tileSize
		sem <- struct{}{}
		wg.Add(1)
		go func(k, iTile int) {
			defer func() { <-sem; wg.Done() }()
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			localGW := localGradWSlices[k]
			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embeddingDim
				outBase := embeddingOutBase(i, gradStride)
				if outBase+embeddingDim > len(gradOutput.Data) {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					localGW[rowBase+j] += gradOutput.Data[outBase+j]
				}
			}
		}(k, iTile)
	}
	wg.Wait()

	for k := 0; k < numTiles; k++ {
		for i, v := range localGradWSlices[k] {
			gradWeights.Data[i] += v
		}
	}
	return gradInput, gradWeights
}
