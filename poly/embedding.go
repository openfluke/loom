package poly

import (
	"runtime"
	"sync"
)

// EmbeddingForwardPolymorphic performs an embedding lookup using purely generic T execution.
func EmbeddingForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if useEmbeddingNativeExact(layer) {
		return EmbeddingForwardNativeExact(layer, input)
	}
	return EmbeddingForwardTiled(layer, input)
}

// EmbeddingBackwardPolymorphic computes gradients for embedding lookup.
func EmbeddingBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if useEmbeddingNativeExact(layer) {
		return EmbeddingBackwardNativeExact(layer, gradOutput, input, preAct)
	}
	return EmbeddingBackwardTiled(layer, gradOutput, input, preAct)
}

// EmbeddingForwardTiled implements a loop-blocked embedding lookup for cache efficiency (multi-core).
func EmbeddingForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return embeddingForwardTiledParallel(layer, input)
}

func embeddingForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embeddingDim)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

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

// EmbeddingBackwardTiled implements a loop-blocked gradient calculation for embeddings (multi-core).
func EmbeddingBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return embeddingBackwardTiledParallel(layer, gradOutput, input, preAct)
}

func embeddingBackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](vocabSize, embeddingDim)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	numTiles := (seqLen + tileSize - 1) / tileSize
	localGradWSlices := make([][]T, numTiles)
	for k := 0; k < numTiles; k++ {
		localGradWSlices[k] = make([]T, vocabSize*embeddingDim)
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
				outBase := i * embeddingDim
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
