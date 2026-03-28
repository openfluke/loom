package poly

import (
	"runtime"
	"sync"
)

// EmbeddingForwardPolymorphic performs an embedding lookup across any numerical type.
func EmbeddingForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
		return EmbeddingForwardTiled(layer, input)
	}
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)

	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embeddingDim)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					val := rawW[tokenID*embeddingDim+j]
					preAct.Data[i*embeddingDim+j] = T(val)
					postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					val := rawW[tokenID*embeddingDim+j]
					preAct.Data[i*embeddingDim+j] = T(val)
					postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					val := rawW[tokenID*embeddingDim+j]
					preAct.Data[i*embeddingDim+j] = T(val)
					postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					val := rawW[tokenID*embeddingDim+j]
					preAct.Data[i*embeddingDim+j] = T(val)
					postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					val := rawW[tokenID*embeddingDim+j]
					preAct.Data[i*embeddingDim+j] = T(val)
					postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for i := 0; i < seqLen; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				for j := 0; j < embeddingDim; j++ {
					val := rawW[tokenID*embeddingDim+j]
					preAct.Data[i*embeddingDim+j] = T(val)
					postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
				}
			}
			return preAct, postAct
		}
	}

	// Universal fallback
	scaleW := layer.WeightStore.Scale
	if scaleW == 0 {
		scaleW = 1.0
	}
	wData := CastWeights[float32](weights)

	for i := 0; i < seqLen; i++ {
		tokenID := int(input.Data[i])
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}
		for j := 0; j < embeddingDim; j++ {
			val := SimulatePrecision(wData[tokenID*embeddingDim+j], layer.DType, scaleW)
			preAct.Data[i*embeddingDim+j] = T(val)
			postAct.Data[i*embeddingDim+j] = Activate(T(val), layer.Activation)
		}
	}

	return preAct, postAct
}

// EmbeddingBackwardPolymorphic computes gradients for embedding lookup.
func EmbeddingBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
		return EmbeddingBackwardTiled(layer, gradOutput, input, preAct)
	}
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](vocabSize, embeddingDim)

	// Direct lookup backward
	for i := 0; i < seqLen; i++ {
		tokenID := int(input.Data[i])
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}
		for j := 0; j < embeddingDim; j++ {
			gradWeights.Data[tokenID*embeddingDim+j] += gradOutput.Data[i*embeddingDim+j]
		}
	}

	return gradInput, gradWeights
}

// EmbeddingForwardTiled implements a loop-blocked embedding lookup for cache efficiency.
func EmbeddingForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
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
	wData := CastWeights[float32](weights)
	scaleW := layer.WeightStore.Scale
	if scaleW == 0 {
		scaleW = 1.0
	}

	if layer.EnableMultiCoreTiling {
		embeddingForwardTiledParallel(input.Data, wData, preAct.Data, vocabSize, embeddingDim, seqLen, layer.DType, scaleW, tileSize)
	} else {
		// Double-blocked lookup
		for iTile := 0; iTile < seqLen; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > seqLen {
				iEnd = seqLen
			}
			for jTile := 0; jTile < embeddingDim; jTile += tileSize {
				jEnd := jTile + tileSize
				if jEnd > embeddingDim {
					jEnd = embeddingDim
				}
				for i := iTile; i < iEnd; i++ {
					tokenID := int(input.Data[i])
					if tokenID < 0 || tokenID >= vocabSize {
						continue
					}
					rowBase := tokenID * embeddingDim
					outBase := i * embeddingDim
					for j := jTile; j < jEnd; j++ {
						val := SimulatePrecision(wData[rowBase+j], layer.DType, scaleW)
						preAct.Data[outBase+j] = T(val)
					}
				}
			}
		}
	}

	for i := range preAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

// embeddingForwardTiledParallel runs token lookups in parallel across sequence tiles.
// Each goroutine owns a non-overlapping slice of sequence positions so there are no races.
func embeddingForwardTiledParallel[T Numeric](input []T, wData []float32, out []T, vocabSize, embeddingDim, seqLen int, dtype DType, scale float32, tileSize int) {
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
				tokenID := int(input[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embeddingDim
				outBase := i * embeddingDim
				for j := 0; j < embeddingDim; j++ {
					out[outBase+j] = T(SimulatePrecision(wData[rowBase+j], dtype, scale))
				}
			}
		}(iTile)
	}
	wg.Wait()
}

// EmbeddingBackwardTiled implements a loop-blocked gradient calculation for embeddings.
func EmbeddingBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 32
	}

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](vocabSize, embeddingDim)

	for iTile := 0; iTile < seqLen; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > seqLen {
			iEnd = seqLen
		}

		for jTile := 0; jTile < embeddingDim; jTile += tileSize {
			jEnd := jTile + tileSize
			if jEnd > embeddingDim {
				jEnd = embeddingDim
			}

			for i := iTile; i < iEnd; i++ {
				tokenID := int(input.Data[i])
				if tokenID < 0 || tokenID >= vocabSize {
					continue
				}
				rowBase := tokenID * embeddingDim
				outBase := i * embeddingDim
				for j := jTile; j < jEnd; j++ {
					gradWeights.Data[rowBase+j] += gradOutput.Data[outBase+j]
				}
			}
		}
	}

	return gradInput, gradWeights
}

// embeddingBackwardTiledParallel computes embedding backward in parallel over seqLen tiles.
// Each goroutine accumulates gradWeights into a local buffer (token IDs may collide across
// sequence positions). After WaitGroup, local buffers are merged.
// gradInput is always zero for embeddings — no backprop through token IDs.
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

	// Merge local gradient buffers
	for k := 0; k < numTiles; k++ {
		for i, v := range localGradWSlices[k] {
			gradWeights.Data[i] += v
		}
	}

	return gradInput, gradWeights
}
