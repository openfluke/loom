package poly

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
	tileSize := layer.TileSize
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

			// Core tile
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

	// Final Activation pass
	for i := 0; i < len(preAct.Data); i++ {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}

	return preAct, postAct
}

// EmbeddingBackwardTiled implements a loop-blocked gradient calculation for embeddings.
func EmbeddingBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)
	tileSize := layer.TileSize
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
