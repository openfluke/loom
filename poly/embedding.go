package poly

// EmbeddingForwardPolymorphic performs an embedding lookup across any numerical type.
func EmbeddingForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)

	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embeddingDim)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

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
	if scaleW == 0 { scaleW = 1.0 }
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
	vocabSize := layer.VocabSize
	embeddingDim := layer.EmbeddingDim
	seqLen := len(input.Data)

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](vocabSize * embeddingDim)

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
