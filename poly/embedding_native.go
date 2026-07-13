package poly

import "math"

// embedding_native.go — embedding lookup/update in storage dtype (no bulk GetActive decode).

func useEmbeddingNativeExact(layer *VolumetricLayer) bool {
	return useLayerNativeExact(layer) && layer.Type == LayerEmbedding
}

func useEmbeddingTrueNative(layer *VolumetricLayer) bool {
	return useEmbeddingNativeExact(layer) && IsTrueNativeDType(layer.DType)
}

func embeddingDim(layer *VolumetricLayer) int {
	if layer.EmbeddingDim > 0 {
		return layer.EmbeddingDim
	}
	if layer.OutputHeight > 0 {
		return layer.OutputHeight
	}
	return 1
}

func embeddingTokenCount(input *Tensor[float32]) int {
	if input == nil {
		return 0
	}
	return len(input.Data)
}

// embeddingGradStride returns embedding width implied by gradOutput shape.
func embeddingGradStride(gradOutput *Tensor[float32], embDim int) int {
	if gradOutput == nil || len(gradOutput.Shape) == 0 {
		return embDim
	}
	if len(gradOutput.Shape) >= 2 {
		return gradOutput.Shape[len(gradOutput.Shape)-1]
	}
	return embDim
}

func embeddingOutBase(tokenIdx, gradStride int) int {
	return tokenIdx * gradStride
}

func EmbeddingForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return EmbeddingForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if pre, post, simdOK := tryEmbeddingForwardNativeSimd(layer, in); simdOK {
			preF, postF = pre, post
		}
	}
	if preF == nil {
		if useEmbeddingTrueNative(layer) {
			preF, postF = embeddingForwardIntegerNative(layer, in)
		} else {
			preF, postF = embeddingForwardNativeMAC(layer, in)
		}
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return EmbeddingForwardTiled(layer, input)
	}
	return pre, post
}

func EmbeddingBackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	if !okIn || !okGO {
		return EmbeddingBackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, gw, simdOK := tryEmbeddingBackwardNativeSimd(layer, goT, in); simdOK {
			giF, gwF = gi, gw
		}
	}
	if giF == nil {
		if useEmbeddingTrueNative(layer) {
			giF, gwF = embeddingBackwardIntegerNative(layer, goT, in)
		} else {
			giF, gwF = embeddingBackwardNativeMAC(layer, goT, in)
		}
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return EmbeddingBackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func embeddingForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return embeddingForwardNativeMAC(layer, input)
	}

	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	seqLen := embeddingTokenCount(input)
	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embDim)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

	for i := 0; i < seqLen; i++ {
		tokenID := int(input.Data[i])
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}
		rowBase := tokenID * embDim
		outBase := i * embDim
		for j := 0; j < embDim; j++ {
			if rowBase+j >= len(w) {
				break
			}
			v := float32(w[rowBase+j]) * scale
			preAct.Data[outBase+j] = v
			postAct.Data[outBase+j] = Activate(v, layer.Activation)
		}
	}
	return preAct, postAct
}

func embeddingForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	seqLen := embeddingTokenCount(input)
	outShape := append([]int{}, input.Shape...)
	outShape = append(outShape, embDim)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

	wCount := layer.WeightStore.WeightCount(layer.DType)

	for i := 0; i < seqLen; i++ {
		tokenID := int(input.Data[i])
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}
		rowBase := tokenID * embDim
		outBase := i * embDim
		for j := 0; j < embDim; j++ {
			idx := rowBase + j
			if idx >= wCount {
				break
			}
			v := nativeWeightValueF32(layer.WeightStore, layer.DType, idx)
			preAct.Data[outBase+j] = v
			postAct.Data[outBase+j] = Activate(v, layer.Activation)
		}
	}
	return preAct, postAct
}

func embeddingBackwardIntegerNative(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return embeddingBackwardNativeMAC(layer, gradOutput, input)
	}

	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	gradStride := embeddingGradStride(gradOutput, embDim)
	seqLen := embeddingTokenCount(input)
	wCount := len(w)

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](wCount)
	gradW := make([]int32, wCount)

	for i := 0; i < seqLen; i++ {
		tokenID := int(input.Data[i])
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}
		rowBase := tokenID * embDim
		outBase := embeddingOutBase(i, gradStride)
		if outBase+embDim > len(gradOutput.Data) {
			continue
		}
		for j := 0; j < embDim; j++ {
			idx := rowBase + j
			if idx >= wCount {
				break
			}
			g := int32(math.Round(float64(gradOutput.Data[outBase+j]) / float64(scale)))
			gradW[idx] += g
		}
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights
}

func embeddingBackwardNativeMAC(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	vocabSize := layer.VocabSize
	embDim := embeddingDim(layer)
	gradStride := embeddingGradStride(gradOutput, embDim)
	seqLen := embeddingTokenCount(input)
	wCount := layer.WeightStore.WeightCount(layer.DType)
	if wCount <= 0 {
		wCount = vocabSize * embDim
	}

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](wCount)
	gwAcc := make([]float64, wCount)

	for i := 0; i < seqLen; i++ {
		tokenID := int(input.Data[i])
		if tokenID < 0 || tokenID >= vocabSize {
			continue
		}
		rowBase := tokenID * embDim
		outBase := embeddingOutBase(i, gradStride)
		if outBase+embDim > len(gradOutput.Data) {
			continue
		}
		for j := 0; j < embDim; j++ {
			idx := rowBase + j
			if idx >= wCount {
				break
			}
			gwAcc[idx] += float64(gradOutput.Data[outBase+j])
		}
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gwAcc[i])
	}
	return gradInput, gradWeights
}
