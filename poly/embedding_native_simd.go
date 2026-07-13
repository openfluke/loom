package poly

import "github.com/openfluke/loom/poly/simd"

// embedding_native_simd.go — native-exact embedding SIMD: MAC dtypes via parallel lookup; integers via scalar lookup.

func tryEmbeddingForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useEmbeddingTrueNative(layer) {
		pre, post := embeddingForwardIntegerNative(layer, input)
		return pre, post, true
	}
	return embeddingForwardNativeMACSimd(layer, input)
}

func tryEmbeddingBackwardNativeSimd(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useEmbeddingTrueNative(layer) {
		gi, gw := embeddingBackwardIntegerNative(layer, gradOutput, input)
		return gi, gw, true
	}
	return embeddingBackwardNativeMACSimd(layer, gradOutput, input)
}

func embeddingForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		count = layer.VocabSize * embeddingDim(layer)
	}
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	preAct, postAct = embeddingForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func embeddingBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	gradInput, gradWeights = embeddingBackwardSimdF32WithWeights(layer, gradOutput, input)
	return gradInput, gradWeights, true
}
