package poly

import "github.com/openfluke/loom/poly/simd"

// Native MHA SIMD:
//   - MAC dtypes (FP8, Float16, Int32, …): materialize f32 weights → mha_*SimdF32WithWeights
//   - True integer (Int8, Int4, …): scalar integer native loops with simd.DotI8Tile on every int8 MAC

// MHANativeSimdApplies reports whether native-exact SIMD is active for this layer when simd forward is on.
func MHANativeSimdApplies(layer *VolumetricLayer) bool {
	return layer != nil && useMHANativeExact(layer) && simd.SimdEnabled()
}

// tryMHAForwardNativeSimd runs native-exact MHA forward with SIMD (MAC dtypes via f32 tiles).
func tryMHAForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() || !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	if useMHATrueNative(layer) {
		// Integer path: mhaForwardIntegerNative uses simd.DotI8Tile via int8HeadDot/int8DotRowAcc.
		return nil, nil, false
	}
	return mhaForwardNativeMACSimd(layer, input)
}

// tryMHABackwardNativeSimd runs native-exact MHA backward with SIMD (MAC dtypes via f32 tiles).
func tryMHABackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() || !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	if useMHATrueNative(layer) {
		return nil, nil, false
	}
	return mhaBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func mhaForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	preAct, postAct = mhaForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func mhaBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	gradInput, gradWeights = mhaBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}
