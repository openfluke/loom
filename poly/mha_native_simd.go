package poly

import "github.com/openfluke/loom/poly/simd"

// mha_native_simd.go — native-exact MHA SIMD via cached f32 weights + Plan 9 DotTile/saxpy kernels.
//
// MHA time is dominated by attention (O(seq²)), not weight decode. Integer-native
// int8 attention/backward cannot match QAT's f32 SIMD saxpy path, so native SIMD
// uses cached NativeSimdF32Weights + the same mha*SimdF32WithWeights kernels as QAT.
// True integer-native (int8 attention) remains in mha_native.go for scalar/non-SIMD.

// MHANativeSimdApplies reports whether native-exact SIMD is active for this layer when simd forward is on.
func MHANativeSimdApplies(layer *VolumetricLayer) bool {
	return layer != nil && useMHANativeExact(layer) && simd.SimdEnabled()
}

func tryMHAForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() || !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	if usePackedTernaryCPU(layer) {
		pre, post := mhaForwardPackedTernaryCPUAsF32(layer, input)
		return pre, post, true
	}
	return mhaForwardNativeMACSimd(layer, input)
}

func tryMHABackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() || !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	return mhaBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func mhaForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		preAct, postAct = mhaForwardSimdF32(layer, input)
		return preAct, postAct, true
	}
	ws := layer.WeightStore
	count := ws.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := ws.NativeSimdF32Weights(layer.DType)
	if wData == nil {
		return nil, nil, false
	}
	preAct, postAct = mhaForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func mhaBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		gradInput, gradWeights = mhaBackwardSimdF32(layer, gradOutput, input, preAct)
		return gradInput, gradWeights, true
	}
	ws := layer.WeightStore
	count := ws.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := ws.NativeSimdF32Weights(layer.DType)
	if wData == nil {
		return nil, nil, false
	}
	gradInput, gradWeights = mhaBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}
