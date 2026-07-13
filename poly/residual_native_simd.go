package poly

import "github.com/openfluke/loom/poly/simd"

// residual_native_simd.go — native-exact residual SIMD: parallel tiled element-wise add.

func tryResidualForwardNativeSimd(layer *VolumetricLayer, input, skip *Tensor[float32]) (postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, false
	}
	if skip == nil || len(skip.Data) != len(input.Data) {
		return nil, false
	}
	return residualForwardSimdF32(layer, input, skip), true
}

func tryResidualBackwardNativeSimd(layer *VolumetricLayer, gradOutput *Tensor[float32]) (gradInput *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, false
	}
	return residualBackwardNative(gradOutput), true
}

func residualForwardSimdF32(layer *VolumetricLayer, input, skip *Tensor[float32]) *Tensor[float32] {
	output := NewTensor[float32](input.Shape...)
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 1024
	}
	residualForwardTiledParallel(input.Data, skip.Data, output.Data, tileSize)
	return output
}
