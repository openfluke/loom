package poly

import (
	asmdense "github.com/openfluke/loom/poly/asm/dense"
)

func layerUseAsmForward(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	if layer.UseAsmForward {
		return true
	}
	if layer.Network != nil && layer.Network.UseAsmForward {
		return true
	}
	return false
}

// denseForwardAsm runs the asm/dense forward path for any Numeric activation/weight type.
func denseForwardAsm[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if usePackedTernaryCPU(layer) {
		return DenseForwardPackedTernaryCPU(layer, input)
	}

	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	tileSize := layer.GetCPUTileSize(layer.DType)
	asmdense.Forward(
		preAct.Data, input.Data, wData,
		batchSize, inputSize, outputSize,
		layer.EnableMultiCoreTiling, tileSize,
	)

	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}
