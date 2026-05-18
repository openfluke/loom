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

// denseForwardAsm runs the asm/dense forward path.
// Low-bit / integer DTypes use native integer matmul in asm (no FP inside the dot).
// Float DTypes use float tiled asm.
func denseForwardAsm[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if isNativeQuantDType(layer.DType) {
		return denseForwardAsmNative(layer, input)
	}
	return denseForwardAsmFloatPath(layer, input)
}

func denseForwardAsmFloatPath[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
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
	denseForwardAsmTyped(layer, input, preAct, wData, batchSize, inputSize, outputSize, tileSize)

	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

func denseForwardAsmTyped[T Numeric](
	layer *VolumetricLayer,
	input *Tensor[T],
	preAct *Tensor[T],
	wData []T,
	batch, inputSize, outputSize, tileSize int,
) {
	asmdense.Forward(
		preAct.Data, input.Data, wData,
		batch, inputSize, outputSize,
		layer.EnableMultiCoreTiling, tileSize,
	)
}
