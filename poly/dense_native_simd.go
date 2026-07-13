package poly

import (
	"math/rand"

	"github.com/openfluke/loom/poly/simd"
)

// dense_native_simd.go — native-exact Dense SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8/u8 kernels.

// DenseNativeSimdApplies reports whether native-exact SIMD is active when simd forward is on.
func DenseNativeSimdApplies(layer *VolumetricLayer) bool {
	return layer != nil && useDenseNativeExact(layer) && simd.SimdEnabled()
}

func tryDenseForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if usePackedTernaryCPU(layer) {
		pre, post := DenseForwardPackedTernaryCPU(layer, input)
		return pre, post, true
	}
	if useDenseTrueNative(layer) {
		return denseForwardIntegerNativeSimd(layer, input)
	}
	return denseForwardNativeMACSimd(layer, input)
}

func tryDenseBackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	// Packed ternary forward uses BitNet matvec; backward gradients match MAC SIMD on cached f32 weights.
	if usePackedTernaryCPU(layer) {
		return denseBackwardNativeMACSimd(layer, gradOutput, input, preAct)
	}
	if useDenseTrueNative(layer) {
		return denseBackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return denseBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func denseForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		preAct, postAct = denseForwardSimdF32(layer, input)
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
	preAct, postAct = denseForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func denseBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		gradInput, gradWeights = denseBackwardSimdF32(layer, gradOutput, input, preAct)
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
	gradInput, gradWeights = denseBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func denseForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}

	batch := input.Shape[0]
	inSz := layer.InputHeight
	outSz := layer.OutputHeight
	cache := ensureDenseExactCache(layer, batch, inSz, outSz)

	for b := 0; b < batch; b++ {
		row := input.Data[b*inSz : (b+1)*inSz]
		codes := quantizeRowF32ToI8(row, scale)
		copy(cache.InputI8[b*inSz:(b+1)*inSz], codes)
	}

	preAct = NewTensor[float32](batch, outSz)
	postAct = NewTensor[float32](batch, outSz)

	switch layer.DType {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		w := ws.NativeSimdU8Weights(layer.DType)
		if w == nil {
			return nil, nil, false
		}
		for i, c := range cache.InputI8 {
			cache.InputU8[i] = uint8(clampU8(int32(c)))
		}
		trueUint8DenseForwardSimd(w, cache.InputU8, batch, inSz, outSz, cache.PreI8, cache.PostI8)
	default:
		w := ws.NativeSimdI8Weights(layer.DType)
		if w == nil {
			return nil, nil, false
		}
		trueInt8DenseForwardSimd(w, cache.InputI8, batch, inSz, outSz, cache.PreI8, cache.PostI8, layer.Activation)
	}

	dequantI8Row(cache.PostI8, scale, postAct.Data)
	copy(preAct.Data, postAct.Data)
	if layer.Activation == ActivationLinear {
		copy(cache.PreI8, cache.PostI8)
	}
	return preAct, postAct, true
}

func denseBackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	_ = preAct
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return nil, nil, false
	}

	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch := input.Shape[0]
	inSz := layer.InputHeight
	outSz := layer.OutputHeight
	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)

	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)

	gradInput = NewTensor[float32](batch, inSz)
	gradWeights = NewTensor[float32](outSz, inSz)

	switch layer.DType {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		w := ws.NativeSimdU8Weights(layer.DType)
		for i, c := range cache.InputI8 {
			cache.InputU8[i] = uint8(clampU8(int32(c)))
		}
		gOutU8 := make([]uint8, len(gradOutI8))
		for i, c := range gradOutI8 {
			gOutU8[i] = uint8(clampU8(int32(c)))
		}
		giU8, wU8 := trueUint8DenseBackwardSimd(w, cache.InputU8, gOutU8, batch, inSz, outSz, lrShift)
		ws.Versions[layer.DType] = wU8
		dequantU8Row(giU8, scale, gradInput.Data)
	default:
		w := ws.NativeSimdI8Weights(layer.DType)
		giI8, wI8 := trueInt8DenseBackwardSimd(
			w, cache.InputI8, gradOutI8,
			batch, inSz, outSz,
			lrShift, layer.Activation, cache.PreI8,
		)
		raw := make([]uint8, len(wI8))
		for i, v := range wI8 {
			raw[i] = uint8(v)
		}
		ws.Versions[layer.DType] = raw
		ws.Master = nil
		dequantI8Row(giI8, scale, gradInput.Data)
	}

	cache.WeightsUpdated = true
	ws.GPUWeights = make(map[DType]any)
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, layer.DType)
	}
	ws.invalidateNativeSimdCache(layer.DType)
	return gradInput, gradWeights, true
}

// trueInt8DenseForwardSimd: int8 MAC forward with simd.DotI8Tile on every output row.
func trueInt8DenseForwardSimd(weights, inputs []int8, batch, inSz, outSz int, pre, post []int8, act ActivationType) {
	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			acc := simd.DotI8Tile(weights, inputs, o*inSz, inOff, inSz, 0)
			pre[outOff+o] = clampI8(acc >> 8)
			if act == ActivationReLU && pre[outOff+o] <= 0 {
				post[outOff+o] = 0
			} else {
				post[outOff+o] = pre[outOff+o]
			}
		}
	}
}

// trueUint8DenseForwardSimd: uint8 MAC forward with simd.DotU8Tile on every output row.
func trueUint8DenseForwardSimd(weights, inputs []uint8, batch, inSz, outSz int, pre, post []int8) {
	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			acc := simd.DotU8Tile(weights, inputs, o*inSz, inOff, inSz, 0)
			pre[outOff+o] = int8(clampU8(acc >> 8))
			post[outOff+o] = pre[outOff+o]
		}
	}
}

// trueInt8DenseBackwardSimd: int8 backward with simd saxpy kernels on ∂W and ∂X.
func trueInt8DenseBackwardSimd(
	weights []int8,
	inputs []int8,
	gradOutput []int8,
	batch, inSz, outSz int,
	lrBitShift uint,
	act ActivationType,
	preI8 []int8,
) ([]int8, []int8) {
	gradInput := make([]int8, batch*inSz)
	gradInputAcc := make([]int32, batch*inSz)
	gradWeights := make([]int32, outSz*inSz)

	gradPre := make([]int8, batch*outSz)
	for i := range gradOutput {
		g := gradOutput[i]
		if act == ActivationReLU && preI8[i] <= 0 {
			g = 0
		}
		gradPre[i] = g
	}

	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			gOut := int32(gradPre[outOff+o])
			rowOff := o * inSz
			inRow := inputs[inOff : inOff+inSz]
			simd.SaxpyI8ScaleI32Acc(gradWeights, rowOff, inRow, gOut, inSz)
			simd.SaxpyI8ShiftedInputGradAcc(gradInputAcc[inOff:inOff+inSz], weights, rowOff, gOut, inSz)
		}
	}
	for i, acc := range gradInputAcc {
		gradInput[i] = clampI8(acc)
	}

	mask := int32((1 << lrBitShift) - 1)
	for i := range weights {
		scaledGrad := gradWeights[i] >> lrBitShift
		remainder := gradWeights[i] & mask
		if remainder > rand.Int31n(1<<lrBitShift) {
			scaledGrad++
		}
		next := int32(weights[i]) - scaledGrad
		weights[i] = clampI8(next)
	}
	return gradInput, weights
}

func trueUint8DenseBackwardSimd(weights []uint8, inputs, gradOutput []uint8, batch, inSz, outSz int, lrBitShift uint) ([]uint8, []uint8) {
	gradInputAcc := make([]int32, batch*inSz)
	gradWeights := make([]int32, outSz*inSz)

	for b := 0; b < batch; b++ {
		inOff := b * inSz
		outOff := b * outSz
		for o := 0; o < outSz; o++ {
			gOut := int32(gradOutput[outOff+o])
			rowOff := o * inSz
			inRow := inputs[inOff : inOff+inSz]
			simd.SaxpyU8ScaleI32Acc(gradWeights, rowOff, inRow, gOut, inSz)
			simd.SaxpyU8ShiftedInputGradAcc(gradInputAcc[inOff:inOff+inSz], weights, rowOff, gOut, inSz)
		}
	}
	gradInput := make([]uint8, batch*inSz)
	for i, acc := range gradInputAcc {
		gradInput[i] = clampU8(acc)
	}
	mask := int32((1 << lrBitShift) - 1)
	for i := range weights {
		scaledGrad := gradWeights[i] >> lrBitShift
		if (gradWeights[i] & mask) > rand.Int31n(1<<lrBitShift) {
			scaledGrad++
		}
		next := int32(weights[i]) - scaledGrad
		weights[i] = clampU8(next)
	}
	return gradInput, weights
}
