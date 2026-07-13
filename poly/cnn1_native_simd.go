package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

// cnn1_native_simd.go — native-exact CNN1 SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8 kernels.

func tryCNN1ForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useBitpackedCPUCNN1(layer) {
		pre, post := CNN1ForwardPackedCPU(layer, input)
		return pre, post, true
	}
	if useCNNTrueNative(layer) {
		return cnn1ForwardIntegerNativeSimd(layer, input)
	}
	return cnn1ForwardNativeMACSimd(layer, input)
}

func tryCNN1BackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	// Packed forward uses bit-decode matvec; backward gradients match MAC SIMD on cached f32 weights.
	if useBitpackedCPUCNN1(layer) {
		return cnn1BackwardNativeMACSimd(layer, gradOutput, input, preAct)
	}
	if useCNNTrueNative(layer) {
		return cnn1BackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return cnn1BackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func cnn1ForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		preAct, postAct = cnn1ForwardSimdF32(layer, input)
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
	preAct, postAct = cnn1ForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func cnn1BackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		gradInput, gradWeights = cnn1BackwardSimdF32(layer, gradOutput, input, preAct)
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
	gradInput, gradWeights = cnn1BackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func cnn1BuildPatchI8(input []float32, patch []int8, b, o, inC, seqLen, kSize, stride, padding int, scale float32) {
	inBatch := b * inC * seqLen
	pOff := 0
	for ic := 0; ic < inC; ic++ {
		for k := 0; k < kSize; k++ {
			inPos := o*stride + k - padding
			if inPos >= 0 && inPos < seqLen {
				inIdx := inBatch + ic*seqLen + inPos
				patch[pOff] = clampI8(int32(math.Round(float64(input[inIdx]) / float64(scale))))
			} else {
				patch[pOff] = 0
			}
			pOff++
		}
	}
}

func cnn1ForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return nil, nil, false
	}

	batch := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize

	preAct = NewTensor[float32](batch, filters, outLen)
	postAct = NewTensor[float32](batch, filters, outLen)
	patch := make([]int8, kernelVol)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			wOff := f * kernelVol
			for o := 0; o < outLen; o++ {
				cnn1BuildPatchI8(input.Data, patch, b, o, inC, seqLen, kSize, stride, padding, scale)
				acc := simd.DotI8Tile(w, patch, wOff, 0, kernelVol, 0)
				out := float32(clampI8(acc>>8)) * scale
				outIdx := b*filters*outLen + f*outLen + o
				preAct.Data[outIdx] = out
				postAct.Data[outIdx] = Activate(out, layer.Activation)
			}
		}
	}
	return preAct, postAct, true
}

func cnn1BackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return nil, nil, false
	}

	batch := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize

	gradInput = NewTensor[float32](batch, inC, seqLen)
	gradWeights = NewTensor[float32](len(w))
	gradW := make([]int32, len(w))
	giAcc := make([]int32, len(gradInput.Data))
	patch := make([]int8, kernelVol)

	for b := 0; b < batch; b++ {
		inBatch := b * inC * seqLen
		for f := 0; f < filters; f++ {
			wOff := f * kernelVol
			for o := 0; o < outLen; o++ {
				outIdx := b*filters*outLen + f*outLen + o
				g := int32(math.Round(float64(gradOutput.Data[outIdx]) / float64(scale)))
				if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
					g = 0
				}
				if g == 0 {
					continue
				}
				cnn1BuildPatchI8(input.Data, patch, b, o, inC, seqLen, kSize, stride, padding, scale)
				simd.SaxpyI8ScaleI32Acc(gradW, wOff, patch, g, kernelVol)

				inPos0 := o*stride - padding
				if stride == 1 && inPos0 >= 0 && inPos0+kSize <= seqLen {
					for ic := 0; ic < inC; ic++ {
						giOff := inBatch + ic*seqLen + inPos0
						icWOff := wOff + ic*kSize
						simd.SaxpyI8ShiftedInputGradAcc(giAcc[giOff:giOff+kSize], w, icWOff, g, kSize)
					}
					continue
				}
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							wIdx := wOff + ic*kSize + k
							inIdx := inBatch + ic*seqLen + inPos
							giAcc[inIdx] += (int32(w[wIdx]) * g) >> 8
						}
					}
				}
			}
		}
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(clampI8(giAcc[i])) * scale
	}
	return gradInput, gradWeights, true
}
