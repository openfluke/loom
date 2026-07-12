package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

// cnn2_native_simd.go — native-exact CNN2 SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8 kernels.

func tryCNN2ForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useCNNTrueNative(layer) {
		return cnn2ForwardIntegerNativeSimd(layer, input)
	}
	return cnn2ForwardNativeMACSimd(layer, input)
}

func tryCNN2BackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useCNNTrueNative(layer) {
		return cnn2BackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return cnn2BackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func cnn2ForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	preAct, postAct = cnn2ForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func cnn2BackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	gradInput, gradWeights = cnn2BackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func cnn2BuildPatchI8(patch []int8, input []float32, b, oh, ow, inC, inH, inW, kSize, stride, padding int, scale float32) {
	inBatch := b * inC * inH * inW
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inH*inW
		pBase := ic * kSize * kSize
		for kh := 0; kh < kSize; kh++ {
			ih := kh0 + kh
			for kw := 0; kw < kSize; kw++ {
				iw := kw0 + kw
				if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
					patch[pBase+kh*kSize+kw] = clampI8(int32(math.Round(float64(input[icInBase+ih*inW+iw]) / float64(scale))))
				} else {
					patch[pBase+kh*kSize+kw] = 0
				}
			}
		}
	}
}

func cnn2BuildPatchI8Fast(patch []int8, input []float32, b, oh, ow, inC, inH, inW, kSize, stride, padding int, scale float32) {
	inBatch := b * inC * inH * inW
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inH*inW
		pBase := ic * kSize * kSize
		for kh := 0; kh < kSize; kh++ {
			rowStart := icInBase + (kh0+kh)*inW + kw0
			pRow := patch[pBase+kh*kSize : pBase+(kh+1)*kSize]
			for kw := 0; kw < kSize; kw++ {
				pRow[kw] = clampI8(int32(math.Round(float64(input[rowStart+kw]) / float64(scale))))
			}
		}
	}
}

func cnn2FillPatchI8ForConv(patch []int8, input []float32, b, oh, ow, inC, inH, inW, kSize, stride, padding int, scale float32) {
	if cnn2SimdPatchFits(oh, ow, inH, inW, kSize, stride, padding) {
		cnn2BuildPatchI8Fast(patch, input, b, oh, ow, inC, inH, inW, kSize, stride, padding, scale)
		return
	}
	cnn2BuildPatchI8(patch, input, b, oh, ow, inC, inH, inW, kSize, stride, padding, scale)
}

func cnn2ForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
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
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize

	preAct = NewTensor[float32](batch, filters, outH, outW)
	postAct = NewTensor[float32](batch, filters, outH, outW)
	patch := make([]int8, kernelVol)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			wOff := f * kernelVol
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					cnn2FillPatchI8ForConv(patch, input.Data, b, oh, ow, inC, inH, inW, kSize, stride, padding, scale)
					acc := simd.DotI8Tile(w, patch, wOff, 0, kernelVol, 0)
					out := float32(clampI8(acc>>8)) * scale
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					preAct.Data[outIdx] = out
					postAct.Data[outIdx] = Activate(out, layer.Activation)
				}
			}
		}
	}
	return preAct, postAct, true
}

func cnn2BackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
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
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize

	gradInput = NewTensor[float32](batch, inC, inH, inW)
	gradWeights = NewTensor[float32](len(w))
	gradW := make([]int32, len(w))
	giAcc := make([]int32, len(gradInput.Data))
	patch := make([]int8, kernelVol)

	for b := 0; b < batch; b++ {
		inBatch := b * inC * inH * inW
		for f := 0; f < filters; f++ {
			wOff := f * kernelVol
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					g := int32(math.Round(float64(gradOutput.Data[outIdx]) / float64(scale)))
					if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
						g = 0
					}
					if g == 0 {
						continue
					}
					cnn2FillPatchI8ForConv(patch, input.Data, b, oh, ow, inC, inH, inW, kSize, stride, padding, scale)
					simd.SaxpyI8ScaleI32Acc(gradW, wOff, patch, g, kernelVol)

					kh0 := oh*stride - padding
					kw0 := ow*stride - padding
					if cnn2SimdPatchFits(oh, ow, inH, inW, kSize, stride, padding) {
						for ic := 0; ic < inC; ic++ {
							icInBase := inBatch + ic*inH*inW
							wIC := wOff + ic*kSize*kSize
							for kh := 0; kh < kSize; kh++ {
								giOff := icInBase + (kh0+kh)*inW + kw0
								wRow := wIC + kh*kSize
								simd.SaxpyI8ShiftedInputGradAcc(giAcc[giOff:giOff+kSize], w, wRow, g, kSize)
							}
						}
						continue
					}
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							ih := kh0 + kh
							for kw := 0; kw < kSize; kw++ {
								iw := kw0 + kw
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									wIdx := cnn2NativeWeightIndex(filters, inC, kSize, f, ic, kh, kw)
									inIdx := inBatch + ic*inH*inW + ih*inW + iw
									giAcc[inIdx] += (int32(w[wIdx]) * g) >> 8
								}
							}
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
