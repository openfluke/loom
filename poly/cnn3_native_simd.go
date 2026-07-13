package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

// cnn3_native_simd.go — native-exact CNN3 SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8 kernels.

func tryCNN3ForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useCNNTrueNative(layer) {
		return cnn3ForwardIntegerNativeSimd(layer, input)
	}
	return cnn3ForwardNativeMACSimd(layer, input)
}

func tryCNN3BackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useCNNTrueNative(layer) {
		return cnn3BackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return cnn3BackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func cnn3ForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	preAct, postAct = cnn3ForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func cnn3BackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	gradInput, gradWeights = cnn3BackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func cnn3BuildPatchI8(patch []int8, input []float32, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding int, scale float32) {
	inBatch := b * inC * inD * inH * inW
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	kVol := kSize * kSize * kSize
	kDStride := kSize * kSize
	kHStride := kSize

	kd0 := od*stride - padding
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inCStride
		pBase := ic * kVol
		for kd := 0; kd < kSize; kd++ {
			id := kd0 + kd
			for kh := 0; kh < kSize; kh++ {
				ih := kh0 + kh
				for kw := 0; kw < kSize; kw++ {
					iw := kw0 + kw
					if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
						patch[pBase+kd*kDStride+kh*kHStride+kw] = clampI8(int32(math.Round(float64(input[icInBase+id*inDStride+ih*inHStride+iw]) / float64(scale))))
					} else {
						patch[pBase+kd*kDStride+kh*kHStride+kw] = 0
					}
				}
			}
		}
	}
}

func cnn3BuildPatchI8Fast(patch []int8, input []float32, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding int, scale float32) {
	inBatch := b * inC * inD * inH * inW
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	kVol := kSize * kSize * kSize
	kDStride := kSize * kSize
	kHStride := kSize

	kd0 := od*stride - padding
	kh0 := oh*stride - padding
	kw0 := ow*stride - padding
	for ic := 0; ic < inC; ic++ {
		icInBase := inBatch + ic*inCStride
		pBase := ic * kVol
		for kd := 0; kd < kSize; kd++ {
			idInBase := icInBase + (kd0+kd)*inDStride
			for kh := 0; kh < kSize; kh++ {
				rowStart := idInBase + (kh0+kh)*inHStride + kw0
				pRow := patch[pBase+kd*kDStride+kh*kHStride : pBase+kd*kDStride+(kh+1)*kHStride]
				for kw := 0; kw < kSize; kw++ {
					pRow[kw] = clampI8(int32(math.Round(float64(input[rowStart+kw]) / float64(scale))))
				}
			}
		}
	}
}

func cnn3FillPatchI8ForConv(patch []int8, input []float32, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding int, scale float32) {
	if cnn3SimdPatchFits(od, oh, ow, inD, inH, inW, kSize, stride, padding) {
		cnn3BuildPatchI8Fast(patch, input, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, scale)
		return
	}
	cnn3BuildPatchI8(patch, input, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, scale)
}

func cnn3ForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
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
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize * kSize

	preAct = NewTensor[float32](batch, filters, outD, outH, outW)
	postAct = NewTensor[float32](batch, filters, outD, outH, outW)
	patch := make([]int8, kernelVol)
	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			wOff := f * kernelVol
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						cnn3FillPatchI8ForConv(patch, input.Data, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, scale)
						acc := simd.DotI8Tile(w, patch, wOff, 0, kernelVol, 0)
						out := float32(clampI8(acc>>8)) * scale
						outIdx := b*filters*outFStride + f*outFStride + od*outDStride + oh*outHStride + ow
						preAct.Data[outIdx] = out
						postAct.Data[outIdx] = Activate(out, layer.Activation)
					}
				}
			}
		}
	}
	return preAct, postAct, true
}

func cnn3BackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
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
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	kernelVol := inC * kSize * kSize * kSize
	filtCStride := kSize * kSize * kSize
	filtDStride := kSize * kSize
	filtHStride := kSize
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW
	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	gradInput = NewTensor[float32](batch, inC, inD, inH, inW)
	gradWeights = NewTensor[float32](len(w))
	gradW := make([]int32, len(w))
	giAcc := make([]int32, len(gradInput.Data))
	patch := make([]int8, kernelVol)

	for b := 0; b < batch; b++ {
		inBatch := b * inC * inCStride
		for f := 0; f < filters; f++ {
			wOff := f * kernelVol
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						outIdx := b*filters*outFStride + f*outFStride + od*outDStride + oh*outHStride + ow
						g := int32(math.Round(float64(gradOutput.Data[outIdx]) / float64(scale)))
						if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
							g = 0
						}
						if g == 0 {
							continue
						}
						cnn3FillPatchI8ForConv(patch, input.Data, b, od, oh, ow, inC, inD, inH, inW, kSize, stride, padding, scale)
						simd.SaxpyI8ScaleI32Acc(gradW, wOff, patch, g, kernelVol)

						kd0 := od*stride - padding
						kh0 := oh*stride - padding
						kw0 := ow*stride - padding
						if cnn3SimdPatchFits(od, oh, ow, inD, inH, inW, kSize, stride, padding) {
							for ic := 0; ic < inC; ic++ {
								icInBase := inBatch + ic*inCStride
								wIC := wOff + ic*filtCStride
								for kd := 0; kd < kSize; kd++ {
									idInBase := icInBase + (kd0+kd)*inDStride
									wKD := wIC + kd*filtDStride
									for kh := 0; kh < kSize; kh++ {
										giOff := idInBase + (kh0+kh)*inHStride + kw0
										wRow := wKD + kh*filtHStride
										simd.SaxpyI8ShiftedInputGradAcc(giAcc[giOff:giOff+kSize], w, wRow, g, kSize)
									}
								}
							}
							continue
						}
						for ic := 0; ic < inC; ic++ {
							for kd := 0; kd < kSize; kd++ {
								id := kd0 + kd
								for kh := 0; kh < kSize; kh++ {
									ih := kh0 + kh
									for kw := 0; kw < kSize; kw++ {
										iw := kw0 + kw
										if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											wIdx := cnn3NativeWeightIndex(filters, inC, kSize, f, ic, kd, kh, kw)
											inIdx := inBatch + ic*inCStride + id*inDStride + ih*inHStride + iw
											giAcc[inIdx] += (int32(w[wIdx]) * g) >> 8
										}
									}
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
