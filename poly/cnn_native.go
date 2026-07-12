package poly

import "math"

// cnn_native.go — CNN1/2/3 integer-native + per-dot native MAC (no bulk GetActive decode).

func useCNNNativeExact(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	switch layer.Type {
	case LayerCNN1, LayerCNN2, LayerCNN3:
		return useLayerNativeExact(layer)
	default:
		return false
	}
}

func cnnNativeWeightIndex(filters, inC, kSize, f, ic, k int) int {
	return f*inC*kSize + ic*kSize + k
}

func cnn2NativeWeightIndex(filters, inC, kSize, f, ic, kh, kw int) int {
	return f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
}

func cnn1ForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return cnn1ForwardNativeMAC(layer, input)
	}

	batch := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[float32](batch, filters, outLen)
	postAct = NewTensor[float32](batch, filters, outLen)
	patch := make([]int8, inC*kSize)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				pOff := 0
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							patch[pOff] = clampI8(int32(math.Round(float64(input.Data[inIdx]) / float64(scale))))
						} else {
							patch[pOff] = 0
						}
						pOff++
					}
				}
				var acc int32
				pOff = 0
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						wIdx := cnnNativeWeightIndex(filters, inC, kSize, f, ic, k)
						acc += int32(w[wIdx]) * int32(patch[pOff])
						pOff++
					}
				}
				out := float32(clampI8(acc>>8)) * scale
				outIdx := b*filters*outLen + f*outLen + o
				preAct.Data[outIdx] = out
				postAct.Data[outIdx] = Activate(out, layer.Activation)
			}
		}
	}
	return preAct, postAct
}

func cnn1ForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	batch := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[float32](batch, filters, outLen)
	postAct = NewTensor[float32](batch, filters, outLen)
	patch := make([]float32, inC*kSize)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				pOff := 0
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							patch[pOff] = input.Data[inIdx]
						} else {
							patch[pOff] = 0
						}
						pOff++
					}
				}
				var sum float64
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						wIdx := cnnNativeWeightIndex(filters, inC, kSize, f, ic, k)
						sum += float64(patch[ic*kSize+k]) * float64(nativeWeightValueF32(layer.WeightStore, layer.DType, wIdx))
					}
				}
				outIdx := b*filters*outLen + f*outLen + o
				preAct.Data[outIdx] = float32(sum)
				postAct.Data[outIdx] = Activate(float32(sum), layer.Activation)
			}
		}
	}
	return preAct, postAct
}

func cnn1BackwardIntegerNative(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return cnn1BackwardNativeMAC(layer, gradOutput, input, preAct)
	}

	batch := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[float32](batch, inC, seqLen)
	gradWeights = NewTensor[float32](len(w))
	gradW := make([]int32, len(w))
	patch := make([]int8, inC*kSize)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				outIdx := b*filters*outLen + f*outLen + o
				g := int32(math.Round(float64(gradOutput.Data[outIdx]) / float64(scale)))
				if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
					g = 0
				}
				pOff := 0
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							patch[pOff] = clampI8(int32(math.Round(float64(input.Data[inIdx]) / float64(scale))))
						} else {
							patch[pOff] = 0
						}
						pOff++
					}
				}
				for i := range patch {
					wIdx := cnnNativeWeightIndex(filters, inC, kSize, f, i/kSize, i%kSize)
					gradW[wIdx] += int32(patch[i]) * g
				}
				pOff = 0
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							wIdx := cnnNativeWeightIndex(filters, inC, kSize, f, ic, k)
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							gradInput.Data[inIdx] += float32(clampI8((int32(w[wIdx])*g)>>8)) * scale
						}
						pOff++
					}
				}
			}
		}
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights
}

func cnn1BackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	batch := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	wCount := layer.WeightStore.WeightCount(layer.DType)

	gradInput = NewTensor[float32](batch, inC, seqLen)
	gradWeights = NewTensor[float32](wCount)
	gwAcc := make([]float64, wCount)
	giAcc := make([]float64, len(gradInput.Data))
	patch := make([]float32, inC*kSize)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				outIdx := b*filters*outLen + f*outLen + o
				g := float64(gradOutput.Data[outIdx])
				if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
					g = 0
				}
				pOff := 0
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							patch[pOff] = input.Data[inIdx]
						} else {
							patch[pOff] = 0
						}
						pOff++
					}
				}
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						wIdx := cnnNativeWeightIndex(filters, inC, kSize, f, ic, k)
						gwAcc[wIdx] += nativeGradW(layer, patch[ic*kSize+k], g)
					}
				}
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							wIdx := cnnNativeWeightIndex(filters, inC, kSize, f, ic, k)
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							giAcc[inIdx] += nativeGradX(layer, wIdx, g)
						}
					}
				}
			}
		}
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gwAcc[i])
	}
	for i := range gradInput.Data {
		gradInput.Data[i] = float32(giAcc[i])
	}
	return gradInput, gradWeights
}

func cnn2ForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return cnn2ForwardNativeMAC(layer, input)
	}

	batch := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[float32](batch, filters, outH, outW)
	postAct = NewTensor[float32](batch, filters, outH, outW)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var acc int32
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									wIdx := cnn2NativeWeightIndex(filters, inC, kSize, f, ic, kh, kw)
									inQ := clampI8(int32(math.Round(float64(input.Data[inIdx]) / float64(scale))))
									acc += int32(w[wIdx]) * int32(inQ)
								}
							}
						}
					}
					out := float32(clampI8(acc>>8)) * scale
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					preAct.Data[outIdx] = out
					postAct.Data[outIdx] = Activate(out, layer.Activation)
				}
			}
		}
	}
	return preAct, postAct
}

func cnn2BackwardIntegerNative(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return cnn2BackwardNativeMAC(layer, gradOutput, input, preAct)
	}

	batch := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[float32](batch, inC, inH, inW)
	gradWeights = NewTensor[float32](len(w))
	gradW := make([]int32, len(w))

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					g := int32(math.Round(float64(gradOutput.Data[outIdx]) / float64(scale)))
					if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
						g = 0
					}
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									wIdx := cnn2NativeWeightIndex(filters, inC, kSize, f, ic, kh, kw)
									inQ := clampI8(int32(math.Round(float64(input.Data[inIdx]) / float64(scale))))
									gradW[wIdx] += int32(inQ) * g
									gradInput.Data[inIdx] += float32(clampI8((int32(w[wIdx])*g)>>8)) * scale
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
	return gradInput, gradWeights
}

func cnn2ForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	batch := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[float32](batch, filters, outH, outW)
	postAct = NewTensor[float32](batch, filters, outH, outW)

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var sum float64
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									wIdx := cnn2NativeWeightIndex(filters, inC, kSize, f, ic, kh, kw)
									sum += float64(input.Data[inIdx]) * float64(nativeWeightValueF32(layer.WeightStore, layer.DType, wIdx))
								}
							}
						}
					}
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					preAct.Data[outIdx] = float32(sum)
					postAct.Data[outIdx] = Activate(float32(sum), layer.Activation)
				}
			}
		}
	}
	return preAct, postAct
}

func cnn2BackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	batch := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	wCount := layer.WeightStore.WeightCount(layer.DType)

	gradInput = NewTensor[float32](batch, inC, inH, inW)
	gradWeights = NewTensor[float32](wCount)
	gwAcc := make([]float64, wCount)
	giAcc := make([]float64, len(gradInput.Data))

	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					g := float64(gradOutput.Data[outIdx])
					if layer.Activation == ActivationReLU && preAct.Data[outIdx] <= 0 {
						g = 0
					}
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									wIdx := cnn2NativeWeightIndex(filters, inC, kSize, f, ic, kh, kw)
									gwAcc[wIdx] += nativeGradW(layer, input.Data[inIdx], g)
									giAcc[inIdx] += nativeGradX(layer, wIdx, g)
								}
							}
						}
					}
				}
			}
		}
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gwAcc[i])
	}
	for i := range gradInput.Data {
		gradInput.Data[i] = float32(giAcc[i])
	}
	return gradInput, gradWeights
}

func cnn3ForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	return cnn2ForwardNativeMAC(layer, input)
}

func cnn3BackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	return cnn2BackwardNativeMAC(layer, gradOutput, input, preAct)
}

func useCNNTrueNative(layer *VolumetricLayer) bool {
	return useCNNNativeExact(layer) && IsTrueNativeDType(layer.DType)
}

func CNN1ForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return CNN1ForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if pre, post, simdOK := tryCNN1ForwardNativeSimd(layer, in); simdOK {
			preF, postF = pre, post
		}
	}
	if preF == nil {
		if useCNNTrueNative(layer) {
			preF, postF = cnn1ForwardIntegerNative(layer, in)
		} else {
			preF, postF = cnn1ForwardNativeMAC(layer, in)
		}
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return CNN1ForwardTiled(layer, input)
	}
	return pre, post
}

func CNN1BackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return CNN1BackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, gw, simdOK := tryCNN1BackwardNativeSimd(layer, goT, in, preF); simdOK {
			giF, gwF = gi, gw
		}
	}
	if giF == nil {
		if useCNNTrueNative(layer) {
			giF, gwF = cnn1BackwardIntegerNative(layer, goT, in, preF)
		} else {
			giF, gwF = cnn1BackwardNativeMAC(layer, goT, in, preF)
		}
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return CNN1BackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func CNN2ForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return CNN2ForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if useCNNTrueNative(layer) {
		preF, postF = cnn2ForwardIntegerNative(layer, in)
	} else {
		preF, postF = cnn2ForwardNativeMAC(layer, in)
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return CNN2ForwardTiled(layer, input)
	}
	return pre, post
}

func CNN2BackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return CNN2BackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if useCNNTrueNative(layer) {
		giF, gwF = cnn2BackwardIntegerNative(layer, goT, in, preF)
	} else {
		giF, gwF = cnn2BackwardNativeMAC(layer, goT, in, preF)
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return CNN2BackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func CNN3ForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return CNN2ForwardNativeExact(layer, input)
}

func CNN3BackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return CNN2BackwardNativeExact(layer, gradOutput, input, preAct)
}
