package poly

import "math"

// rnn_native.go — RNN integer-native + per-dot native MAC training.

func useRNNNativeExact(layer *VolumetricLayer) bool {
	return useLayerNativeExact(layer) && layer.Type == LayerRNN
}

func useRNNTrueNative(layer *VolumetricLayer) bool {
	return useRNNNativeExact(layer) && IsTrueNativeDType(layer.DType)
}

func RNNForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return RNNForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if pre, post, simdOK := tryRNNForwardNativeSimd(layer, in); simdOK {
			preF, postF = pre, post
		}
	}
	if preF == nil {
		if useRNNTrueNative(layer) {
			preF, postF = rnnForwardIntegerNative(layer, in)
		} else {
			preF, postF = rnnForwardNativeMAC(layer, in)
		}
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return RNNForwardTiled(layer, input)
	}
	return pre, post
}

func RNNBackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return RNNBackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, gw, simdOK := tryRNNBackwardNativeSimd(layer, goT, in, preF); simdOK {
			giF, gwF = gi, gw
		}
	}
	if giF == nil {
		if useRNNTrueNative(layer) {
			giF, gwF = rnnBackwardIntegerNative(layer, goT, in, preF)
		} else {
			giF, gwF = rnnBackwardNativeMAC(layer, goT, in, preF)
		}
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return RNNBackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func rnnForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ihSz, hhSz := hid*inSz, hid*hid

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return rnnForwardNativeMAC(layer, input)
	}

	cache := ensureDenseExactCache(layer, batch*seqLen, inSz, hid)
	preAct = NewTensor[float32](batch, seqLen, hid)
	postAct = NewTensor[float32](batch, seqLen, hid)

	hPrev := make([]int8, hid)

	for b := 0; b < batch; b++ {
		for h := range hPrev {
			hPrev[h] = 0
		}
		for t := 0; t < seqLen; t++ {
			inRow := input.Data[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			inI8 := quantizeRowF32ToI8(inRow, scale)
			copy(cache.InputI8[(b*seqLen+t)*inSz:(b*seqLen+t+1)*inSz], inI8)

			for o := 0; o < hid; o++ {
				var acc int32
				acc += int32(w[ihSz+hhSz+o])
				for i := 0; i < inSz; i++ {
					acc += int32(w[o*inSz+i]) * int32(inI8[i])
				}
				for i := 0; i < hid; i++ {
					acc += int32(w[ihSz+o*hid+i]) * int32(hPrev[i])
				}
				acc >>= 8
				pre := clampI8(acc)
				post := clampI8(int32(math.Tanh(float64(pre)) * 127))
				cache.PreI8[(b*seqLen+t)*hid+o] = pre
				cache.PostI8[(b*seqLen+t)*hid+o] = post
				hPrev[o] = post
				idx := b*seqLen*hid + t*hid + o
				preAct.Data[idx] = float32(pre) * scale
				postAct.Data[idx] = float32(post) * scale
			}
		}
	}
	return preAct, postAct
}

func rnnBackwardIntegerNative(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return rnnBackwardNativeMAC(layer, gradOutput, input, preAct)
	}
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ihSz, hhSz := hid*inSz, hid*hid

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return rnnBackwardNativeMAC(layer, gradOutput, input, preAct)
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))

	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)

	for b := 0; b < batch; b++ {
		gH := make([]int32, hid)
		for t := seqLen - 1; t >= 0; t-- {
			nextGH := make([]int32, hid)
			for o := 0; o < hid; o++ {
				idx := b*seqLen*hid + t*hid + o
				pre := float64(cache.PreI8[idx])
				hVal := math.Tanh(pre * float64(scale))
				gPre := (float64(gH[o]) + float64(gradOutI8[idx])) * (1.0 - hVal*hVal)
				g := int32(gPre / float64(scale))

				gradW[ihSz+hhSz+o] += g
				inI8 := cache.InputI8[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
				int8AccumWeightGrad(gradW, w, inI8, g, o*inSz, inSz)
				for i := 0; i < inSz; i++ {
					gradInput.Data[b*seqLen*inSz+t*inSz+i] += float32(clampI8((int32(w[o*inSz+i])*g)>>8)) * scale
				}
				for hp := 0; hp < hid; hp++ {
					var hPrev int8
					if t > 0 {
						hPrev = cache.PostI8[b*seqLen*hid+(t-1)*hid+hp]
					}
					gradW[ihSz+o*hid+hp] += int32(hPrev) * g
					nextGH[hp] += (int32(w[ihSz+o*hid+hp]) * g) >> 8
				}
			}
			gH = nextGH
		}
	}

	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights
}

func rnnForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ihSz, hhSz := hid*inSz, hid*hid
	preAct = NewTensor[float32](batch, seqLen, hid)
	postAct = NewTensor[float32](batch, seqLen, hid)
	hPrev := make([]float32, hid)

	for b := 0; b < batch; b++ {
		for i := range hPrev {
			hPrev[i] = 0
		}
		for t := 0; t < seqLen; t++ {
			inRow := input.Data[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			for o := 0; o < hid; o++ {
				sum := nativeBiasAt(layer, ihSz+hhSz+o)
				sum += nativeDotRow(layer, inRow, o*inSz, inSz)
				for i := 0; i < hid; i++ {
					sum += hPrev[i] * nativeWeightValueF32(layer.WeightStore, layer.DType, ihSz+o*hid+i)
				}
				idx := b*seqLen*hid + t*hid + o
				preAct.Data[idx] = sum
				postAct.Data[idx] = float32(math.Tanh(float64(sum)))
				hPrev[o] = postAct.Data[idx]
			}
		}
	}
	return preAct, postAct
}

func rnnBackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ihSz, hhSz := hid*inSz, hid*hid
	gradInput = NewTensor[float32](batch, seqLen, inSz)
	gradWeights = NewTensor[float32](layer.WeightStore.WeightCount(layer.DType))
	gwAcc := make([]float64, len(gradWeights.Data))
	giAcc := make([]float64, len(gradInput.Data))

	for b := 0; b < batch; b++ {
		gH := make([]float64, hid)
		for t := seqLen - 1; t >= 0; t-- {
			nextGH := make([]float64, hid)
			for o := 0; o < hid; o++ {
				idx := b*seqLen*hid + t*hid + o
				hVal := math.Tanh(float64(preAct.Data[idx]))
				gPre := (gH[o] + float64(gradOutput.Data[idx])) * (1.0 - hVal*hVal)
				inRow := input.Data[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
				gwAcc[ihSz+hhSz+o] += gPre
				for i := 0; i < inSz; i++ {
					gwAcc[o*inSz+i] += nativeGradW(layer, inRow[i], gPre)
					giAcc[b*seqLen*inSz+t*inSz+i] += nativeGradX(layer, o*inSz+i, gPre)
				}
				for hp := 0; hp < hid; hp++ {
					var hPrev float64
					if t > 0 {
						hPrev = float64(preAct.Data[b*seqLen*hid+(t-1)*hid+hp])
					}
					gwAcc[ihSz+o*hid+hp] += gPre * hPrev
					nextGH[hp] += nativeGradX(layer, ihSz+o*hid+hp, gPre)
				}
			}
			gH = nextGH
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
