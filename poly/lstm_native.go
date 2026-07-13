package poly

import "math"

// lstm_native.go — LSTM integer-native + per-dot native MAC training.

func useLSTMNativeExact(layer *VolumetricLayer) bool {
	return useLayerNativeExact(layer) && layer.Type == LayerLSTM
}

func useLSTMTrueNative(layer *VolumetricLayer) bool {
	return useLSTMNativeExact(layer) && IsTrueNativeDType(layer.DType)
}

func LSTMForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return LSTMForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if pre, post, simdOK := tryLSTMForwardNativeSimd(layer, in); simdOK {
			preF, postF = pre, post
		}
	}
	if preF == nil {
		if useLSTMTrueNative(layer) {
			preF, postF = lstmForwardIntegerNative(layer, in)
		} else {
			preF, postF = lstmForwardNativeMAC(layer, in)
		}
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return LSTMForwardTiled(layer, input)
	}
	return pre, post
}

func LSTMBackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return LSTMBackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, gw, simdOK := tryLSTMBackwardNativeSimd(layer, goT, in, preF); simdOK {
			giF, gwF = gi, gw
		}
	}
	if giF == nil {
		if useLSTMTrueNative(layer) {
			giF, gwF = lstmBackwardIntegerNative(layer, goT, in, preF)
		} else {
			giF, gwF = lstmBackwardNativeMAC(layer, goT, in, preF)
		}
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return LSTMBackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func lstmGateLayout(hid, inSz int) (ih, hh, b, gateSz int) {
	ih = hid * inSz
	hh = hid * hid
	b = hid
	gateSz = ih + hh + b
	return
}

func lstmGateWeights(w []int8, gateSz, gate int) []int8 {
	return w[gate*gateSz : (gate+1)*gateSz]
}

func lstmForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return lstmForwardNativeMAC(layer, input)
	}

	cache := ensureDenseExactCache(layer, batch*seqLen, inSz, hid)
	preAct = NewTensor[float32](batch, seqLen, 5*hid)
	postAct = NewTensor[float32](batch, seqLen, hid)
	hPrev := make([]int8, hid)
	cPrev := make([]float64, hid)

	wI := lstmGateWeights(w, gateSz, 0)
	wF := lstmGateWeights(w, gateSz, 1)
	wG := lstmGateWeights(w, gateSz, 2)
	wO := lstmGateWeights(w, gateSz, 3)

	for bIdx := 0; bIdx < batch; bIdx++ {
		for i := range hPrev {
			hPrev[i] = 0
			cPrev[i] = 0
		}
		for t := 0; t < seqLen; t++ {
			inRow := input.Data[bIdx*seqLen*inSz+t*inSz : bIdx*seqLen*inSz+(t+1)*inSz]
			inI8 := quantizeRowF32ToI8(inRow, scale)
			copy(cache.InputI8[(bIdx*seqLen+t)*inSz:(bIdx*seqLen+t+1)*inSz], inI8)
			pBase := bIdx*seqLen*5*hid + t*5*hid

			for o := 0; o < hid; o++ {
				gates := [4]float64{}
				gateW := [4][]int8{wI, wF, wG, wO}
				for g := 0; g < 4; g++ {
					acc := int32(gateW[g][ih+hh+o])
					acc += int8DotRowAcc(gateW[g], inI8, o*inSz, inSz)
					for i := 0; i < hid; i++ {
						acc += int32(gateW[g][ih+o*hid+i]) * int32(hPrev[i])
					}
					gates[g] = float64(clampI8(acc>>8)) * float64(scale)
					preAct.Data[pBase+o+hid*g] = float32(gates[g])
				}
				iG := 1.0 / (1.0 + math.Exp(-gates[0]))
				fG := 1.0 / (1.0 + math.Exp(-gates[1]))
				gG := math.Tanh(gates[2])
				oG := 1.0 / (1.0 + math.Exp(-gates[3]))
				cCur := fG*cPrev[o] + iG*gG
				hOut := oG * math.Tanh(cCur)
				cPrev[o] = cCur
				hPrev[o] = clampI8(int32(math.Round(hOut / float64(scale))))
				cache.PostI8[(bIdx*seqLen+t)*hid+o] = hPrev[o]
				postAct.Data[bIdx*seqLen*hid+t*hid+o] = float32(hPrev[o]) * scale
				preAct.Data[pBase+4*hid+o] = float32(cCur)
			}
		}
	}
	_ = cache
	return preAct, postAct
}

func lstmForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)
	preAct = NewTensor[float32](batch, seqLen, 5*hid)
	postAct = NewTensor[float32](batch, seqLen, hid)
	hPrev := make([]float32, hid)
	cPrev := make([]float64, hid)
	gateOff := [4]int{0, gateSz, 2 * gateSz, 3 * gateSz}

	for b := 0; b < batch; b++ {
		for i := range hPrev {
			hPrev[i] = 0
			cPrev[i] = 0
		}
		for t := 0; t < seqLen; t++ {
			inRow := input.Data[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			pBase := b*seqLen*5*hid + t*5*hid
			for o := 0; o < hid; o++ {
				sums := [4]float64{}
				for g := 0; g < 4; g++ {
					base := gateOff[g]
					sums[g] = float64(nativeBiasAt(layer, base+ih+hh+o))
					sums[g] += float64(nativeDotRow(layer, inRow, base+o*inSz, inSz))
					for i := 0; i < hid; i++ {
						sums[g] += float64(hPrev[i]) * float64(nativeWeightValueF32(layer.WeightStore, layer.DType, base+ih+o*hid+i))
					}
					preAct.Data[pBase+o+hid*g] = float32(sums[g])
				}
				iA := 1.0 / (1.0 + math.Exp(-sums[0]))
				fA := 1.0 / (1.0 + math.Exp(-sums[1]))
				cCur := iA*math.Tanh(sums[2]) + fA*cPrev[o]
				oA := 1.0 / (1.0 + math.Exp(-sums[3]))
				hOut := float32(oA * math.Tanh(cCur))
				cPrev[o] = cCur
				hPrev[o] = hOut
				postAct.Data[b*seqLen*hid+t*hid+o] = hOut
				preAct.Data[pBase+4*hid+o] = float32(cCur)
			}
		}
	}
	return preAct, postAct
}

func lstmBackwardIntegerNative(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return lstmBackwardNativeMAC(layer, gradOutput, input, preAct)
	}
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return lstmBackwardNativeMAC(layer, gradOutput, input, preAct)
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))

	wI := lstmGateWeights(w, gateSz, 0)
	wF := lstmGateWeights(w, gateSz, 1)
	wG := lstmGateWeights(w, gateSz, 2)
	wO := lstmGateWeights(w, gateSz, 3)
	gateW := [4][]int8{wI, wF, wG, wO}
	gateBase := [4]int{0, gateSz, 2 * gateSz, 3 * gateSz}

	for b := 0; b < batch; b++ {
		gradH := make([]float64, hid)
		gradC := make([]float64, hid)
		for t := seqLen - 1; t >= 0; t-- {
			nextGradH := make([]float64, hid)
			nextGradC := make([]float64, hid)
			pIdx := b*seqLen*5*hid + t*5*hid
			deltas := make([]float64, 4*hid)

			for o := 0; o < hid; o++ {
				dh := gradH[o] + float64(gradOutput.Data[b*seqLen*hid+t*hid+o])
				iS := float64(preAct.Data[pIdx+o])
				fS := float64(preAct.Data[pIdx+hid+o])
				gS := float64(preAct.Data[pIdx+2*hid+o])
				oS := float64(preAct.Data[pIdx+3*hid+o])
				cC := float64(preAct.Data[pIdx+4*hid+o])
				iG := 1.0 / (1.0 + math.Exp(-iS))
				fG := 1.0 / (1.0 + math.Exp(-fS))
				oG := 1.0 / (1.0 + math.Exp(-oS))
				gG, cT := math.Tanh(gS), math.Tanh(cC)
				cP := 0.0
				if t > 0 {
					cP = float64(preAct.Data[pIdx-5*hid+4*hid+o])
				}
				dc := gradC[o] + dh*oG*(1.0-cT*cT)
				deltas[o] = dc * gG * iG * (1.0 - iG)
				deltas[hid+o] = dc * cP * fG * (1.0 - fG)
				deltas[2*hid+o] = dc * iG * (1.0 - gG*gG)
				deltas[3*hid+o] = dh * cT * oG * (1.0 - oG)
				nextGradC[o] = dc * fG
			}

			inI8 := cache.InputI8[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			for g := 0; g < 4; g++ {
				base := gateBase[g]
				for o := 0; o < hid; o++ {
					d := int32(deltas[g*hid+o] / float64(scale))
					gradW[base+ih+hh+o] += d
					int8AccumWeightGrad(gradW, gateW[g], inI8, d, o*inSz, inSz)
					for i := 0; i < inSz; i++ {
						gradInput.Data[b*seqLen*inSz+t*inSz+i] += float32(clampI8((int32(gateW[g][o*inSz+i])*d)>>8)) * scale
					}
					for hp := 0; hp < hid; hp++ {
						var hPrev int8
						if t > 0 {
							hPrev = cache.PostI8[b*seqLen*hid+(t-1)*hid+hp]
						}
						gradW[base+ih+o*hid+hp] += int32(hPrev) * d
						nextGradH[hp] += float64(clampI8((int32(gateW[g][ih+o*hid+hp])*d)>>8)) * float64(scale)
					}
				}
			}
			gradH, gradC = nextGradH, nextGradC
		}
	}

	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights
}

func lstmBackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)
	wCount := layer.WeightStore.WeightCount(layer.DType)
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](wCount)
	gwAcc := make([]float64, wCount)
	giAcc := make([]float64, len(gradInput.Data))
	gateBase := [4]int{0, gateSz, 2 * gateSz, 3 * gateSz}

	for b := 0; b < batch; b++ {
		gradH := make([]float64, hid)
		gradC := make([]float64, hid)
		for t := seqLen - 1; t >= 0; t-- {
			nextGradH := make([]float64, hid)
			nextGradC := make([]float64, hid)
			pIdx := b*seqLen*5*hid + t*5*hid
			deltas := make([]float64, 4*hid)

			for o := 0; o < hid; o++ {
				dh := gradH[o] + float64(gradOutput.Data[b*seqLen*hid+t*hid+o])
				iS := float64(preAct.Data[pIdx+o])
				fS := float64(preAct.Data[pIdx+hid+o])
				gS := float64(preAct.Data[pIdx+2*hid+o])
				oS := float64(preAct.Data[pIdx+3*hid+o])
				cC := float64(preAct.Data[pIdx+4*hid+o])
				iG := 1.0 / (1.0 + math.Exp(-iS))
				fG := 1.0 / (1.0 + math.Exp(-fS))
				oG := 1.0 / (1.0 + math.Exp(-oS))
				gG, cT := math.Tanh(gS), math.Tanh(cC)
				cP := 0.0
				if t > 0 {
					cP = float64(preAct.Data[pIdx-5*hid+4*hid+o])
				}
				dc := gradC[o] + dh*oG*(1.0-cT*cT)
				deltas[o] = dc * gG * iG * (1.0 - iG)
				deltas[hid+o] = dc * cP * fG * (1.0 - fG)
				deltas[2*hid+o] = dc * iG * (1.0 - gG*gG)
				deltas[3*hid+o] = dh * cT * oG * (1.0 - oG)
				nextGradC[o] = dc * fG
			}

			inRow := input.Data[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			for g := 0; g < 4; g++ {
				base := gateBase[g]
				for o := 0; o < hid; o++ {
					d := deltas[g*hid+o]
					gwAcc[base+ih+hh+o] += d
					for i := 0; i < inSz; i++ {
						wIdx := base + o*inSz + i
						gwAcc[wIdx] += nativeGradW(layer, inRow[i], d)
						giAcc[b*seqLen*inSz+t*inSz+i] += nativeGradX(layer, wIdx, d)
					}
					for hp := 0; hp < hid; hp++ {
						wIdx := base + ih + o*hid + hp
						var hPrev float64
						if t > 0 {
							hPrev = float64(preAct.Data[b*seqLen*hid+(t-1)*hid+hp])
						}
						gwAcc[wIdx] += d * hPrev
						nextGradH[hp] += nativeGradX(layer, wIdx, d)
					}
				}
			}
			gradH, gradC = nextGradH, nextGradC
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
