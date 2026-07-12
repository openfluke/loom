package poly

import "math"

// swiglu_native.go — SwiGLU native training: int8 MAC + in-place update, or per-dot native MAC.

func useSwiGLUNativeExact(layer *VolumetricLayer) bool {
	return useLayerNativeExact(layer) && layer.Type == LayerSwiGLU
}

func useSwiGLUTrueNative(layer *VolumetricLayer) bool {
	return useSwiGLUNativeExact(layer) && IsTrueNativeDType(layer.DType)
}

// SwiGLUForwardNativeExact runs SwiGLU in storage dtype (integer-native or per-dot MAC).
func SwiGLUForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return SwiGLUForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if pre, post, simdOK := trySwiGLUForwardNativeSimd(layer, in); simdOK {
			preF, postF = pre, post
		}
	}
	if preF == nil {
		if useSwiGLUTrueNative(layer) {
			preF, postF = swigluForwardIntegerNative(layer, in)
		} else {
			preF, postF = swigluForwardNativeMAC(layer, in)
		}
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return SwiGLUForwardTiled(layer, input)
	}
	return pre, post
}

// SwiGLUBackwardNativeExact runs SwiGLU backward in storage dtype.
func SwiGLUBackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return SwiGLUBackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if layerUseSimdForward(layer) {
		if gi, gw, simdOK := trySwiGLUBackwardNativeSimd(layer, goT, in, preF); simdOK {
			giF, gwF = gi, gw
		}
	}
	if giF == nil {
		if useSwiGLUTrueNative(layer) {
			giF, gwF = swigluBackwardIntegerNative(layer, goT, in, preF)
		} else {
			giF, gwF = swigluBackwardNativeMAC(layer, goT, in, preF)
		}
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return SwiGLUBackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func swigluWeightLayout(inSz, interSz int) (gateW, upW, downW, gateB, upB, downB int) {
	wBlock := inSz * interSz
	gateW = 0
	upW = wBlock
	downW = 2 * wBlock
	gateB = 3 * wBlock
	upB = gateB + interSz
	downB = upB + interSz
	return
}

func swigluForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	inSz, interSz := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inSz
	gateW, upW, downW, gateB, upB, downB := swigluWeightLayout(inSz, interSz)

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return swigluForwardNativeMAC(layer, input)
	}

	cache := ensureDenseExactCache(layer, seqLen, inSz, interSz)
	preAct = NewTensor[float32](seqLen, interSz)
	postAct = NewTensor[float32](seqLen, inSz)

	interI8 := make([]int8, interSz)

	for s := 0; s < seqLen; s++ {
		inRow := input.Data[s*inSz : (s+1)*inSz]
		inI8 := quantizeRowF32ToI8(inRow, scale)
		copy(cache.InputI8[s*inSz:(s+1)*inSz], inI8)

		for o := 0; o < interSz; o++ {
			gAcc := int8DotRowAcc(w, inI8, gateW+o*inSz, inSz) + int32(w[gateB+o])
			uAcc := int8DotRowAcc(w, inI8, upW+o*inSz, inSz) + int32(w[upB+o])
			gf := float32(clampI8(gAcc>>8)) * scale
			uf := float32(clampI8(uAcc>>8)) * scale
			v := swigluGateProduct(float64(gf), float64(uf), layer.Activation)
			interI8[o] = clampI8(int32(math.Round(float64(v) / float64(scale))))
			cache.PreI8[s*interSz+o] = interI8[o]
			cache.PostI8[s*interSz+o] = interI8[o]
			preAct.Data[s*interSz+o] = float32(interI8[o]) * scale
		}

		for o := 0; o < inSz; o++ {
			acc := int8DotRowAcc(w, interI8, downW+o*interSz, interSz) + int32(w[downB+o])
			out := clampI8(acc >> 8)
			postAct.Data[s*inSz+o] = float32(out) * scale
		}
	}
	return preAct, postAct
}

func swigluBackwardIntegerNative(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return swigluBackwardNativeMAC(layer, gradOutput, input, preAct)
	}
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	inSz, interSz := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inSz
	gateW, upW, downW, gateB, upB, downB := swigluWeightLayout(inSz, interSz)
	_ = gateB
	_ = upB
	_ = downB

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return swigluBackwardNativeMAC(layer, gradOutput, input, preAct)
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))

	gradInterI8 := make([]int8, seqLen*interSz)
	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)

	for s := 0; s < seqLen; s++ {
		interI8 := cache.PreI8[s*interSz : (s+1)*interSz]
		gOutRow := gradOutI8[s*inSz : (s+1)*inSz]
		inI8 := cache.InputI8[s*inSz : (s+1)*inSz]

		for o := 0; o < inSz; o++ {
			g := int32(gOutRow[o])
			int8AccumWeightGrad(gradW, w, interI8, g, downW+o*interSz, interSz)
			for i := 0; i < interSz; i++ {
				gradInterI8[s*interSz+i] = clampI8(int32(gradInterI8[s*interSz+i]) + (int32(w[downW+o*interSz+i])*g)>>8)
			}
		}

		for o := 0; o < interSz; o++ {
			gi := int32(gradInterI8[s*interSz+o])
			gf := float32(interI8[o]) * scale
			gAcc := int8DotRowAcc(w, inI8, gateW+o*inSz, inSz) + int32(w[gateB+o])
			uAcc := int8DotRowAcc(w, inI8, upW+o*inSz, inSz) + int32(w[upB+o])
			gateVal := float32(clampI8(gAcc>>8)) * scale
			upVal := float32(clampI8(uAcc>>8)) * scale
			sig := 1.0 / (1.0 + math.Exp(-float64(gateVal)))
			dSilu := sig * (1.0 + float64(gateVal)*(1.0-sig))
			gradGate := int32(float64(gi) * dSilu * float64(upVal) / float64(scale))
			gradUp := int32(float64(gi) * float64(gf) / float64(scale))

			int8AccumWeightGrad(gradW, w, inI8, gradGate, gateW+o*inSz, inSz)
			int8AccumWeightGrad(gradW, w, inI8, gradUp, upW+o*inSz, inSz)
			for i := 0; i < inSz; i++ {
				gradInput.Data[s*inSz+i] += float32(clampI8((int32(w[gateW+o*inSz+i])*gradGate+(int32(w[upW+o*inSz+i])*gradUp))>>8)) * scale
			}
		}
	}

	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights
}

func swigluForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	inSz, interSz := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inSz
	gateW, upW, downW, gateB, upB, downB := swigluWeightLayout(inSz, interSz)

	preAct = NewTensor[float32](seqLen, interSz)
	postAct = NewTensor[float32](seqLen, inSz)
	inter := make([]float32, interSz)

	for s := 0; s < seqLen; s++ {
		inRow := input.Data[s*inSz : (s+1)*inSz]
		for o := 0; o < interSz; o++ {
			g := nativeDotRow(layer, inRow, gateW+o*inSz, inSz) + nativeBiasAt(layer, gateB+o)
			u := nativeDotRow(layer, inRow, upW+o*inSz, inSz) + nativeBiasAt(layer, upB+o)
			v := float32(swigluGateProduct(float64(g), float64(u), layer.Activation))
			inter[o] = v
			preAct.Data[s*interSz+o] = v
		}
		for o := 0; o < inSz; o++ {
			sum := nativeDotRow(layer, inter, downW+o*interSz, interSz) + nativeBiasAt(layer, downB+o)
			postAct.Data[s*inSz+o] = sum
		}
	}
	return preAct, postAct
}

func swigluBackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	inSz, interSz := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inSz
	gateW, upW, downW, gateB, upB, downB := swigluWeightLayout(inSz, interSz)
	_ = gateB
	_ = upB
	_ = downB

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](layer.WeightStore.WeightCount(layer.DType))
	gwAcc := make([]float64, len(gradWeights.Data))
	giAcc := make([]float64, len(gradInput.Data))
	gradInter := make([]float64, seqLen*interSz)

	for s := 0; s < seqLen; s++ {
		inRow := input.Data[s*inSz : (s+1)*inSz]
		inter := preAct.Data[s*interSz : (s+1)*interSz]
		gOut := gradOutput.Data[s*inSz : (s+1)*inSz]

		for o := 0; o < inSz; o++ {
			g := float64(gOut[o])
			for i := 0; i < interSz; i++ {
				wIdx := downW + o*interSz + i
				gwAcc[wIdx] += nativeGradW(layer, inter[i], g)
				gradInter[s*interSz+i] += nativeGradX(layer, wIdx, g)
			}
		}

		for o := 0; o < interSz; o++ {
			gi := gradInter[s*interSz+o]
			g := nativeDotRow(layer, inRow, gateW+o*inSz, inSz) + nativeBiasAt(layer, gateB+o)
			u := nativeDotRow(layer, inRow, upW+o*inSz, inSz) + nativeBiasAt(layer, upB+o)
			sig := 1.0 / (1.0 + math.Exp(-float64(g)))
			dSilu := sig * (1.0 + float64(g)*(1.0-sig))
			gradGate := gi * dSilu * float64(u)
			gradUp := gi * float64(inter[o])
			for i := 0; i < inSz; i++ {
				gwAcc[gateW+o*inSz+i] += nativeGradW(layer, inRow[i], gradGate)
				gwAcc[upW+o*inSz+i] += nativeGradW(layer, inRow[i], gradUp)
				giAcc[s*inSz+i] += nativeGradX(layer, gateW+o*inSz+i, gradGate) + nativeGradX(layer, upW+o*inSz+i, gradUp)
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
