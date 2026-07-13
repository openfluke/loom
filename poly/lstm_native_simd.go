package poly

import (
	"math"
	"math/rand"

	"github.com/openfluke/loom/poly/simd"
)

// lstm_native_simd.go — native-exact LSTM SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8/u8 kernels.

func lstmTrueNativeUsesU8(dtype DType) bool {
	switch dtype {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		return true
	default:
		return false
	}
}

func tryLSTMForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useLSTMTrueNative(layer) {
		if lstmTrueNativeUsesU8(layer.DType) {
			return lstmForwardUIntegerNativeSimd(layer, input)
		}
		return lstmForwardIntegerNativeSimd(layer, input)
	}
	return lstmForwardNativeMACSimd(layer, input)
}

func tryLSTMBackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useLSTMTrueNative(layer) {
		if lstmTrueNativeUsesU8(layer.DType) {
			return lstmBackwardUIntegerNativeSimd(layer, gradOutput, input, preAct)
		}
		return lstmBackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return lstmBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func lstmForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		preAct, postAct = lstmForwardSimdF32(layer, input)
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
	preAct, postAct = lstmForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func lstmBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		gradInput, gradWeights = lstmBackwardSimdF32(layer, gradOutput, input, preAct)
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
	gradInput, gradWeights = lstmBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func lstmForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := ws.NativeSimdI8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)

	cache := ensureDenseExactCache(layer, batch*seqLen, inSz, hid)
	preAct = NewTensor[float32](batch, seqLen, 5*hid)
	postAct = NewTensor[float32](batch, seqLen, hid)
	hPrev := make([]int8, hid)
	cPrev := make([]float64, hid)

	wI := lstmGateWeights(w, gateSz, 0)
	wF := lstmGateWeights(w, gateSz, 1)
	wG := lstmGateWeights(w, gateSz, 2)
	wO := lstmGateWeights(w, gateSz, 3)
	gateW := [4][]int8{wI, wF, wG, wO}

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
				for g := 0; g < 4; g++ {
					acc := int32(gateW[g][ih+hh+o])
					acc += simd.DotI8Tile(gateW[g], inI8, o*inSz, 0, inSz, 0)
					acc += simd.DotI8Tile(gateW[g], hPrev, ih+o*hid, 0, hid, 0)
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
	return preAct, postAct, true
}

func lstmBackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return nil, nil, false
	}
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)

	w := ws.NativeSimdI8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))
	dxRow := make([]int32, inSz)

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
			giOff := b*seqLen*inSz + t*inSz
			for g := 0; g < 4; g++ {
				base := gateBase[g]
				for o := 0; o < hid; o++ {
					d := int32(deltas[g*hid+o] / float64(scale))
					gradW[base+ih+hh+o] += d
					simd.SaxpyI8ScaleI32Acc(gradW, base+o*inSz, inI8, d, inSz)
					for i := range dxRow {
						dxRow[i] = 0
					}
					simd.SaxpyI8ShiftedInputGradAcc(dxRow, gateW[g], o*inSz, d, inSz)
					for i := 0; i < inSz; i++ {
						gradInput.Data[giOff+i] += float32(clampI8(dxRow[i])) * scale
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

	applyStochasticNativeI8Update(layer.DType, w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights, true
}

func lstmForwardUIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := ws.NativeSimdU8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}
	wI8 := ws.NativeSimdI8Weights(layer.DType)

	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)

	cache := ensureDenseExactCache(layer, batch*seqLen, inSz, hid)
	preAct = NewTensor[float32](batch, seqLen, 5*hid)
	postAct = NewTensor[float32](batch, seqLen, hid)
	hPrev := make([]int8, hid)
	cPrev := make([]float64, hid)

	for bIdx := 0; bIdx < batch; bIdx++ {
		for i := range hPrev {
			hPrev[i] = 0
			cPrev[i] = 0
		}
		for t := 0; t < seqLen; t++ {
			inRow := input.Data[bIdx*seqLen*inSz+t*inSz : bIdx*seqLen*inSz+(t+1)*inSz]
			inI8 := quantizeRowF32ToI8(inRow, scale)
			inOff := (bIdx*seqLen + t) * inSz
			for i, c := range inI8 {
				cache.InputI8[inOff+i] = c
				cache.InputU8[inOff+i] = uint8(clampU8(int32(c)))
			}
			inU8 := cache.InputU8[inOff : inOff+inSz]
			pBase := bIdx*seqLen*5*hid + t*5*hid

			for o := 0; o < hid; o++ {
				gates := [4]float64{}
				for g := 0; g < 4; g++ {
					gwU8 := lstmGateWeightsU8(w, gateSz, g)
					acc := int32(gwU8[ih+hh+o])
					acc = simd.DotU8Tile(gwU8, inU8, o*inSz, 0, inSz, acc)
					if wI8 != nil {
						gwI8 := lstmGateWeights(wI8, gateSz, g)
						acc += simd.DotI8Tile(gwI8, hPrev, ih+o*hid, 0, hid, 0)
					}
					gates[g] = float64(clampI8(acc >> 8)) * float64(scale)
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
	return preAct, postAct, true
}

func lstmBackwardUIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 {
		return nil, nil, false
	}
	ws := layer.WeightStore
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ih, hh, _, gateSz := lstmGateLayout(hid, inSz)

	w := ws.NativeSimdU8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))
	giAcc := make([]int32, len(gradInput.Data))
	dxRow := make([]int32, inSz)
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

			inOff := (b*seqLen + t) * inSz
			inU8 := cache.InputU8[inOff : inOff+inSz]
			giOff := b*seqLen*inSz + t*inSz
			for g := 0; g < 4; g++ {
				base := gateBase[g]
				gwU8 := lstmGateWeightsU8(w, gateSz, g)
				for o := 0; o < hid; o++ {
					d := int32(deltas[g*hid+o] / float64(scale))
					gradW[base+ih+hh+o] += d
					simd.SaxpyU8ScaleI32Acc(gradW, base+o*inSz, inU8, d, inSz)
					for i := range dxRow {
						dxRow[i] = 0
					}
					simd.SaxpyU8ShiftedInputGradAcc(dxRow, gwU8, o*inSz, d, inSz)
					for i := 0; i < inSz; i++ {
						giAcc[giOff+i] += dxRow[i]
					}
					for hp := 0; hp < hid; hp++ {
						var hPrev int8
						if t > 0 {
							hPrev = cache.PostI8[b*seqLen*hid+(t-1)*hid+hp]
						}
						gradW[base+ih+o*hid+hp] += int32(hPrev) * d
						nextGradH[hp] += float64(clampI8((int32(gwU8[ih+o*hid+hp])*d)>>8)) * float64(scale)
					}
				}
			}
			gradH, gradC = nextGradH, nextGradC
		}
	}

	mask := int32((1 << lrShift) - 1)
	for i := range w {
		scaledGrad := gradW[i] >> lrShift
		if (gradW[i] & mask) > rand.Int31n(1<<lrShift) {
			scaledGrad++
		}
		w[i] = clampNativeU8Weight(layer.DType, clampU8(int32(w[i])-scaledGrad))
	}
	ws.Versions[layer.DType] = w
	ws.Master = nil
	cache.WeightsUpdated = true
	ws.GPUWeights = make(map[DType]any)
	if ws.CPUPacked != nil {
		delete(ws.CPUPacked, layer.DType)
	}
	ws.invalidateNativeSimdCache(layer.DType)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))

	giU8 := make([]uint8, len(giAcc))
	for i, acc := range giAcc {
		giU8[i] = clampU8(acc)
	}
	dequantU8Row(giU8, scale, gradInput.Data)
	return gradInput, gradWeights, true
}
