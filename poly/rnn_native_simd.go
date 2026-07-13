package poly

import (
	"math"
	"math/rand"

	"github.com/openfluke/loom/poly/simd"
)

// rnn_native_simd.go — native-exact RNN SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8/u8 kernels.

func rnnTrueNativeUsesU8(dtype DType) bool {
	switch dtype {
	case DTypeUint8, DTypeUint4, DTypeUint2:
		return true
	default:
		return false
	}
}

func tryRNNForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useRNNTrueNative(layer) {
		if rnnTrueNativeUsesU8(layer.DType) {
			return rnnForwardUIntegerNativeSimd(layer, input)
		}
		return rnnForwardIntegerNativeSimd(layer, input)
	}
	return rnnForwardNativeMACSimd(layer, input)
}

func tryRNNBackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if useRNNTrueNative(layer) {
		if rnnTrueNativeUsesU8(layer.DType) {
			return rnnBackwardUIntegerNativeSimd(layer, gradOutput, input, preAct)
		}
		return rnnBackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return rnnBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func rnnForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		preAct, postAct = rnnForwardSimdF32(layer, input)
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
	preAct, postAct = rnnForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func rnnBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		gradInput, gradWeights = rnnBackwardSimdF32(layer, gradOutput, input, preAct)
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
	gradInput, gradWeights = rnnBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func rnnForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
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

	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ihSz, hhSz := hid*inSz, hid*hid

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
				acc := int32(w[ihSz+hhSz+o])
				acc += simd.DotI8Tile(w, inI8, o*inSz, 0, inSz, 0)
				acc += simd.DotI8Tile(w, hPrev, ihSz+o*hid, 0, hid, 0)
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
	return preAct, postAct, true
}

func rnnBackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
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
	ihSz, hhSz := hid*inSz, hid*hid

	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return nil, nil, false
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))
	giAcc := make([]float32, len(gradInput.Data))
	dxRow := make([]int32, inSz)

	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)

	for b := 0; b < batch; b++ {
		gH := make([]int32, hid)
		for t := seqLen - 1; t >= 0; t-- {
			nextGH := make([]int32, hid)
			inI8 := cache.InputI8[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			giOff := b*seqLen*inSz + t*inSz

			for o := 0; o < hid; o++ {
				idx := b*seqLen*hid + t*hid + o
				pre := float64(cache.PreI8[idx])
				hVal := math.Tanh(pre * float64(scale))
				gPre := (float64(gH[o]) + float64(gradOutI8[idx])) * (1.0 - hVal*hVal)
				g := int32(gPre / float64(scale))

				gradW[ihSz+hhSz+o] += g
				simd.SaxpyI8ScaleI32Acc(gradW, o*inSz, inI8, g, inSz)
				for i := range dxRow {
					dxRow[i] = 0
				}
				simd.SaxpyI8ShiftedInputGradAcc(dxRow, w, o*inSz, g, inSz)
				for i := 0; i < inSz; i++ {
					giAcc[giOff+i] += float32(clampI8(dxRow[i])) * scale
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

	applyStochasticNativeI8Update(layer.DType, w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))

	for i := range gradInput.Data {
		gradInput.Data[i] = giAcc[i]
	}
	return gradInput, gradWeights, true
}

func rnnForwardUIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
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

	batch, inSz, hid, seqLen := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	ihSz, hhSz := hid*inSz, hid*hid

	cache := ensureDenseExactCache(layer, batch*seqLen, inSz, hid)
	preAct = NewTensor[float32](batch, seqLen, hid)
	postAct = NewTensor[float32](batch, seqLen, hid)
	hPrev := make([]int8, hid)
	wI8 := ws.NativeSimdI8Weights(layer.DType)

	for b := 0; b < batch; b++ {
		for h := range hPrev {
			hPrev[h] = 0
		}
		for t := 0; t < seqLen; t++ {
			inRow := input.Data[b*seqLen*inSz+t*inSz : b*seqLen*inSz+(t+1)*inSz]
			inI8 := quantizeRowF32ToI8(inRow, scale)
			inOff := (b*seqLen + t) * inSz
			for i, c := range inI8 {
				cache.InputI8[inOff+i] = c
				cache.InputU8[inOff+i] = uint8(clampU8(int32(c)))
			}
			inU8 := cache.InputU8[inOff : inOff+inSz]

			for o := 0; o < hid; o++ {
				acc := int32(w[ihSz+hhSz+o])
				acc = simd.DotU8Tile(w, inU8, o*inSz, 0, inSz, acc)
				if wI8 != nil {
					acc += simd.DotI8Tile(wI8, hPrev, ihSz+o*hid, 0, hid, 0)
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
	return preAct, postAct, true
}

func rnnBackwardUIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
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
	ihSz, hhSz := hid*inSz, hid*hid

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

	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)
	gradOutU8 := make([]uint8, len(gradOutI8))
	for i, c := range gradOutI8 {
		gradOutU8[i] = uint8(clampU8(int32(c)))
	}

	for b := 0; b < batch; b++ {
		gH := make([]int32, hid)
		for t := seqLen - 1; t >= 0; t-- {
			nextGH := make([]int32, hid)
			inOff := (b*seqLen + t) * inSz
			inU8 := cache.InputU8[inOff : inOff+inSz]
			giOff := b*seqLen*inSz + t*inSz

			for o := 0; o < hid; o++ {
				idx := b*seqLen*hid + t*hid + o
				pre := float64(cache.PreI8[idx])
				hVal := math.Tanh(pre * float64(scale))
				gPre := (float64(gH[o]) + float64(gradOutU8[idx])) * (1.0 - hVal*hVal)
				g := int32(gPre / float64(scale))

				gradW[ihSz+hhSz+o] += g
				simd.SaxpyU8ScaleI32Acc(gradW, o*inSz, inU8, g, inSz)
				for i := range dxRow {
					dxRow[i] = 0
				}
				simd.SaxpyU8ShiftedInputGradAcc(dxRow, w, o*inSz, g, inSz)
				for i := 0; i < inSz; i++ {
					giAcc[giOff+i] += dxRow[i]
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
