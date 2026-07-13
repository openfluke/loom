package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

// swiglu_native_simd.go — native-exact SwiGLU SIMD: MAC dtypes via f32 tiles; integers via AVX2/NEON int8 kernels.

func trySwiGLUForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	if usePackedTernaryCPU(layer) {
		pre, post := SwiGLUForwardPackedTernaryCPU(layer, input)
		return pre, post, true
	}
	if useSwiGLUTrueNative(layer) {
		return swigluForwardIntegerNativeSimd(layer, input)
	}
	if !swigluLayerSimdViable(layer) {
		return nil, nil, false
	}
	return swigluForwardNativeMACSimd(layer, input)
}

func trySwiGLUBackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() {
		return nil, nil, false
	}
	// Packed ternary forward uses BitNet matvec; backward gradients match MAC SIMD on cached f32 weights.
	if usePackedTernaryCPU(layer) {
		if !swigluLayerSimdViable(layer) {
			return nil, nil, false
		}
		return swigluBackwardNativeMACSimd(layer, gradOutput, input, preAct)
	}
	if useSwiGLUTrueNative(layer) {
		return swigluBackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	if !swigluLayerSimdViable(layer) {
		return nil, nil, false
	}
	return swigluBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func swigluForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		preAct, postAct = swigluForwardSimdF32(layer, input)
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
	preAct, postAct = swigluForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func swigluBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if layer.DType == DTypeFloat32 {
		gradInput, gradWeights = swigluBackwardSimdF32(layer, gradOutput, input, preAct)
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
	gradInput, gradWeights = swigluBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

func swigluForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	inSz, interSz := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inSz
	gateW, upW, downW, gateB, upB, downB := swigluWeightLayout(inSz, interSz)

	w := ws.NativeSimdI8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	cache := ensureDenseExactCache(layer, seqLen, inSz, interSz)
	preAct = NewTensor[float32](seqLen, interSz)
	postAct = NewTensor[float32](seqLen, inSz)
	interI8 := make([]int8, interSz)

	for s := 0; s < seqLen; s++ {
		inOff := s * inSz
		inRow := input.Data[inOff : inOff+inSz]
		inI8 := quantizeRowF32ToI8(inRow, scale)
		copy(cache.InputI8[inOff:inOff+inSz], inI8)

		for o := 0; o < interSz; o++ {
			gAcc := simd.DotI8Tile(w, inI8, gateW+o*inSz, 0, inSz, 0) + int32(w[gateB+o])
			uAcc := simd.DotI8Tile(w, inI8, upW+o*inSz, 0, inSz, 0) + int32(w[upB+o])
			gf := float32(clampI8(gAcc>>8)) * scale
			uf := float32(clampI8(uAcc>>8)) * scale
			v := swigluGateProduct(float64(gf), float64(uf), layer.Activation)
			interI8[o] = clampI8(int32(math.Round(float64(v) / float64(scale))))
			cache.PreI8[s*interSz+o] = interI8[o]
			cache.PostI8[s*interSz+o] = interI8[o]
			preAct.Data[s*interSz+o] = float32(interI8[o]) * scale
		}

		for o := 0; o < inSz; o++ {
			acc := simd.DotI8Tile(w, interI8, downW+o*interSz, 0, interSz, 0) + int32(w[downB+o])
			postAct.Data[inOff+o] = float32(clampI8(acc>>8)) * scale
		}
	}
	return preAct, postAct, true
}

func swigluBackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
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
	inSz, interSz := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inSz
	gateW, upW, downW, gateB, upB, downB := swigluWeightLayout(inSz, interSz)
	_ = gateB
	_ = upB
	_ = downB

	w := ws.NativeSimdI8Weights(layer.DType)
	if w == nil {
		return nil, nil, false
	}

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))

	gradInterAcc := make([]int32, seqLen*interSz)
	gradOutI8 := quantizeRowF32ToI8(gradOutput.Data, scale)
	gradInAcc := make([]int32, inSz)

	for s := 0; s < seqLen; s++ {
		interI8 := cache.PreI8[s*interSz : (s+1)*interSz]
		gOutRow := gradOutI8[s*inSz : (s+1)*inSz]
		inI8 := cache.InputI8[s*inSz : (s+1)*inSz]
		gradInterRow := gradInterAcc[s*interSz : (s+1)*interSz]

		for o := 0; o < inSz; o++ {
			g := int32(gOutRow[o])
			simd.SaxpyI8ScaleI32Acc(gradW, downW+o*interSz, interI8, g, interSz)
			simd.SaxpyI8ShiftedInputGradAcc(gradInterRow, w, downW+o*interSz, g, interSz)
		}
		for i := 0; i < interSz; i++ {
			gradInterRow[i] = int32(clampI8(gradInterRow[i]))
		}

		clear(gradInAcc)
		for o := 0; o < interSz; o++ {
			gi := gradInterRow[o]
			gf := float32(interI8[o]) * scale
			gAcc := simd.DotI8Tile(w, inI8, gateW+o*inSz, 0, inSz, 0) + int32(w[gateB+o])
			uAcc := simd.DotI8Tile(w, inI8, upW+o*inSz, 0, inSz, 0) + int32(w[upB+o])
			gateVal := float32(clampI8(gAcc>>8)) * scale
			upVal := float32(clampI8(uAcc>>8)) * scale
			sig := 1.0 / (1.0 + math.Exp(-float64(gateVal)))
			dSilu := sig * (1.0 + float64(gateVal)*(1.0-sig))
			gradGate := int32(float64(gi) * dSilu * float64(upVal) / float64(scale))
			gradUp := int32(float64(gi) * float64(gf) / float64(scale))

			simd.SaxpyI8ScaleI32Acc(gradW, gateW+o*inSz, inI8, gradGate, inSz)
			simd.SaxpyI8ScaleI32Acc(gradW, upW+o*inSz, inI8, gradUp, inSz)
			simd.SaxpyI8ShiftedInputGradAcc(gradInAcc, w, gateW+o*inSz, gradGate, inSz)
			simd.SaxpyI8ShiftedInputGradAcc(gradInAcc, w, upW+o*inSz, gradUp, inSz)
		}
		for i := 0; i < inSz; i++ {
			gradInput.Data[s*inSz+i] = float32(clampI8(gradInAcc[i])) * scale
		}
	}

	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights, true
}
