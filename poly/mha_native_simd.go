package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

// tryMHAForwardNativeSimd runs native-exact MHA forward with SIMD dot tiles.
func tryMHAForwardNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() || !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	if useMHATrueNative(layer) {
		return mhaForwardIntegerNativeSimd(layer, input)
	}
	return mhaForwardNativeMACSimd(layer, input)
}

// tryMHABackwardNativeSimd runs native-exact MHA backward with SIMD dot tiles.
func tryMHABackwardNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	if !layerUseSimdForward(layer) || !simd.SimdEnabled() || !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	if useMHATrueNative(layer) {
		return mhaBackwardIntegerNativeSimd(layer, gradOutput, input, preAct)
	}
	return mhaBackwardNativeMACSimd(layer, gradOutput, input, preAct)
}

func mhaForwardNativeMACSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	preAct, postAct = mhaForwardSimdF32WithWeights(layer, input, wData)
	return preAct, postAct, true
}

func mhaBackwardNativeMACSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	wctx := newNativeWeightCtx(layer)
	count := layer.WeightStore.WeightCount(layer.DType)
	if count <= 0 {
		return nil, nil, false
	}
	wData := wctx.materializeF32Weights(count)
	gradInput, gradWeights = mhaBackwardSimdF32WithWeights(layer, gradOutput, input, preAct, wData)
	return gradInput, gradWeights, true
}

// int8HeadDotSimd matches int8HeadDot using simd.DotTile on float32(int8) values.
func int8HeadDotSimd(q, k []int8, qOff, kOff, headDim int, qScratch, kScratch []float32) int32 {
	for d := 0; d < headDim; d++ {
		qScratch[d] = float32(q[qOff+d])
		kScratch[d] = float32(k[kOff+d])
	}
	return int32(int64(mhaSimdDot(qScratch, kScratch, headDim)) >> 8)
}

func mhaForwardIntegerNativeSimd(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32], ok bool) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return mhaForwardNativeMACSimd(layer, input)
	}

	dModel, numHeads, numKVHeads, headDim, qDim, kvDim := mhaLayerDims(layer)
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

	qwStart, kwStart, vwStart, owStart, qbStart, kbStart, vbStart, obStart := mhaWeightOffsets(layer, dModel, qDim, kvDim)
	_ = vbStart
	cache := ensureMHAExactCache(layer, lay.batch, seqLen, dModel, qDim, msl, kvDim)

	mhaPrepareKVForForward[float32](layer, lay, msl, kvDim)
	cacheK := layer.KVCacheK.(*Tensor[float32])
	cacheV := layer.KVCacheV.(*Tensor[float32])

	kvStart := layer.KVOffset
	invSqrtHead := 1.0 / math.Sqrt(float64(headDim))
	qHeadF32 := make([]float32, headDim)
	kHeadF32 := make([]float32, headDim)

	for b := 0; b < lay.batch; b++ {
		base := lay.base(b)
		seqBase := kvStart + b*seqLen
		qI8 := cache.QI8[b*seqLen*qDim : b*seqLen*qDim+seqLen*qDim]
		kI8 := cache.KI8[b*msl*kvDim : b*msl*kvDim+msl*kvDim]
		vI8 := cache.VI8[b*msl*kvDim : b*msl*kvDim+msl*kvDim]
		qkEps := mhaQKNormEpsilon(layer)
		kScratch := make([]float32, kvDim)
		vScratch := make([]float32, kvDim)

		for s := 0; s < seqLen; s++ {
			pos := seqBase + s
			inBase := lay.inIdx(b, s, 0)
			inRow := input.Data[inBase : inBase+dModel]
			inI8 := quantizeRowF32ToI8(inRow, scale)
			inOff := b*seqLen*dModel + s*dModel
			copy(cache.InputI8[inOff:inOff+dModel], inI8)
			kRow := cacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := cacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]

			qOff := b*seqLen*qDim + s*qDim
			Q := cache.QF64[qOff : qOff+qDim]
			qProj := mhaInt8ProjectRow(w, inI8, qwStart, qbStart, qDim, dModel, scale)
			copy(Q, qProj)
			kProj := mhaInt8ProjectRow(w, inI8, kwStart, kbStart, kvDim, dModel, scale)
			vProj := mhaInt8ProjectRow(w, inI8, vwStart, vbStart, kvDim, dModel, scale)
			for i := 0; i < kvDim; i++ {
				kScratch[i] = float32(kProj[i])
				vScratch[i] = float32(vProj[i])
			}
			copy(kRow, kScratch)
			copy(vRow, vScratch)

			if len(layer.QNormWeight) > 0 {
				applyPerHeadRMSNormFloat64(Q, layer.QNormWeight, numHeads, headDim, qkEps)
			}
			if len(layer.KNormWeight) > 0 {
				applyPerHeadRMSNormTensor(kRow, layer.KNormWeight, numKVHeads, headDim, qkEps)
			}

			theta := mhaRoPETheta(layer)
			half := headDim / 2
			for h := 0; h < numHeads; h++ {
				for d := 0; d < half; d++ {
					angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
					c, sVal := math.Cos(angle), math.Sin(angle)
					hOff := h*headDim + d
					v0, v1 := Q[hOff], Q[hOff+half]
					Q[hOff] = v0*c - v1*sVal
					Q[hOff+half] = v0*sVal + v1*c
				}
			}
			for h := 0; h < numKVHeads; h++ {
				for d := 0; d < half; d++ {
					angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
					c, sVal := math.Cos(angle), math.Sin(angle)
					kOff := h*headDim + d
					v0, v1 := float64(kRow[kOff]), float64(kRow[kOff+half])
					kRow[kOff] = float32(v0*c - v1*sVal)
					kRow[kOff+half] = float32(v0*sVal + v1*c)
				}
			}

			for i := 0; i < qDim; i++ {
				qI8[s*qDim+i] = clampI8(int32(math.Round(Q[i] / float64(scale))))
			}
			kBufOff := (pos % msl) * kvDim
			for i := 0; i < kvDim; i++ {
				kI8[kBufOff+i] = clampI8(int32(math.Round(float64(kRow[i]) / float64(scale))))
				vI8[kBufOff+i] = clampI8(int32(math.Round(float64(vRow[i]) / float64(scale))))
			}
		}

		headsPerKV := numHeads / numKVHeads
		attnOut := make([]float64, seqLen*qDim)
		intScores := make([]int32, seqLen)

		for s := 0; s < seqLen; s++ {
			currentTotalPos := seqBase + s
			nPos := currentTotalPos + 1
			if cap(intScores) < nPos {
				intScores = make([]int32, nPos)
			} else {
				intScores = intScores[:nPos]
			}
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for kPos := 0; kPos < nPos; kPos++ {
					kIdx := kPos % msl
					intScores[kPos] = int8HeadDotSimd(
						qI8, kI8,
						s*qDim+h*headDim, kIdx*kvDim+kvHead*headDim,
						headDim, qHeadF32, kHeadF32,
					)
				}
				probs := intAttnSoftmaxU8(intScores, invSqrtHead)
				for d := 0; d < headDim; d++ {
					var acc int32
					for kPos := 0; kPos < nPos; kPos++ {
						kIdx := kPos % msl
						acc += int32(probs[kPos]) * int32(vI8[kIdx*kvDim+kvHead*headDim+d])
					}
					attnOut[s*qDim+h*headDim+d] = float64(clampI8(acc>>8)) * float64(scale)
				}
			}
		}

		for s := 0; s < seqLen; s++ {
			attnOff := b*seqLen*qDim + s*qDim
			for i := 0; i < dModel && i < qDim; i++ {
				preAct.Data[base+s*dModel+i] = float32(attnOut[s*qDim+i])
			}
			for j := 0; j < qDim; j++ {
				cache.PreI8[attnOff+j] = clampI8(int32(math.Round(attnOut[s*qDim+j] / float64(scale))))
			}
			oProj := mhaInt8ProjectRow(w, cache.PreI8[attnOff:attnOff+qDim], owStart, obStart, dModel, qDim, scale)
			for i := 0; i < dModel; i++ {
				postAct.Data[base+s*dModel+i] = float32(oProj[i])
			}
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct, true
}

func mhaBackwardIntegerNativeSimd(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32], ok bool) {
	// Reuse scalar int8 projection backward; attention backward uses SIMD Q·K dots.
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return mhaBackwardNativeMACSimd(layer, gradOutput, input, preAct)
	}
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 || len(cache.QF64) == 0 || len(cache.QI8) == 0 {
		return mhaBackwardNativeMACSimd(layer, gradOutput, input, preAct)
	}

	dModel, numHeads, numKVHeads, headDim, qDim, kvDim := mhaLayerDims(layer)
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}
	qwStart, kwStart, vwStart, owStart, qbStart, kbStart, vbStart, obStart := mhaWeightOffsets(layer, dModel, qDim, kvDim)
	_ = obStart
	invSqrtHead := 1.0 / math.Sqrt(float64(headDim))

	lrShift := learningRateToBitShift(layer.Network.ExactNativeLR)
	gradW := make([]int32, len(w))
	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](len(w))
	giAcc := make([]int32, len(gradInput.Data))

	kvEnd := layer.KVOffset
	qHeadF32 := make([]float32, headDim)
	kHeadF32 := make([]float32, headDim)

	for b := 0; b < lay.batch; b++ {
		seqBase := kvEnd - lay.batch*seqLen + b*seqLen
		gradPre := make([]float64, seqLen*qDim)
		qI8 := cache.QI8[b*seqLen*qDim : b*seqLen*qDim+seqLen*qDim]
		kI8 := cache.KI8[b*msl*kvDim : b*msl*kvDim+msl*kvDim]
		vI8 := cache.VI8[b*msl*kvDim : b*msl*kvDim+msl*kvDim]

		for s := 0; s < seqLen; s++ {
			attnOff := b*seqLen*qDim + s*qDim
			attnI8 := cache.PreI8[attnOff : attnOff+qDim]
			for i := 0; i < dModel; i++ {
				g := gradF64ToI32(float64(gradOutput.Data[lay.outIdx(b, s, i)]), scale)
				gradW[obStart+i] += g
				int8AccumWeightGrad(gradW, w, attnI8, g, owStart+i*qDim, qDim)
				for j := 0; j < qDim; j++ {
					gradPre[s*qDim+j] += float64(clampI8((int32(w[owStart+i*qDim+j])*g)>>8)) * float64(scale)
				}
			}
		}

		gQ, gK, gV := mhaAttentionBackwardIntNativeSimd(
			layer, seqLen, msl, seqBase,
			qI8, kI8, vI8, gradPre,
			qDim, kvDim, headDim, numHeads, numKVHeads,
			invSqrtHead, scale,
			qHeadF32, kHeadF32,
		)

		for s := 0; s < seqLen; s++ {
			inOff := lay.inIdx(b, s, 0)
			inI8 := cache.InputI8[b*seqLen*dModel+s*dModel : b*seqLen*dModel+(s+1)*dModel]
			for i := 0; i < qDim; i++ {
				g := gradF64ToI32(gQ[s*qDim+i], scale)
				mhaInt8LinearBackwardRow(gradW, w, inI8, g, qwStart+i*dModel, qbStart+i, dModel, giAcc, inOff)
			}
			kIdx := (seqBase + s) % msl
			for i := 0; i < kvDim; i++ {
				dk := gradF64ToI32(gK[kIdx*kvDim+i], scale)
				dv := gradF64ToI32(gV[kIdx*kvDim+i], scale)
				mhaInt8LinearBackwardRow(gradW, w, inI8, dk, kwStart+i*dModel, kbStart+i, dModel, giAcc, inOff)
				mhaInt8LinearBackwardRow(gradW, w, inI8, dv, vwStart+i*dModel, vbStart+i, dModel, giAcc, inOff)
			}
		}
	}

	for i, acc := range giAcc {
		gradInput.Data[i] = float32(acc) * scale
	}
	applyStochasticInt8Update(w, gradW, lrShift)
	publishInt8Weights(ws, layer.DType, w)
	markLayerNativeWeightsUpdated(layer, ws, layer.DType, ws.Versions[layer.DType].([]uint8))
	return gradInput, gradWeights, true
}

func mhaAttentionBackwardIntNativeSimd(
	layer *VolumetricLayer,
	seqLen, msl, seqBase int,
	qI8, kI8, vI8 []int8,
	gradPre []float64,
	qDim, kvDim, headDim, numHeads, numKVHeads int,
	invSqrtHead float64,
	scale float32,
	qScratch, kScratch []float32,
) (gQ, gK, gV []float64) {
	headsPerKV := numHeads / numKVHeads
	attnScale := invSqrtHead
	invScale := 1.0 / float64(scale)
	macNorm := 1.0 / 256.0

	gQ = make([]float64, seqLen*qDim)
	gK = make([]float64, msl*kvDim)
	gV = make([]float64, msl*kvDim)
	intScores := make([]int32, seqLen)

	for h := 0; h < numHeads; h++ {
		kvHead := h / headsPerKV
		for qPos := 0; qPos < seqLen; qPos++ {
			nPos := seqBase + qPos + 1
			if cap(intScores) < nPos {
				intScores = make([]int32, nPos)
			} else {
				intScores = intScores[:nPos]
			}
			for kPos := 0; kPos < nPos; kPos++ {
				kIdx := kPos % msl
				intScores[kPos] = int8HeadDotSimd(
					qI8, kI8,
					qPos*qDim+h*headDim, kIdx*kvDim+kvHead*headDim,
					headDim, qScratch, kScratch,
				)
			}
			probsU8 := intAttnSoftmaxU8(intScores, invSqrtHead)
			probs := make([]float64, nPos)
			for kPos := 0; kPos < nPos; kPos++ {
				probs[kPos] = float64(probsU8[kPos]) / 255.0
			}
			for d := 0; d < headDim; d++ {
				dy := gradPre[qPos*qDim+h*headDim+d] * invScale
				var dSSum float64
				for kPos := 0; kPos < nPos; kPos++ {
					vIdx := kPos % msl
					vCode := float64(vI8[vIdx*kvDim+kvHead*headDim+d])
					gV[vIdx*kvDim+kvHead*headDim+d] += probs[kPos] * dy * float64(scale)
					dSSum += vCode * dy * probs[kPos]
				}
				for kPos := 0; kPos < nPos; kPos++ {
					kIdx := kPos % msl
					kCode := float64(kI8[kIdx*kvDim+kvHead*headDim+d])
					qCode := float64(qI8[qPos*qDim+h*headDim+d])
					vCode := float64(vI8[kIdx*kvDim+kvHead*headDim+d])
					dScore := (probs[kPos]*dy*vCode - probs[kPos]*dSSum) * attnScale * macNorm
					gQ[qPos*qDim+h*headDim+d] += dScore * kCode * float64(scale)
					gK[kIdx*kvDim+kvHead*headDim+d] += dScore * qCode * float64(scale)
				}
			}
		}
	}

	mhaRoPEBackwardGrad(layer, seqLen, msl, seqBase, gQ, gK, qDim, kvDim, headDim, numHeads, numKVHeads)
	return gQ, gK, gV
}
