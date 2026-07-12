package poly

import "math"

// mha_native.go — MHA native training:
//   - Integer dtypes: int8 Q/K/V/O projections, int8 Q·K + uint8 softmax attention, stochastic int8 update
//   - Other dtypes: cached native weight slices + dtype-aware MAC (no per-element GetNative)

func useMHANativeExact(layer *VolumetricLayer) bool {
	return useLayerNativeExact(layer) && layer.Type == LayerMultiHeadAttention
}

func useMHATrueNative(layer *VolumetricLayer) bool {
	return useMHANativeExact(layer) && IsTrueNativeDType(layer.DType)
}

func mhaWeightOffsets(layer *VolumetricLayer, dModel, qDim, kvDim int) (qw, kw, vw, ow, qb, kb, vb, ob int) {
	qw = 0
	kw = qw + qDim*dModel
	vw = kw + kvDim*dModel
	ow = vw + kvDim*dModel
	qb = ow + dModel*qDim
	kb = qb + qDim
	vb = kb + kvDim
	ob = vb + kvDim
	return
}

func mhaNativeProjectRow(ctx *nativeWeightCtx, inRow []float32, wStart, bStart, outDim, inDim int, out []float64) {
	mhaProjectRowCtx(ctx, inRow, wStart, bStart, outDim, inDim, out)
}

func mhaInt8ProjectRow(w []int8, inI8 []int8, wStart, bStart, outDim, inDim int, scale float32) []float64 {
	out := make([]float64, outDim)
	for o := 0; o < outDim; o++ {
		acc := int32(w[bStart+o])
		acc += int8DotRowAcc(w, inI8, wStart+o*inDim, inDim)
		out[o] = float64(clampI8(acc>>8)) * float64(scale)
	}
	return out
}

func ensureMHAExactCache(layer *VolumetricLayer, batch, seqLen, dModel, qDim, msl, kvDim int) *DenseExactCache {
	if layer.ExactDense == nil {
		layer.ExactDense = &DenseExactCache{}
	}
	c := layer.ExactDense
	needIn := batch * seqLen * dModel
	needAttn := batch * seqLen * qDim
	needKV := batch * msl * kvDim
	growI8 := func(buf *[]int8, need int) {
		if cap(*buf) < need {
			*buf = make([]int8, need)
		} else {
			*buf = (*buf)[:need]
		}
	}
	growI8(&c.InputI8, needIn)
	growI8(&c.PreI8, needAttn)
	growI8(&c.QI8, needAttn)
	growI8(&c.KI8, needKV)
	growI8(&c.VI8, needKV)
	if cap(c.QF64) < needAttn {
		c.QF64 = make([]float64, needAttn)
	} else {
		c.QF64 = c.QF64[:needAttn]
	}
	return c
}

func int8HeadDot(q, k []int8, qOff, kOff, headDim int) int32 {
	var acc int32
	for d := 0; d < headDim; d++ {
		acc += int32(q[qOff+d]) * int32(k[kOff+d])
	}
	return acc >> 8
}

func intAttnSoftmaxU8(scores []int32, invSqrtHead float64) []uint8 {
	n := len(scores)
	probs := make([]uint8, n)
	if n == 0 {
		return probs
	}
	maxS := scores[0]
	for _, s := range scores[1:] {
		if s > maxS {
			maxS = s
		}
	}
	var expSum float64
	exps := make([]float64, n)
	for i, s := range scores {
		exps[i] = math.Exp(float64(s-maxS) * invSqrtHead / 256.0)
		expSum += exps[i]
	}
	if expSum == 0 {
		probs[0] = 255
		return probs
	}
	var sum uint16
	for i, e := range exps {
		p := uint8(math.Round(e / expSum * 255))
		probs[i] = p
		sum += uint16(p)
	}
	if sum != 255 {
		diff := int(255) - int(sum)
		probs[n-1] = uint8(int(probs[n-1]) + diff)
	}
	return probs
}

// mhaRoPEBackwardGrad applies inverse RoPE to Q/K gradients.
func mhaRoPEBackwardGrad(layer *VolumetricLayer, seqLen, msl, seqBase int, gQ, gK []float64, qDim, kvDim, headDim, numHeads, numKVHeads int) {
	half := headDim / 2
	theta := mhaRoPETheta(layer)
	for s := 0; s < seqLen; s++ {
		pos := seqBase + s
		for h := 0; h < numHeads; h++ {
			for d := 0; d < half; d++ {
				angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
				c, sVal := math.Cos(angle), math.Sin(angle)
				qOff := s*qDim + h*headDim + d
				v0, v1 := gQ[qOff], gQ[qOff+half]
				gQ[qOff] = v0*c + v1*sVal
				gQ[qOff+half] = -v0*sVal + v1*c
			}
		}
		kIdx := pos % msl
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < half; d++ {
				angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
				c, sVal := math.Cos(angle), math.Sin(angle)
				kOff := kIdx*kvDim + h*headDim + d
				v0, v1 := gK[kOff], gK[kOff+half]
				gK[kOff] = v0*c + v1*sVal
				gK[kOff+half] = -v0*sVal + v1*c
			}
		}
	}
}

// mhaAttentionBackwardFloat computes gQ/gK/gV from gradPre through softmax attention (float).
func mhaAttentionBackwardFloat(
	layer *VolumetricLayer,
	lay mhaLayout,
	seqLen, msl int,
	seqBase int,
	Q []float64,
	gradPre []float64,
	cacheK, cacheV *Tensor[float32],
) (gQ, gK, gV []float64) {
	dModel, numHeads, numKVHeads, headDim, qDim, kvDim := mhaLayerDims(layer)
	headsPerKV := numHeads / numKVHeads
	attnScale := 1.0 / math.Sqrt(float64(headDim))
	gQ = make([]float64, seqLen*qDim)
	gK = make([]float64, msl*kvDim)
	gV = make([]float64, msl*kvDim)

	for h := 0; h < numHeads; h++ {
		kvHead := h / headsPerKV
		for qPos := 0; qPos < seqLen; qPos++ {
			currentTotalPos := seqBase + qPos
			nPos := currentTotalPos + 1
			scores := make([]float64, nPos)
			probs := make([]float64, nPos)
			maxScore := float64(-1e9)
			for kPos := 0; kPos < nPos; kPos++ {
				kIdx := kPos % msl
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += Q[qPos*qDim+h*headDim+d] * float64(cacheK.Data[kIdx*kvDim+kvHead*headDim+d])
				}
				score := dot * attnScale
				scores[kPos] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var expSum float64
			for kPos := 0; kPos < nPos; kPos++ {
				probs[kPos] = math.Exp(scores[kPos] - maxScore)
				expSum += probs[kPos]
			}
			for kPos := 0; kPos < nPos; kPos++ {
				probs[kPos] /= expSum
			}
			for d := 0; d < headDim; d++ {
				dy := gradPre[qPos*qDim+h*headDim+d]
				var dSSum float64
				for kPos := 0; kPos < nPos; kPos++ {
					vIdx := kPos % msl
					gV[vIdx*kvDim+kvHead*headDim+d] += probs[kPos] * dy
					dSSum += float64(cacheV.Data[vIdx*kvDim+kvHead*headDim+d]) * dy * probs[kPos]
				}
				for kPos := 0; kPos < nPos; kPos++ {
					kIdx := kPos % msl
					dScore := (probs[kPos]*dy*float64(cacheV.Data[kIdx*kvDim+kvHead*headDim+d]) - probs[kPos]*dSSum) * attnScale
					gQ[qPos*qDim+h*headDim+d] += dScore * float64(cacheK.Data[kIdx*kvDim+kvHead*headDim+d])
					gK[kIdx*kvDim+kvHead*headDim+d] += dScore * Q[qPos*qDim+h*headDim+d]
				}
			}
		}
	}

	mhaRoPEBackwardGrad(layer, seqLen, msl, seqBase, gQ, gK, qDim, kvDim, headDim, numHeads, numKVHeads)
	_ = dModel
	_ = lay
	return gQ, gK, gV
}

// mhaAttentionBackwardIntNative backprops through int8 Q·K scores + uint8 softmax (code space).
// gradPre is in float activation units (post-dequant); Q/K/V slices are int8 codes.
func mhaAttentionBackwardIntNative(
	layer *VolumetricLayer,
	seqLen, msl, seqBase int,
	qI8, kI8, vI8 []int8,
	gradPre []float64,
	qDim, kvDim, headDim, numHeads, numKVHeads int,
	invSqrtHead float64,
	scale float32,
) (gQ, gK, gV []float64) {
	headsPerKV := numHeads / numKVHeads
	attnScale := invSqrtHead
	invScale := 1.0 / float64(scale)
	// int8 MAC uses >>8; attention scores live in that fixed-point space.
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
				intScores[kPos] = int8HeadDot(qI8, kI8, qPos*qDim+h*headDim, kIdx*kvDim+kvHead*headDim, headDim)
			}
			probsU8 := intAttnSoftmaxU8(intScores, invSqrtHead)
			probs := make([]float64, nPos)
			for kPos := 0; kPos < nPos; kPos++ {
				probs[kPos] = float64(probsU8[kPos]) / 255.0
			}
			for d := 0; d < headDim; d++ {
				// gradPre is dL/d(attn_float); convert to code-space sensitivity.
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

func mhaInt8LinearBackwardRow(
	gradW []int32,
	w []int8,
	inI8 []int8,
	gradOut int32,
	wStart, bIdx, inDim int,
	gradIn []int32,
	inOff int,
) {
	gradW[bIdx] += gradOut
	int8AccumWeightGrad(gradW, w, inI8, gradOut, wStart, inDim)
	if gradIn != nil && inOff+inDim <= len(gradIn) {
		int8AccumInputGrad(gradIn[inOff:inOff+inDim], w, gradOut, wStart, inDim)
	}
}

func MHAForwardNativeExact[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return MHAForwardTiled(layer, input)
	}
	var preF, postF *Tensor[float32]
	if useMHATrueNative(layer) {
		preF, postF = mhaForwardIntegerNative(layer, in)
	} else {
		preF, postF = mhaForwardNativeMAC(layer, in)
	}
	pre, post, ok2 := nativeTensorsAs[T](preF, postF)
	if !ok2 {
		return MHAForwardTiled(layer, input)
	}
	return pre, post
}

func MHABackwardNativeExact[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return MHABackwardTiled(layer, gradOutput, input, preAct)
	}
	var giF, gwF *Tensor[float32]
	if useMHATrueNative(layer) {
		giF, gwF = mhaBackwardIntegerNative(layer, goT, in, preF)
	} else {
		giF, gwF = mhaBackwardNativeMAC(layer, goT, in, preF)
	}
	gi, okGI := nativeTensorAs[T](giF)
	gw, okGW := nativeTensorAs[T](gwF)
	if !okGI || !okGW {
		return MHABackwardTiled(layer, gradOutput, input, preAct)
	}
	return gi, gw
}

func mhaForwardNativeMAC(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	wctx := newNativeWeightCtx(layer)
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

	mhaPrepareKVForForward[float32](layer, lay, msl, kvDim)
	cacheK := layer.KVCacheK.(*Tensor[float32])
	cacheV := layer.KVCacheV.(*Tensor[float32])

	kvStart := layer.KVOffset
	for b := 0; b < lay.batch; b++ {
		base := lay.base(b)
		seqBase := kvStart + b*seqLen
		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		qScratch := make([]float64, qDim)
		kScratch := make([]float32, kvDim)
		vScratch := make([]float32, kvDim)

		for s := 0; s < seqLen; s++ {
			pos := seqBase + s
			inRow := input.Data[base+s*dModel : base+(s+1)*dModel]
			kRow := cacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := cacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]

			mhaNativeProjectRow(&wctx, inRow, qwStart, qbStart, qDim, dModel, qScratch)
			for i := 0; i < qDim; i++ {
				Q[s*qDim+i] = qScratch[i]
			}
			mhaNativeProjectRow(&wctx, inRow, kwStart, kbStart, kvDim, dModel, qScratch[:kvDim])
			for i := 0; i < kvDim; i++ {
				kScratch[i] = float32(qScratch[i])
			}
			mhaNativeProjectRow(&wctx, inRow, vwStart, vbStart, kvDim, dModel, qScratch[:kvDim])
			for i := 0; i < kvDim; i++ {
				vScratch[i] = float32(qScratch[i])
			}
			copy(kRow, kScratch)
			copy(vRow, vScratch)

			if len(layer.QNormWeight) > 0 {
				applyPerHeadRMSNormFloat64(Q[s*qDim:(s+1)*qDim], layer.QNormWeight, numHeads, headDim, qkEps)
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
					qOff := s*qDim + h*headDim + d
					v0, v1 := Q[qOff], Q[qOff+half]
					Q[qOff] = v0*c - v1*sVal
					Q[qOff+half] = v0*sVal + v1*c
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
		}

		headsPerKV := numHeads / numKVHeads
		scale := 1.0 / math.Sqrt(float64(headDim))
		attnOut := make([]float64, seqLen*qDim)

		for s := 0; s < seqLen; s++ {
			currentTotalPos := seqBase + s
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				scores := make([]float64, currentTotalPos+1)
				maxScore := float64(-1e9)
				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					kIdx := kPos % msl
					var dot float64
					for d := 0; d < headDim; d++ {
						dot += Q[s*qDim+h*headDim+d] * float64(cacheK.Data[kIdx*kvDim+kvHead*headDim+d])
					}
					score := dot * scale
					scores[kPos] = score
					if score > maxScore {
						maxScore = score
					}
				}
				var expSum float64
				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					scores[kPos] = math.Exp(scores[kPos] - maxScore)
					expSum += scores[kPos]
				}
				for d := 0; d < headDim; d++ {
					var sum float64
					for kPos := 0; kPos <= currentTotalPos; kPos++ {
						sum += scores[kPos] * float64(cacheV.Data[(kPos%msl)*kvDim+kvHead*headDim+d])
					}
					attnOut[s*qDim+h*headDim+d] = sum / expSum
				}
			}
		}

		attnScratch := make([]float32, qDim)
		for s := 0; s < seqLen; s++ {
			for i := 0; i < dModel; i++ {
				preAct.Data[base+s*dModel+i] = float32(attnOut[s*qDim+i])
			}
			for j := 0; j < qDim; j++ {
				attnScratch[j] = float32(attnOut[s*qDim+j])
			}
			mhaNativeProjectRow(&wctx, attnScratch, owStart, obStart, dModel, qDim, qScratch[:dModel])
			for i := 0; i < dModel; i++ {
				postAct.Data[base+s*dModel+i] = float32(qScratch[i])
			}
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct
}

func mhaForwardIntegerNative(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return mhaForwardNativeMAC(layer, input)
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
		intScores := make([]int32, seqLen) // max positions <= seqLen for single-layer cache

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
					intScores[kPos] = int8HeadDot(qI8, kI8, s*qDim+h*headDim, kIdx*kvDim+kvHead*headDim, headDim)
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
	return preAct, postAct
}

func mhaBackwardNativeMAC(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	wctx := newNativeWeightCtx(layer)
	dModel, numHeads, _, headDim, qDim, kvDim := mhaLayerDims(layer)
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](layer.WeightStore.WeightCount(layer.DType))
	qwStart, kwStart, vwStart, owStart, qbStart, kbStart, vbStart, obStart := mhaWeightOffsets(layer, dModel, qDim, kvDim)

	giAcc := make([]float64, len(gradInput.Data))
	gwAcc := make([]float64, len(gradWeights.Data))

	cacheK := layer.KVCacheK.(*Tensor[float32])
	cacheV := layer.KVCacheV.(*Tensor[float32])
	kvEnd := layer.KVOffset

	for b := 0; b < lay.batch; b++ {
		seqBase := kvEnd - lay.batch*seqLen + b*seqLen
		gradPre := make([]float64, seqLen*qDim)

		// O projection backward — attn output (preAct) × gradOutput
		for s := 0; s < seqLen; s++ {
			outBase := lay.outIdx(b, s, 0)
			for i := 0; i < dModel; i++ {
				dy := float64(gradOutput.Data[outBase+i])
				gwAcc[obStart+i] += dy
				for j := 0; j < qDim; j++ {
					attnVal := 0.0
					if j < dModel {
						attnVal = float64(preAct.Data[outBase+j])
					}
					wIdx := owStart + i*qDim + j
					gwAcc[wIdx] += attnVal * dy
					gradPre[s*qDim+j] += dy * wctx.weightF64(wIdx)
				}
			}
		}

		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		for s := 0; s < seqLen; s++ {
			inBase := lay.inIdx(b, s, 0)
			inRow := input.Data[inBase : inBase+dModel]
			mhaNativeProjectRow(&wctx, inRow, qwStart, qbStart, qDim, dModel, Q[s*qDim:(s+1)*qDim])
			if len(layer.QNormWeight) > 0 {
				applyPerHeadRMSNormFloat64(Q[s*qDim:(s+1)*qDim], layer.QNormWeight, numHeads, headDim, qkEps)
			}
			theta := mhaRoPETheta(layer)
			half := headDim / 2
			pos := seqBase + s
			for h := 0; h < numHeads; h++ {
				for d := 0; d < half; d++ {
					angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
					c, sVal := math.Cos(angle), math.Sin(angle)
					qOff := s*qDim + h*headDim + d
					v0, v1 := Q[qOff], Q[qOff+half]
					Q[qOff] = v0*c - v1*sVal
					Q[qOff+half] = v0*sVal + v1*c
				}
			}
		}

		gQ, gK, gV := mhaAttentionBackwardFloat(layer, lay, seqLen, msl, seqBase, Q, gradPre, cacheK, cacheV)

		for s := 0; s < seqLen; s++ {
			kIdx := (seqBase + s) % msl
			inBase := lay.inIdx(b, s, 0)
			inRow := input.Data[inBase : inBase+dModel]
			for i := 0; i < qDim; i++ {
				dq := gQ[s*qDim+i]
				gwAcc[qbStart+i] += dq
				for j := 0; j < dModel; j++ {
					wIdx := qwStart + i*dModel + j
					inIdx := lay.inIdx(b, s, j)
					gwAcc[wIdx] += wctx.gradWTerm(inRow[j], dq)
					giAcc[inIdx] += wctx.gradXAt(wIdx, dq)
				}
			}
			for i := 0; i < kvDim; i++ {
				dk, dv := gK[kIdx*kvDim+i], gV[kIdx*kvDim+i]
				gwAcc[kbStart+i] += dk
				gwAcc[vbStart+i] += dv
				for j := 0; j < dModel; j++ {
					inIdx := lay.inIdx(b, s, j)
					kwIdx := kwStart + i*dModel + j
					vwIdx := vwStart + i*dModel + j
					gwAcc[kwIdx] += wctx.gradWTerm(inRow[j], dk)
					gwAcc[vwIdx] += wctx.gradWTerm(inRow[j], dv)
					giAcc[inIdx] += wctx.gradXAt(kwIdx, dk) + wctx.gradXAt(vwIdx, dv)
				}
			}
		}
	}

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(giAcc[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gwAcc[i])
	}
	return gradInput, gradWeights
}

func mhaBackwardIntegerNative(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	ws := layer.WeightStore
	ws.Morph(layer.DType)
	scale := ws.Scale
	if scale == 0 {
		scale = 1
	}
	w := nativeWeightsI8(ws, layer.DType)
	if w == nil {
		return mhaBackwardNativeMAC(layer, gradOutput, input, preAct)
	}
	cache := layer.ExactDense
	if cache == nil || len(cache.InputI8) == 0 || len(cache.QF64) == 0 || len(cache.QI8) == 0 {
		return mhaBackwardNativeMAC(layer, gradOutput, input, preAct)
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

	for b := 0; b < lay.batch; b++ {
		seqBase := kvEnd - lay.batch*seqLen + b*seqLen
		gradPre := make([]float64, seqLen*qDim)
		qI8 := cache.QI8[b*seqLen*qDim : b*seqLen*qDim+seqLen*qDim]
		kI8 := cache.KI8[b*msl*kvDim : b*msl*kvDim+msl*kvDim]
		vI8 := cache.VI8[b*msl*kvDim : b*msl*kvDim+msl*kvDim]

		// O projection backward — int8 MAC (input = cached attn int8)
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

		gQ, gK, gV := mhaAttentionBackwardIntNative(
			layer, seqLen, msl, seqBase,
			qI8, kI8, vI8, gradPre,
			qDim, kvDim, headDim, numHeads, numKVHeads,
			invSqrtHead, scale,
		)

		// Q/K/V projection backward — int8 MAC
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
	return gradInput, gradWeights
}
