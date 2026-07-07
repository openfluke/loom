package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

// mhaRoPETheta matches wgpu_forward.go: default 10000 when the layer leaves RoPE base unset.
func mhaRoPETheta(layer *VolumetricLayer) float64 {
	if layer != nil && layer.RoPEFreqBase > 0 {
		return layer.RoPEFreqBase
	}
	return 10000.0
}

// mhaQKNormEpsilon matches GPU DispatchRMSNorm for Q/K: prefer layer RMSNormEps (from config),
// same default as block norms when unset.
func mhaQKNormEpsilon(layer *VolumetricLayer) float64 {
	if layer != nil && layer.RMSNormEps > 0 {
		return layer.RMSNormEps
	}
	return 1e-6
}

// MHAForwardPolymorphic performs Multi-Head Attention using strictly generic arithmetic.
func MHAForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if usePackedTernaryCPU(layer) {
		return MHAForwardPackedTernaryCPU(layer, input)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryMHAForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	if layer.UseTiling && layer.TileSize > 0 {
		return mhaForwardTiledGeneric(layer, input)
	}

	dModel := layer.DModel
	numHeads := layer.NumHeads
	numKVHeads := layer.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := layer.HeadDim
	qDim := layer.QueryDim
	if qDim == 0 {
		qDim = numHeads * headDim
	}
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}
	kvDim := numKVHeads * headDim

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	qbStart := owStart + dModel*qDim
	kbStart := qbStart + qDim
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	qW, kW, vW, oW := wData[qwStart:kwStart], wData[kwStart:vwStart], wData[vwStart:owStart], wData[owStart:qbStart]
	qB, kB, vB, oB := wData[qbStart:kbStart], wData[kbStart:vbStart], wData[vbStart:obStart], wData[obStart:obStart+dModel]

	mhaPrepareKVForForward[T](layer, lay, msl, kvDim)
	cacheK := layer.KVCacheK.(*Tensor[T])
	cacheV := layer.KVCacheV.(*Tensor[T])

	kvStart := layer.KVOffset
	for b := 0; b < lay.batch; b++ {
		base := lay.base(b)
		seqBase := kvStart + b*seqLen

		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		for s := 0; s < seqLen; s++ {
			pos := seqBase + s
			kRow := cacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := cacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]

			for i := 0; i < qDim; i++ {
				sum := float64(qB[i])
				for j := 0; j < dModel; j++ {
					sum += float64(input.Data[base+s*dModel+j]) * float64(qW[i*dModel+j])
				}
				Q[s*qDim+i] = sum
			}

			for i := 0; i < kvDim; i++ {
				sumK, sumV := float64(kB[i]), float64(vB[i])
				for j := 0; j < dModel; j++ {
					inVal := float64(input.Data[base+s*dModel+j])
					sumK += inVal * float64(kW[i*dModel+j])
					sumV += inVal * float64(vW[i*dModel+j])
				}
				kRow[i] = T(sumK)
				vRow[i] = T(sumV)
			}

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
					kRow[kOff] = T(v0*c - v1*sVal)
					kRow[kOff+half] = T(v0*sVal + v1*c)
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

		for s := 0; s < seqLen; s++ {
			for i := 0; i < dModel; i++ {
				preAct.Data[base+s*dModel+i] = T(attnOut[s*qDim+i])
				sum := float64(oB[i])
				for j := 0; j < qDim; j++ {
					sum += attnOut[s*qDim+j] * float64(oW[i*qDim+j])
				}
				postAct.Data[base+s*dModel+i] = T(sum)
			}
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct
}

func MHAForwardPackedTernaryCPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	dModel := layer.DModel
	numHeads := layer.NumHeads
	numKVHeads := layer.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := layer.HeadDim
	qDim := layer.QueryDim
	if qDim == 0 {
		qDim = numHeads * headDim
	}
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}
	kvDim := numKVHeads * headDim

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	qbStart := owStart + dModel*qDim
	kbStart := qbStart + qDim
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	qW, okQ := layer.WeightStore.GetBitNetTernaryMatrix(qwStart, qDim, dModel)
	kW, okK := layer.WeightStore.GetBitNetTernaryMatrix(kwStart, kvDim, dModel)
	vW, okV := layer.WeightStore.GetBitNetTernaryMatrix(vwStart, kvDim, dModel)
	oW, okO := layer.WeightStore.GetBitNetTernaryMatrix(owStart, dModel, qDim)
	if !okQ || !okK || !okV || !okO {
		// Fall back with exact dtype disabled to avoid recursively re-entering this fast path.
		exact := layer.Network.UseExactDType
		layer.Network.UseExactDType = false
		pre, post := MHAForwardPolymorphic(layer, input)
		layer.Network.UseExactDType = exact
		return pre, post
	}

	mhaPrepareKVForForward[T](layer, lay, msl, kvDim)
	cacheK := layer.KVCacheK.(*Tensor[T])
	cacheV := layer.KVCacheV.(*Tensor[T])

	kvStart := layer.KVOffset
	for b := 0; b < lay.batch; b++ {
		base := lay.base(b)
		seqBase := kvStart + b*seqLen

		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		tmpQ := make([]float64, qDim)
		tmpK := make([]float64, kvDim)
		tmpV := make([]float64, kvDim)
		inputQ := make([]int8, dModel)
		for s := 0; s < seqLen; s++ {
			pos := seqBase + s
			inputRow := input.Data[base+s*dModel : base+(s+1)*dModel]
			kRow := cacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := cacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]

			inputQ, activationMax := bitNetQuantizeActivationNumeric(inputRow, inputQ)
			bitNetTernaryMatVecQuantized(qW, inputQ, activationMax, tmpQ)
			for i := 0; i < qDim; i++ {
				Q[s*qDim+i] = tmpQ[i] + bitNetTernaryBias(layer.WeightStore, qbStart+i)
			}

			bitNetTernaryMatVecQuantized(kW, inputQ, activationMax, tmpK)
			bitNetTernaryMatVecQuantized(vW, inputQ, activationMax, tmpV)
			for i := 0; i < kvDim; i++ {
				kRow[i] = T(tmpK[i] + bitNetTernaryBias(layer.WeightStore, kbStart+i))
				vRow[i] = T(tmpV[i] + bitNetTernaryBias(layer.WeightStore, vbStart+i))
			}

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
					kRow[kOff] = T(v0*c - v1*sVal)
					kRow[kOff+half] = T(v0*sVal + v1*c)
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

		tmpO := make([]float64, dModel)
		attnQ := make([]int8, qDim)
		for s := 0; s < seqLen; s++ {
			attnRow := attnOut[s*qDim : (s+1)*qDim]
			bitNetRMSNormFloat64Weighted(attnRow, layer.InnerNormWeight, layer.RMSNormEps)
			attnQ, activationMax := bitNetQuantizeActivationFloat64(attnRow, attnQ)
			bitNetTernaryMatVecQuantized(oW, attnQ, activationMax, tmpO)
			for i := 0; i < dModel; i++ {
				preAct.Data[base+s*dModel+i] = T(attnRow[i])
				postAct.Data[base+s*dModel+i] = T(tmpO[i] + bitNetTernaryBias(layer.WeightStore, obStart+i))
			}
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct
}

// MHABackwardPolymorphic calculates gradients for the MHA layer.
func MHABackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	dModel := layer.DModel
	numHeads := layer.NumHeads
	numKVHeads := layer.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := layer.HeadDim
	qDim := layer.QueryDim
	if qDim == 0 {
		qDim = numHeads * headDim
	}
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}
	kvDim := numKVHeads * headDim

	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](layer.WeightStore.WeightCount(layer.DType))

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	qbStart := owStart + dModel*qDim
	kbStart := qbStart + qDim
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	qW, kW, vW, oW := wData[qwStart:kwStart], wData[kwStart:vwStart], wData[vwStart:owStart], wData[owStart:qbStart]

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	kvEnd := layer.KVOffset
	for b := 0; b < lay.batch; b++ {
		seqBase := kvEnd - lay.batch*seqLen + b*seqLen
		gradPre := make([]float64, seqLen*qDim)
		for s := 0; s < seqLen; s++ {
			for i := 0; i < dModel; i++ {
				dy := float64(gradOutput.Data[lay.outIdx(b, s, i)])
				gw64[obStart+i] += dy
				for j := 0; j < qDim; j++ {
					preVal := 0.0
					if j < dModel {
						preVal = float64(preAct.Data[lay.outIdx(b, s, j)])
					}
					gw64[owStart+i*qDim+j] += preVal * dy
					gradPre[s*qDim+j] += dy * float64(oW[i*qDim+j])
				}
			}
		}

		cacheK := layer.KVCacheK.(*Tensor[T])
		cacheV := layer.KVCacheV.(*Tensor[T])

		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		for s := 0; s < seqLen; s++ {
			for i := 0; i < qDim; i++ {
				sum := float64(wData[qbStart+i])
				for j := 0; j < dModel; j++ {
					sum += float64(input.Data[lay.inIdx(b, s, j)]) * float64(qW[i*dModel+j])
				}
				Q[s*qDim+i] = sum
			}
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

		headsPerKV := numHeads / numKVHeads
		scale := 1.0 / math.Sqrt(float64(headDim))

		gQ := make([]float64, seqLen*qDim)
		gK := make([]float64, msl*kvDim)
		gV := make([]float64, msl*kvDim)

		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKV
			for qPos := 0; qPos < seqLen; qPos++ {
				currentTotalPos := seqBase + qPos
				scores := make([]float64, currentTotalPos+1)
				maxScore := float64(-1e9)
				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					kIdx := kPos % msl
					var dot float64
					for d := 0; d < headDim; d++ {
						dot += Q[qPos*qDim+h*headDim+d] * float64(cacheK.Data[kIdx*kvDim+kvHead*headDim+d])
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
				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					scores[kPos] /= expSum
				}

				for d := 0; d < headDim; d++ {
					dy := gradPre[qPos*qDim+h*headDim+d]
					var dS_sum float64
					for kPos := 0; kPos <= currentTotalPos; kPos++ {
						vIdx := kPos % msl
						gV[vIdx*kvDim+kvHead*headDim+d] += scores[kPos] * dy
						dS_sum += float64(cacheV.Data[vIdx*kvDim+kvHead*headDim+d]) * dy * scores[kPos]
					}

					for kPos := 0; kPos <= currentTotalPos; kPos++ {
						kIdx := kPos % msl
						dScore := (scores[kPos]*dy*float64(cacheV.Data[kIdx*kvDim+kvHead*headDim+d]) - scores[kPos]*dS_sum) * scale

						gQ[qPos*qDim+h*headDim+d] += dScore * float64(cacheK.Data[kIdx*kvDim+kvHead*headDim+d])
						gK[kIdx*kvDim+kvHead*headDim+d] += dScore * Q[qPos*qDim+h*headDim+d]
					}
				}
			}
		}

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

		for s := 0; s < seqLen; s++ {
			kIdx := (seqBase + s) % msl
			for i := 0; i < qDim; i++ {
				dq := gQ[s*qDim+i]
				gw64[qbStart+i] += dq
				for j := 0; j < dModel; j++ {
					inIdx := lay.inIdx(b, s, j)
					gw64[qwStart+i*dModel+j] += float64(input.Data[inIdx]) * dq
					gi64[inIdx] += dq * float64(qW[i*dModel+j])
				}
			}
			for i := 0; i < kvDim; i++ {
				dk, dv := gK[kIdx*kvDim+i], gV[kIdx*kvDim+i]
				gw64[kbStart+i] += dk
				gw64[vbStart+i] += dv
				for j := 0; j < dModel; j++ {
					inIdx := lay.inIdx(b, s, j)
					x := float64(input.Data[inIdx])
					gw64[kwStart+i*dModel+j] += x * dk
					gw64[vwStart+i*dModel+j] += x * dv
					gi64[inIdx] += dk*float64(kW[i*dModel+j]) + dv*float64(vW[i*dModel+j])
				}
			}
		}
	}

	for i := range gradInput.Data {
		gradInput.Data[i] = T(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gw64[i])
	}

	return gradInput, gradWeights
}

func applyPerHeadRMSNormFloat64(data []float64, gamma []float32, numHeads, headDim int, eps float64) {
	if numHeads <= 0 || headDim <= 0 || len(data) < numHeads*headDim {
		return
	}
	if eps <= 0 {
		eps = 1e-6
	}
	for h := 0; h < numHeads; h++ {
		start := h * headDim
		end := start + headDim
		var sumSq float64
		for i := start; i < end; i++ {
			v := data[i]
			sumSq += v * v
		}
		rms := math.Sqrt(sumSq/float64(headDim) + eps)
		if rms == 0 {
			continue
		}
		inv := 1.0 / rms
		for d := 0; d < headDim; d++ {
			scale := 1.0
			if len(gamma) == headDim {
				scale = float64(gamma[d])
			} else if len(gamma) == numHeads*headDim {
				scale = float64(gamma[h*headDim+d])
			}
			data[start+d] = data[start+d] * inv * scale
		}
	}
}

func applyPerHeadRMSNormTensor[T Numeric](data []T, gamma []float32, numHeads, headDim int, eps float64) {
	if numHeads <= 0 || headDim <= 0 || len(data) < numHeads*headDim {
		return
	}
	if eps <= 0 {
		eps = 1e-6
	}
	for h := 0; h < numHeads; h++ {
		start := h * headDim
		end := start + headDim
		var sumSq float64
		for i := start; i < end; i++ {
			v := float64(data[i])
			sumSq += v * v
		}
		rms := math.Sqrt(sumSq/float64(headDim) + eps)
		if rms == 0 {
			continue
		}
		inv := 1.0 / rms
		for d := 0; d < headDim; d++ {
			scale := 1.0
			if len(gamma) == headDim {
				scale = float64(gamma[d])
			} else if len(gamma) == numHeads*headDim {
				scale = float64(gamma[h*headDim+d])
			}
			data[start+d] = T(float64(data[start+d]) * inv * scale)
		}
	}
}

// MHAForwardTiled performs a tiled Multi-Head Attention forward pass.
func MHAForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return mhaForwardTiledGeneric(layer, input)
}

// MHAForwardTiledParallel is an alias for the tiled forward path.
func MHAForwardTiledParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return mhaForwardTiledGeneric(layer, input)
}

func mhaForwardTiledGeneric[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	layer.EnsureRuntimeTileSizes()
	tileSize := layer.GetCPUTileSize(layer.DType)

	dModel, numHeads, numKVHeads, headDim, qDim, kvDim := mhaLayerDims(layer)
	qkvTile := capMHAProjTileToLayer(tileSize, dModel, maxInt(qDim, kvDim))
	oTile := capMHAProjTileToLayer(tileSize, qDim, dModel)

	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	qbStart := owStart + dModel*qDim
	kbStart := qbStart + qDim
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	wsScale := float32(1)
	if layer.WeightStore != nil && layer.WeightStore.Scale != 0 {
		wsScale = layer.WeightStore.Scale
	}

	mhaPrepareKVForForward[T](layer, lay, msl, kvDim)
	cacheK := layer.KVCacheK.(*Tensor[T])
	cacheV := layer.KVCacheV.(*Tensor[T])

	kvStart := layer.KVOffset
	for b := 0; b < lay.batch; b++ {
		base := lay.base(b)
		seqBase := kvStart + b*seqLen

		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		qScratch := make([]float64, qDim)
		kScratch := make([]T, kvDim)
		vScratch := make([]T, kvDim)

		for s := 0; s < seqLen; s++ {
			pos := seqBase + s
			inRow := input.Data[base+s*dModel : base+(s+1)*dModel]
			kRow := cacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := cacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]

			mhaTiledProject(inRow, wData, qwStart, qbStart, qScratch, dModel, qDim, 1, layer.DType, wsScale, qkvTile)
			for i := 0; i < qDim; i++ {
				Q[s*qDim+i] = float64(qScratch[i])
			}

			mhaTiledProject(inRow, wData, kwStart, kbStart, kScratch, dModel, kvDim, 1, layer.DType, wsScale, qkvTile)
			mhaTiledProject(inRow, wData, vwStart, vbStart, vScratch, dModel, kvDim, 1, layer.DType, wsScale, qkvTile)
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
					kRow[kOff] = T(v0*c - v1*sVal)
					kRow[kOff+half] = T(v0*sVal + v1*c)
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

		attnScratch := make([]T, qDim)
		oScratch := make([]T, dModel)
		for s := 0; s < seqLen; s++ {
			for i := 0; i < dModel; i++ {
				preAct.Data[base+s*dModel+i] = T(attnOut[s*qDim+i])
			}
			for j := 0; j < qDim; j++ {
				attnScratch[j] = T(attnOut[s*qDim+j])
			}
			mhaTiledProject(attnScratch, wData, owStart, obStart, oScratch, qDim, dModel, 1, layer.DType, wsScale, oTile)
			copy(postAct.Data[base+s*dModel:base+(s+1)*dModel], oScratch)
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct
}

// MHABackwardTiled computes the backward pass for multihead attention using tiled matrix multiplication.
func MHABackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	orig := layer.UseTiling
	layer.UseTiling = false
	gI, gW := MHABackwardPolymorphic(layer, gradOutput, input, preAct)
	layer.UseTiling = orig
	return gI, gW
}

// mhaTiledProject projects inputs to Q, K, or V within generic tiled bounds.
func mhaTiledProject[TIn Numeric, TOut Numeric, TW Numeric](input []TIn, wData []TW, projWStart, projBStart int, output []TOut, inDim, outDim, seqLen int, dtype DType, scale float32, tileSize int) {
	projW := wData[projWStart:]
	projB := wData[projBStart:]

	for s := 0; s < seqLen; s++ {
		for oTile := 0; oTile < outDim; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outDim {
				oEnd = outDim
			}
			for iTile := 0; iTile < inDim; iTile += tileSize {
				iEnd := iTile + tileSize
				if iEnd > inDim {
					iEnd = inDim
				}

				for o := oTile; o < oEnd; o++ {
					sum := float64(0)
					if iTile == 0 {
						sum = float64(projB[o])
					} else {
						sum = float64(output[s*outDim+o])
					}
					for i := iTile; i < iEnd; i++ {
						sum += float64(input[s*inDim+i]) * float64(projW[o*inDim+i])
					}
					output[s*outDim+o] = TOut(sum)
				}
			}
		}
	}
}

// mhaTiledProjectBackward computes gradient pass of tiled projections.
func mhaTiledProjectBackward[TIn Numeric, TGrad Numeric](layer *VolumetricLayer, gradOut *Tensor[TGrad], curAct *Tensor[TIn], weights any, wData []TIn, gradInter *Tensor[TGrad], gradWeights *Tensor[TGrad], outDim, inDim, seqLen, projWStart, projBStart, tileSize int) {
	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			gradWeights.Data[projBStart+o] += gradOut.Data[s*outDim+o]
		}
	}
}
