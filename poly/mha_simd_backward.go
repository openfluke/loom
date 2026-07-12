package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

func tryMHABackwardSimd[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T], ok bool) {
	in, okIn := any(input).(*Tensor[float32])
	goT, okGO := any(gradOutput).(*Tensor[float32])
	preF, okPre := any(preAct).(*Tensor[float32])
	if !okIn || !okGO || !okPre {
		return nil, nil, false
	}
	if !mhaLayerSimdViable(layer) {
		return nil, nil, false
	}
	giF, gwF := mhaBackwardSimdF32(layer, goT, in, preF)
	gi, okGI := simdTensorAsBackward[T](giF)
	gw, okGW := simdTensorAsBackward[T](gwF)
	if !okGI || !okGW {
		return nil, nil, false
	}
	return gi, gw, true
}

func mhaBackwardSimdF32(layer *VolumetricLayer, gradOutput, input, preAct *Tensor[float32]) (gradInput, gradWeights *Tensor[float32]) {
	dModel, numHeads, numKVHeads, headDim, qDim, kvDim := mhaLayerDims(layer)
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}

	gradInput = NewTensor[float32](input.Shape...)
	gradWeights = NewTensor[float32](layer.WeightStore.WeightCount(layer.DType))

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)

	qwStart := 0
	kwStart := qwStart + qDim*dModel
	vwStart := kwStart + kvDim*dModel
	owStart := vwStart + kvDim*dModel
	qbStart := owStart + dModel*qDim
	kbStart := qbStart + qDim
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	qW := wData[qwStart:kwStart]
	kW := wData[kwStart:vwStart]
	vW := wData[vwStart:owStart]
	oW := wData[owStart:qbStart]

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))

	qkvTile := layer.GetCPUSimdTileSize(layer.DType)
	qkvTile = capMHAProjTileToLayer(qkvTile, dModel, maxInt(qDim, kvDim))

	kvEnd := layer.KVOffset
	cacheK := layer.KVCacheK.(*Tensor[float32])
	cacheV := layer.KVCacheV.(*Tensor[float32])

	for b := 0; b < lay.batch; b++ {
		seqBase := kvEnd - lay.batch*seqLen + b*seqLen
		gradPre := make([]float64, seqLen*qDim)

		for s := 0; s < seqLen; s++ {
			outIdx := lay.outIdx(b, s, 0)
			preIdx := lay.outIdx(b, s, 0)
			mhaSimdOProjBackwardOne(
				gradPre[s*qDim:(s+1)*qDim],
				gradOutput.Data[outIdx:outIdx+dModel],
				preAct.Data[preIdx:preIdx+dModel],
				oW, gw64, owStart, obStart, dModel, qDim,
			)
		}

		Q := make([]float64, seqLen*qDim)
		qkEps := mhaQKNormEpsilon(layer)
		qScratch := make([]float64, qDim)
		for s := 0; s < seqLen; s++ {
			inIdx := lay.inIdx(b, s, 0)
			inRow := input.Data[inIdx : inIdx+dModel]
			mhaSimdProject(inRow, qW, wData, qbStart, dModel, qDim, qkvTile, qScratch)
			copy(Q[s*qDim:(s+1)*qDim], qScratch)

			if len(layer.QNormWeight) > 0 {
				applyPerHeadRMSNormFloat64(Q[s*qDim:(s+1)*qDim], layer.QNormWeight, numHeads, headDim, qkEps)
			}
			pos := seqBase + s
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
		}

		headsPerKV := numHeads / numKVHeads
		scale := 1.0 / math.Sqrt(float64(headDim))

		gQ := make([]float64, seqLen*qDim)
		gK := make([]float64, msl*kvDim)
		gV := make([]float64, msl*kvDim)

		qHeadF32 := make([]float32, headDim)
		kHeadF32 := make([]float32, headDim)

		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKV
			for qPos := 0; qPos < seqLen; qPos++ {
				currentTotalPos := seqBase + qPos
				scores := make([]float64, currentTotalPos+1)
				maxScore := float64(-1e9)

				qOff := qPos*qDim + h*headDim
				for d := 0; d < headDim; d++ {
					qHeadF32[d] = float32(Q[qOff+d])
				}

				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					kIdx := kPos % msl
					kBase := kIdx*kvDim + kvHead*headDim
					for d := 0; d < headDim; d++ {
						kHeadF32[d] = cacheK.Data[kBase+d]
					}
					score := mhaSimdDot(qHeadF32, kHeadF32, headDim) * scale
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
						kBase := kIdx*kvDim + kvHead*headDim
						dScore := (scores[kPos]*dy*float64(cacheV.Data[kBase+d]) - scores[kPos]*dS_sum) * scale

						gQ[qOff+d] += dScore * float64(cacheK.Data[kBase+d])
						gK[kBase+d] += dScore * Q[qOff+d]
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
			inIdx := lay.inIdx(b, s, 0)
			inRow := input.Data[inIdx : inIdx+dModel]
			mhaSimdQKVProjBackwardOne(gi64[inIdx:inIdx+dModel], inRow, gQ[s*qDim:(s+1)*qDim], qW, gw64, qwStart, qbStart, dModel, qDim)

			kIdx := (seqBase + s) % msl
			mhaSimdQKVProjBackwardOne(gi64[inIdx:inIdx+dModel], inRow, gK[kIdx*kvDim:(kIdx+1)*kvDim], kW, gw64, kwStart, kbStart, dModel, kvDim)
			mhaSimdQKVProjBackwardOne(gi64[inIdx:inIdx+dModel], inRow, gV[kIdx*kvDim:(kIdx+1)*kvDim], vW, gw64, vwStart, vbStart, dModel, kvDim)
		}
	}

	for i := range gradInput.Data {
		gradInput.Data[i] = float32(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = float32(gw64[i])
	}
	return gradInput, gradWeights
}

// mhaSimdOProjBackwardOne: ∂L/∂O projection for one token (saxpy on O weight rows + gradPre).
func mhaSimdOProjBackwardOne(gradPre []float64, gradOut, preAct []float32, oW []float32, gw64 []float64, owStart, obStart, dModel, qDim int) {
	prePad := make([]float32, qDim)
	for j := 0; j < dModel && j < len(preAct); j++ {
		prePad[j] = preAct[j]
	}
	for i := 0; i < dModel; i++ {
		dy := float64(gradOut[i])
		gw64[obStart+i] += dy
		rowOff := owStart + i*qDim
		simd.SaxpyF32AccF64(gw64[rowOff:rowOff+qDim], dy, prePad, qDim)
		simd.SaxpyF32AccF64(gradPre, dy, oW[i*qDim:(i+1)*qDim], qDim)
	}
}

// mhaSimdQKVProjBackwardOne: ∂L/∂Q/K/V input projection for one token (saxpy dW rows + dX).
func mhaSimdQKVProjBackwardOne(gi64 []float64, input []float32, g []float64, w []float32, gw64 []float64, wStart, bStart, dModel, outDim int) {
	for i := 0; i < outDim; i++ {
		gi := g[i]
		gw64[bStart+i] += gi
		wRow := i * dModel
		gwOff := wStart + wRow
		simd.SaxpyF32AccF64(gw64[gwOff:gwOff+dModel], gi, input, dModel)
		simd.SaxpyF32AccF64(gi64, gi, w[wRow:wRow+dModel], dModel)
	}
}
