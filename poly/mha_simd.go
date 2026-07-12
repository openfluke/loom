package poly

import (
	"math"

	"github.com/openfluke/loom/poly/simd"
)

func tryMHAForwardSimd[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T], ok bool) {
	in, ok := any(input).(*Tensor[float32])
	if !ok {
		return nil, nil, false
	}
	preF, postF := mhaForwardSimdF32(layer, in)
	return simdTensorsAs[T](preF, postF)
}

func mhaLayerDims(layer *VolumetricLayer) (dModel, numHeads, numKVHeads, headDim, qDim, kvDim int) {
	dModel = layer.DModel
	numHeads = layer.NumHeads
	numKVHeads = layer.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim = layer.HeadDim
	if headDim == 0 && numHeads > 0 {
		headDim = dModel / numHeads
	}
	qDim = layer.QueryDim
	if qDim == 0 {
		qDim = numHeads * headDim
	}
	kvDim = numKVHeads * headDim
	return
}

func mhaLayerSimdViable(layer *VolumetricLayer) bool {
	if layer == nil {
		return false
	}
	minDim := DenseSimdMinDim()
	dModel, _, _, headDim, qDim, _ := mhaLayerDims(layer)
	if dModel < minDim && headDim < minDim && qDim < minDim {
		return false
	}
	return true
}

func mhaForwardSimdF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	layer.EnsureRuntimeTileSizes()

	if usePackedTernaryCPU(layer) {
		return mhaForwardPackedTernaryCPUAsF32(layer, input)
	}
	if !mhaLayerSimdViable(layer) {
		return mhaForwardScalarFallbackF32(layer, input)
	}

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[float32](weights)
	return mhaForwardSimdF32WithWeights(layer, input, wData)
}

func mhaForwardSimdF32WithWeights(layer *VolumetricLayer, input *Tensor[float32], wData []float32) (preAct, postAct *Tensor[float32]) {
	dModel, numHeads, numKVHeads, headDim, qDim, kvDim := mhaLayerDims(layer)
	lay := mhaParseLayout(layer, input)
	seqLen := lay.seqLen
	msl := layer.MaxSeqLen
	if msl == 0 {
		msl = 512
	}

	tileSize := layer.GetCPUSimdTileSize(layer.DType)
	qkvTile := capMHAProjTileToLayer(tileSize, dModel, maxInt(qDim, kvDim))
	oTile := capMHAProjTileToLayer(tileSize, qDim, dModel)

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[float32](outShape...)
	postAct = NewTensor[float32](outShape...)

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
		kScratch := make([]float64, kvDim)
		vScratch := make([]float64, kvDim)

		for s := 0; s < seqLen; s++ {
			pos := seqBase + s
			inRow := input.Data[base+s*dModel : base+(s+1)*dModel]
			kRow := cacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
			vRow := cacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]

			mhaSimdProject(inRow, qW, wData, qbStart, dModel, qDim, qkvTile, qScratch)
			copy(Q[s*qDim:(s+1)*qDim], qScratch)

			mhaSimdProject(inRow, kW, wData, kbStart, dModel, kvDim, qkvTile, kScratch)
			mhaSimdProject(inRow, vW, wData, vbStart, dModel, kvDim, qkvTile, vScratch)
			for i := 0; i < kvDim; i++ {
				kRow[i] = float32(kScratch[i])
				vRow[i] = float32(vScratch[i])
			}

			if len(layer.QNormWeight) > 0 {
				applyPerHeadRMSNormFloat64(Q[s*qDim:(s+1)*qDim], layer.QNormWeight, numHeads, headDim, qkEps)
			}
			if len(layer.KNormWeight) > 0 {
				applyPerHeadRMSNormTensor(kRow, layer.KNormWeight, numKVHeads, headDim, qkEps)
			}

			mhaApplyRoPEQK(Q[s*qDim:(s+1)*qDim], kRow, pos, numHeads, numKVHeads, headDim, mhaRoPETheta(layer))
		}

		headsPerKV := numHeads / numKVHeads
		scale := 1.0 / math.Sqrt(float64(headDim))
		attnOut := make([]float64, seqLen*qDim)
		qHeadF32 := make([]float32, headDim)
		kHeadF32 := make([]float32, headDim)

		for s := 0; s < seqLen; s++ {
			currentTotalPos := seqBase + s
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				scores := make([]float64, currentTotalPos+1)
				maxScore := float64(-1e9)

				qOff := s*qDim + h*headDim
				for d := 0; d < headDim; d++ {
					qHeadF32[d] = float32(Q[qOff+d])
				}

				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					kIdx := kPos % msl
					kBase := kIdx*kvDim + kvHead*headDim
					for d := 0; d < headDim; d++ {
						kHeadF32[d] = cacheK.Data[kBase+d]
					}
					dot := mhaSimdDot(qHeadF32, kHeadF32, headDim)
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
						kIdx := kPos % msl
						sum += scores[kPos] * float64(cacheV.Data[kIdx*kvDim+kvHead*headDim+d])
					}
					attnOut[s*qDim+h*headDim+d] = sum / expSum
				}
			}
		}

		attnF32 := make([]float32, qDim)
		oScratch := make([]float64, dModel)
		for s := 0; s < seqLen; s++ {
			for j := 0; j < qDim; j++ {
				attnF32[j] = float32(attnOut[s*qDim+j])
				if j < dModel {
					preAct.Data[base+s*dModel+j] = attnF32[j]
				}
			}
			mhaSimdProject(attnF32, oW, wData, obStart, qDim, dModel, oTile, oScratch)
			for i := 0; i < dModel; i++ {
				postAct.Data[base+s*dModel+i] = float32(oScratch[i])
			}
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct
}

func mhaForwardScalarFallbackF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	wasLayer := layer.UseSimdForward
	wasNet := false
	if layer.Network != nil {
		wasNet = layer.Network.UseSimdForward
		layer.Network.UseSimdForward = false
	}
	layer.UseSimdForward = false
	pre, post := MHAForwardPolymorphic(layer, input)
	layer.UseSimdForward = wasLayer
	if layer.Network != nil {
		layer.Network.UseSimdForward = wasNet
	}
	return pre, post
}

func mhaForwardPackedTernaryCPUAsF32(layer *VolumetricLayer, input *Tensor[float32]) (preAct, postAct *Tensor[float32]) {
	pre, post := MHAForwardPackedTernaryCPU(layer, input)
	return pre, post
}

// mhaSimdProject is a per-token GEMV (input · Q/K/V/O weights). Like the SwiGLU
// path, we call the DotTile kernel ONCE over the full inner dimension per output
// row instead of once per tiny inner tile — the tile-per-call reduction overhead
// otherwise erases the SIMD win at decode. tileSize is now unused; full-range
// float64 reduction is bit-identical across arm64/amd64 (see poly/simd/dot.go).
func mhaSimdProject(input, weights []float32, wData []float32, bStart, inDim, outDim, tileSize int, output []float64) {
	for o := 0; o < outDim; o++ {
		rowOff := o * inDim
		output[o] = simd.DotTile(input, weights[rowOff:rowOff+inDim], 0, inDim, float64(wData[bStart+o]))
	}
}

func mhaSimdDot(q, k []float32, headDim int) float64 {
	if headDim <= 0 {
		return 0
	}
	if simd.SimdEnabled() && headDim >= 4 {
		return simd.DotTile(q, k, 0, headDim, 0)
	}
	var sum float64
	for d := 0; d < headDim; d++ {
		sum += float64(q[d]) * float64(k[d])
	}
	return sum
}

func mhaApplyRoPEQK(q []float64, kRow []float32, pos, numHeads, numKVHeads, headDim int, theta float64) {
	half := headDim / 2
	for h := 0; h < numHeads; h++ {
		for d := 0; d < half; d++ {
			angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
			c, sVal := math.Cos(angle), math.Sin(angle)
			qOff := h*headDim + d
			v0, v1 := q[qOff], q[qOff+half]
			q[qOff] = v0*c - v1*sVal
			q[qOff+half] = v0*sVal + v1*c
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
