package poly

// q4_cpu.go — CPU inference against baked Q4_0 .entity weights without expanding Master to FP32.
// Matches wgpu ShaderTiledDenseQ4 indexing (block=32, 8 nibbles per u32 word).
//
// SIMD path uses fused Q4 GEMV (simd.DotQ4_0Row): AVX2 (amd64) / NEON (arm64) unpack
// nibbles in-register per 32-weight block — no full-row FP32 dequant, no Master inflate.

import (
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func usePackedQ4CPU(layer *VolumetricLayer) bool {
	return layer != nil &&
		layer.Network != nil &&
		layer.Network.UsePackedQ4CPU &&
		!layer.Network.UseGPU &&
		layer.WeightStore != nil &&
		layer.WeightStore.HasAnyQ4_0()
}

func usePackedQ4SIMD(layer *VolumetricLayer) bool {
	return usePackedQ4CPU(layer) && layerUseSimdForward(layer) && simd.SimdEnabled()
}

// GemvQ4_0Packed computes out[o] = bias[o] + sum_i in[i]*W[o,i] with W stored as PackQ4_0GPU.
func GemvQ4_0Packed(scales []float32, packed []uint32, in []float32, bias []float32, out []float64, outRows, inCols int) {
	gemvQ4_0PackedRows(scales, packed, in, bias, out, 0, outRows, inCols, false)
}

// gemvQ4_0Packed is the internal alias used by layer forwards.
func gemvQ4_0Packed(scales []float32, packed []uint32, in []float32, bias []float32, out []float64, outRows, inCols int) {
	GemvQ4_0Packed(scales, packed, in, bias, out, outRows, inCols)
}

func gemvQ4_0PackedSIMD(scales []float32, packed []uint32, in []float32, bias []float32, out []float64, outRows, inCols int) {
	gemvQ4_0PackedRows(scales, packed, in, bias, out, 0, outRows, inCols, true)
}

// GemvQ4_0PackedParallel is the multi-row Q4 GEMV (optional SIMD).
func GemvQ4_0PackedParallel(scales []float32, packed []uint32, in []float32, bias []float32, out []float64, outRows, inCols int, useSimd bool) {
	gemvQ4_0PackedParallel(scales, packed, in, bias, out, outRows, inCols, useSimd)
}

func gemvQ4_0PackedParallel(scales []float32, packed []uint32, in []float32, bias []float32, out []float64, outRows, inCols int, useSimd bool) {
	if len(out) < outRows || len(in) < inCols {
		return
	}
	if outRows < 64 || runtime.NumCPU() < 2 {
		gemvQ4_0PackedRows(scales, packed, in, bias, out, 0, outRows, inCols, useSimd)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ4_0PackedRows(scales, packed, in, bias, out, lo, hi, inCols, useSimd)
		}(o0, o1)
	}
	wg.Wait()
}

// gemvQ4_0PackedParallelF32 is the LM-head fast path: writes float32 logits directly.
func gemvQ4_0PackedParallelF32(scales []float32, packed []uint32, in []float32, out []float32, outRows, inCols int, useSimd bool) {
	if len(out) < outRows || len(in) < inCols {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if outRows < 64 || workers < 2 {
		gemvQ4_0PackedRowsF32(scales, packed, in, out, 0, outRows, inCols, useSimd)
		return
	}
	if workers > outRows {
		workers = outRows
	}
	chunk := (outRows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		o0 := w * chunk
		o1 := o0 + chunk
		if o1 > outRows {
			o1 = outRows
		}
		if o0 >= o1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			gemvQ4_0PackedRowsF32(scales, packed, in, out, lo, hi, inCols, useSimd)
		}(o0, o1)
	}
	wg.Wait()
}

func gemvQ4_0PackedRowsF32(scales []float32, packed []uint32, in []float32, out []float32, rowLo, rowHi, inCols int, useSimd bool) {
	o := rowLo
	if useSimd && inCols%32 == 0 && simd.SimdEnabled() {
		for ; o+3 < rowHi; o += 4 {
			simd.DotQ4_0Rows4(in, scales, packed, o*inCols, inCols, out[o:o+4])
		}
	}
	for ; o < rowHi; o++ {
		baseW := o * inCols
		if useSimd {
			out[o] = float32(simd.DotQ4_0Row(in, scales, packed, baseW, inCols, 0))
			continue
		}
		out[o] = float32(gemvQ4_0PackedRowGo(in, scales, packed, baseW, inCols, 0))
	}
}

// gemvQ4_0PackedRows covers output rows [rowLo, rowHi).
// useSimd: fused Q4 AVX2/NEON DotQ4_0Row (no full-row FP32 scratch).
func gemvQ4_0PackedRows(scales []float32, packed []uint32, in []float32, bias []float32, out []float64, rowLo, rowHi, inCols int, useSimd bool) {
	o := rowLo
	if useSimd && bias == nil && inCols%32 == 0 && simd.SimdEnabled() {
		var tmp [4]float32
		for ; o+3 < rowHi; o += 4 {
			simd.DotQ4_0Rows4(in, scales, packed, o*inCols, inCols, tmp[:])
			out[o] = float64(tmp[0])
			out[o+1] = float64(tmp[1])
			out[o+2] = float64(tmp[2])
			out[o+3] = float64(tmp[3])
		}
	}
	for ; o < rowHi; o++ {
		sum := 0.0
		if bias != nil && o < len(bias) {
			sum = float64(bias[o])
		}
		baseW := o * inCols
		if useSimd {
			out[o] = simd.DotQ4_0Row(in, scales, packed, baseW, inCols, sum)
			continue
		}
		out[o] = simdDotQ4_0RowScalar(in, scales, packed, baseW, inCols, sum)
	}
}

// simdDotQ4_0RowScalar keeps the portable fused kernel without requiring SimdEnabled().
func simdDotQ4_0RowScalar(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	return gemvQ4_0PackedRowGo(in, scales, packed, baseW, n, prev)
}

func gemvQ4_0PackedRowGo(in []float32, scales []float32, packed []uint32, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	limit := n / 8
	for k := 0; k < limit; k++ {
		globalIdx := baseW + i
		scale := float64(scales[globalIdx/32])
		w := packed[globalIdx/8]
		acc := 0.0
		for nib := 0; nib < 8; nib++ {
			q := int32((w >> (uint(nib) * 4)) & 0xF)
			if q > 7 {
				q -= 16
			}
			acc += float64(in[i+nib]) * float64(q)
		}
		sum += acc * scale
		i += 8
	}
	for ; i < n; i++ {
		globalIdx := baseW + i
		scale := float64(scales[globalIdx/32])
		nibble := (globalIdx % 8) * 4
		q := int32((packed[globalIdx/8] >> uint(nibble)) & 0xF)
		if q > 7 {
			q -= 16
		}
		sum += float64(in[i]) * float64(q) * scale
	}
	return sum
}

func SwiGLUForwardPackedQ4CPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	ws := layer.WeightStore
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	wSize := inputSize * intermediateSize

	gateS, gateP := ws.Q4_0Scales[DType(100)], ws.Q4_0Packed[DType(100)]
	upS, upP := ws.Q4_0Scales[DType(101)], ws.Q4_0Packed[DType(101)]
	downS, downP := ws.Q4_0Scales[DType(102)], ws.Q4_0Packed[DType(102)]
	if len(gateP) == 0 || len(upP) == 0 || len(downP) == 0 {
		return swigluForwardTiledParallel(layer, input)
	}

	gateBStart := 3 * wSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize
	biasTail := 2*intermediateSize + inputSize
	var gateB, upB, downB []float32
	switch {
	case len(ws.Master) == biasTail:
		gateB = ws.Master[0:intermediateSize]
		upB = ws.Master[intermediateSize : 2*intermediateSize]
		downB = ws.Master[2*intermediateSize : biasTail]
	case len(ws.Master) >= downBStart+inputSize:
		gateB = ws.Master[gateBStart:upBStart]
		upB = ws.Master[upBStart:downBStart]
		downB = ws.Master[downBStart : downBStart+inputSize]
	}

	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)
	gate := make([]float64, intermediateSize)
	up := make([]float64, intermediateSize)
	down := make([]float64, inputSize)
	inF := make([]float32, inputSize)
	preF := make([]float32, intermediateSize)

	act := layer.Activation
	if act == 0 {
		act = ActivationSilu
	}
	useSimd := usePackedQ4SIMD(layer)

	for s := 0; s < seqLen; s++ {
		row := input.Data[s*inputSize : (s+1)*inputSize]
		for i := 0; i < inputSize; i++ {
			inF[i] = float32(row[i])
		}
		gemvQ4_0PackedParallel(gateS, gateP, inF, gateB, gate, intermediateSize, inputSize, useSimd)
		gemvQ4_0PackedParallel(upS, upP, inF, upB, up, intermediateSize, inputSize, useSimd)
		for o := 0; o < intermediateSize; o++ {
			v := swigluGateProduct(gate[o], up[o], act)
			preAct.Data[s*intermediateSize+o] = T(v)
			preF[o] = float32(v)
		}
		if len(layer.InnerNormWeight) > 0 {
			preRow := preAct.Data[s*intermediateSize : (s+1)*intermediateSize]
			bitNetRMSNormTensorRowWeighted(preRow, layer.InnerNormWeight, layer.RMSNormEps)
			for o := 0; o < intermediateSize; o++ {
				preF[o] = float32(preRow[o])
			}
		}
		gemvQ4_0PackedParallel(downS, downP, preF, downB, down, inputSize, intermediateSize, useSimd)
		for i := 0; i < inputSize; i++ {
			postAct.Data[s*inputSize+i] = T(down[i])
		}
	}
	return preAct, postAct
}

func MHAForwardPackedQ4CPU[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	ws := layer.WeightStore
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

	qS, qP := ws.Q4_0Scales[WeightMHAQuery], ws.Q4_0Packed[WeightMHAQuery]
	kS, kP := ws.Q4_0Scales[WeightMHAKey], ws.Q4_0Packed[WeightMHAKey]
	vS, vP := ws.Q4_0Scales[WeightMHAValue], ws.Q4_0Packed[WeightMHAValue]
	oS, oP := ws.Q4_0Scales[WeightMHAProjection], ws.Q4_0Packed[WeightMHAProjection]
	if len(qP) == 0 || len(kP) == 0 || len(vP) == 0 || len(oP) == 0 {
		return MHAForwardTiled(layer, input)
	}

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	mhaPrepareKVForForward[T](layer, lay, msl, kvDim)
	cacheK := layer.KVCacheK.(*Tensor[T])
	cacheV := layer.KVCacheV.(*Tensor[T])

	inF := make([]float32, dModel)
	qOut := make([]float64, qDim)
	kOut := make([]float64, kvDim)
	vOut := make([]float64, kvDim)
	oIn := make([]float32, qDim)
	oOut := make([]float64, dModel)
	useSimd := usePackedQ4SIMD(layer)

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
			for j := 0; j < dModel; j++ {
				inF[j] = float32(input.Data[base+s*dModel+j])
			}
			gemvQ4_0PackedParallel(qS, qP, inF, nil, qOut, qDim, dModel, useSimd)
			gemvQ4_0PackedParallel(kS, kP, inF, nil, kOut, kvDim, dModel, useSimd)
			gemvQ4_0PackedParallel(vS, vP, inF, nil, vOut, kvDim, dModel, useSimd)
			copy(Q[s*qDim:(s+1)*qDim], qOut)
			for i := 0; i < kvDim; i++ {
				kRow[i] = T(kOut[i])
				vRow[i] = T(vOut[i])
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
		scoresBuf := make([]float64, seqBase+seqLen+1) // reuse; grow if needed
		qHeadF := make([]float32, headDim)
		kHeadF := make([]float32, headDim)

		for s := 0; s < seqLen; s++ {
			currentTotalPos := seqBase + s
			need := currentTotalPos + 1
			if len(scoresBuf) < need {
				scoresBuf = make([]float64, need)
			}
			scores := scoresBuf[:need]
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				maxScore := float64(-1e9)
				qBase := s*qDim + h*headDim
				useF32Attn := useSimd && headDim >= 8
				if useF32Attn {
					for d := 0; d < headDim; d++ {
						qHeadF[d] = float32(Q[qBase+d])
					}
				}
				for kPos := 0; kPos <= currentTotalPos; kPos++ {
					kIdx := kPos % msl
					var dot float64
					kOff := kIdx*kvDim + kvHead*headDim
					if useF32Attn {
						if kf, ok := any(cacheK.Data).([]float32); ok {
							dot = simd.DotTile(qHeadF, kf[kOff:kOff+headDim], 0, headDim, 0)
						} else {
							for d := 0; d < headDim; d++ {
								kHeadF[d] = float32(cacheK.Data[kOff+d])
							}
							dot = simd.DotTile(qHeadF, kHeadF, 0, headDim, 0)
						}
					} else {
						for d := 0; d < headDim; d++ {
							dot += Q[qBase+d] * float64(cacheK.Data[kOff+d])
						}
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
			for j := 0; j < qDim; j++ {
				oIn[j] = float32(attnOut[s*qDim+j])
			}
			for i := 0; i < dModel; i++ {
				if i < qDim {
					preAct.Data[base+s*dModel+i] = T(attnOut[s*qDim+i])
				}
			}
			gemvQ4_0PackedParallel(oS, oP, oIn, nil, oOut, dModel, qDim, useSimd)
			for i := 0; i < dModel; i++ {
				postAct.Data[base+s*dModel+i] = T(oOut[i])
			}
		}
	}
	layer.KVOffset = kvStart + lay.batch*seqLen
	return preAct, postAct
}
