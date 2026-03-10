package poly

import (
	"math"
)

// MHAForwardPolymorphic performs Multi-Head Attention across any numerical type.
func MHAForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	dModel := layer.DModel
	numHeads := layer.NumHeads
	numKVHeads := layer.NumKVHeads
	if numKVHeads == 0 { numKVHeads = numHeads }
	headDim := layer.HeadDim
	seqLen := len(input.Data) / dModel
	msl := layer.MaxSeqLen
	if msl == 0 { msl = 512 }
	kvDim := numKVHeads * headDim
	
	if layer.UseTiling && layer.TileSize > 0 {
		return MHAForwardTiled(layer, input)
	}
	
	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	qwStart := 0
	kwStart := dModel * dModel
	vwStart := dModel * (dModel + kvDim)
	owStart := dModel * (dModel + 2 * kvDim)
	
	qbStart := owStart + dModel * dModel
	kbStart := qbStart + dModel
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			qW := rawW[qwStart : qwStart + dModel * dModel]
			kW := rawW[kwStart : kwStart + dModel * kvDim]
			vW := rawW[vwStart : vwStart + dModel * kvDim]
			oW := rawW[owStart : owStart + dModel * dModel]
			
			qB := rawW[qbStart : qbStart + dModel]
			kB := rawW[kbStart : kbStart + kvDim]
			vB := rawW[vbStart : vbStart + kvDim]
			oB := rawW[obStart : obStart + dModel]

			Q := make([]float64, seqLen * dModel)
			K := make([]float64, seqLen * kvDim)
			V := make([]float64, seqLen * kvDim)
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					sum := float64(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float64(input.Data[s*dModel+j]) * float64(qW[i*dModel+j])
					}
					Q[s*dModel+i] = sum
				}
				for i := 0; i < kvDim; i++ {
					sumK := float64(kB[i])
					sumV := float64(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float64(input.Data[s*dModel+j])
						sumK += inVal * float64(kW[i*kvDim+j])
						sumV += inVal * float64(vW[i*kvDim+j])
					}
					K[s*kvDim+i] = sumK
					V[s*kvDim+i] = sumV
				}
			}

			if layer.RoPEFreqBase > 0 {
				half := headDim / 2
				theta := float64(layer.RoPEFreqBase)
				for s := 0; s < seqLen; s++ {
					pos := s
					for h := 0; h < numHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							qOff := s*dModel + h*headDim + d
							v0, v1 := Q[qOff], Q[qOff+half]
							Q[qOff] = v0*c - v1*sVal
							Q[qOff+half] = v0*sVal + v1*c
						}
					}
					for h := 0; h < numKVHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							kOff := s*kvDim + h*headDim + d
							v0, v1 := K[kOff], K[kOff+half]
							K[kOff] = v0*c - v1*sVal
							K[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float64(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float64, seqLen * dModel)
			
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for qPos := 0; qPos < seqLen; qPos++ {
					scores := make([]float64, qPos+1)
					maxScore := float64(-1e9)
					
					for kPos := 0; kPos <= qPos; kPos++ {
						var dot float64
						for d := 0; d < headDim; d++ {
							dot += Q[qPos*dModel+h*headDim+d] * K[kPos*kvDim+kvHead*headDim+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float64
					for kPos := 0; kPos <= qPos; kPos++ {
						scores[kPos] = float64(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float64
						for kPos := 0; kPos <= qPos; kPos++ {
							sum += scores[kPos] * V[kPos*kvDim+kvHead*headDim+d]
						}
						attnOut[qPos*dModel + h*headDim + d] = sum / expSum
					}
				}
			}

			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
					sum := float64(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * float64(oW[i*dModel+j])
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			qW := rawW[qwStart : qwStart + dModel * dModel]
			kW := rawW[kwStart : kwStart + dModel * kvDim]
			vW := rawW[vwStart : vwStart + dModel * kvDim]
			oW := rawW[owStart : owStart + dModel * dModel]
			
			qB := rawW[qbStart : qbStart + dModel]
			kB := rawW[kbStart : kbStart + kvDim]
			vB := rawW[vbStart : vbStart + kvDim]
			oB := rawW[obStart : obStart + dModel]

			// Initialize KV Cache if needed
			if layer.KVCacheK == nil {
				layer.KVCacheK = NewTensor[float32](msl, kvDim)
				layer.KVCacheV = NewTensor[float32](msl, kvDim)
				layer.KVOffset = 0
			}

			Q := make([]float32, seqLen * dModel)
			
			for s := 0; s < seqLen; s++ {
				// Current position in the global sequence
				pos := layer.KVOffset + s
				
				// 1. Projections
				for i := 0; i < dModel; i++ {
					sum := float32(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float32(input.Data[s*dModel+j]) * qW[i*dModel+j]
					}
					Q[s*dModel+i] = sum
				}
				
				kRow := layer.KVCacheK.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
				vRow := layer.KVCacheV.Data[(pos%msl)*kvDim : (pos%msl+1)*kvDim]
				
				for i := 0; i < kvDim; i++ {
					sumK := float32(kB[i])
					sumV := float32(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float32(input.Data[s*dModel+j])
						sumK += inVal * kW[i*dModel+j]
						sumV += inVal * vW[i*dModel+j]
					}
					kRow[i] = sumK
					vRow[i] = sumV
				}

				// 2. RoPE
				if layer.RoPEFreqBase > 0 {
					half := headDim / 2
					theta := float64(layer.RoPEFreqBase)
					// Apply to Q
					for h := 0; h < numHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
							qOff := s*dModel + h*headDim + d
							v0, v1 := Q[qOff], Q[qOff+half]
							Q[qOff] = v0*c - v1*sVal
							Q[qOff+half] = v0*sVal + v1*c
						}
					}
					// Apply to K (current row in cache)
					for h := 0; h < numKVHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
							kOff := h*headDim + d
							v0, v1 := kRow[kOff], kRow[kOff+half]
							kRow[kOff] = v0*c - v1*sVal
							kRow[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float32(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float32, seqLen * dModel)
			
			for s := 0; s < seqLen; s++ {
				currentTotalPos := layer.KVOffset + s
				for h := 0; h < numHeads; h++ {
					kvHead := h / headsPerKV
					
					// Attention against all cached tokens
					scores := make([]float32, currentTotalPos+1)
					maxScore := float32(-1e9)
					
					for kPos := 0; kPos <= currentTotalPos; kPos++ {
						kIdx := kPos % msl
						var dot float32
						qBase := s*dModel + h*headDim
						kBase := kIdx*kvDim + kvHead*headDim
						for d := 0; d < headDim; d++ {
							dot += Q[qBase+d] * layer.KVCacheK.Data[kBase+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float32
					for kPos := 0; kPos <= currentTotalPos; kPos++ {
						scores[kPos] = float32(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float32
						for kPos := 0; kPos <= currentTotalPos; kPos++ {
							vIdx := kPos % msl
							vBase := vIdx*kvDim + kvHead*headDim
							sum += scores[kPos] * layer.KVCacheV.Data[vBase+d]
						}
						attnOut[s*dModel + h*headDim + d] = sum / expSum
					}
				}
			}

			// 3. Final Projection
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
					sum := float32(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * oW[i*dModel+j]
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}

			layer.KVOffset += seqLen
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			qW := rawW[qwStart : qwStart + dModel * dModel]
			kW := rawW[kwStart : kwStart + dModel * kvDim]
			vW := rawW[vwStart : vwStart + dModel * kvDim]
			oW := rawW[owStart : owStart + dModel * dModel]
			
			qB := rawW[qbStart : qbStart + dModel]
			kB := rawW[kbStart : kbStart + kvDim]
			vB := rawW[vbStart : vbStart + kvDim]
			oB := rawW[obStart : obStart + dModel]

			Q := make([]float64, seqLen * dModel)
			K := make([]float64, seqLen * kvDim)
			V := make([]float64, seqLen * kvDim)
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					sum := float64(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float64(input.Data[s*dModel+j]) * float64(qW[i*dModel+j])
					}
					Q[s*dModel+i] = sum
				}
				for i := 0; i < kvDim; i++ {
					sumK := float64(kB[i])
					sumV := float64(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float64(input.Data[s*dModel+j])
						sumK += inVal * float64(kW[i*kvDim+j])
						sumV += inVal * float64(vW[i*kvDim+j])
					}
					K[s*kvDim+i] = sumK
					V[s*kvDim+i] = sumV
				}
			}

			if layer.RoPEFreqBase > 0 {
				half := headDim / 2
				theta := float64(layer.RoPEFreqBase)
				for s := 0; s < seqLen; s++ {
					pos := s
					for h := 0; h < numHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							qOff := s*dModel + h*headDim + d
							v0, v1 := Q[qOff], Q[qOff+half]
							Q[qOff] = v0*c - v1*sVal
							Q[qOff+half] = v0*sVal + v1*c
						}
					}
					for h := 0; h < numKVHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							kOff := s*kvDim + h*headDim + d
							v0, v1 := K[kOff], K[kOff+half]
							K[kOff] = v0*c - v1*sVal
							K[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float64(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float64, seqLen * dModel)
			
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for qPos := 0; qPos < seqLen; qPos++ {
					scores := make([]float64, qPos+1)
					maxScore := float64(-1e9)
					
					for kPos := 0; kPos <= qPos; kPos++ {
						var dot float64
						for d := 0; d < headDim; d++ {
							dot += Q[qPos*dModel+h*headDim+d] * K[kPos*kvDim+kvHead*headDim+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float64
					for kPos := 0; kPos <= qPos; kPos++ {
						scores[kPos] = float64(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float64
						for kPos := 0; kPos <= qPos; kPos++ {
							sum += scores[kPos] * V[kPos*kvDim+kvHead*headDim+d]
						}
						attnOut[qPos*dModel + h*headDim + d] = sum / expSum
					}
				}
			}

			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
					sum := float64(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * float64(oW[i*dModel+j])
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			qW := rawW[qwStart : qwStart + dModel * dModel]
			kW := rawW[kwStart : kwStart + dModel * kvDim]
			vW := rawW[vwStart : vwStart + dModel * kvDim]
			oW := rawW[owStart : owStart + dModel * dModel]
			
			qB := rawW[qbStart : qbStart + dModel]
			kB := rawW[kbStart : kbStart + kvDim]
			vB := rawW[vbStart : vbStart + kvDim]
			oB := rawW[obStart : obStart + dModel]

			Q := make([]float64, seqLen * dModel)
			K := make([]float64, seqLen * kvDim)
			V := make([]float64, seqLen * kvDim)
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					sum := float64(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float64(input.Data[s*dModel+j]) * float64(qW[i*dModel+j])
					}
					Q[s*dModel+i] = sum
				}
				for i := 0; i < kvDim; i++ {
					sumK := float64(kB[i])
					sumV := float64(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float64(input.Data[s*dModel+j])
						sumK += inVal * float64(kW[i*kvDim+j])
						sumV += inVal * float64(vW[i*kvDim+j])
					}
					K[s*kvDim+i] = sumK
					V[s*kvDim+i] = sumV
				}
			}

			if layer.RoPEFreqBase > 0 {
				half := headDim / 2
				theta := float64(layer.RoPEFreqBase)
				for s := 0; s < seqLen; s++ {
					pos := s
					for h := 0; h < numHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							qOff := s*dModel + h*headDim + d
							v0, v1 := Q[qOff], Q[qOff+half]
							Q[qOff] = v0*c - v1*sVal
							Q[qOff+half] = v0*sVal + v1*c
						}
					}
					for h := 0; h < numKVHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							kOff := s*kvDim + h*headDim + d
							v0, v1 := K[kOff], K[kOff+half]
							K[kOff] = v0*c - v1*sVal
							K[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float64(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float64, seqLen * dModel)
			
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for qPos := 0; qPos < seqLen; qPos++ {
					scores := make([]float64, qPos+1)
					maxScore := float64(-1e9)
					
					for kPos := 0; kPos <= qPos; kPos++ {
						var dot float64
						for d := 0; d < headDim; d++ {
							dot += Q[qPos*dModel+h*headDim+d] * K[kPos*kvDim+kvHead*headDim+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float64
					for kPos := 0; kPos <= qPos; kPos++ {
						scores[kPos] = float64(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float64
						for kPos := 0; kPos <= qPos; kPos++ {
							sum += scores[kPos] * V[kPos*kvDim+kvHead*headDim+d]
						}
						attnOut[qPos*dModel + h*headDim + d] = sum / expSum
					}
				}
			}

			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
					sum := float64(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * float64(oW[i*dModel+j])
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			qW := rawW[qwStart : qwStart + dModel * dModel]
			kW := rawW[kwStart : kwStart + dModel * kvDim]
			vW := rawW[vwStart : vwStart + dModel * kvDim]
			oW := rawW[owStart : owStart + dModel * dModel]
			
			qB := rawW[qbStart : qbStart + dModel]
			kB := rawW[kbStart : kbStart + kvDim]
			vB := rawW[vbStart : vbStart + kvDim]
			oB := rawW[obStart : obStart + dModel]

			Q := make([]float64, seqLen * dModel)
			K := make([]float64, seqLen * kvDim)
			V := make([]float64, seqLen * kvDim)
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					sum := float64(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float64(input.Data[s*dModel+j]) * float64(qW[i*dModel+j])
					}
					Q[s*dModel+i] = sum
				}
				for i := 0; i < kvDim; i++ {
					sumK := float64(kB[i])
					sumV := float64(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float64(input.Data[s*dModel+j])
						sumK += inVal * float64(kW[i*kvDim+j])
						sumV += inVal * float64(vW[i*kvDim+j])
					}
					K[s*kvDim+i] = sumK
					V[s*kvDim+i] = sumV
				}
			}

			if layer.RoPEFreqBase > 0 {
				half := headDim / 2
				theta := float64(layer.RoPEFreqBase)
				for s := 0; s < seqLen; s++ {
					pos := s
					for h := 0; h < numHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							qOff := s*dModel + h*headDim + d
							v0, v1 := Q[qOff], Q[qOff+half]
							Q[qOff] = v0*c - v1*sVal
							Q[qOff+half] = v0*sVal + v1*c
						}
					}
					for h := 0; h < numKVHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							kOff := s*kvDim + h*headDim + d
							v0, v1 := K[kOff], K[kOff+half]
							K[kOff] = v0*c - v1*sVal
							K[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float64(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float64, seqLen * dModel)
			
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for qPos := 0; qPos < seqLen; qPos++ {
					scores := make([]float64, qPos+1)
					maxScore := float64(-1e9)
					
					for kPos := 0; kPos <= qPos; kPos++ {
						var dot float64
						for d := 0; d < headDim; d++ {
							dot += Q[qPos*dModel+h*headDim+d] * K[kPos*kvDim+kvHead*headDim+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float64
					for kPos := 0; kPos <= qPos; kPos++ {
						scores[kPos] = float64(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float64
						for kPos := 0; kPos <= qPos; kPos++ {
							sum += scores[kPos] * V[kPos*kvDim+kvHead*headDim+d]
						}
						attnOut[qPos*dModel + h*headDim + d] = sum / expSum
					}
				}
			}

			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
					sum := float64(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * float64(oW[j*dModel+i])
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			qW := rawW[qwStart : qwStart + dModel * dModel]
			kW := rawW[kwStart : kwStart + dModel * kvDim]
			vW := rawW[vwStart : vwStart + dModel * kvDim]
			oW := rawW[owStart : owStart + dModel * dModel]
			
			qB := rawW[qbStart : qbStart + dModel]
			kB := rawW[kbStart : kbStart + kvDim]
			vB := rawW[vbStart : vbStart + kvDim]
			oB := rawW[obStart : obStart + dModel]

			Q := make([]float64, seqLen * dModel)
			K := make([]float64, seqLen * kvDim)
			V := make([]float64, seqLen * kvDim)
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					sum := float64(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float64(input.Data[s*dModel+j]) * float64(qW[j*dModel+i])
					}
					Q[s*dModel+i] = sum
				}
				for i := 0; i < kvDim; i++ {
					sumK := float64(kB[i])
					sumV := float64(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float64(input.Data[s*dModel+j])
						sumK += inVal * float64(kW[j*kvDim+i])
						sumV += inVal * float64(vW[j*kvDim+i])
					}
					K[s*kvDim+i] = sumK
					V[s*kvDim+i] = sumV
				}
			}

			if layer.RoPEFreqBase > 0 {
				half := headDim / 2
				theta := float64(layer.RoPEFreqBase)
				for s := 0; s < seqLen; s++ {
					pos := s
					for h := 0; h < numHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							qOff := s*dModel + h*headDim + d
							v0, v1 := Q[qOff], Q[qOff+half]
							Q[qOff] = v0*c - v1*sVal
							Q[qOff+half] = v0*sVal + v1*c
						}
					}
					for h := 0; h < numKVHeads; h++ {
						for d := 0; d < half; d++ {
							freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
							angle := freq * float64(pos)
							c := float64(math.Cos(angle))
							sVal := float64(math.Sin(angle))
							
							kOff := s*kvDim + h*headDim + d
							v0, v1 := K[kOff], K[kOff+half]
							K[kOff] = v0*c - v1*sVal
							K[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float64(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float64, seqLen * dModel)
			
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for qPos := 0; qPos < seqLen; qPos++ {
					scores := make([]float64, qPos+1)
					maxScore := float64(-1e9)
					
					for kPos := 0; kPos <= qPos; kPos++ {
						var dot float64
						for d := 0; d < headDim; d++ {
							dot += Q[qPos*dModel+h*headDim+d] * K[kPos*kvDim+kvHead*headDim+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float64
					for kPos := 0; kPos <= qPos; kPos++ {
						scores[kPos] = float64(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float64
						for kPos := 0; kPos <= qPos; kPos++ {
							sum += scores[kPos] * V[kPos*kvDim+kvHead*headDim+d]
						}
						attnOut[qPos*dModel + h*headDim + d] = sum / expSum
					}
				}
			}

			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
					sum := float64(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * float64(oW[j*dModel+i])
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	}

	scaleW := layer.WeightStore.Scale
	if scaleW == 0 { scaleW = 1.0 }
	wData := CastWeights[float32](weights)
	
	qW := wData[qwStart : qwStart + dModel * dModel]
	kW := wData[kwStart : kwStart + dModel * kvDim]
	vW := wData[vwStart : vwStart + dModel * kvDim]
	oW := wData[owStart : owStart + dModel * dModel]
	
	qB := wData[qbStart : qbStart + dModel]
	kB := wData[kbStart : kbStart + kvDim]
	vB := wData[vbStart : vbStart + kvDim]
	oB := wData[obStart : obStart + dModel]

	Q := make([]float32, seqLen * dModel)
	K := make([]float32, seqLen * kvDim)
	V := make([]float32, seqLen * kvDim)
	
	for s := 0; s < seqLen; s++ {
		for i := 0; i < dModel; i++ {
			sum := SimulatePrecision(qB[i], layer.DType, scaleW)
			for j := 0; j < dModel; j++ {
				sum += float32(input.Data[s*dModel+j]) * SimulatePrecision(qW[j*dModel+i], layer.DType, scaleW)
			}
			Q[s*dModel+i] = sum
		}
		for i := 0; i < kvDim; i++ {
			sumK := SimulatePrecision(kB[i], layer.DType, scaleW)
			sumV := SimulatePrecision(vB[i], layer.DType, scaleW)
			for j := 0; j < dModel; j++ {
				val := float32(input.Data[s*dModel+j])
				sumK += val * SimulatePrecision(kW[j*kvDim+i], layer.DType, scaleW)
				sumV += val * SimulatePrecision(vW[j*kvDim+i], layer.DType, scaleW)
			}
			K[s*kvDim+i] = sumK
			V[s*kvDim+i] = sumV
		}
	}

	if layer.RoPEFreqBase > 0 {
		half := headDim / 2
		theta := float64(layer.RoPEFreqBase)
		for s := 0; s < seqLen; s++ {
			pos := s
			for h := 0; h < numHeads; h++ {
				for d := 0; d < half; d++ {
					freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
					angle := freq * float64(pos)
					c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
					off := s*dModel + h*headDim + d
					v0, v1 := Q[off], Q[off+half]
					Q[off] = v0*c - v1*sVal
					Q[off+half] = v0*sVal + v1*c
				}
			}
			for h := 0; h < numKVHeads; h++ {
				for d := 0; d < half; d++ {
					freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
					angle := freq * float64(pos)
					c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
					off := s*kvDim + h*headDim + d
					v0, v1 := K[off], K[off+half]
					K[off] = v0*c - v1*sVal
					K[off+half] = v0*sVal + v1*c
				}
			}
		}
	}

	headsPerKV := numHeads / numKVHeads
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut := make([]float32, seqLen * dModel)
	for h := 0; h < numHeads; h++ {
		kvH := h / headsPerKV
		for qP := 0; qP < seqLen; qP++ {
			scores := make([]float32, qP+1)
			maxS := float32(-1e9)
			for kP := 0; kP <= qP; kP++ {
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += Q[qP*dModel+h*headDim+d] * K[kP*kvDim+kvH*headDim+d]
				}
				scores[kP] = dot * attnScale
				if scores[kP] > maxS { maxS = scores[kP] }
			}
			var eSum float32
			for kP := 0; kP <= qP; kP++ {
				scores[kP] = float32(math.Exp(float64(scores[kP] - maxS)))
				eSum += scores[kP]
			}
			for d := 0; d < headDim; d++ {
				var sum float32
				for kP := 0; kP <= qP; kP++ {
					sum += scores[kP] * V[kP*kvDim+kvH*headDim+d]
				}
				attnOut[qP*dModel+h*headDim+d] = sum / eSum
			}
		}
	}

	for s := 0; s < seqLen; s++ {
		for i := 0; i < dModel; i++ {
			preAct.Data[s*dModel+i] = T(attnOut[s*dModel+i])
			sum := SimulatePrecision(oB[i], layer.DType, scaleW)
			for j := 0; j < dModel; j++ {
				sum += attnOut[s*dModel+j] * SimulatePrecision(oW[j*dModel+i], layer.DType, scaleW)
			}
			postAct.Data[s*dModel+i] = T(sum)
		}
	}

	return preAct, postAct
}

// MHABackwardPolymorphic handles BPTT-style gradients for MHA.
func MHABackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	return gradInput, gradWeights
}

// MHAForwardTiled performs an optimized, tiled forward pass for MHA.
func MHAForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	dModel := layer.DModel
	numHeads := layer.NumHeads
	numKVHeads := layer.NumKVHeads
	if numKVHeads == 0 { numKVHeads = numHeads }
	headDim := layer.HeadDim
	seqLen := len(input.Data) / dModel
	msl := layer.MaxSeqLen
	if msl == 0 { msl = 512 }
	kvDim := numKVHeads * headDim
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 32 }

	outShape := append([]int{}, input.Shape[:len(input.Shape)-1]...)
	outShape = append(outShape, dModel)
	preAct = NewTensor[T](outShape...)
	postAct = NewTensor[T](outShape...)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	// Initialize KV Cache if needed
	if layer.KVCacheK == nil {
		layer.KVCacheK = NewTensor[float32](msl, kvDim)
		layer.KVCacheV = NewTensor[float32](msl, kvDim)
		layer.KVOffset = 0
	}

	qwStart := 0
	kwStart := dModel * dModel
	vwStart := dModel * (dModel + kvDim)
	owStart := dModel * (dModel + 2 * kvDim)
	qbStart := owStart + dModel * dModel
	kbStart := qbStart + dModel
	vbStart := kbStart + kvDim
	obStart := vbStart + kvDim

	// Local Q buffer
	Q := make([]float32, seqLen * dModel)

	scaleW := layer.WeightStore.Scale
	if scaleW == 0 { scaleW = 1.0 }

	// Tiled Q Projection (Local)
	mhaTiledProject(input.Data, weights, qwStart, qbStart, Q, dModel, dModel, seqLen, layer.DType, scaleW, tileSize)
	
	// Tiled K and V Projections directly into Cache
	for s := 0; s < seqLen; s++ {
		pos := (layer.KVOffset + s) % msl
		kRow := layer.KVCacheK.Data[pos*kvDim : (pos+1)*kvDim]
		vRow := layer.KVCacheV.Data[pos*kvDim : (pos+1)*kvDim]
		
		// We use a temporary slice of one step for the projection helper
		inStep := input.Data[s*dModel : (s+1)*dModel]
		mhaTiledProject(inStep, weights, kwStart, kbStart, kRow, dModel, kvDim, 1, layer.DType, scaleW, tileSize)
		mhaTiledProject(inStep, weights, vwStart, vbStart, vRow, dModel, kvDim, 1, layer.DType, scaleW, tileSize)

		// Apply RoPE to current step K in cache
		if layer.RoPEFreqBase > 0 {
			half := headDim / 2
			theta := float64(layer.RoPEFreqBase)
			globalPos := layer.KVOffset + s
			for h := 0; h < numKVHeads; h++ {
				for d := 0; d < half; d++ {
					freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
					angle := freq * float64(globalPos)
					c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
					off := h*headDim + d
					v0, v1 := kRow[off], kRow[off+half]
					kRow[off] = v0*c - v1*sVal
					kRow[off+half] = v0*sVal + v1*c
				}
			}
			// Apply RoPE to Q (local)
			for h := 0; h < numHeads; h++ {
				for d := 0; d < half; d++ {
					freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
					angle := freq * float64(globalPos)
					c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
					off := s*dModel + h*headDim + d
					v0, v1 := Q[off], Q[off+half]
					Q[off] = v0*c - v1*sVal
					Q[off+half] = v0*sVal + v1*c
				}
			}
		}
	}

	// Tiled Attention using KV Cache
	attnOut := make([]float32, seqLen * dModel)
	mhaTiledAttention(Q, layer.KVCacheK.Data, layer.KVCacheV.Data, attnOut, seqLen, numHeads, numKVHeads, headDim, dModel, kvDim, layer.KVOffset, msl, tileSize)

	// Tiled Output Projection
	mhaTiledProject(attnOut, weights, owStart, obStart, postAct.Data, dModel, dModel, seqLen, layer.DType, scaleW, tileSize)

	for i := range preAct.Data {
		preAct.Data[i] = T(attnOut[i])
	}

	layer.KVOffset += seqLen
	return preAct, postAct
}

func mhaTiledProject[TIn Numeric, TOut Numeric](input []TIn, weights any, wStart, bStart int, output []TOut, inDim, outDim, seqLen int, dtype DType, scale float32, tileSize int) {
	// Specialized fast-paths
	switch dtype {
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
		// Float16/BFloat16 are stored as masked Float32 in this framework.
		if wData, ok := weights.([]float32); ok {
			mhaTiledProjectFloat32(input, wData, wStart, bStart, output, inDim, outDim, seqLen, tileSize)
			return
		}
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt4, DTypeUint4, DTypeFP4:
		// These use []int8 representation (unpacked in RAM for this framework)
		if wData, ok := weights.([]int8); ok {
			mhaTiledProjectInt8(input, wData, wStart, bStart, output, inDim, outDim, seqLen, scale, tileSize)
			return
		}
	case DTypeBinary:
		if wData, ok := weights.([]int8); ok {
			mhaTiledProjectBinary(input, wData, wStart, bStart, output, inDim, outDim, seqLen, scale, tileSize)
			return
		}
	}

	// Fallback to generic tiled projection
	wData := CastWeights[float32](weights)
	if wData == nil { return }
	wBlock := wData[wStart:]
	bBlock := wData[bStart:]

	for oTile := 0; oTile < outDim; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outDim { oEnd = outDim }
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim { iEnd = inDim }

			for s := 0; s < seqLen; s++ {
				for o := oTile; o < oEnd; o++ {
					var sum float32
					if iTile == 0 {
						sum = bBlock[o] // Raw bias
					} else {
						sum = float32(output[s*outDim+o])
					}
					for i := iTile; i < iEnd; i++ {
						sum += float32(input[s*inDim+i]) * SimulatePrecision(wBlock[o*inDim+i], dtype, scale)
					}
					output[s*outDim+o] = TOut(sum)
				}
			}
		}
	}
}

func mhaTiledProjectFloat32[TIn Numeric, TOut Numeric](input []TIn, wData []float32, wStart, bStart int, output []TOut, inDim, outDim, seqLen int, tileSize int) {
	wBlock := wData[wStart:]
	bBlock := wData[bStart:]

	// Local buffer for input tile
	inTileBuf := make([]float32, tileSize)

	for iTile := 0; iTile < inDim; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > inDim { iEnd = inDim }
		currentITileSize := iEnd - iTile

		for s := 0; s < seqLen; s++ {
			// Load input tile once
			for i := 0; i < currentITileSize; i++ {
				inTileBuf[i] = float32(input[s*inDim+iTile+i])
			}

			for oTile := 0; oTile < outDim; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outDim { oEnd = outDim }

				for o := oTile; o < oEnd; o++ {
					var sum float32
					if iTile == 0 {
						sum = bBlock[o]
					} else {
						sum = float32(output[s*outDim+o])
					}
					
					rowOff := o * inDim + iTile
					// Unrolled dot product
					i := 0
					for ; i <= currentITileSize-4; i += 4 {
						sum += inTileBuf[i] * wBlock[rowOff+i]
						sum += inTileBuf[i+1] * wBlock[rowOff+i+1]
						sum += inTileBuf[i+2] * wBlock[rowOff+i+2]
						sum += inTileBuf[i+3] * wBlock[rowOff+i+3]
					}
					for ; i < currentITileSize; i++ {
						sum += inTileBuf[i] * wBlock[rowOff+i]
					}
					output[s*outDim+o] = TOut(sum)
				}
			}
		}
	}
}

func mhaTiledProjectInt8[TIn Numeric, TOut Numeric](input []TIn, wData []int8, wStart, bStart int, output []TOut, inDim, outDim, seqLen int, scale float32, tileSize int) {
	wBlock := wData[wStart:]
	// NOTE: Biases are typically kept in Master (float32) for accuracy.
	// We handle it by casting or having a separate bias slice if the framework supports it.
	// In poly, WeightStore.Master is siempre available for biases if needed.
	// But WeightStore.GetActive only returns the active type.
	// To keep it clean, we'll use SimulatePrecision fallback for bias if bBlock is missing.
	
	// Local buffer for input tile
	inTileBuf := make([]float32, tileSize)

	for iTile := 0; iTile < inDim; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > inDim { iEnd = inDim }
		currentITileSize := iEnd - iTile

		for s := 0; s < seqLen; s++ {
			// Load input tile once
			for i := 0; i < currentITileSize; i++ {
				inTileBuf[i] = float32(input[s*inDim+iTile+i])
			}

			for oTile := 0; oTile < outDim; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outDim { oEnd = outDim }

				for o := oTile; o < oEnd; o++ {
					var sum float32
					if iTile == 0 {
						sum = 0 // Biases typically zero or in separate master weights
					} else {
						sum = float32(output[s*outDim+o])
					}
					
					rowOff := o * inDim + iTile
					// Unrolled dot product
					i := 0
					for ; i <= currentITileSize-4; i += 4 {
						sum += inTileBuf[i] * (float32(wBlock[rowOff+i]) * scale)
						sum += inTileBuf[i+1] * (float32(wBlock[rowOff+i+1]) * scale)
						sum += inTileBuf[i+2] * (float32(wBlock[rowOff+i+2]) * scale)
						sum += inTileBuf[i+3] * (float32(wBlock[rowOff+i+3]) * scale)
					}
					for ; i < currentITileSize; i++ {
						sum += inTileBuf[i] * (float32(wBlock[rowOff+i]) * scale)
					}
					output[s*outDim+o] = TOut(sum)
				}
			}
		}
	}
}

func mhaTiledProjectBinary[TIn Numeric, TOut Numeric](input []TIn, wData []int8, wStart, bStart int, output []TOut, inDim, outDim, seqLen int, scale float32, tileSize int) {
	wBlock := wData[wStart:]
	
	for oTile := 0; oTile < outDim; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outDim { oEnd = outDim }
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim { iEnd = inDim }

			for s := 0; s < seqLen; s++ {
				for o := oTile; o < oEnd; o++ {
					var sum float32
					if iTile == 0 {
						sum = 0
					} else {
						sum = float32(output[s*outDim+o])
					}
					for i := iTile; i < iEnd; i++ {
						if wBlock[o*inDim+i] > 0 {
							sum += float32(input[s*inDim+i]) * scale
						} else {
							sum -= float32(input[s*inDim+i]) * scale
						}
					}
					output[s*outDim+o] = TOut(sum)
				}
			}
		}
	}
}

func applyRoPETiled(Q, K []float32, seqLen, numHeads, numKVHeads, headDim, dModel, kvDim int, freqBase float64) {
	half := headDim / 2
	for s := 0; s < seqLen; s++ {
		for h := 0; h < numHeads; h++ {
			for d := 0; d < half; d++ {
				freq := 1.0 / math.Pow(freqBase, float64(2*d)/float64(headDim))
				angle := freq * float64(s)
				c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
				off := s*dModel + h*headDim + d
				v0, v1 := Q[off], Q[off+half]
				Q[off] = v0*c - v1*sVal
				Q[off+half] = v0*sVal + v1*c
			}
		}
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < half; d++ {
				freq := 1.0 / math.Pow(freqBase, float64(2*d)/float64(headDim))
				angle := freq * float64(s)
				c, sVal := float32(math.Cos(angle)), float32(math.Sin(angle))
				off := s*kvDim + h*headDim + d
				v0, v1 := K[off], K[off+half]
				K[off] = v0*c - v1*sVal
				K[off+half] = v0*sVal + v1*c
			}
		}
	}
}

func mhaTiledAttention(Q, KRows, VRows, attnOut []float32, seqLen, numHeads, numKVHeads, headDim, dModel, kvDim, kvOffset, msl, tileSize int) {
	headsPerKV := numHeads / numKVHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// SCRATCHPADS: Pre-allocate buffers to avoid GC churn
	// maxS, denom, and vAcc need to be tracked PER head/query for the current token(s)
	// Since seqLen is usually 1 during decoding, we focus on that.
	numQueries := seqLen * numHeads
	maxSArr := make([]float32, numQueries)
	denomArr := make([]float32, numQueries)
	vAccArr := make([]float32, numQueries*headDim)
	for i := range maxSArr { maxSArr[i] = -1e9 }

	tileScores := make([]float32, tileSize)

	// KV-FIRST TILING: Outer loops iterate over KV tiles to keep them in cache
	for kvH := 0; kvH < numKVHeads; kvH++ {
		hStart := kvH * headsPerKV
		hEnd := hStart + headsPerKV

		for kTile := 0; kTile <= kvOffset + seqLen - 1; kTile += tileSize {
			kEnd := kTile + tileSize
			if kEnd > kvOffset+seqLen { kEnd = kvOffset + seqLen }
			
			// Process each query in this GQA group
			for s := 0; s < seqLen; s++ {
				currentTotalPos := kvOffset + s
				// Only process tiles that are within the causal mask for this query
				if kTile > currentTotalPos { continue }
				
				tileKEnd := kEnd
				if tileKEnd > currentTotalPos + 1 { tileKEnd = currentTotalPos + 1 }
				actualTileSize := tileKEnd - kTile
				if actualTileSize <= 0 { continue }

				for h := hStart; h < hEnd; h++ {
					qIdx := s*numHeads + h
					qBase := s*dModel + h*headDim
					
					// 1. Compute scores for this KV tile
					tileMax := float32(-1e9)
					scores := tileScores[:actualTileSize]
					
					for kP := 0; kP < actualTileSize; kP++ {
						kIdx := (kTile + kP) % msl
						kBase := kIdx*kvDim + kvH*headDim
						
						var dot float32
						d := 0
						for ; d <= headDim-4; d += 4 {
							dot += Q[qBase+d] * KRows[kBase+d]
							dot += Q[qBase+d+1] * KRows[kBase+d+1]
							dot += Q[qBase+d+2] * KRows[kBase+d+2]
							dot += Q[qBase+d+3] * KRows[kBase+d+3]
						}
						for ; d < headDim; d++ {
							dot += Q[qBase+d] * KRows[kBase+d]
						}
						
						sVal := dot * scale
						scores[kP] = sVal
						if sVal > tileMax { tileMax = sVal }
					}

					// 2. Online Softmax Rescaling
					oldMax := maxSArr[qIdx]
					if tileMax > oldMax { maxSArr[qIdx] = tileMax }
					newMax := maxSArr[qIdx]
					
					expPrev := float32(math.Exp(float64(oldMax - newMax)))
					vAccBase := qIdx * headDim
					for d := 0; d < headDim; d++ { vAccArr[vAccBase+d] *= expPrev }
					denomArr[qIdx] *= expPrev

					// 3. Accumulate new tile
					var tileDenom float32
					for i := 0; i < actualTileSize; i++ {
						ev := float32(math.Exp(float64(scores[i] - newMax)))
						scores[i] = ev
						tileDenom += ev
					}

					for kP := 0; kP < actualTileSize; kP++ {
						vIdx := (kTile + kP) % msl
						vBase := vIdx*kvDim + kvH*headDim
						sVal := scores[kP]
						
						d := 0
						for ; d <= headDim-4; d += 4 {
							vAccArr[vAccBase+d] += sVal * VRows[vBase+d]
							vAccArr[vAccBase+d+1] += sVal * VRows[vBase+d+1]
							vAccArr[vAccBase+d+2] += sVal * VRows[vBase+d+2]
							vAccArr[vAccBase+d+3] += sVal * VRows[vBase+d+3]
						}
						for ; d < headDim; d++ {
							vAccArr[vAccBase+d] += sVal * VRows[vBase+d]
						}
					}
					denomArr[qIdx] += tileDenom
				}
			}
		}
	}

	// Final normalization and output writing
	for qIdx := 0; qIdx < numQueries; qIdx++ {
		s := qIdx / numHeads
		h := qIdx % numHeads
		d := denomArr[qIdx]
		if d > 0 {
			invDenom := 1.0 / d
			vAccBase := qIdx * headDim
			outBase := s*dModel + h*headDim
			for i := 0; i < headDim; i++ {
				attnOut[outBase+i] = vAccArr[vAccBase+i] * invDenom
			}
		}
	}
}
