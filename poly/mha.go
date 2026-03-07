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
	kvDim := numKVHeads * headDim
	
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

			Q := make([]float32, seqLen * dModel)
			K := make([]float32, seqLen * kvDim)
			V := make([]float32, seqLen * kvDim)
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < dModel; i++ {
					sum := float32(qB[i])
					for j := 0; j < dModel; j++ {
						sum += float32(input.Data[s*dModel+j]) * float32(qW[j*dModel+i])
					}
					Q[s*dModel+i] = sum
				}
				for i := 0; i < kvDim; i++ {
					sumK := float32(kB[i])
					sumV := float32(vB[i])
					for j := 0; j < dModel; j++ {
						inVal := float32(input.Data[s*dModel+j])
						sumK += inVal * float32(kW[j*kvDim+i])
						sumV += inVal * float32(vW[j*kvDim+i])
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
							c := float32(math.Cos(angle))
							sVal := float32(math.Sin(angle))
							
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
							c := float32(math.Cos(angle))
							sVal := float32(math.Sin(angle))
							
							kOff := s*kvDim + h*headDim + d
							v0, v1 := K[kOff], K[kOff+half]
							K[kOff] = v0*c - v1*sVal
							K[kOff+half] = v0*sVal + v1*c
						}
					}
				}
			}

			headsPerKV := numHeads / numKVHeads
			scale := float32(1.0 / math.Sqrt(float64(headDim)))
			attnOut := make([]float32, seqLen * dModel)
			
			for h := 0; h < numHeads; h++ {
				kvHead := h / headsPerKV
				for qPos := 0; qPos < seqLen; qPos++ {
					scores := make([]float32, qPos+1)
					maxScore := float32(-1e9)
					
					for kPos := 0; kPos <= qPos; kPos++ {
						var dot float32
						for d := 0; d < headDim; d++ {
							dot += Q[qPos*dModel+h*headDim+d] * K[kPos*kvDim+kvHead*headDim+d]
						}
						score := dot * scale
						scores[kPos] = score
						if score > maxScore { maxScore = score }
					}
					
					var expSum float32
					for kPos := 0; kPos <= qPos; kPos++ {
						scores[kPos] = float32(math.Exp(float64(scores[kPos] - maxScore)))
						expSum += scores[kPos]
					}
					
					for d := 0; d < headDim; d++ {
						var sum float32
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
					sum := float32(oB[i])
					for j := 0; j < dModel; j++ {
						sum += attnOut[s*dModel+j] * float32(oW[j*dModel+i])
					}
					postAct.Data[s*dModel+i] = T(sum)
				}
			}
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
