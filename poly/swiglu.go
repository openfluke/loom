package poly

import (
	"math"
)

// SwiGLUForwardPolymorphic performs SwiGLU gated activation: silu(gate) * up then down_proj.
func SwiGLUForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	
	// preAct stores [siluGate * up] intermediates for backward pass
	preAct = NewTensor[T](seqLen, intermediateSize)
	postAct = NewTensor[T](seqLen, inputSize)
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	
	gateBStart := 2 * inputSize * intermediateSize + intermediateSize * inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float64(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float64
					for j := 0; j < intermediateSize; j++ {
						sum += float64(preAct.Data[s*intermediateSize+j]) * float64(downW[i*intermediateSize+j])
					}
					sum += float64(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float32
					for j := 0; j < inputSize; j++ {
						inVal := float32(input.Data[s*inputSize+j])
						sumGate += inVal * float32(gateW[i*inputSize+j])
						sumUp   += inVal * float32(upW[i*inputSize+j])
					}
					sumGate += float32(gateB[i])
					sumUp   += float32(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float32(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float32
					for j := 0; j < intermediateSize; j++ {
						sum += float32(preAct.Data[s*intermediateSize+j]) * float32(downW[i*intermediateSize+j])
					}
					sum += float32(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float64(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float64
					for j := 0; j < intermediateSize; j++ {
						sum += float64(preAct.Data[s*intermediateSize+j]) * float64(downW[i*intermediateSize+j])
					}
					sum += float64(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float64(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float64
					for j := 0; j < intermediateSize; j++ {
						sum += float64(preAct.Data[s*intermediateSize+j]) * float64(downW[i*intermediateSize+j])
					}
					sum += float64(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float64(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float64
					for j := 0; j < intermediateSize; j++ {
						sum += float64(preAct.Data[s*intermediateSize+j]) * float64(downW[i*intermediateSize+j])
					}
					sum += float64(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]
			downB := rawW[downBStart : downBStart + inputSize]

			for s := 0; s < seqLen; s++ {
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					sig := 1.0 / (1.0 + math.Exp(-float64(sumGate)))
					silu := float64(float64(sumGate) * sig)
					preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
				}
			}
			
			for s := 0; s < seqLen; s++ {
				for i := 0; i < inputSize; i++ {
					var sum float64
					for j := 0; j < intermediateSize; j++ {
						sum += float64(preAct.Data[s*intermediateSize+j]) * float64(downW[i*intermediateSize+j])
					}
					sum += float64(downB[i])
					postAct.Data[s*inputSize+i] = T(sum)
				}
			}
			return preAct, postAct
		}
	}

	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }
	wData := CastWeights[float32](weights)
	
	gateW := wData[gateWStart : gateWStart + inputSize * intermediateSize]
	upW   := wData[upWStart : upWStart + inputSize * intermediateSize]
	downW := wData[downWStart : downWStart + intermediateSize * inputSize]
	
	gateB := wData[gateBStart : gateBStart + intermediateSize]
	upB   := wData[upBStart : upBStart + intermediateSize]
	downB := wData[downBStart : downBStart + inputSize]

	for s := 0; s < seqLen; s++ {
		for i := 0; i < intermediateSize; i++ {
			var sumGate, sumUp float32
			for j := 0; j < inputSize; j++ {
				inVal := float32(input.Data[s*inputSize+j])
				sumGate += inVal * SimulatePrecision(gateW[j*intermediateSize+i], layer.DType, scale)
				sumUp   += inVal * SimulatePrecision(upW[j*intermediateSize+i], layer.DType, scale)
			}
			sumGate += SimulatePrecision(gateB[i], layer.DType, scale)
			sumUp   += SimulatePrecision(upB[i], layer.DType, scale)
			
			sig := float32(1.0 / (1.0 + math.Exp(-float64(sumGate))))
			silu := sumGate * sig
			preAct.Data[s*intermediateSize+i] = T(silu * sumUp)
		}
	}
	
	for s := 0; s < seqLen; s++ {
		for i := 0; i < inputSize; i++ {
			var sum float32
			for j := 0; j < intermediateSize; j++ {
				sum += float32(preAct.Data[s*intermediateSize+j]) * SimulatePrecision(downW[j*inputSize+i], layer.DType, scale)
			}
			sum += SimulatePrecision(downB[i], layer.DType, scale)
			postAct.Data[s*inputSize+i] = T(sum)
		}
	}

	return preAct, postAct
}

// SwiGLUBackwardPolymorphic calculates gradients for SwiGLU.
func SwiGLUBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	inputSize, intermediateSize := layer.InputHeight, layer.OutputHeight
	seqLen := len(input.Data) / inputSize
	
	gradInput = NewTensor[T](input.Shape...)
	gradWeights = NewTensor[T](len(layer.WeightStore.Master))
	
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	
	gateWStart := 0
	upWStart := inputSize * intermediateSize
	downWStart := 2 * inputSize * intermediateSize
	
	gateBStart := 2 * inputSize * intermediateSize + intermediateSize * inputSize
	upBStart := gateBStart + intermediateSize
	downBStart := upBStart + intermediateSize

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float32
					for j := 0; j < inputSize; j++ {
						inVal := float32(input.Data[s*inputSize+j])
						sumGate += inVal * float32(gateW[i*inputSize+j])
						sumUp   += inVal * float32(upW[i*inputSize+j])
					}
					sumGate += float32(gateB[i])
					sumUp   += float32(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			gateW := rawW[gateWStart : gateWStart + inputSize * intermediateSize]
			upW   := rawW[upWStart : upWStart + inputSize * intermediateSize]
			downW := rawW[downWStart : downWStart + intermediateSize * inputSize]
			
			gateB := rawW[gateBStart : gateBStart + intermediateSize]
			upB   := rawW[upBStart : upBStart + intermediateSize]

			for s := 0; s < seqLen; s++ {
				gradInterRow := make([]float64, intermediateSize)
				for i := 0; i < inputSize; i++ {
					dy := float64(gradOutput.Data[s*inputSize+i])
					gradWeights.Data[downBStart+i] += T(dy)
					for j := 0; j < intermediateSize; j++ {
						act := float64(preAct.Data[s*intermediateSize+j])
						gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
						gradInterRow[j] += dy * float64(downW[j*inputSize+i])
					}
				}
				
				for i := 0; i < intermediateSize; i++ {
					var sumGate, sumUp float64
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						sumGate += inVal * float64(gateW[i*inputSize+j])
						sumUp   += inVal * float64(upW[i*inputSize+j])
					}
					sumGate += float64(gateB[i])
					sumUp   += float64(upB[i])
					
					x := float64(sumGate)
					sig := 1.0 / (1.0 + math.Exp(-x))
					silu := x * sig
					dSilu := sig * (1.0 + x*(1.0-sig))
					
					gi := gradInterRow[i]
					dUp := gi * silu
					dGate := gi * float64(sumUp) * dSilu
					
					gradWeights.Data[upBStart+i] += T(dUp)
					gradWeights.Data[gateBStart+i] += T(dGate)
					
					for j := 0; j < inputSize; j++ {
						inVal := float64(input.Data[s*inputSize+j])
						gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
						gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
						
						gradInput.Data[s*inputSize+j] += T(dUp * float64(upW[j*intermediateSize+i]) + dGate * float64(gateW[j*intermediateSize+i]))
					}
				}
			}
			return gradInput, gradWeights
		}
	}

	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }
	wData := CastWeights[float32](weights)

	gateW := wData[gateWStart : gateWStart + inputSize * intermediateSize]
	upW   := wData[upWStart : upWStart + inputSize * intermediateSize]
	downW := wData[downWStart : downWStart + intermediateSize * inputSize]
	
	gateB := wData[gateBStart : gateBStart + intermediateSize]
	upB   := wData[upBStart : upBStart + intermediateSize]

	for s := 0; s < seqLen; s++ {
		gradInterRow := make([]float64, intermediateSize)
		for i := 0; i < inputSize; i++ {
			dy := float64(gradOutput.Data[s*inputSize+i])
			gradWeights.Data[downBStart+i] += T(dy)
			for j := 0; j < intermediateSize; j++ {
				act := float64(preAct.Data[s*intermediateSize+j])
				gradWeights.Data[downWStart + j*inputSize+i] += T(act * dy)
				gradInterRow[j] += dy * float64(SimulatePrecision(downW[j*inputSize+i], layer.DType, scale))
			}
		}
		
		for i := 0; i < intermediateSize; i++ {
			var sumGate, sumUp float32
			for j := 0; j < inputSize; j++ {
				inVal := float32(input.Data[s*inputSize+j])
				sumGate += inVal * SimulatePrecision(gateW[j*intermediateSize+i], layer.DType, scale)
				sumUp   += inVal * SimulatePrecision(upW[j*intermediateSize+i], layer.DType, scale)
			}
			sumGate += SimulatePrecision(gateB[i], layer.DType, scale)
			sumUp   += SimulatePrecision(upB[i], layer.DType, scale)
			
			x := float64(sumGate)
			sig := 1.0 / (1.0 + math.Exp(-x))
			silu := x * sig
			dSilu := sig * (1.0 + x*(1.0-sig))
			
			gi := gradInterRow[i]
			dUp := gi * silu
			dGate := gi * float64(sumUp) * dSilu
			
			gradWeights.Data[upBStart+i] += T(dUp)
			gradWeights.Data[gateBStart+i] += T(dGate)
			
			for j := 0; j < inputSize; j++ {
				inVal := float64(input.Data[s*inputSize+j])
				gradWeights.Data[upWStart + j*intermediateSize+i] += T(inVal * dUp)
				gradWeights.Data[gateWStart + j*intermediateSize+i] += T(inVal * dGate)
				
				gradInput.Data[s*inputSize+j] += T(dUp * float64(SimulatePrecision(upW[j*intermediateSize+i], layer.DType, scale)) + 
												  dGate * float64(SimulatePrecision(gateW[j*intermediateSize+i], layer.DType, scale)))
			}
		}
	}

	return gradInput, gradWeights
}
