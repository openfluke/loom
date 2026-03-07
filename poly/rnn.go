package poly

import (
	"math"
)

// RNNForwardPolymorphic performs a forward pass through an RNN layer.
// It handles precision transitions and all 21 numerical types.
func RNNForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	preAct, postAct = NewTensor[T](batchSize, seqLength, hiddenSize), NewTensor[T](batchSize, seqLength, hiddenSize)

	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	ihSize, hhSize := hiddenSize*inputSize, hiddenSize*hiddenSize

	// EXHAUSTIVE NATIVE FAST-PATHS
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			wIH, wHH, bH := rawW[0:ihSize], rawW[ihSize:ihSize+hhSize], rawW[ihSize+hhSize:]
			for b := 0; b < batchSize; b++ {
				hPrev := make([]float64, hiddenSize)
				for t := 0; t < seqLength; t++ {
					for h := 0; h < hiddenSize; h++ {
						sum := bH[h]
						it := b*seqLength*inputSize + t*inputSize
						for i := 0; i < inputSize; i++ {
							sum += float64(input.Data[it+i]) * wIH[h*inputSize+i]
						}
						for hp := 0; hp < hiddenSize; hp++ {
							sum += hPrev[hp] * wHH[h*hiddenSize+hp]
						}
						preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(sum)
						hCurr := math.Tanh(sum)
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hCurr)
						hPrev[h] = hCurr
					}
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			wIH, wHH, bH := rawW[0:ihSize], rawW[ihSize:ihSize+hhSize], rawW[ihSize+hhSize:]
			for b := 0; b < batchSize; b++ {
				hPrev := make([]float32, hiddenSize)
				for t := 0; t < seqLength; t++ {
					for h := 0; h < hiddenSize; h++ {
						sum := bH[h]
						it := b*seqLength*inputSize + t*inputSize
						for i := 0; i < inputSize; i++ {
							sum += float32(input.Data[it+i]) * wIH[h*inputSize+i]
						}
						for hp := 0; hp < hiddenSize; hp++ {
							sum += hPrev[hp] * wHH[h*hiddenSize+hp]
						}
						preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(sum)
						hCurr := float32(math.Tanh(float64(sum)))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hCurr)
						hPrev[h] = hCurr
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			wIH, wHH, bH := rawW[0:ihSize], rawW[ihSize:ihSize+hhSize], rawW[ihSize+hhSize:]
			for b := 0; b < batchSize; b++ {
				hPrev := make([]int64, hiddenSize)
				for t := 0; t < seqLength; t++ {
					for h := 0; h < hiddenSize; h++ {
						sum := bH[h]
						it := b*seqLength*inputSize + t*inputSize
						for i := 0; i < inputSize; i++ {
							sum += int64(input.Data[it+i]) * wIH[h*inputSize+i]
						}
						for hp := 0; hp < hiddenSize; hp++ {
							sum += hPrev[hp] * wHH[h*hiddenSize+hp]
						}
						preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(sum)
						hCurr := int64(math.Tanh(float64(sum)))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hCurr)
						hPrev[h] = hCurr
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			wIH, wHH, bH := rawW[0:ihSize], rawW[ihSize:ihSize+hhSize], rawW[ihSize+hhSize:]
			for b := 0; b < batchSize; b++ {
				hPrev := make([]int32, hiddenSize)
				for t := 0; t < seqLength; t++ {
					for h := 0; h < hiddenSize; h++ {
						sum := bH[h]
						it := b*seqLength*inputSize + t*inputSize
						for i := 0; i < inputSize; i++ {
							sum += int32(input.Data[it+i]) * wIH[h*inputSize+i]
						}
						for hp := 0; hp < hiddenSize; hp++ {
							sum += hPrev[hp] * wHH[h*hiddenSize+hp]
						}
						preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(sum)
						hCurr := int32(math.Tanh(float64(sum)))
						postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hCurr)
						hPrev[h] = hCurr
					}
				}
			}
			return preAct, postAct
		}
	}

	// UNIVERSAL POLYMORPHIC FALLTHROUGH
	wData := CastWeights[T](weights)
	wIH, wHH, bH := wData[0:ihSize], wData[ihSize:ihSize+hhSize], wData[ihSize+hhSize:]
	for b := 0; b < batchSize; b++ {
		hPrev := make([]float32, hiddenSize)
		for t := 0; t < seqLength; t++ {
			for h := 0; h < hiddenSize; h++ {
				sum := SimulatePrecision(float32(bH[h]), layer.DType, scale)
				it := b*seqLength*inputSize+t*inputSize
				for i := 0; i < inputSize; i++ {
					sum += float32(input.Data[it+i]) * SimulatePrecision(float32(wIH[h*inputSize+i]), layer.DType, scale)
				}
				for hp := 0; hp < hiddenSize; hp++ {
					sum += hPrev[hp] * SimulatePrecision(float32(wHH[h*hiddenSize+hp]), layer.DType, scale)
				}
				preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(sum)
				hCurr := float32(math.Tanh(float64(sum)))
				postAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h] = T(hCurr)
				hPrev[h] = hCurr
			}
		}
	}
	return preAct, postAct
}

// RNNBackwardPolymorphic calculates gradients for the RNN layer using BPTT.
func RNNBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize, inputSize, hiddenSize, seqLength := input.Shape[0], layer.InputHeight, layer.OutputHeight, layer.SeqLength
	gradInput, gradWeights = NewTensor[T](batchSize, seqLength, inputSize), NewTensor[T](len(layer.WeightStore.Master))
	ihSize, hhSize := hiddenSize*inputSize, hiddenSize*hiddenSize

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	// EXHAUSTIVE NATIVE BACKWARD FAST-PATHS
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			wIH, wHH := rawW[0:ihSize], rawW[ihSize:ihSize+hhSize]
			for b := 0; b < batchSize; b++ {
				gH := make([]float64, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGH := make([]float64, hiddenSize)
					for h := 0; h < hiddenSize; h++ {
						gPre := (gH[h] + float64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])) * (1.0 - math.Pow(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h])), 2))
						gradWeights.Data[ihSize+hhSize+h] += T(gPre)
						it := b*seqLength*inputSize+t*inputSize
						for i := 0; i < inputSize; i++ {
							gradWeights.Data[h*inputSize+i] += T(gPre * float64(input.Data[it+i]))
							gradInput.Data[it+i] += T(wIH[h*inputSize+i] * gPre)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hPrev := float64(0); if t > 0 { hPrev = math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+(t-1)*hiddenSize+hp])) }
							gradWeights.Data[ihSize+h*hiddenSize+hp] += T(gPre * hPrev)
							nextGH[hp] += wHH[h*hiddenSize+hp] * gPre
						}
					}
					gH = nextGH
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			wIH, wHH := rawW[0:ihSize], rawW[ihSize:ihSize+hhSize]
			for b := 0; b < batchSize; b++ {
				gH := make([]float32, hiddenSize)
				for t := seqLength - 1; t >= 0; t-- {
					nextGH := make([]float32, hiddenSize)
					for h := 0; h < hiddenSize; h++ {
						gPre := (gH[h] + float32(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])) * (1.0 - float32(math.Pow(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h])), 2)))
						gradWeights.Data[ihSize+hhSize+h] += T(gPre)
						it := b*seqLength*inputSize+t*inputSize
						for i := 0; i < inputSize; i++ {
							gradWeights.Data[h*inputSize+i] += T(gPre * float32(input.Data[it+i]))
							gradInput.Data[it+i] += T(wIH[h*inputSize+i] * gPre)
						}
						for hp := 0; hp < hiddenSize; hp++ {
							hPrev := float32(0); if t > 0 { hPrev = float32(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+(t-1)*hiddenSize+hp]))) }
							gradWeights.Data[ihSize+h*hiddenSize+hp] += T(gPre * hPrev)
							nextGH[hp] += wHH[h*hiddenSize+hp] * gPre
						}
					}
					gH = nextGH
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64, DTypeInt32, DTypeUint32:
		wData := CastWeights[int64](weights); wIH, wHH := wData[0:ihSize], wData[ihSize:ihSize+hhSize]
		for b := 0; b < batchSize; b++ {
			gH := make([]int64, hiddenSize)
			for t := seqLength - 1; t >= 0; t-- {
				nextGH := make([]int64, hiddenSize)
				for h := 0; h < hiddenSize; h++ {
					hVal := int64(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h])))
					gPre := (gH[h] + int64(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])) * (1 - hVal*hVal)
					gradWeights.Data[ihSize+hhSize+h] += T(gPre)
					it := b*seqLength*inputSize+t*inputSize
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[h*inputSize+i] += T(gPre * int64(input.Data[it+i]))
						gradInput.Data[it+i] += T(wIH[h*inputSize+i] * gPre)
					}
					for hp := 0; hp < hiddenSize; hp++ {
						hPrev := int64(0); if t > 0 { hPrev = int64(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+(t-1)*hiddenSize+hp]))) }
						gradWeights.Data[ihSize+h*hiddenSize+hp] += T(gPre * hPrev)
						nextGH[hp] += wHH[h*hiddenSize+hp] * gPre
					}
				}
				gH = nextGH
			}
		}
		return gradInput, gradWeights
	}

	// UNIVERSAL FALLTHROUGH
	wData := CastWeights[T](weights); wIH, wHH := wData[0:ihSize], wData[ihSize:ihSize+hhSize]
	for b := 0; b < batchSize; b++ {
		gH := make([]float32, hiddenSize)
		for t := seqLength - 1; t >= 0; t-- {
			nextGH := make([]float32, hiddenSize)
			for h := 0; h < hiddenSize; h++ {
				hVal := float32(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+t*hiddenSize+h])))
				gPre := (gH[h] + float32(gradOutput.Data[b*seqLength*hiddenSize+t*hiddenSize+h])) * (1.0 - hVal*hVal)
				gradWeights.Data[ihSize+hhSize+h] += T(gPre)
				it := b*seqLength*inputSize+t*inputSize
				for i := 0; i < inputSize; i++ {
					gradWeights.Data[h*inputSize+i] += T(gPre * float32(input.Data[it+i]))
					gradInput.Data[it+i] += T(float32(wIH[h*inputSize+i]) * gPre)
				}
				for hp := 0; hp < hiddenSize; hp++ {
					hPrev := float32(0); if t > 0 { hPrev = float32(math.Tanh(float64(preAct.Data[b*seqLength*hiddenSize+(t-1)*hiddenSize+hp]))) }
					gradWeights.Data[ihSize+h*hiddenSize+hp] += T(gPre * hPrev)
					nextGH[hp] += float32(wHH[h*hiddenSize+hp]) * gPre
				}
			}
			gH = nextGH
		}
	}
	return gradInput, gradWeights
}
