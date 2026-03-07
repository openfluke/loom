package poly

import (
)

// DenseForwardPolymorphic performs a forward pass through a dense layer.
// It handles precision transitions (e.g., FP32 input to FP4 layer).
func DenseForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)

	// Simulation of QAT/Precision scaling
	scale := layer.WeightStore.Scale
	if scale == 0 {
		scale = 1.0
	}

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	// EXHAUSTIVE NATIVE FAST-PATHS
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					var sum float64
					for i := 0; i < inputSize; i++ {
						sum += float64(input.Data[b*inputSize+i]) * rawW[i*outputSize+o]
					}
					preAct.Data[b*outputSize+o] = T(sum)
					postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					var sum float32
					for i := 0; i < inputSize; i++ {
						sum += float32(input.Data[b*inputSize+i]) * rawW[i*outputSize+o]
					}
					preAct.Data[b*outputSize+o] = T(sum)
					postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					var sum int64
					for i := 0; i < inputSize; i++ {
						sum += int64(input.Data[b*inputSize+i]) * rawW[i*outputSize+o]
					}
					preAct.Data[b*outputSize+o] = T(sum)
					postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					var sum int32
					for i := 0; i < inputSize; i++ {
						sum += int32(input.Data[b*inputSize+i]) * rawW[i*outputSize+o]
					}
					preAct.Data[b*outputSize+o] = T(sum)
					postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					var sum int32
					for i := 0; i < inputSize; i++ {
						sum += int32(input.Data[b*inputSize+i]) * int32(rawW[i*outputSize+o])
					}
					preAct.Data[b*outputSize+o] = T(sum)
					postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					var sum int32
					for i := 0; i < inputSize; i++ {
						sum += int32(input.Data[b*inputSize+i]) * int32(rawW[i*outputSize+o])
					}
					preAct.Data[b*outputSize+o] = T(sum)
					postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
				}
			}
			return preAct, postAct
		}
	}

	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			var sum float32
			for i := 0; i < inputSize; i++ {
				val := float32(input.Data[b*inputSize+i])
				wVal := float32(wData[i*outputSize+o])
				
				wVal = SimulatePrecision(wVal, layer.DType, scale)
				
				sum += val * wVal
			}
			preAct.Data[b*outputSize+o] = T(sum)
			postAct.Data[b*outputSize+o] = Activate(T(sum), layer.Activation)
		}
	}

	return preAct, postAct
}

// DenseBackwardPolymorphic calculates gradients for the dense layer.
func DenseBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight

	gradInput = NewTensor[T](batchSize, inputSize)
	gradWeights = NewTensor[T](inputSize, outputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	// Local grad for pre-activation
	gradPre := make([]float32, batchSize*outputSize)
	for i := 0; i < len(gradOutput.Data); i++ {
		gradPre[i] = float32(gradOutput.Data[i]) * float32(ActivateDerivative(preAct.Data[i], layer.Activation))
	}

	// EXHAUSTIVE NATIVE BACKWARD FAST-PATHS
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					g := float64(gradPre[b*outputSize+o])
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[i*outputSize+o] += T(float64(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[i*outputSize+o] * g)
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					g := gradPre[b*outputSize+o]
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[i*outputSize+o] += T(float32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[i*outputSize+o] * g)
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					g := int64(gradPre[b*outputSize+o])
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[i*outputSize+o] += T(int64(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[i*outputSize+o] * g)
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					g := int32(gradPre[b*outputSize+o])
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[i*outputSize+o] += T(int32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[i*outputSize+o] * g)
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					g := int32(gradPre[b*outputSize+o])
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[i*outputSize+o] += T(int32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(int32(rawW[i*outputSize+o]) * g)
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outputSize; o++ {
					g := int32(gradPre[b*outputSize+o])
					for i := 0; i < inputSize; i++ {
						gradWeights.Data[i*outputSize+o] += T(int32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(int32(rawW[i*outputSize+o]) * g)
					}
				}
			}
			return gradInput, gradWeights
		}
	}

	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			g := gradPre[b*outputSize+o]
			for i := 0; i < inputSize; i++ {
				inVal := float32(input.Data[b*inputSize+i])
				wVal := float32(wData[i*outputSize+o])
				
				// STE Gradient
				gradWeights.Data[i*outputSize+o] += T(inVal * g)
				gradInput.Data[b*inputSize+i] += T(wVal * g)
			}
		}
	}

	return gradInput, gradWeights
}

