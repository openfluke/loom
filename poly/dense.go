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

	if layer.UseTiling && layer.TileSize > 0 {
		return DenseForwardTiled(layer, input)
	}

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
						sum += float64(input.Data[b*inputSize+i]) * rawW[o*inputSize+i]
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
						sum += float32(input.Data[b*inputSize+i]) * rawW[o*inputSize+i]
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
						sum += int64(input.Data[b*inputSize+i]) * rawW[o*inputSize+i]
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
						sum += int32(input.Data[b*inputSize+i]) * rawW[o*inputSize+i]
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
						sum += int32(input.Data[b*inputSize+i]) * int32(rawW[o*inputSize+i])
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
						sum += int32(input.Data[b*inputSize+i]) * int32(rawW[o*inputSize+i])
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
				wVal := float32(wData[o*inputSize+i])
				
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
	gradWeights = NewTensor[T](outputSize, inputSize)

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
						gradWeights.Data[o*inputSize+i] += T(float64(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[o*inputSize+i] * g)
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
						gradWeights.Data[o*inputSize+i] += T(float32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[o*inputSize+i] * g)
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
						gradWeights.Data[o*inputSize+i] += T(int64(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[o*inputSize+i] * g)
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
						gradWeights.Data[o*inputSize+i] += T(int32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(rawW[o*inputSize+i] * g)
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
						gradWeights.Data[o*inputSize+i] += T(int32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(int32(rawW[o*inputSize+i]) * g)
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
						gradWeights.Data[o*inputSize+i] += T(int32(input.Data[b*inputSize+i]) * g)
						gradInput.Data[b*inputSize+i] += T(int32(rawW[o*inputSize+i]) * g)
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
				wVal := float32(wData[o*inputSize+i])
				
				// STE Gradient
				gradWeights.Data[o*inputSize+i] += T(inVal * g)
				gradInput.Data[b*inputSize+i] += T(wVal * g)
			}
		}
	}

	return gradInput, gradWeights
}


// DenseForwardTiled performs a tiled forward pass for the dense layer.
func DenseForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 32 }

	preAct = NewTensor[T](batchSize, outputSize)
	postAct = NewTensor[T](batchSize, outputSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	// Specialized fast-paths for tiling
	switch layer.DType {
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
		if rawW, ok := weights.([]float32); ok {
			denseForwardTiledFloat32(layer, input, preAct, rawW, tileSize)
			for i := range postAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8, DTypeFP8E4M3, DTypeFP8E5M2, DTypeInt4, DTypeUint4, DTypeFP4:
		if rawW, ok := weights.([]int8); ok {
			denseForwardTiledInt8(layer, input, preAct, rawW, tileSize)
			for i := range postAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeBinary:
		// Attempt to use packed binary weights if available, otherwise fallback to unpacked float32 path
		// For now, we use the unpacked int8 representation we established
		if rawW, ok := weights.([]int8); ok {
			denseForwardTiledBinary(layer, input, preAct, rawW, tileSize)
			for i := range postAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	}

	// Generic tiling fallback
	wData := CastWeights[T](weights)
	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }

	for oTile := 0; oTile < outputSize; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outputSize { oEnd = outputSize }
		for iTile := 0; iTile < inputSize; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inputSize { iEnd = inputSize }

			for o := oTile; o < oEnd; o++ {
				for b := 0; b < batchSize; b++ {
					sum := float32(preAct.Data[b*outputSize+o])
					for i := iTile; i < iEnd; i++ {
						inVal := float32(input.Data[b*inputSize+i])
						wVal := float32(wData[o*inputSize+i])
						wVal = SimulatePrecision(wVal, layer.DType, scale)
						sum += inVal * wVal
					}
					preAct.Data[b*outputSize+o] = T(sum)
				}
			}
		}
	}

	for i := range postAct.Data {
		postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
	}
	return preAct, postAct
}

func denseForwardTiledFloat32[T Numeric](layer *VolumetricLayer, input *Tensor[T], preAct *Tensor[T], weights []float32, tileSize int) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight

	// Local buffer for input tile to avoid redundant reads
	inTileBuf := make([]float32, tileSize)

	for iTile := 0; iTile < inputSize; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > inputSize { iEnd = inputSize }
		currentITileSize := iEnd - iTile

		for b := 0; b < batchSize; b++ {
			// Load input tile once
			for i := 0; i < currentITileSize; i++ {
				inTileBuf[i] = float32(input.Data[b*inputSize+iTile+i])
			}

			for oTile := 0; oTile < outputSize; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outputSize { oEnd = outputSize }

				for o := oTile; o < oEnd; o++ {
					var sum float32
					if iTile > 0 {
						sum = float32(preAct.Data[b*outputSize+o])
					}
					
					rowOff := o * inputSize + iTile
					// Unrolled dot product using buffered input
					i := 0
					for ; i <= currentITileSize-4; i += 4 {
						sum += inTileBuf[i] * weights[rowOff+i]
						sum += inTileBuf[i+1] * weights[rowOff+i+1]
						sum += inTileBuf[i+2] * weights[rowOff+i+2]
						sum += inTileBuf[i+3] * weights[rowOff+i+3]
					}
					for ; i < currentITileSize; i++ {
						sum += inTileBuf[i] * weights[rowOff+i]
					}
					preAct.Data[b*outputSize+o] = T(sum)
				}
			}
		}
	}
}

func denseForwardTiledInt8[T Numeric](layer *VolumetricLayer, input *Tensor[T], preAct *Tensor[T], weights []int8, tileSize int) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }

	// Local buffer for input tile
	inTileBuf := make([]float32, tileSize)

	for iTile := 0; iTile < inputSize; iTile += tileSize {
		iEnd := iTile + tileSize
		if iEnd > inputSize { iEnd = inputSize }
		currentITileSize := iEnd - iTile

		for b := 0; b < batchSize; b++ {
			// Load input tile once
			for i := 0; i < currentITileSize; i++ {
				inTileBuf[i] = float32(input.Data[b*inputSize+iTile+i])
			}

			for oTile := 0; oTile < outputSize; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outputSize { oEnd = outputSize }

				for o := oTile; o < oEnd; o++ {
					var sum float32
					if iTile > 0 {
						sum = float32(preAct.Data[b*outputSize+o])
					}
					
					rowOff := o * inputSize + iTile
					// Unrolled dot product using buffered input
					i := 0
					for ; i <= currentITileSize-4; i += 4 {
						sum += inTileBuf[i] * (float32(weights[rowOff+i]) * scale)
						sum += inTileBuf[i+1] * (float32(weights[rowOff+i+1]) * scale)
						sum += inTileBuf[i+2] * (float32(weights[rowOff+i+2]) * scale)
						sum += inTileBuf[i+3] * (float32(weights[rowOff+i+3]) * scale)
					}
					for ; i < currentITileSize; i++ {
						sum += inTileBuf[i] * (float32(weights[rowOff+i]) * scale)
					}
					preAct.Data[b*outputSize+o] = T(sum)
				}
			}
		}
	}
}

func denseForwardTiledBinary[T Numeric](layer *VolumetricLayer, input *Tensor[T], preAct *Tensor[T], weights []int8, tileSize int) {
	batchSize := input.Shape[0]
	inputSize := layer.InputHeight
	outputSize := layer.OutputHeight
	scale := layer.WeightStore.Scale
	if scale == 0 { scale = 1.0 }

	for oTile := 0; oTile < outputSize; oTile += tileSize {
		oEnd := oTile + tileSize
		if oEnd > outputSize { oEnd = outputSize }
		for iTile := 0; iTile < inputSize; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inputSize { iEnd = inputSize }

			for o := oTile; o < oEnd; o++ {
				for b := 0; b < batchSize; b++ {
					var sum float32
					if iTile > 0 {
						sum = float32(preAct.Data[b*outputSize+o])
					}
					for i := iTile; i < iEnd; i++ {
						if weights[o*inputSize+i] > 0 {
							sum += float32(input.Data[b*inputSize+i])
						} else {
							sum -= float32(input.Data[b*inputSize+i])
						}
					}
					// Only multiply by scale at the very end of the input dimension
					if iEnd == inputSize {
						preAct.Data[b*outputSize+o] = T(sum * scale)
					} else {
						preAct.Data[b*outputSize+o] = T(sum)
					}
				}
			}
		}
	}
}
