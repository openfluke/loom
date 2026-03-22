package poly

import (
	"runtime"
	"sync"
)

// =============================================================================
// CNN1 (1D Convolution) Polymorphic
// =============================================================================

// CNN1ForwardPolymorphic performs a forward pass through a 1D convolutional layer.
func CNN1ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseTiling && layer.GetCPUTileSize(layer.DType) > 0 {
		return CNN1ForwardTiled(layer, input)
	}
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

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
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum float64
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float64(input.Data[inIdx]) * rawW[kWIdx]
								}
							}
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum float32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float32(input.Data[inIdx]) * rawW[kWIdx]
								}
							}
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum float64
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float64(input.Data[inIdx]) * float64(rawW[kWIdx])
								}
							}
						}
						if scale != 1.0 {
							sum *= float64(scale)
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum float32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float32(input.Data[inIdx]) * float32(rawW[kWIdx])
								}
							}
						}
						if scale != 1.0 {
							sum *= scale
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum float32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float32(input.Data[inIdx]) * float32(rawW[kWIdx])
								}
							}
						}
						if scale != 1.0 {
							sum *= scale
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum float32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float32(input.Data[inIdx]) * float32(rawW[kWIdx])
								}
							}
						}
						if scale != 1.0 {
							sum *= scale
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
			return preAct, postAct
		}
	}

	// UNIVERSAL POLYMORPHIC FALLTHROUGH (Handling 21 Types)
	wData := CastWeights[T](weights)
	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				var sum float32
				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							kWIdx := f*inC*kSize + ic*kSize + k

							if bWeights, ok := weights.([]uint64); ok {
								isSet := (bWeights[kWIdx/64] >> (uint(kWIdx) % 64)) & 1
								if isSet != 0 {
									sum += float32(input.Data[inIdx])
								} else {
									sum -= float32(input.Data[inIdx])
								}
							} else if wData != nil {
								sum += float32(input.Data[inIdx]) * float32(wData[kWIdx])
							}
						}
					}
				}
				if scale != 1.0 {
					sum *= scale
				}
				outIdx := b*filters*outLen + f*outLen + o
				preAct.Data[outIdx] = T(sum)
				postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
			}
		}
	}
	return preAct, postAct
}

// CNN1BackwardPolymorphic calculates gradients for a 1D convolutional layer.
func CNN1BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.UseTiling && layer.GetCPUTileSize(layer.DType) > 0 {
		return CNN1BackwardTiled(layer, gradOutput, input, preAct)
	}
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[T](batchSize, inC, seqLen)
	gradWeights = NewTensor[T](filters, inC, kSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	// EXHAUSTIVE NATIVE FAST-PATHS
	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									gradInput.Data[inIdx] += T(gOut * rawW[kWIdx])
									gradWeights.Data[kWIdx] += T(gOut * float64(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									gradInput.Data[inIdx] += T(gOut * rawW[kWIdx])
									gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									gradInput.Data[inIdx] += T(gOut * float64(rawW[kWIdx]))
									gradWeights.Data[kWIdx] += T(gOut * float64(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									gradInput.Data[inIdx] += T(gOut * float32(rawW[kWIdx]))
									gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									gradInput.Data[inIdx] += T(gOut * float32(rawW[kWIdx]))
									gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									gradInput.Data[inIdx] += T(gOut * float32(rawW[kWIdx]))
									gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
			return gradInput, gradWeights
		}
	}
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				outIdx := b*filters*outLen + f*outLen + o
				gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))

				for ic := 0; ic < inC; ic++ {
					for k := 0; k < kSize; k++ {
						inPos := o*stride + k - padding
						if inPos >= 0 && inPos < seqLen {
							inIdx := b*inC*seqLen + ic*seqLen + inPos
							kWIdx := f*inC*kSize + ic*kSize + k

							gradInput.Data[inIdx] += T(gOut * float32(wData[kWIdx]))
							gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
						}
					}
				}
			}
		}
	}
	return gradInput, gradWeights
}

// CNN1ForwardTiled is the dispatcher: routes to parallel or single-core tiled based on EnableMultiCoreTiling.
func CNN1ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	scale := layer.WeightStore.Scale
	if scale == 0 {
		scale = 1.0
	}

	if layer.EnableMultiCoreTiling {
		switch w := weights.(type) {
		case []float32:
			return cnn1ForwardTiledGenericParallel[T, float32](layer, input, w, 1.0)
		case []int8:
			return cnn1ForwardTiledGenericParallel[T, int8](layer, input, w, scale)
		case []float64:
			return cnn1ForwardTiledF64Parallel[T](layer, input, w)
		case []int64:
			return cnn1ForwardTiledGenericParallel[T, int64](layer, input, w, scale)
		case []int32:
			return cnn1ForwardTiledGenericParallel[T, int32](layer, input, w, scale)
		case []int16:
			return cnn1ForwardTiledGenericParallel[T, int16](layer, input, w, scale)
		default:
			wData := CastWeights[T](weights)
			return cnn1ForwardTiledGenericParallel[T, T](layer, input, wData, scale)
		}
	}

	switch w := weights.(type) {
	case []float32:
		return cnn1ForwardTiledGeneric[T, float32](layer, input, w, 1.0)
	case []int8:
		return cnn1ForwardTiledGeneric[T, int8](layer, input, w, scale)
	case []float64:
		return cnn1ForwardTiledF64[T](layer, input, w)
	case []int64:
		return cnn1ForwardTiledGeneric[T, int64](layer, input, w, scale)
	case []int32:
		return cnn1ForwardTiledGeneric[T, int32](layer, input, w, scale)
	case []int16:
		return cnn1ForwardTiledGeneric[T, int16](layer, input, w, scale)
	default:
		wData := CastWeights[T](weights)
		return cnn1ForwardTiledGeneric[T, T](layer, input, wData, scale)
	}
}

// cnn1ForwardTiledGeneric is a single-core L1-tiled CNN1 forward for typed weight slices.
func cnn1ForwardTiledGeneric[T Numeric, W Numeric](layer *VolumetricLayer, input *Tensor[T], weights []W, scale float32) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 16
	}

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	for b := 0; b < batchSize; b++ {
		for oTile := 0; oTile < outLen; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outLen {
				oEnd = outLen
			}
			for fTile := 0; fTile < filters; fTile += tileSize {
				fEnd := fTile + tileSize
				if fEnd > filters {
					fEnd = filters
				}
				for o := oTile; o < oEnd; o++ {
					for f := fTile; f < fEnd; f++ {
						var sum float32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float32(input.Data[inIdx]) * float32(weights[kWIdx])
								}
							}
						}
						if scale != 1.0 {
							sum *= scale
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
					}
				}
			}
		}
		outBBase := b * filters * outLen
		for i := 0; i < filters*outLen; i++ {
			idx := outBBase + i
			postAct.Data[idx] = Activate(preAct.Data[idx], layer.Activation)
		}
	}
	return preAct, postAct
}

// cnn1ForwardTiledF64 is a single-core float64-accumulating L1-tiled forward for CNN1.
func cnn1ForwardTiledF64[T Numeric](layer *VolumetricLayer, input *Tensor[T], weights []float64) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 16
	}

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	for b := 0; b < batchSize; b++ {
		for oTile := 0; oTile < outLen; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outLen {
				oEnd = outLen
			}
			for fTile := 0; fTile < filters; fTile += tileSize {
				fEnd := fTile + tileSize
				if fEnd > filters {
					fEnd = filters
				}
				for o := oTile; o < oEnd; o++ {
					for f := fTile; f < fEnd; f++ {
						var sum float64
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += float64(input.Data[inIdx]) * weights[kWIdx]
								}
							}
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
					}
				}
			}
		}
		outBBase := b * filters * outLen
		for i := 0; i < filters*outLen; i++ {
			idx := outBBase + i
			postAct.Data[idx] = Activate(preAct.Data[idx], layer.Activation)
		}
	}
	return preAct, postAct
}

// cnn1ForwardTiledGenericParallel is a multi-core L1-tiled CNN1 forward.
func cnn1ForwardTiledGenericParallel[T Numeric, W Numeric](layer *VolumetricLayer, input *Tensor[T], weights []W, scale float32) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
				for o := 0; o < outLen; o++ {
					var sum float32
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								kWIdx := f*inC*kSize + ic*kSize + k
								sum += float32(input.Data[inIdx]) * float32(weights[kWIdx])
							}
						}
					}
					if scale != 1.0 {
						sum *= scale
					}
					outIdx := b*filters*outLen + f*outLen + o
					preAct.Data[outIdx] = T(sum)
					postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
				}
			}(b, f)
		}
	}
	wg.Wait()
	return preAct, postAct
}

// cnn1ForwardTiledF64Parallel is multi-core float64 tiled forward for CNN1.
func cnn1ForwardTiledF64Parallel[T Numeric](layer *VolumetricLayer, input *Tensor[T], weights []float64) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
				for o := 0; o < outLen; o++ {
					var sum float64
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								kWIdx := f*inC*kSize + ic*kSize + k
								sum += float64(input.Data[inIdx]) * weights[kWIdx]
							}
						}
					}
					outIdx := b*filters*outLen + f*outLen + o
					preAct.Data[outIdx] = T(sum)
					postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
				}
			}(b, f)
		}
	}
	wg.Wait()
	return preAct, postAct
}

// CNN1BackwardTiled implements a loop-blocked backward pass for CNN1.
func CNN1BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 16
	}

	gradInput = NewTensor[T](batchSize, inC, seqLen)
	gradWeights = NewTensor[T](filters, inC, kSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for fTile := 0; fTile < filters; fTile += tileSize {
			fEnd := fTile + tileSize
			if fEnd > filters {
				fEnd = filters
			}

			for oTile := 0; oTile < outLen; oTile += tileSize {
				oEnd := oTile + tileSize
				if oEnd > outLen {
					oEnd = outLen
				}

				for f := fTile; f < fEnd; f++ {
					for o := oTile; o < oEnd; o++ {
						outIdx := b*filters*outLen + f*outLen + o
						gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))

						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k

									gradInput.Data[inIdx] += T(gOut * float32(wData[kWIdx]))
									gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
								}
							}
						}
					}
				}
			}
		}
	}
	return gradInput, gradWeights
}

// CNN1BackwardTiledParallel is the multi-core backward for CNN1.
// DX pass: parallelise over (batch, input-channel) pairs.
// DW pass: parallelise over filters.
func CNN1BackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[T](batchSize, inC, seqLen)
	gradWeights = NewTensor[T](filters, inC, kSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	// DX pass — each (b, ic) goroutine accumulates gradInput[b, ic, :]
	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < inC; ic++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, ic int) {
				defer func() { <-sem; wg.Done() }()
				for inPos := 0; inPos < seqLen; inPos++ {
					var sum float32
					for f := 0; f < filters; f++ {
						for k := 0; k < kSize; k++ {
							o := inPos + padding - k
							if o >= 0 && o%stride == 0 {
								o /= stride
								if o < outLen {
									outIdx := b*filters*outLen + f*outLen + o
									gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += gOut * float32(wData[kWIdx])
								}
							}
						}
					}
					gradInput.Data[b*inC*seqLen+ic*seqLen+inPos] += T(sum)
				}
			}(b, ic)
		}
	}
	wg.Wait()

	// DW pass — each filter goroutine accumulates gradWeights[f, :, :]
	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outLen; o++ {
					outIdx := b*filters*outLen + f*outLen + o
					gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								kWIdx := f*inC*kSize + ic*kSize + k
								gradWeights.Data[kWIdx] += T(gOut * float32(input.Data[inIdx]))
							}
						}
					}
				}
			}
		}(f)
	}
	wg.Wait()

	return gradInput, gradWeights
}
