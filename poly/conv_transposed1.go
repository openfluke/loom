package poly

// ConvTransposed1DForwardPolymorphic performs a forward pass through a 1D transposed convolutional layer.
func ConvTransposed1DForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inW, inC := layer.InputWidth, layer.InputChannels
	outW, filters := layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outW)
	postAct = NewTensor[T](batchSize, filters, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := float64(input.Data[b*inC*inW + ic*inW + iw])
						for f := 0; f < filters; f++ {
							for k := 0; k < kSize; k++ {
								ow := iw*stride - padding + k
								if ow >= 0 && ow < outW {
									outIdx := b*filters*outW + f*outW + ow
									kWIdx := f*inC*kSize + ic*kSize + k
									preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
								}
							}
						}
					}
				}
			}
			for i := range preAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := float32(input.Data[b*inC*inW + ic*inW + iw])
						for f := 0; f < filters; f++ {
							for k := 0; k < kSize; k++ {
								ow := iw*stride - padding + k
								if ow >= 0 && ow < outW {
									outIdx := b*filters*outW + f*outW + ow
									kWIdx := f*inC*kSize + ic*kSize + k
									preAct.Data[outIdx] += T(inputVal * float32(rawW[kWIdx]))
								}
							}
						}
					}
				}
			}
			for i := range preAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := float64(input.Data[b*inC*inW + ic*inW + iw])
						for f := 0; f < filters; f++ {
							for k := 0; k < kSize; k++ {
								ow := iw*stride - padding + k
								if ow >= 0 && ow < outW {
									outIdx := b*filters*outW + f*outW + ow
									kWIdx := f*inC*kSize + ic*kSize + k
									preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
								}
							}
						}
					}
				}
			}
			for i := range preAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := float64(input.Data[b*inC*inW + ic*inW + iw])
						for f := 0; f < filters; f++ {
							for k := 0; k < kSize; k++ {
								ow := iw*stride - padding + k
								if ow >= 0 && ow < outW {
									outIdx := b*filters*outW + f*outW + ow
									kWIdx := f*inC*kSize + ic*kSize + k
									preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
								}
							}
						}
					}
				}
			}
			for i := range preAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := float64(input.Data[b*inC*inW + ic*inW + iw])
						for f := 0; f < filters; f++ {
							for k := 0; k < kSize; k++ {
								ow := iw*stride - padding + k
								if ow >= 0 && ow < outW {
									outIdx := b*filters*outW + f*outW + ow
									kWIdx := f*inC*kSize + ic*kSize + k
									preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
								}
							}
						}
					}
				}
			}
			for i := range preAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := float64(input.Data[b*inC*inW + ic*inW + iw])
						for f := 0; f < filters; f++ {
							for k := 0; k < kSize; k++ {
								ow := iw*stride - padding + k
								if ow >= 0 && ow < outW {
									outIdx := b*filters*outW + f*outW + ow
									kWIdx := f*inC*kSize + ic*kSize + k
									preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
								}
							}
						}
					}
				}
			}
			for i := range preAct.Data {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
			return preAct, postAct
		}
	}
	return preAct, postAct
}

func ConvTransposed1DBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return NewTensor[T](input.Shape...), NewTensor[T](len(layer.WeightStore.Master))
}
