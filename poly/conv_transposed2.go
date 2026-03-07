package poly

// ConvTransposed2DForwardPolymorphic performs a forward pass through a 2D transposed convolutional layer.
func ConvTransposed2DForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outH, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }

	switch layer.DType {
	case DTypeFloat64:
		if rawW, ok := weights.([]float64); ok {
			for b := 0; b < batchSize; b++ {
				for ic := 0; ic < inC; ic++ {
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := float64(input.Data[b*inC*inH*inW + ic*inH*inW + ih*inW + iw])
							for f := 0; f < filters; f++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										oh := ih*stride - padding + kh
										ow := iw*stride - padding + kw
										if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
										}
									}
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
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := float32(input.Data[b*inC*inH*inW + ic*inH*inW + ih*inW + iw])
							for f := 0; f < filters; f++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										oh := ih*stride - padding + kh
										ow := iw*stride - padding + kw
										if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											preAct.Data[outIdx] += T(inputVal * float32(rawW[kWIdx]))
										}
									}
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
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := float64(input.Data[b*inC*inH*inW + ic*inH*inW + ih*inW + iw])
							for f := 0; f < filters; f++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										oh := ih*stride - padding + kh
										ow := iw*stride - padding + kw
										if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
										}
									}
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
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := float64(input.Data[b*inC*inH*inW + ic*inH*inW + ih*inW + iw])
							for f := 0; f < filters; f++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										oh := ih*stride - padding + kh
										ow := iw*stride - padding + kw
										if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
										}
									}
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
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := float64(input.Data[b*inC*inH*inW + ic*inH*inW + ih*inW + iw])
							for f := 0; f < filters; f++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										oh := ih*stride - padding + kh
										ow := iw*stride - padding + kw
										if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
										}
									}
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
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := float64(input.Data[b*inC*inH*inW + ic*inH*inW + ih*inW + iw])
							for f := 0; f < filters; f++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										oh := ih*stride - padding + kh
										ow := iw*stride - padding + kw
										if oh >= 0 && oh < outH && ow >= 0 && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											preAct.Data[outIdx] += T(inputVal * float64(rawW[kWIdx]))
										}
									}
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

func ConvTransposed2DBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return NewTensor[T](input.Shape...), NewTensor[T](len(layer.WeightStore.Master))
}
