package poly



// =============================================================================
// CNN1 (1D Convolution) Polymorphic
// =============================================================================

// CNN1ForwardPolymorphic performs a forward pass through a 1D convolutional layer.
func CNN1ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
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
						var sum int64
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += int64(input.Data[inIdx]) * rawW[kWIdx]
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
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum int32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += int32(input.Data[inIdx]) * rawW[kWIdx]
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
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum int32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += int32(input.Data[inIdx]) * int32(rawW[kWIdx])
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
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for o := 0; o < outLen; o++ {
						var sum int32
						for ic := 0; ic < inC; ic++ {
							for k := 0; k < kSize; k++ {
								inPos := o*stride + k - padding
								if inPos >= 0 && inPos < seqLen {
									inIdx := b*inC*seqLen + ic*seqLen + inPos
									kWIdx := f*inC*kSize + ic*kSize + k
									sum += int32(input.Data[inIdx]) * int32(rawW[kWIdx])
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
	}

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

							val := float32(input.Data[inIdx])
							wVal := float32(wData[kWIdx])

							wVal = SimulatePrecision(wVal, layer.DType, scale)
							sum += val * wVal
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

// CNN1BackwardPolymorphic calculates gradients for a 1D convolutional layer.
func CNN1BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
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

// CNN1ForwardTiled implements a loop-blocked forward pass for CNN1.
func CNN1ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 {
		tileSize = 16
	}

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		// Output tiles (Spatial Blocking)
		for oTile := 0; oTile < outLen; oTile += tileSize {
			oEnd := oTile + tileSize
			if oEnd > outLen {
				oEnd = outLen
			}

			// Filter tiles
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
									sum += float32(input.Data[inIdx]) * float32(wData[kWIdx])
								}
							}
						}
						outIdx := b*filters*outLen + f*outLen + o
						preAct.Data[outIdx] = T(sum)
					}
				}
			}
		}
		// Final Activation pass (Cache-friendly)
		outBBase := b * filters * outLen
		for i := 0; i < filters*outLen; i++ {
			idx := outBBase + i
			postAct.Data[idx] = Activate(preAct.Data[idx], layer.Activation)
		}
	}
	return preAct, postAct
}

// CNN1BackwardTiled implements a loop-blocked backward pass for CNN1.
func CNN1BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
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
