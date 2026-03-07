package poly



// =============================================================================
// CNN2 (2D Convolution) Polymorphic
// =============================================================================

// CNN2ForwardPolymorphic performs a forward pass through a 2D convolutional layer.
func CNN2ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outH, outW)

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
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							var sum float64
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += float64(input.Data[inIdx]) * rawW[kWIdx]
										}
									}
								}
							}
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							preAct.Data[outIdx] = T(sum)
							postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
						}
					}
				}
			}
			return preAct, postAct
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							var sum float32
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += float32(input.Data[inIdx]) * rawW[kWIdx]
										}
									}
								}
							}
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							preAct.Data[outIdx] = T(sum)
							postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
						}
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							var sum int64
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += int64(input.Data[inIdx]) * rawW[kWIdx]
										}
									}
								}
							}
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							preAct.Data[outIdx] = T(sum)
							postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
						}
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							var sum int32
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += int32(input.Data[inIdx]) * rawW[kWIdx]
										}
									}
								}
							}
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							preAct.Data[outIdx] = T(sum)
							postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
						}
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							var sum int32
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += int32(input.Data[inIdx]) * int32(rawW[kWIdx])
										}
									}
								}
							}
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							preAct.Data[outIdx] = T(sum)
							postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
						}
					}
				}
			}
			return preAct, postAct
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							var sum int32
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += int32(input.Data[inIdx]) * int32(rawW[kWIdx])
										}
									}
								}
							}
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							preAct.Data[outIdx] = T(sum)
							postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
						}
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
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var sum float32
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw

									val := float32(input.Data[inIdx])
									wVal := float32(wData[kWIdx])

									wVal = SimulatePrecision(wVal, layer.DType, scale)
									sum += val * wVal
								}
							}
						}
					}
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					preAct.Data[outIdx] = T(sum)
					postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
				}
			}
		}
	}
	return preAct, postAct
}

// CNN2BackwardPolymorphic calculates gradients for a 2D convolutional layer.
func CNN2BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[T](batchSize, inC, inH, inW)
	gradWeights = NewTensor[T](filters, inC, kSize, kSize)

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
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											gradInput.Data[inIdx] += T(gOut * rawW[kWIdx])
											gradWeights.Data[kWIdx] += T(gOut * float64(input.Data[inIdx]))
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
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											gradInput.Data[inIdx] += T(gOut * rawW[kWIdx])
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
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											gradInput.Data[inIdx] += T(gOut * float64(rawW[kWIdx]))
											gradWeights.Data[kWIdx] += T(gOut * float64(input.Data[inIdx]))
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
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											gradInput.Data[inIdx] += T(gOut * float32(rawW[kWIdx]))
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
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											gradInput.Data[inIdx] += T(gOut * float32(rawW[kWIdx]))
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
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
							gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											gradInput.Data[inIdx] += T(gOut * float32(rawW[kWIdx]))
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
	}

	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))

					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw

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
