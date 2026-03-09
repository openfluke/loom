package poly



// =============================================================================
// CNN2 (2D Convolution) Polymorphic
// =============================================================================

// CNN2ForwardPolymorphic performs a forward pass through a 2D convolutional layer.
func CNN2ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
		return CNN2ForwardTiled(layer, input)
	}

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
									kWIdx := f*inC*kSize*inH*inW + ic*kSize*kSize + kh*kSize + kw

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
	if layer.UseTiling && layer.TileSize > 0 {
		return CNN2BackwardTiled(layer, gradOutput, input, preAct)
	}

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

// CNN2ForwardTiled implements a loop-blocked forward pass for CNN2.
func CNN2ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 16 }

	preAct = NewTensor[T](batchSize, filters, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outH, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		// Output tiles (Spatial Blocking)
		for ohTile := 0; ohTile < outH; ohTile += tileSize {
			ohEnd := ohTile + tileSize
			if ohEnd > outH { ohEnd = outH }
			
			for owTile := 0; owTile < outW; owTile += tileSize {
				owEnd := owTile + tileSize
				if owEnd > outW { owEnd = outW }

				// Filter tiles
				for fTile := 0; fTile < filters; fTile += tileSize {
					fEnd := fTile + tileSize
					if fEnd > filters { fEnd = filters }

					for oh := ohTile; oh < ohEnd; oh++ {
						for ow := owTile; ow < owEnd; ow++ {
							for f := fTile; f < fEnd; f++ {
								var sum float32
								for ic := 0; ic < inC; ic++ {
									for kh := 0; kh < kSize; kh++ {
										ih := oh*stride + kh - padding
										if ih < 0 || ih >= inH { continue }
										
										inBase := b*inC*inH*inW + ic*inH*inW + ih*inW
										kWBase := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize
										
										for kw := 0; kw < kSize; kw++ {
											iw := ow*stride + kw - padding
											if iw >= 0 && iw < inW {
												// OPTIMIZED INNER LOOP: No simulation/overhead
												sum += float32(input.Data[inBase + iw]) * float32(wData[kWBase + kw])
											}
										}
									}
								}
								outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
								preAct.Data[outIdx] += T(sum)
							}
						}
					}
				}
			}
		}
		// Final Activation pass (Cache-friendly)
		for i := range preAct.Data {
			postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
		}
	}
	return preAct, postAct
}

// CNN2BackwardTiled implements a loop-blocked backward pass for CNN2.
func CNN2BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 16 }

	gradInput = NewTensor[T](batchSize, inC, inH, inW)
	gradWeights = NewTensor[T](filters, inC, kSize, kSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for fTile := 0; fTile < filters; fTile += tileSize {
			fEnd := fTile + tileSize
			if fEnd > filters { fEnd = filters }
			
			for ohTile := 0; ohTile < outH; ohTile += tileSize {
				ohEnd := ohTile + tileSize
				if ohEnd > outH { ohEnd = outH }
				
				for owTile := 0; owTile < outW; owTile += tileSize {
					owEnd := owTile + tileSize
					if owEnd > outW { owEnd = outW }

					for icTile := 0; icTile < inC; icTile += tileSize {
						icEnd := icTile + tileSize
						if icEnd > inC { icEnd = inC }
						
						for f := fTile; f < fEnd; f++ {
							for oh := ohTile; oh < ohEnd; oh++ {
								for ow := owTile; ow < owEnd; ow++ {
									outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
									gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
									
									for ic := icTile; ic < icEnd; ic++ {
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
				}
			}
		}
	}
	return gradInput, gradWeights
}
