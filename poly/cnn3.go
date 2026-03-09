package poly

import (
)

// =============================================================================
// CNN3 (3D Convolution) Polymorphic
// =============================================================================

// CNN3ForwardPolymorphic performs a forward pass through a 3D convolutional layer.
func CNN3ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
		return CNN3ForwardTiled(layer, input)
	}
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outD, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outD, outH, outW)

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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								var sum float64
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += float64(input.Data[inIdx]) * rawW[kWIdx]
												}
											}
										}
									}
								}
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								preAct.Data[outIdx] = T(sum)
								postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
							}
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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								var sum float32
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += float32(input.Data[inIdx]) * rawW[kWIdx]
												}
											}
										}
									}
								}
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								preAct.Data[outIdx] = T(sum)
								postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
							}
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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								var sum int64
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += int64(input.Data[inIdx]) * rawW[kWIdx]
												}
											}
										}
									}
								}
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								preAct.Data[outIdx] = T(sum)
								postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
							}
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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								var sum int32
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += int32(input.Data[inIdx]) * rawW[kWIdx]
												}
											}
										}
									}
								}
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								preAct.Data[outIdx] = T(sum)
								postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
							}
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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								var sum int32
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += int32(input.Data[inIdx]) * int32(rawW[kWIdx])
												}
											}
										}
									}
								}
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								preAct.Data[outIdx] = T(sum)
								postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
							}
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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								var sum int32
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += int32(input.Data[inIdx]) * int32(rawW[kWIdx])
												}
											}
										}
									}
								}
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								preAct.Data[outIdx] = T(sum)
								postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
							}
						}
					}
				}
			}
			return preAct, postAct
		}
	}

	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						var sum float32
						for ic := 0; ic < inC; ic++ {
							for kd := 0; kd < kSize; kd++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										id := od*stride + kd - padding
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw

											val := float32(input.Data[inIdx])
											var wVal float32
											if bWeights, ok := weights.([]uint64); ok {
												isSet := (bWeights[kWIdx/64] >> (uint(kWIdx) % 64)) & 1
												if isSet != 0 {
													wVal = scale
												} else {
													wVal = -scale
												}
											} else if wData != nil {
												wVal = float32(wData[kWIdx])
												// Precision Simulation
												wVal = SimulatePrecision(wVal, layer.DType, scale)
											}
											sum += val * wVal
										}
									}
								}
							}
						}
						outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}
		}
	}
	return preAct, postAct
}

// CNN3BackwardPolymorphic calculates gradients for a 3D convolutional layer.
func CNN3BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layer.UseTiling && layer.TileSize > 0 {
		return CNN3BackwardTiled(layer, gradOutput, input, preAct)
	}
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[T](batchSize, inC, inD, inH, inW)
	gradWeights = NewTensor[T](filters, inC, kSize, kSize, kSize)

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
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
				}
			}
			return gradInput, gradWeights
		}
	case DTypeFloat32:
		if rawW, ok := weights.([]float32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt64, DTypeUint64:
		if rawW, ok := weights.([]int64); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt32, DTypeUint32:
		if rawW, ok := weights.([]int32); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt16, DTypeUint16:
		if rawW, ok := weights.([]int16); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
				}
			}
			return gradInput, gradWeights
		}
	case DTypeInt8, DTypeUint8:
		if rawW, ok := weights.([]int8); ok {
			for b := 0; b < batchSize; b++ {
				for f := 0; f < filters; f++ {
					for od := 0; od < outD; od++ {
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
								gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
								for ic := 0; ic < inC; ic++ {
									for kd := 0; kd < kSize; kd++ {
										for kh := 0; kh < kSize; kh++ {
											for kw := 0; kw < kSize; kw++ {
												id := od*stride + kd - padding
												ih := oh*stride + kh - padding
												iw := ow*stride + kw - padding
												if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
													inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
				}
			}
			return gradInput, gradWeights
		}
	}

	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
						gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))

						for ic := 0; ic < inC; ic++ {
							for kd := 0; kd < kSize; kd++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										id := od*stride + kd - padding
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding
										if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
											kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw

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
	return gradInput, gradWeights
}

// CNN3ForwardTiled implements a loop-blocked forward pass for CNN3.
func CNN3ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}

	scale := layer.WeightStore.Scale
	if scale == 0 {
		scale = 1.0
	}

	switch w := weights.(type) {
	case []float32:
		return cnn3ForwardTiledGeneric[T, float32](layer, input, w, 1.0)
	case []int8:
		return cnn3ForwardTiledGeneric[T, int8](layer, input, w, scale)
	case []float64:
		return cnn3ForwardTiledGeneric[T, float64](layer, input, w, 1.0)
	case []int32:
		return cnn3ForwardTiledGeneric[T, int32](layer, input, w, scale)
	case []int16:
		return cnn3ForwardTiledGeneric[T, int16](layer, input, w, scale)
	case []uint64:
		// Specialized bit-packed binary kernel
		return cnn3ForwardTiledBinaryPacked[T](layer, input, w, scale)
	default:
		// Fallback for non-standard or packed types via CastWeights (the "old" way)
		wData := CastWeights[T](weights)
		return cnn3ForwardTiledGeneric[T, T](layer, input, wData, 1.0)
	}
}

// cnn3ForwardTiledGeneric provides the core loop-blocked logic for any weight storage type.
func cnn3ForwardTiledGeneric[T Numeric, W Numeric](layer *VolumetricLayer, input *Tensor[T], weights []W, scale float32) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 {
		tileSize = 8
	}

	preAct = NewTensor[T](batchSize, filters, outD, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outD, outH, outW)

	// Pre-calculate strides
	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW

	filtCStride := kSize * kSize * kSize
	filtDStride := kSize * kSize
	filtHStride := kSize

	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	for b := 0; b < batchSize; b++ {
		bInOffset := b * inC * inCStride
		bOutOffset := b * filters * outFStride

		// Spatial blocking (Depth/Height/Width)
		for odTile := 0; odTile < outD; odTile += tileSize {
			odEnd := odTile + tileSize
			if odEnd > outD { odEnd = outD }

			for ohTile := 0; ohTile < outH; ohTile += tileSize {
				ohEnd := ohTile + tileSize
				if ohEnd > outH { ohEnd = outH }

				for owTile := 0; owTile < outW; owTile += tileSize {
					owEnd := owTile + tileSize
					if owEnd > outW { owEnd = outW }

					// Filter tiling
					for fTile := 0; fTile < filters; fTile += tileSize {
						fEnd := fTile + tileSize
						if fEnd > filters { fEnd = filters }

						for od := odTile; od < odEnd; od++ {
							for oh := ohTile; oh < ohEnd; oh++ {
								for ow := owTile; ow < owEnd; ow++ {
									for f := fTile; f < fEnd; f++ {
										var sum float32
										fWeightsOffset := f * inC * filtCStride
										for ic := 0; ic < inC; ic++ {
											icInOffset := bInOffset + ic*inCStride
											icWeightsOffset := fWeightsOffset + ic*filtCStride

											for kd := 0; kd < kSize; kd++ {
												id := od*stride + kd - padding
												if id < 0 || id >= inD { continue }

												kdInOffset := icInOffset + id*inDStride
												kdWeightsOffset := icWeightsOffset + kd*filtDStride

												for kh := 0; kh < kSize; kh++ {
													ih := oh*stride + kh - padding
													if ih < 0 || ih >= inH { continue }

													inBase := kdInOffset + ih*inHStride
													kWBase := kdWeightsOffset + kh*filtHStride

													for kw := 0; kw < kSize; kw++ {
														iw := ow*stride + kw - padding
														if iw >= 0 && iw < inW {
															sum += float32(input.Data[inBase+iw]) * float32(weights[kWBase+kw])
														}
													}
												}
											}
										}
										outIdx := bOutOffset + f*outFStride + od*outDStride + oh*outHStride + ow
										if scale != 1.0 {
											sum *= scale
										}
										preAct.Data[outIdx] = T(sum)
									}
								}
							}
						}
					}
				}
			}
		}
		// Cache-friendly activation commit
		outBBase := b * filters * outFStride
		for i := 0; i < filters*outFStride; i++ {
			idx := outBBase + i
			postAct.Data[idx] = Activate(preAct.Data[idx], layer.Activation)
		}
	}
	return preAct, postAct
}

// CNN3BackwardTiled implements a loop-blocked backward pass for CNN3.
func CNN3BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 8 }

	gradInput = NewTensor[T](batchSize, inC, inD, inH, inW)
	gradWeights = NewTensor[T](filters, inC, kSize, kSize, kSize)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil { weights = layer.WeightStore.Master }
	wData := CastWeights[T](weights)

	for b := 0; b < batchSize; b++ {
		for fTile := 0; fTile < filters; fTile += tileSize {
			fEnd := fTile + tileSize
			if fEnd > filters { fEnd = filters }
			
			for odTile := 0; odTile < outD; odTile += tileSize {
				odEnd := odTile + tileSize
				if odEnd > outD { odEnd = outD }

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
								for od := odTile; od < odEnd; od++ {
									for oh := ohTile; oh < ohEnd; oh++ {
										for ow := owTile; ow < owEnd; ow++ {
											outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
											gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
											
											for ic := icTile; ic < icEnd; ic++ {
												for kd := 0; kd < kSize; kd++ {
													for kh := 0; kh < kSize; kh++ {
														for kw := 0; kw < kSize; kw++ {
															id := od*stride + kd - padding
															ih := oh*stride + kh - padding
															iw := ow*stride + kw - padding
															if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
																inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
																kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
																
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
			}
		}
	}
	return gradInput, gradWeights
}

// cnn3ForwardTiledBinaryPacked implement optimized logic for bit-packed binary weights.
func cnn3ForwardTiledBinaryPacked[T Numeric](layer *VolumetricLayer, input *Tensor[T], weights []uint64, scale float32) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 {
		tileSize = 8
	}

	preAct = NewTensor[T](batchSize, filters, outD, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outD, outH, outW)

	inCStride := inD * inH * inW
	inDStride := inH * inW
	inHStride := inW

	filtCStride := kSize * kSize * kSize
	filtDStride := kSize * kSize
	filtHStride := kSize

	outFStride := outD * outH * outW
	outDStride := outH * outW
	outHStride := outW

	for b := 0; b < batchSize; b++ {
		bInOffset := b * inC * inCStride
		bOutOffset := b * filters * outFStride

		for odTile := 0; odTile < outD; odTile += tileSize {
			odEnd := odTile + tileSize
			if odEnd > outD { odEnd = outD }

			for ohTile := 0; ohTile < outH; ohTile += tileSize {
				ohEnd := ohTile + tileSize
				if ohEnd > outH { ohEnd = outH }

				for owTile := 0; owTile < outW; owTile += tileSize {
					owEnd := owTile + tileSize
					if owEnd > outW { owEnd = outW }

					for fTile := 0; fTile < filters; fTile += tileSize {
						fEnd := fTile + tileSize
						if fEnd > filters { fEnd = filters }

						for od := odTile; od < odEnd; od++ {
							for oh := ohTile; oh < ohEnd; oh++ {
								for ow := owTile; ow < owEnd; ow++ {
									for f := fTile; f < fEnd; f++ {
										var sum float32
										fWeightsOffset := f * inC * filtCStride
										for ic := 0; ic < inC; ic++ {
											icInOffset := bInOffset + ic*inCStride
											icWeightsOffset := fWeightsOffset + ic*filtCStride

											for kd := 0; kd < kSize; kd++ {
												id := od*stride + kd - padding
												if id < 0 || id >= inD { continue }

												kdInOffset := icInOffset + id*inDStride
												kdWeightsOffset := icWeightsOffset + kd*filtDStride

												for kh := 0; kh < kSize; kh++ {
													ih := oh*stride + kh - padding
													if ih < 0 || ih >= inH { continue }

													inBase := kdInOffset + ih*inHStride
													kWBase := kdWeightsOffset + kh*filtHStride

													for kw := 0; kw < kSize; kw++ {
														iw := ow*stride + kw - padding
														if iw >= 0 && iw < inW {
															wIdx := kWBase + kw
															isSet := (weights[wIdx/64] >> (uint(wIdx) % 64)) & 1
															if isSet != 0 {
																sum += float32(input.Data[inBase+iw])
															} else {
																sum -= float32(input.Data[inBase+iw])
															}
														}
													}
												}
											}
										}
										outIdx := bOutOffset + f*outFStride + od*outDStride + oh*outHStride + ow
										if scale != 1.0 {
											sum *= scale
										}
										preAct.Data[outIdx] = T(sum)
									}
								}
							}
						}
					}
				}
			}
		}
		outBBase := b * filters * outFStride
		for i := 0; i < filters*outFStride; i++ {
			idx := outBBase + i
			postAct.Data[idx] = Activate(preAct.Data[idx], layer.Activation)
		}
	}
	return preAct, postAct
}
