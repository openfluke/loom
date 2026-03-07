package poly

import (
)

// =============================================================================
// CNN3 (3D Convolution) Polymorphic
// =============================================================================

// CNN3ForwardPolymorphic performs a forward pass through a 3D convolutional layer.
func CNN3ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
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
											wVal := float32(wData[kWIdx])

											// Precision Simulation
											wVal = SimulatePrecision(wVal, layer.DType, scale)
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
