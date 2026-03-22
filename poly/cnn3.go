package poly

import (
	"runtime"
	"sync"
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
													sum += float64(input.Data[inIdx]) * float64(rawW[kWIdx])
												}
											}
										}
									}
								}
								if scale != 1.0 {
									sum *= float64(scale)
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
													sum += float32(input.Data[inIdx]) * float32(rawW[kWIdx])
												}
											}
										}
									}
								}
								if scale != 1.0 {
									sum *= scale
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
													sum += float32(input.Data[inIdx]) * float32(rawW[kWIdx])
												}
											}
										}
									}
								}
								if scale != 1.0 {
									sum *= scale
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
													sum += float32(input.Data[inIdx]) * float32(rawW[kWIdx])
												}
											}
										}
									}
								}
								if scale != 1.0 {
									sum *= scale
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
							}
						}
						if scale != 1.0 {
							sum *= scale
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

	if layer.EnableMultiCoreTiling {
		switch w := weights.(type) {
		case []float32:
			return cnn3ForwardTiledGenericParallel[T, float32](layer, input, w, 1.0)
		case []int8:
			return cnn3ForwardTiledGenericParallel[T, int8](layer, input, w, scale)
		case []float64:
			return cnn3ForwardTiledF64Parallel[T](layer, input, w)
		case []int64:
			return cnn3ForwardTiledGenericParallel[T, int64](layer, input, w, scale)
		case []int32:
			return cnn3ForwardTiledGenericParallel[T, int32](layer, input, w, scale)
		case []int16:
			return cnn3ForwardTiledGenericParallel[T, int16](layer, input, w, scale)
		default:
			wData := CastWeights[T](weights)
			return cnn3ForwardTiledGenericParallel[T, T](layer, input, wData, scale)
		}
	}

	switch w := weights.(type) {
	case []float32:
		return cnn3ForwardTiledGeneric[T, float32](layer, input, w, 1.0)
	case []int8:
		return cnn3ForwardTiledGeneric[T, int8](layer, input, w, scale)
	case []float64:
		return cnn3ForwardTiledF64[T](layer, input, w)
	case []int64:
		return cnn3ForwardTiledGeneric[T, int64](layer, input, w, scale)
	case []int32:
		return cnn3ForwardTiledGeneric[T, int32](layer, input, w, scale)
	case []int16:
		return cnn3ForwardTiledGeneric[T, int16](layer, input, w, scale)
	case []uint64:
		return cnn3ForwardTiledBinaryPacked[T](layer, input, w, scale)
	default:
		wData := CastWeights[T](weights)
		return cnn3ForwardTiledGeneric[T, T](layer, input, wData, scale)
	}
}

// cnn3ForwardTiledGenericParallel provides a multi-core parallel loop-blocked logic.
func cnn3ForwardTiledGenericParallel[T Numeric, W Numeric](layer *VolumetricLayer, input *Tensor[T], weights []W, scale float32) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 {
		tileSize = 8 // Fallback if SyncCPU wasn't called
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

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	// Semaphore caps concurrent goroutines to numCPUs so we saturate all
	// cores without spawning more OS threads than there are physical cores.
	sem := make(chan struct{}, numCPUs)

	// One goroutine per filter — gives numCPUs-wide parallelism regardless
	// of tileSize. Each goroutine owns all spatial tiles for its filter.
	for b := 0; b < batchSize; b++ {
		bInOffset := b * inC * inCStride
		bOutOffset := b * filters * outFStride

		for f := 0; f < filters; f++ {
			sem <- struct{}{}
			wg.Add(1)
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
				fWeightsOffset := f * inC * filtCStride

				for odTile := 0; odTile < outD; odTile += tileSize {
					odEnd := odTile + tileSize
					if odEnd > outD {
						odEnd = outD
					}

					for ohTile := 0; ohTile < outH; ohTile += tileSize {
						ohEnd := ohTile + tileSize
						if ohEnd > outH {
							ohEnd = outH
						}

						for owTile := 0; owTile < outW; owTile += tileSize {
							owEnd := owTile + tileSize
							if owEnd > outW {
								owEnd = outW
							}

							for od := odTile; od < odEnd; od++ {
								for oh := ohTile; oh < ohEnd; oh++ {
									for ow := owTile; ow < owEnd; ow++ {
										var sum float32
										outIdx := bOutOffset + f*outFStride + od*outDStride + oh*outHStride + ow

										for ic := 0; ic < inC; ic++ {
											icInOffset := bInOffset + ic*inCStride
											icWeightsOffset := fWeightsOffset + ic*filtCStride

											for kd := 0; kd < kSize; kd++ {
												id := od*stride + kd - padding
												if id < 0 || id >= inD {
													continue
												}
												idInOffset := icInOffset + id*inDStride
												idWeightsOffset := icWeightsOffset + kd*filtDStride

												for kh := 0; kh < kSize; kh++ {
													ih := oh*stride + kh - padding
													if ih < 0 || ih >= inH {
														continue
													}
													ihInOffset := idInOffset + ih*inHStride
													ihWeightsOffset := idWeightsOffset + kh*filtHStride

													for kw := 0; kw < kSize; kw++ {
														iw := ow*stride + kw - padding
														if iw >= 0 && iw < inW {
															sum += float32(input.Data[ihInOffset+iw]) * float32(weights[ihWeightsOffset+kw])
														}
													}
												}
											}
										}

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
			}(b, f)
		}
	}
	wg.Wait()

	// Parallelize activation commit as well
	wg.Add(numCPUs)
	totalElements := batchSize * filters * outFStride
	chunkSize := (totalElements + numCPUs - 1) / numCPUs
	for c := 0; c < numCPUs; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > totalElements {
			end = totalElements
		}
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
		}(start, end)
	}
	wg.Wait()

	return preAct, postAct
}

// cnn3ForwardTiledF64Parallel is the float64-accumulating parallel tiled forward pass.
// Using float64 accumulation preserves precision for float64 weight convolutions.
func cnn3ForwardTiledF64Parallel[T Numeric](layer *VolumetricLayer, input *Tensor[T], weights []float64) (preAct, postAct *Tensor[T]) {
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

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	sem := make(chan struct{}, numCPUs)

	for b := 0; b < batchSize; b++ {
		bInOffset := b * inC * inCStride
		bOutOffset := b * filters * outFStride

		for f := 0; f < filters; f++ {
			sem <- struct{}{}
			wg.Add(1)
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
				fWeightsOffset := f * inC * filtCStride

				for odTile := 0; odTile < outD; odTile += tileSize {
					odEnd := odTile + tileSize
					if odEnd > outD { odEnd = outD }
					for ohTile := 0; ohTile < outH; ohTile += tileSize {
						ohEnd := ohTile + tileSize
						if ohEnd > outH { ohEnd = outH }
						for owTile := 0; owTile < outW; owTile += tileSize {
							owEnd := owTile + tileSize
							if owEnd > outW { owEnd = outW }
							for od := odTile; od < odEnd; od++ {
								for oh := ohTile; oh < ohEnd; oh++ {
									for ow := owTile; ow < owEnd; ow++ {
										var sum float64
										outIdx := bOutOffset + f*outFStride + od*outDStride + oh*outHStride + ow
										for ic := 0; ic < inC; ic++ {
											icInOffset := bInOffset + ic*inCStride
											icWeightsOffset := fWeightsOffset + ic*filtCStride
											for kd := 0; kd < kSize; kd++ {
												id := od*stride + kd - padding
												if id < 0 || id >= inD { continue }
												idInOffset := icInOffset + id*inDStride
												idWeightsOffset := icWeightsOffset + kd*filtDStride
												for kh := 0; kh < kSize; kh++ {
													ih := oh*stride + kh - padding
													if ih < 0 || ih >= inH { continue }
													ihInOffset := idInOffset + ih*inHStride
													ihWeightsOffset := idWeightsOffset + kh*filtHStride
													for kw := 0; kw < kSize; kw++ {
														iw := ow*stride + kw - padding
														if iw >= 0 && iw < inW {
															sum += float64(input.Data[ihInOffset+iw]) * weights[ihWeightsOffset+kw]
														}
													}
												}
											}
										}
										preAct.Data[outIdx] = T(sum)
									}
								}
							}
						}
					}
				}
			}(b, f)
		}
	}
	wg.Wait()

	numCPUs2 := runtime.NumCPU()
	var wg2 sync.WaitGroup
	wg2.Add(numCPUs2)
	totalElements := batchSize * filters * outD * outH * outW
	chunkSize := (totalElements + numCPUs2 - 1) / numCPUs2
	for c := 0; c < numCPUs2; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > totalElements { end = totalElements }
		go func(start, end int) {
			defer wg2.Done()
			for i := start; i < end; i++ {
				postAct.Data[i] = Activate(preAct.Data[i], layer.Activation)
			}
		}(start, end)
	}
	wg2.Wait()
	return preAct, postAct
}

// cnn3ForwardTiledF64 is the float64-accumulating single-core tiled forward pass.
func cnn3ForwardTiledF64[T Numeric](layer *VolumetricLayer, input *Tensor[T], weights []float64) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 8 }

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
						for f := fTile; f < fEnd; f++ {
							fWeightsOffset := f * inC * filtCStride
							for od := odTile; od < odEnd; od++ {
								for oh := ohTile; oh < ohEnd; oh++ {
									for ow := owTile; ow < owEnd; ow++ {
										var sum float64
										outIdx := bOutOffset + f*outFStride + od*outDStride + oh*outHStride + ow
										for ic := 0; ic < inC; ic++ {
											icInOffset := bInOffset + ic*inCStride
											icWeightsOffset := fWeightsOffset + ic*filtCStride
											for kd := 0; kd < kSize; kd++ {
												id := od*stride + kd - padding
												if id < 0 || id >= inD { continue }
												idInOffset := icInOffset + id*inDStride
												idWeightsOffset := icWeightsOffset + kd*filtDStride
												for kh := 0; kh < kSize; kh++ {
													ih := oh*stride + kh - padding
													if ih < 0 || ih >= inH { continue }
													ihInOffset := idInOffset + ih*inHStride
													ihWeightsOffset := idWeightsOffset + kh*filtHStride
													for kw := 0; kw < kSize; kw++ {
														iw := ow*stride + kw - padding
														if iw >= 0 && iw < inW {
															sum += float64(input.Data[ihInOffset+iw]) * weights[ihWeightsOffset+kw]
														}
													}
												}
											}
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

// cnn3ForwardTiledGeneric provides the core loop-blocked logic for any weight storage type.
func cnn3ForwardTiledGeneric[T Numeric, W Numeric](layer *VolumetricLayer, input *Tensor[T], weights []W, scale float32) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.TileSize
	if tileSize <= 0 {
		tileSize = 8 // Fallback if SyncToCPU wasn't called
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

					for f := fTile; f < fEnd; f++ {
						fWeightsOffset := f * inC * filtCStride
						for od := odTile; od < odEnd; od++ {
							for oh := ohTile; oh < ohEnd; oh++ {
								for ow := owTile; ow < owEnd; ow++ {
									var sum float32
									outIdx := bOutOffset + f*outFStride + od*outDStride + oh*outHStride + ow

									for ic := 0; ic < inC; ic++ {
										icInOffset := bInOffset + ic*inCStride
										icWeightsOffset := fWeightsOffset + ic*filtCStride

										for kd := 0; kd < kSize; kd++ {
											id := od*stride + kd - padding
											if id < 0 || id >= inD {
												continue
											}
											idInOffset := icInOffset + id*inDStride
											idWeightsOffset := icWeightsOffset + kd*filtDStride

											for kh := 0; kh < kSize; kh++ {
												ih := oh*stride + kh - padding
												if ih < 0 || ih >= inH {
													continue
												}
												ihInOffset := idInOffset + ih*inHStride
												ihWeightsOffset := idWeightsOffset + kh*filtHStride

												for kw := 0; kw < kSize; kw++ {
													iw := ow*stride + kw - padding
													if iw >= 0 && iw < inW {
														sum += float32(input.Data[ihInOffset+iw]) * float32(weights[ihWeightsOffset+kw])
													}
												}
											}
										}
									}
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

// CNN3BackwardTiledParallel is the multi-core parallel backward pass for CNN3.
// It matches CNN3BackwardTiled numerically but dispatches:
//   - DX (gradInput):    one goroutine per (batch, inputChannel) — no output conflicts.
//   - DW (gradWeights):  one goroutine per filter          — no output conflicts.
// Both passes are bounded by a runtime.NumCPU()-wide semaphore for core saturation.
// Scale is NOT applied (matching CNN3BackwardTiled and DispatchCNN3BackwardDX behaviour).
func CNN3BackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
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
	wData := CastWeights[T](weights)

	numCPUs := runtime.NumCPU()
	sem := make(chan struct{}, numCPUs)
	var wg sync.WaitGroup

	// --- DX pass: parallel over (batch, inputChannel) ---
	// Each (b, ic) goroutine computes all gradInput[b, ic, id, ih, iw] independently
	// by summing contributions from every filter and kernel position (gather pattern).
	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < inC; ic++ {
			sem <- struct{}{}
			wg.Add(1)
			go func(b, ic int) {
				defer func() { <-sem; wg.Done() }()
				for id := 0; id < inD; id++ {
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							var sum float32
							for f := 0; f < filters; f++ {
								for kd := 0; kd < kSize; kd++ {
									for kh := 0; kh < kSize; kh++ {
										for kw := 0; kw < kSize; kw++ {
											vd := id + padding - kd
											vh := ih + padding - kh
											vw := iw + padding - kw
											if vd >= 0 && vd%stride == 0 &&
												vh >= 0 && vh%stride == 0 &&
												vw >= 0 && vw%stride == 0 {
												od, oh, ow := vd/stride, vh/stride, vw/stride
												if od < outD && oh < outH && ow < outW {
													outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
													gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += gOut * float32(wData[kWIdx])
												}
											}
										}
									}
								}
							}
							inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
							gradInput.Data[inIdx] = T(sum)
						}
					}
				}
			}(b, ic)
		}
	}
	wg.Wait()

	// --- DW pass: parallel over filters ---
	// Each filter-f goroutine computes all gradWeights[f, ic, kd, kh, kw] independently
	// by summing over all (batch, output spatial) positions (scatter pattern per filter).
	for f := 0; f < filters; f++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			for b := 0; b < batchSize; b++ {
				for od := 0; od < outD; od++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
							gOut := float32(gradOutput.Data[outIdx]) * float32(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
							for ic := 0; ic < inC; ic++ {
								for kd := 0; kd < kSize; kd++ {
									for kh := 0; kh < kSize; kh++ {
										for kw := 0; kw < kSize; kw++ {
											id_ := od*stride + kd - padding
											ih_ := oh*stride + kh - padding
											iw_ := ow*stride + kw - padding
											if id_ >= 0 && id_ < inD && ih_ >= 0 && ih_ < inH && iw_ >= 0 && iw_ < inW {
												inIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id_*inH*inW + ih_*inW + iw_
												kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
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
		}(f)
	}
	wg.Wait()

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
		tileSize = 8 // Fallback if SyncToCPU wasn't called
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
