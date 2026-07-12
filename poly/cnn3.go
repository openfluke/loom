package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// =============================================================================
// CNN3 (3D Convolution) Polymorphic
// =============================================================================

// CNN3ForwardPolymorphic performs a forward pass through a 3D convolutional layer.
func CNN3ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryCNN3ForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return CNN3ForwardTiled(layer, input)
}

// CNN3BackwardPolymorphic calculates gradients for a 3D convolutional layer.
func CNN3BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if gi, gw, ok := tryCNN3BackwardSimd(layer, gradOutput, input, preAct); ok {
			return gi, gw
		}
	}
	return CNN3BackwardTiled(layer, gradOutput, input, preAct)
}

// CNN3ForwardTiled runs multi-core tiled CNN3 forward.
func CNN3ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return cnn3ForwardTiledGenericParallel(layer, input)
}

func cnn3ForwardTiledGenericParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inD, inH, inW, inC := layer.InputDepth, layer.InputHeight, layer.InputWidth, layer.InputChannels
	outD, outH, outW, filters := layer.OutputDepth, layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding
	tileSize := layer.GetCPUTileSize(layer.DType)
	if tileSize <= 0 {
		tileSize = 8
	}

	preAct = NewTensor[T](batchSize, filters, outD, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outD, outH, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

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
										var sum float64
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
															sum += float64(input.Data[ihInOffset+iw]) * float64(wData[ihWeightsOffset+kw])
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

// CNN3BackwardTiled implements multi-core tiled backward for CNN3.
func CNN3BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return cnn3BackwardTiledGenericParallel(layer, gradOutput, input, preAct)
}

func cnn3BackwardTiledGenericParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
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

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < inC; ic++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, ic int) {
				defer func() { <-sem; wg.Done() }()
				for id := 0; id < inD; id++ {
					for ih := 0; ih < inH; ih++ {
						for iw := 0; iw < inW; iw++ {
							var sum float64
							for f := 0; f < filters; f++ {
								for kd := 0; kd < kSize; kd++ {
									for kh := 0; kh < kSize; kh++ {
										for kw := 0; kw < kSize; kw++ {
											od := id + padding - kd
											oh := ih + padding - kh
											ow := iw + padding - kw
											if od >= 0 && od%stride == 0 && oh >= 0 && oh%stride == 0 && ow >= 0 && ow%stride == 0 {
												od /= stride
												oh /= stride
												ow /= stride
												if od < outD && oh < outH && ow < outW {
													outIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
													gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
													kWIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
													sum += gOut * float64(wData[kWIdx])
												}
											}
										}
									}
								}
							}
							gradInput.Data[b*inC*inD*inH*inW+ic*inD*inH*inW+id*inH*inW+ih*inW+iw] += T(sum)
						}
					}
				}
			}(b, ic)
		}
	}
	wg.Wait()

	gwDataArr := make([]float64, len(gradWeights.Data))
	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			for b := 0; b < batchSize; b++ {
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
												gwDataArr[kWIdx] += gOut * float64(input.Data[inIdx])
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

	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gwDataArr[i])
	}
	return gradInput, gradWeights
}
