package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

// =============================================================================
// CNN2 (2D Convolution) Polymorphic
// =============================================================================

// CNN2ForwardPolymorphic performs a forward pass through a 2D convolutional layer.
func CNN2ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryCNN2ForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return CNN2ForwardTiled(layer, input)
}

// CNN2BackwardPolymorphic calculates gradients for a 2D convolutional layer.
func CNN2BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if gi, gw, ok := tryCNN2BackwardSimd(layer, gradOutput, input, preAct); ok {
			return gi, gw
		}
	}
	return CNN2BackwardTiled(layer, gradOutput, input, preAct)
}

// CNN2ForwardTiled runs multi-core tiled CNN2 forward.
func CNN2ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	return cnn2ForwardTiledGenericParallel(layer, input)
}

// cnn2ForwardTiledGenericParallel provides a multi-core parallel forward pass.
func cnn2ForwardTiledGenericParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	inH, inW, inC := layer.InputHeight, layer.InputWidth, layer.InputChannels
	outH, outW, filters := layer.OutputHeight, layer.OutputWidth, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outH, outW)
	postAct = NewTensor[T](batchSize, filters, outH, outW)

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	wData := CastWeights[T](weights)

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
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
										sum += float64(input.Data[inIdx]) * float64(wData[kWIdx])
									}
								}
							}
						}
						outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
						preAct.Data[outIdx] = T(sum)
						postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
					}
				}
			}(b, f)
		}
	}
	wg.Wait()
	return preAct, postAct
}

// CNN2BackwardTiled implements multi-core tiled backward for CNN2.
func CNN2BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	return cnn2BackwardTiledGenericParallel(layer, gradOutput, input, preAct)
}

func cnn2BackwardTiledGenericParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
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
	wData := CastWeights[T](weights)

	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < inC; ic++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, ic int) {
				defer func() { <-sem; wg.Done() }()
				localSum := make([]float64, inH*inW)
				for ih := 0; ih < inH; ih++ {
					for iw := 0; iw < inW; iw++ {
						var sum float64
						for f := 0; f < filters; f++ {
							for kh := 0; kh < kSize; kh++ {
								for kw := 0; kw < kSize; kw++ {
									oh := ih + padding - kh
									ow := iw + padding - kw
									if oh >= 0 && oh%stride == 0 && ow >= 0 && ow%stride == 0 {
										oh /= stride
										ow /= stride
										if oh < outH && ow < outW {
											outIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
											gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
											kWIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
											sum += gOut * float64(wData[kWIdx])
										}
									}
								}
							}
						}
						localSum[ih*inW+iw] = sum
					}
				}
				mu.Lock()
				for i, v := range localSum {
					gi64[b*inC*inH*inW+ic*inH*inW+i] += v
				}
				mu.Unlock()
			}(b, ic)
		}
	}
	wg.Wait()

	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGW := make([]float64, inC*kSize*kSize)
			for b := 0; b < batchSize; b++ {
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
										localGW[ic*kSize*kSize+kh*kSize+kw] += gOut * float64(input.Data[inIdx])
									}
								}
							}
						}
					}
				}
			}
			mu.Lock()
			for i, v := range localGW {
				gw64[f*inC*kSize*kSize+i] += v
			}
			mu.Unlock()
		}(f)
	}
	wg.Wait()

	for i := range gradInput.Data {
		gradInput.Data[i] = T(gi64[i])
	}
	for i := range gradWeights.Data {
		gradWeights.Data[i] = T(gw64[i])
	}
	return gradInput, gradWeights
}
