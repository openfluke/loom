package poly

import (
	"runtime"
	"sync"

	"github.com/openfluke/loom/poly/simd"
)

func useNativeQuantCNN1(layer *VolumetricLayer) bool {
	return useCNNNativeExact(layer)
}

func cnn1NativeWeights(layer *VolumetricLayer) ([]float32, float64) {
	if layer == nil || layer.WeightStore == nil {
		return nil, 1.0
	}
	if layer.DType == DTypeFP4 {
		if codes, ok := layer.WeightStore.GetNative(layer.DType).([]uint8); ok {
			return decodeFP4Codes(codes, layer.WeightStore.Scale), 1.0
		}
	}
	raw := CastWeights[float32](layer.WeightStore.GetNative(layer.DType))
	scale := float64(layer.WeightStore.Scale)
	if scale == 0 {
		scale = 1.0
	}
	return raw, scale
}

// =============================================================================
// CNN1 (1D Convolution) Polymorphic
// =============================================================================

// CNN1ForwardPolymorphic performs a forward pass through a 1D convolutional layer.
func CNN1ForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if useCNNNativeExact(layer) {
		return CNN1ForwardNativeExact(layer, input)
	}
	if useBitpackedCPUCNN1(layer) {
		return CNN1ForwardPackedCPU(layer, input)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if pre, post, ok := tryCNN1ForwardSimd(layer, input); ok {
			return pre, post
		}
	}
	return CNN1ForwardTiled(layer, input)
}

// CNN1BackwardPolymorphic calculates gradients for a 1D convolutional layer.
func CNN1BackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if useCNNNativeExact(layer) {
		return CNN1BackwardNativeExact(layer, gradOutput, input, preAct)
	}
	if useBitpackedCPUCNN1(layer) {
		return CNN1BackwardPackedCPU(layer, gradOutput, input, preAct)
	}
	if layerUseSimdForward(layer) && simd.SimdEnabled() {
		if gi, gw, ok := tryCNN1BackwardSimd(layer, gradOutput, input, preAct); ok {
			return gi, gw
		}
	}
	return CNN1BackwardTiled(layer, gradOutput, input, preAct)
}

// CNN1ForwardTiled runs the multi-core tiled CNN1 forward (non-packed weights).
func CNN1ForwardTiled[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if useBitpackedCPUCNN1(layer) {
		return cnn1ForwardPackedCPUParallel(layer, input)
	}
	return cnn1ForwardTiledGenericParallel(layer, input)
}

// cnn1ForwardTiledGenericParallel is a multi-core L1-tiled CNN1 forward.
func cnn1ForwardTiledGenericParallel[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	preAct = NewTensor[T](batchSize, filters, outLen)
	postAct = NewTensor[T](batchSize, filters, outLen)

	nativeQuant := useNativeQuantCNN1(layer)
	var rawW []float32
	weightScale := 1.0
	var wData []T
	if nativeQuant {
		rawW, weightScale = cnn1NativeWeights(layer)
	} else {
		weights := layer.WeightStore.GetActive(layer.DType)
		if weights == nil {
			weights = layer.WeightStore.Master
		}
		wData = CastWeights[T](weights)
	}

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, f int) {
				defer func() { <-sem; wg.Done() }()
				for o := 0; o < outLen; o++ {
					var sum float64
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								kWIdx := f*inC*kSize + ic*kSize + k
								if nativeQuant {
									sum += float64(input.Data[inIdx]) * float64(rawW[kWIdx])
								} else {
									sum += float64(input.Data[inIdx]) * float64(wData[kWIdx])
								}
							}
						}
					}
					if nativeQuant {
						sum *= weightScale
					}
					outIdx := b*filters*outLen + f*outLen + o
					preAct.Data[outIdx] = T(sum)
					postAct.Data[outIdx] = Activate(T(sum), layer.Activation)
				}
			}(b, f)
		}
	}
	wg.Wait()
	return preAct, postAct
}

// CNN1BackwardTiled implements the multi-core tiled backward pass for CNN1.
func CNN1BackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if useBitpackedCPUCNN1(layer) {
		return cnn1BackwardPackedCPUParallel(layer, gradOutput, input, preAct)
	}
	return CNN1BackwardTiledParallel(layer, gradOutput, input, preAct)
}

// CNN1BackwardTiledParallel is the multi-core backward for CNN1.
func CNN1BackwardTiledParallel[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	batchSize := input.Shape[0]
	seqLen, inC := layer.InputHeight, layer.InputChannels
	outLen, filters := layer.OutputHeight, layer.Filters
	kSize, stride, padding := layer.KernelSize, layer.Stride, layer.Padding

	gradInput = NewTensor[T](batchSize, inC, seqLen)
	gradWeights = NewTensor[T](filters, inC, kSize)

	nativeQuant := useNativeQuantCNN1(layer)
	var rawW []float32
	weightScale := 1.0
	var wData []T
	if nativeQuant {
		rawW, weightScale = cnn1NativeWeights(layer)
	} else {
		weights := layer.WeightStore.GetActive(layer.DType)
		if weights == nil {
			weights = layer.WeightStore.Master
		}
		wData = CastWeights[T](weights)
	}

	// High-precision buffers for bit-exact parity
	gi64 := make([]float64, len(gradInput.Data))
	gw64 := make([]float64, len(gradWeights.Data))
	var mu sync.Mutex

	sem := make(chan struct{}, runtime.NumCPU())
	var wg sync.WaitGroup

	// DX pass
	for b := 0; b < batchSize; b++ {
		for ic := 0; ic < inC; ic++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(b, ic int) {
				defer func() { <-sem; wg.Done() }()
				localSum := make([]float64, seqLen)
				for inPos := 0; inPos < seqLen; inPos++ {
					var sum float64
					for f := 0; f < filters; f++ {
						for k := 0; k < kSize; k++ {
							o := inPos + padding - k
							if o >= 0 && o%stride == 0 {
								o /= stride
								if o < outLen {
									outIdx := b*filters*outLen + f*outLen + o
									gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
									kWIdx := f*inC*kSize + ic*kSize + k
									if nativeQuant {
										sum += gOut * float64(rawW[kWIdx])
									} else {
										sum += gOut * float64(wData[kWIdx])
									}
								}
							}
						}
					}
					if nativeQuant {
						sum *= weightScale
					}
					localSum[inPos] = sum
				}
				mu.Lock()
				for i, v := range localSum {
					gi64[b*inC*seqLen+ic*seqLen+i] += v
				}
				mu.Unlock()
			}(b, ic)
		}
	}
	wg.Wait()

	// DW pass
	for f := 0; f < filters; f++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(f int) {
			defer func() { <-sem; wg.Done() }()
			localGW := make([]float64, inC*kSize)
			for b := 0; b < batchSize; b++ {
				for o := 0; o < outLen; o++ {
					outIdx := b*filters*outLen + f*outLen + o
					gOut := float64(gradOutput.Data[outIdx]) * float64(ActivateDerivative(preAct.Data[outIdx], layer.Activation))
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kSize; k++ {
							inPos := o*stride + k - padding
							if inPos >= 0 && inPos < seqLen {
								inIdx := b*inC*seqLen + ic*seqLen + inPos
								localGW[ic*kSize+k] += gOut * float64(input.Data[inIdx])
							}
						}
					}
				}
			}
			mu.Lock()
			for i, v := range localGW {
				gw64[f*inC*kSize+i] += v
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
