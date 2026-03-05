package nn

import (
	"math"
	"math/rand"
)

// =============================================================================
// Generic Conv3D Implementation
// =============================================================================

// Conv3DForward performs 3D convolution for any numeric type.
// Input shape: [batch][inChannels][depth][height][width] (flattened)
// Output shape: [batch][filters][outDepth][outHeight][outWidth] (flattened)
func Conv3DForward[T Numeric](
	input, kernel, bias *Tensor[T],
	inD, inH, inW, inC, kSize, stride, padding, filters, outD, outH, outW, batchSize int,
	activation ActivationType,
) (preAct, postAct *Tensor[T]) {
	outputSize := batchSize * filters * outD * outH * outW
	preAct = NewTensor[T](outputSize)
	postAct = NewTensor[T](outputSize)

	// Safety checks
	if input == nil || len(input.Data) == 0 || kernel == nil || len(kernel.Data) == 0 {
		return preAct, postAct
	}

	inputLen := len(input.Data)
	kernelLen := len(kernel.Data)
	biasLen := 0
	if bias != nil {
		biasLen = len(bias.Data)
	}

	// For each batch
	for b := 0; b < batchSize; b++ {
		// For each output filter
		for f := 0; f < filters; f++ {
			// For each output position
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						var sum T
						if f < biasLen {
							sum = bias.Data[f]
						}

						// Convolve over input channels
						for ic := 0; ic < inC; ic++ {
							for kd := 0; kd < kSize; kd++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										id := od*stride + kd - padding
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding

										if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inputIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
											kernelIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw
											if inputIdx < inputLen && kernelIdx < kernelLen {
												sum += input.Data[inputIdx] * kernel.Data[kernelIdx]
											}
										}
									}
								}
							}
						}

						outputIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
						if outputIdx < outputSize {
							preAct.Data[outputIdx] = sum
							postAct.Data[outputIdx] = Activate(sum, activation)
						}
					}
				}
			}
		}
	}

	return preAct, postAct
}

// Conv3DBackward computes gradients for 3D convolution with any numeric type.
func Conv3DBackward[T Numeric](
	gradOutput, input, preActivation, kernel *Tensor[T],
	inD, inH, inW, inC, kSize, stride, padding, filters, outD, outH, outW, batchSize int,
	activation ActivationType,
) (gradInput, gradKernel, gradBias *Tensor[T]) {
	gradInput = NewTensor[T](batchSize * inC * inD * inH * inW)
	gradKernel = NewTensor[T](filters * inC * kSize * kSize * kSize)
	gradBias = NewTensor[T](filters)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for od := 0; od < outD; od++ {
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						outputIdx := b*filters*outD*outH*outW + f*outD*outH*outW + od*outH*outW + oh*outW + ow
						derivative := ActivateDerivative(preActivation.Data[outputIdx], activation)
						gradOut := gradOutput.Data[outputIdx] * derivative

						gradBias.Data[f] += gradOut

						for ic := 0; ic < inC; ic++ {
							for kd := 0; kd < kSize; kd++ {
								for kh := 0; kh < kSize; kh++ {
									for kw := 0; kw < kSize; kw++ {
										id := od*stride + kd - padding
										ih := oh*stride + kh - padding
										iw := ow*stride + kw - padding

										if id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW {
											inputIdx := b*inC*inD*inH*inW + ic*inD*inH*inW + id*inH*inW + ih*inW + iw
											kernelIdx := f*inC*kSize*kSize*kSize + ic*kSize*kSize*kSize + kd*kSize*kSize + kh*kSize + kw

											if inputIdx < len(gradInput.Data) && kernelIdx < len(gradKernel.Data) && outputIdx < len(gradOutput.Data) && inputIdx < len(input.Data) {
												gradInput.Data[inputIdx] += gradOut * kernel.Data[kernelIdx]
												gradKernel.Data[kernelIdx] += gradOut * input.Data[inputIdx]
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

	return gradInput, gradKernel, gradBias
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// InitConv3DLayer initializes a Conv3D layer with random weights
func InitConv3DLayer(
	inputDepth, inputHeight, inputWidth, inputChannels int,
	kernelSize, stride, padding, filters int,
	activation ActivationType,
) LayerConfig {
	if stride < 1 {
		stride = 1
	}

	// Calculate output dimensions
	outputDepth := (inputDepth+2*padding-kernelSize)/stride + 1
	outputHeight := (inputHeight+2*padding-kernelSize)/stride + 1
	outputWidth := (inputWidth+2*padding-kernelSize)/stride + 1

	// Initialize kernel weights (He initialization)
	kernelTotal := filters * inputChannels * kernelSize * kernelSize * kernelSize
	kernel := make([]float32, kernelTotal)
	stddev := float32(math.Sqrt(2.0 / float64(inputChannels*kernelSize*kernelSize*kernelSize)))

	for i := range kernel {
		kernel[i] = float32(rand.NormFloat64()) * stddev
	}

	// Initialize biases to zero
	bias := make([]float32, filters)

	return LayerConfig{
		Type:             LayerConv3D,
		Activation:       activation,
		Conv3DKernelSize: kernelSize,
		Conv3DStride:     stride,
		Conv3DPadding:    padding,
		Conv3DFilters:    filters,
		Conv3DKernel:     kernel,
		Conv3DBias:       bias,
		Conv3DInChannels: inputChannels,
		InputDepth:       inputDepth,
		InputHeight:      inputHeight,
		InputWidth:       inputWidth,
		OutputDepth:      outputDepth,
		OutputHeight:     outputHeight,
		OutputWidth:      outputWidth,
	}
}

// conv3DForwardCPU performs 3D convolution on CPU (for backward compatibility)
func conv3DForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputT := NewTensorFromSlice(input, len(input))
	kernelT := NewTensorFromSlice(config.Conv3DKernel, len(config.Conv3DKernel))
	biasT := NewTensorFromSlice(config.Conv3DBias, len(config.Conv3DBias))

	preAct, postAct := Conv3DForward(
		inputT, kernelT, biasT,
		config.InputDepth, config.InputHeight, config.InputWidth, config.Conv3DInChannels,
		config.Conv3DKernelSize, config.Conv3DStride, config.Conv3DPadding,
		config.Conv3DFilters, config.OutputDepth, config.OutputHeight, config.OutputWidth, batchSize,
		config.Activation,
	)

	return preAct.Data, postAct.Data
}

// conv3DBackwardCPU computes gradients for 3D convolution on CPU (for backward compatibility)
func conv3DBackwardCPU(
	gradOutput []float32,
	input []float32,
	preActivation []float32,
	config *LayerConfig,
	batchSize int,
) (gradInput []float32, gradKernel []float32, gradBias []float32) {
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))
	inputT := NewTensorFromSlice(input, len(input))
	preActT := NewTensorFromSlice(preActivation, len(preActivation))
	kernelT := NewTensorFromSlice(config.Conv3DKernel, len(config.Conv3DKernel))

	gradInputT, gradKernelT, gradBiasT := Conv3DBackward(
		gradOutputT, inputT, preActT, kernelT,
		config.InputDepth, config.InputHeight, config.InputWidth, config.Conv3DInChannels,
		config.Conv3DKernelSize, config.Conv3DStride, config.Conv3DPadding,
		config.Conv3DFilters, config.OutputDepth, config.OutputHeight, config.OutputWidth, batchSize,
		config.Activation,
	)

	return gradInputT.Data, gradKernelT.Data, gradBiasT.Data
}
