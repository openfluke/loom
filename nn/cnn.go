package nn

import (
	"math"
	"math/rand"
)

// InitConv2DLayer initializes a Conv2D layer with random weights
func InitConv2DLayer(
	inputHeight, inputWidth, inputChannels int,
	kernelSize, stride, padding, filters int,
	activation ActivationType,
) LayerConfig {
	// Calculate output dimensions
	outputHeight := (inputHeight+2*padding-kernelSize)/stride + 1
	outputWidth := (inputWidth+2*padding-kernelSize)/stride + 1

	// Initialize kernel weights (He initialization)
	kernelTotal := filters * inputChannels * kernelSize * kernelSize
	kernel := make([]float32, kernelTotal)
	stddev := float32(math.Sqrt(2.0 / float64(inputChannels*kernelSize*kernelSize)))

	for i := range kernel {
		kernel[i] = float32(rand.NormFloat64()) * stddev
	}

	// Initialize biases to zero
	bias := make([]float32, filters)

	return LayerConfig{
		Type:          LayerConv2D,
		Activation:    activation,
		KernelSize:    kernelSize,
		Stride:        stride,
		Padding:       padding,
		Filters:       filters,
		Kernel:        kernel,
		Bias:          bias,
		InputHeight:   inputHeight,
		InputWidth:    inputWidth,
		InputChannels: inputChannels,
		OutputHeight:  outputHeight,
		OutputWidth:   outputWidth,
	}
}

// =============================================================================
// Generic Conv2D Implementation
// =============================================================================

// Conv2DForward performs 2D convolution for any numeric type.
// input shape: [batch][inChannels][height][width] (flattened)
// output shape: [batch][filters][outHeight][outWidth] (flattened)
func Conv2DForward[T Numeric](
	input, kernel, bias *Tensor[T],
	inH, inW, inC, kSize, stride, padding, filters, outH, outW, batchSize int,
	activation ActivationType,
) (preAct, postAct *Tensor[T]) {
	outputSize := batchSize * filters * outH * outW
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
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var sum T
					if f < biasLen {
						sum = bias.Data[f]
					}

					// Convolve over input channels
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding

								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inputIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									kernelIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
									if inputIdx < inputLen && kernelIdx < kernelLen {
										sum += input.Data[inputIdx] * kernel.Data[kernelIdx]
									}
								}
							}
						}
					}

					outputIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					if outputIdx < outputSize {
						preAct.Data[outputIdx] = sum
						postAct.Data[outputIdx] = Activate(sum, activation)
					}
				}
			}
		}
	}

	return preAct, postAct
}

// Conv2DBackward computes gradients for 2D convolution with any numeric type.
func Conv2DBackward[T Numeric](
	gradOutput, input, preActivation, kernel *Tensor[T],
	inH, inW, inC, kSize, stride, padding, filters, outH, outW, batchSize int,
	activation ActivationType,
) (gradInput, gradKernel, gradBias *Tensor[T]) {
	gradInput = NewTensor[T](batchSize * inC * inH * inW)
	gradKernel = NewTensor[T](filters * inC * kSize * kSize)
	gradBias = NewTensor[T](filters)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outputIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					derivative := ActivateDerivative(preActivation.Data[outputIdx], activation)
					gradOut := gradOutput.Data[outputIdx] * derivative

					gradBias.Data[f] += gradOut

					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding

								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inputIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									kernelIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw

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

	return gradInput, gradKernel, gradBias
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// conv2DForwardCPU performs 2D convolution on CPU
func conv2DForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputT := NewTensorFromSlice(input, len(input))
	kernelT := NewTensorFromSlice(config.Kernel, len(config.Kernel))
	biasT := NewTensorFromSlice(config.Bias, len(config.Bias))

	preAct, postAct := Conv2DForward(
		inputT, kernelT, biasT,
		config.InputHeight, config.InputWidth, config.InputChannels,
		config.KernelSize, config.Stride, config.Padding,
		config.Filters, config.OutputHeight, config.OutputWidth,
		batchSize, config.Activation,
	)

	return preAct.Data, postAct.Data
}

// conv2DBackwardCPU computes gradients for 2D convolution on CPU
func conv2DBackwardCPU(
	gradOutput []float32,
	input []float32,
	preActivation []float32,
	config *LayerConfig,
	batchSize int,
) (gradInput []float32, gradKernel []float32, gradBias []float32) {
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))
	inputT := NewTensorFromSlice(input, len(input))
	preActT := NewTensorFromSlice(preActivation, len(preActivation))
	kernelT := NewTensorFromSlice(config.Kernel, len(config.Kernel))

	gradInputT, gradKernelT, gradBiasT := Conv2DBackward(
		gradOutputT, inputT, preActT, kernelT,
		config.InputHeight, config.InputWidth, config.InputChannels,
		config.KernelSize, config.Stride, config.Padding,
		config.Filters, config.OutputHeight, config.OutputWidth,
		batchSize, config.Activation,
	)

	return gradInputT.Data, gradKernelT.Data, gradBiasT.Data
}

// ReshapeTo2D converts flattened 1D data to 2D shape for convolution
func ReshapeTo2D(input []float32, batchSize int) ([]float32, int, int, int) {
	featuresPerBatch := len(input) / batchSize
	size := int(math.Sqrt(float64(featuresPerBatch)))
	channels := 1

	if size*size != featuresPerBatch {
		size = featuresPerBatch
		channels = 1
	}

	return input, channels, size, size
}

// FlattenFrom2D converts 2D conv output back to 1D for dense layers
func FlattenFrom2D(input []float32, batchSize, channels, height, width int) []float32 {
	return input
}

