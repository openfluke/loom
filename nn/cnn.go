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

// conv2DForwardCPU performs 2D convolution on CPU
// input shape: [batch][inChannels][height][width] (flattened)
// output shape: [batch][filters][outHeight][outWidth] (flattened)
// Returns: preActivation (before activation), postActivation (after activation)
func conv2DForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inH := config.InputHeight
	inW := config.InputWidth
	inC := config.InputChannels
	kSize := config.KernelSize
	stride := config.Stride
	padding := config.Padding
	filters := config.Filters
	outH := config.OutputHeight
	outW := config.OutputWidth

	// Output size: batch * filters * outH * outW
	outputSize := batchSize * filters * outH * outW
	preActivation := make([]float32, outputSize)
	postActivation := make([]float32, outputSize)

	// For each batch
	for b := 0; b < batchSize; b++ {
		// For each output filter
		for f := 0; f < filters; f++ {
			// For each output position
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := config.Bias[f]

					// Convolve over input channels
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								// Calculate input position
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding

								// Check bounds
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inputIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									kernelIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw
									sum += input[inputIdx] * config.Kernel[kernelIdx]
								}
							}
						}
					}

					// Store pre-activation (before activation function)
					outputIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow
					preActivation[outputIdx] = sum

					// Apply activation and store post-activation
					postActivation[outputIdx] = activateCPU(sum, config.Activation)
				}
			}
		}
	}

	// Notify observer if present
	if config.Observer != nil {
		notifyObserver(config, "forward", -1, input, postActivation, 0)
	}

	return preActivation, postActivation
}

// conv2DBackwardCPU computes gradients for 2D convolution on CPU
// gradOutput: gradient flowing back from next layer
// input: input from forward pass
// Returns: gradInput (gradient w.r.t. input), gradKernel, gradBias
func conv2DBackwardCPU(
	gradOutput []float32,
	input []float32,
	preActivation []float32,
	config *LayerConfig,
	batchSize int,
) (gradInput []float32, gradKernel []float32, gradBias []float32) {
	inH := config.InputHeight
	inW := config.InputWidth
	inC := config.InputChannels
	kSize := config.KernelSize
	stride := config.Stride
	padding := config.Padding
	filters := config.Filters
	outH := config.OutputHeight
	outW := config.OutputWidth

	// Initialize gradients
	gradInput = make([]float32, batchSize*inC*inH*inW)
	gradKernel = make([]float32, filters*inC*kSize*kSize)
	gradBias = make([]float32, filters)

	// For each batch
	for b := 0; b < batchSize; b++ {
		// For each output filter
		for f := 0; f < filters; f++ {
			// For each output position
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					outputIdx := b*filters*outH*outW + f*outH*outW + oh*outW + ow

					// Apply activation derivative
					preActIdx := outputIdx
					derivative := activateDerivativeCPU(preActivation[preActIdx], config.Activation)
					gradOut := gradOutput[outputIdx] * derivative

					// Accumulate bias gradient
					gradBias[f] += gradOut

					// Backprop through convolution
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kSize; kh++ {
							for kw := 0; kw < kSize; kw++ {
								ih := oh*stride + kh - padding
								iw := ow*stride + kw - padding

								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inputIdx := b*inC*inH*inW + ic*inH*inW + ih*inW + iw
									kernelIdx := f*inC*kSize*kSize + ic*kSize*kSize + kh*kSize + kw

									// Gradient w.r.t. input
									gradInput[inputIdx] += gradOut * config.Kernel[kernelIdx]

									// Gradient w.r.t. kernel
									gradKernel[kernelIdx] += gradOut * input[inputIdx]
								}
							}
						}
					}
				}
			}
		}
	}

	// Notify observer if present
	if config.Observer != nil {
		notifyObserver(config, "backward", -1, nil, gradInput, 0)
	}

	return gradInput, gradKernel, gradBias
}

// ReshapeTo2D converts flattened 1D data to 2D shape for convolution
// Assumes input is [batch][features] and converts to [batch][1][sqrt(features)][sqrt(features)]
func ReshapeTo2D(input []float32, batchSize int) ([]float32, int, int, int) {
	featuresPerBatch := len(input) / batchSize
	// Assume square images with 1 channel
	size := int(math.Sqrt(float64(featuresPerBatch)))
	channels := 1

	if size*size != featuresPerBatch {
		// Not a perfect square, try to find best dimensions
		// For now, just use the input as-is with 1x1 "images"
		size = featuresPerBatch
		channels = 1
	}

	return input, channels, size, size
}

// FlattenFrom2D converts 2D conv output back to 1D for dense layers
func FlattenFrom2D(input []float32, batchSize, channels, height, width int) []float32 {
	// Already flattened in our representation
	return input
}
