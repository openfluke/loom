package nn

import (
	"math"
	"math/rand"
)

// =============================================================================
// Generic Conv1D Implementation
// =============================================================================

// Conv1DForward performs 1D convolution for any numeric type.
// Input shape: [batch][inChannels][seqLen] (flattened)
// Output shape: [batch][filters][outLen] (flattened)
func Conv1DForward[T Numeric](
	input, kernel, bias *Tensor[T],
	seqLen, inChannels, kernelSize, stride, padding, filters, batchSize int,
	activation ActivationType,
) (preAct, postAct *Tensor[T]) {
	// Calculate output length
	outLen := (seqLen+2*padding-kernelSize)/stride + 1
	outputSize := batchSize * filters * outLen
	
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
			for o := 0; o < outLen; o++ {
				var sum T
				if f < biasLen {
					sum = bias.Data[f]
				}

				// Convolve over input channels and kernel
				for ic := 0; ic < inChannels; ic++ {
					for k := 0; k < kernelSize; k++ {
						inPos := o*stride + k - padding

						if inPos >= 0 && inPos < seqLen {
							inputIdx := b*inChannels*seqLen + ic*seqLen + inPos
							kernelIdx := f*inChannels*kernelSize + ic*kernelSize + k
							if inputIdx < inputLen && kernelIdx < kernelLen {
								sum += input.Data[inputIdx] * kernel.Data[kernelIdx]
							}
						}
					}
				}

				outputIdx := b*filters*outLen + f*outLen + o
				if outputIdx < outputSize {
					preAct.Data[outputIdx] = sum
					postAct.Data[outputIdx] = Activate(sum, activation)
				}
			}
		}
	}

	return preAct, postAct
}

// Conv1DBackward computes gradients for 1D convolution with any numeric type.
func Conv1DBackward[T Numeric](
	gradOutput, input, preActivation, kernel *Tensor[T],
	seqLen, inChannels, kernelSize, stride, padding, filters, batchSize int,
	activation ActivationType,
) (gradInput, gradKernel, gradBias *Tensor[T]) {
	outLen := (seqLen+2*padding-kernelSize)/stride + 1

	gradInput = NewTensor[T](batchSize * inChannels * seqLen)
	gradKernel = NewTensor[T](filters * inChannels * kernelSize)
	gradBias = NewTensor[T](filters)

	for b := 0; b < batchSize; b++ {
		for f := 0; f < filters; f++ {
			for o := 0; o < outLen; o++ {
				outputIdx := b*filters*outLen + f*outLen + o
				derivative := ActivateDerivative(preActivation.Data[outputIdx], activation)
				gradOut := gradOutput.Data[outputIdx] * derivative

				gradBias.Data[f] += gradOut

				for ic := 0; ic < inChannels; ic++ {
					for k := 0; k < kernelSize; k++ {
						inPos := o*stride + k - padding

						if inPos >= 0 && inPos < seqLen {
							inputIdx := b*inChannels*seqLen + ic*seqLen + inPos
							kernelIdx := f*inChannels*kernelSize + ic*kernelSize + k

							gradInput.Data[inputIdx] += gradOut * kernel.Data[kernelIdx]
							gradKernel.Data[kernelIdx] += gradOut * input.Data[inputIdx]
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

// InitConv1DLayer initializes a Conv1D layer with random weights
func InitConv1DLayer(
	seqLen, inChannels int,
	kernelSize, stride, padding, filters int,
	activation ActivationType,
) LayerConfig {
	// Calculate output length
	outLen := (seqLen+2*padding-kernelSize)/stride + 1

	// Initialize kernel weights (He initialization)
	kernelTotal := filters * inChannels * kernelSize
	kernel := make([]float32, kernelTotal)
	stddev := float32(math.Sqrt(2.0 / float64(inChannels*kernelSize)))

	for i := range kernel {
		kernel[i] = float32(rand.NormFloat64()) * stddev
	}

	// Initialize biases to zero
	bias := make([]float32, filters)

	return LayerConfig{
		Type:             LayerConv1D,
		Activation:       activation,
		Conv1DKernelSize: kernelSize,
		Conv1DStride:     stride,
		Conv1DPadding:    padding,
		Conv1DFilters:    filters,
		Kernel:           kernel,
		Bias:             bias,
		Conv1DInChannels: inChannels,
		// Store input size for computing output
		InputHeight: seqLen, // Reuse InputHeight for sequence length
		OutputHeight: outLen, // Reuse OutputHeight for output length
	}
}

// conv1DForwardCPU performs 1D convolution on CPU
func conv1DForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputT := NewTensorFromSlice(input, len(input))
	kernelT := NewTensorFromSlice(config.Kernel, len(config.Kernel))
	biasT := NewTensorFromSlice(config.Bias, len(config.Bias))

	// Use InputHeight as seqLen
	seqLen := config.InputHeight
	if seqLen <= 0 {
		// Try to infer from input size
		seqLen = len(input) / (config.Conv1DInChannels * batchSize)
	}

	preAct, postAct := Conv1DForward(
		inputT, kernelT, biasT,
		seqLen, config.Conv1DInChannels,
		config.Conv1DKernelSize, config.Conv1DStride, config.Conv1DPadding,
		config.Conv1DFilters, batchSize, config.Activation,
	)

	return preAct.Data, postAct.Data
}

// conv1DBackwardCPU computes gradients for 1D convolution on CPU
func conv1DBackwardCPU(
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

	seqLen := config.InputHeight
	if seqLen <= 0 {
		seqLen = len(input) / (config.Conv1DInChannels * batchSize)
	}

	gradInputT, gradKernelT, gradBiasT := Conv1DBackward(
		gradOutputT, inputT, preActT, kernelT,
		seqLen, config.Conv1DInChannels,
		config.Conv1DKernelSize, config.Conv1DStride, config.Conv1DPadding,
		config.Conv1DFilters, batchSize, config.Activation,
	)

	return gradInputT.Data, gradKernelT.Data, gradBiasT.Data
}
