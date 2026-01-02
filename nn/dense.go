package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// InitDenseLayer initializes a dense (fully-connected) layer
func InitDenseLayer(inputSize, outputSize int, activation ActivationType) LayerConfig {
	// He initialization for weights
	stddev := float32(math.Sqrt(2.0 / float64(inputSize)))

	weights := make([]float32, inputSize*outputSize)
	for i := range weights {
		weights[i] = float32(rand.NormFloat64()) * stddev
	}

	bias := make([]float32, outputSize)
	// Biases initialized to zero

	return LayerConfig{
		Type:       LayerDense,
		Activation: activation,
		// Store dense layer params in Conv2D fields for now (reuse existing structure)
		InputHeight:  inputSize,  // Reuse as inputSize
		OutputHeight: outputSize, // Reuse as outputSize
		Kernel:       weights,    // Weight matrix [inputSize * outputSize]
		Bias:         bias,       // Bias vector [outputSize]
	}
}

// InitStitchLayer creates a linear dense layer to project dimensionality.
// It is effectively a Dense layer with linear activation.
func InitStitchLayer(inputSize, outputSize int) LayerConfig {
	cfg := InitDenseLayer(inputSize, outputSize, ActivationType(-1)) // Linear activation
	// We might want to initialize valid weights, InitDenseLayer already does He init.
	// That's fine for stitching.
	return cfg
}

// =============================================================================
// Generic Dense Layer Implementation
// =============================================================================

// DenseForward performs forward pass for dense layer with any numeric type.
// input: [batchSize * inputSize]
// weights: [inputSize * outputSize]
// output: [batchSize * outputSize]
func DenseForward[T Numeric](input *Tensor[T], weights, bias *Tensor[T], inputSize, outputSize, batchSize int, activation ActivationType) (preAct, postAct *Tensor[T]) {
	preAct = NewTensor[T](batchSize * outputSize)
	postAct = NewTensor[T](batchSize * outputSize)

	// Matrix multiplication: output = input @ weights + bias
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			var sum T
			for i := 0; i < inputSize; i++ {
				inputIdx := b*inputSize + i
				weightIdx := i*outputSize + o
				sum += input.Data[inputIdx] * weights.Data[weightIdx]
			}
			sum += bias.Data[o]

			outIdx := b*outputSize + o
			preAct.Data[outIdx] = sum
			postAct.Data[outIdx] = Activate(sum, activation)
		}
	}

	return preAct, postAct
}

// DenseBackward performs backward pass for dense layer with any numeric type.
func DenseBackward[T Numeric](gradOutput, input, preAct, weights *Tensor[T], inputSize, outputSize, batchSize int, activation ActivationType) (gradInput, gradWeights, gradBias *Tensor[T]) {
	gradInput = NewTensor[T](batchSize * inputSize)
	gradWeights = NewTensor[T](inputSize * outputSize)
	gradBias = NewTensor[T](outputSize)

	// Apply activation derivative
	gradPreAct := NewTensor[T](len(gradOutput.Data))
	for i := 0; i < len(gradOutput.Data); i++ {
		derivative := ActivateDerivative(preAct.Data[i], activation)
		gradPreAct.Data[i] = gradOutput.Data[i] * derivative
	}

	// Compute gradients
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			outIdx := b*outputSize + o
			grad := gradPreAct.Data[outIdx]

			// Gradient w.r.t bias
			gradBias.Data[o] += grad

			// Gradient w.r.t weights and input
			for i := 0; i < inputSize; i++ {
				inputIdx := b*inputSize + i
				weightIdx := i*outputSize + o

				// Gradient w.r.t weights
				gradWeights.Data[weightIdx] += input.Data[inputIdx] * grad

				// Gradient w.r.t input
				gradInput.Data[inputIdx] += weights.Data[weightIdx] * grad
			}
		}
	}

	return gradInput, gradWeights, gradBias
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// denseForwardCPU performs forward pass for dense layer
// input: [batchSize * inputSize]
// weights: [inputSize * outputSize]
// output: [batchSize * outputSize]
func denseForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputSize := config.InputHeight   // Reused field
	outputSize := config.OutputHeight // Reused field

	// Wrap slices into tensors
	inputTensor := NewTensorFromSlice(input, batchSize*inputSize)
	weightsTensor := NewTensorFromSlice(config.Kernel, inputSize*outputSize)
	biasTensor := NewTensorFromSlice(config.Bias, outputSize)

	// Call generic implementation
	preAct, postAct := DenseForward(inputTensor, weightsTensor, biasTensor, inputSize, outputSize, batchSize, config.Activation)

	return preAct.Data, postAct.Data
}

// denseBackwardCPU performs backward pass for dense layer
func denseBackwardCPU(gradOutput, input, preAct []float32, config *LayerConfig, batchSize int) ([]float32, []float32, []float32) {
	inputSize := config.InputHeight
	outputSize := config.OutputHeight

	// Validate sizes
	expectedInputLen := batchSize * inputSize
	expectedOutputLen := batchSize * outputSize

	if len(input) != expectedInputLen {
		fmt.Printf("[WARNING] Dense backward: input size mismatch: got %d, expected %d (batch=%d, inputSize=%d)\n",
			len(input), expectedInputLen, batchSize, inputSize)
		// Adjust batchSize based on actual input
		batchSize = len(input) / inputSize
	}

	if len(gradOutput) != expectedOutputLen {
		fmt.Printf("[WARNING] Dense backward: gradOutput size mismatch: got %d, expected %d (batch=%d, outputSize=%d)\n",
			len(gradOutput), expectedOutputLen, batchSize, outputSize)
	}

	if len(preAct) != expectedOutputLen {
		fmt.Printf("[WARNING] Dense backward: preAct size mismatch: got %d, expected %d (batch=%d, outputSize=%d)\n",
			len(preAct), expectedOutputLen, batchSize, outputSize)
		// Truncate preAct if it's too long
		if len(preAct) > expectedOutputLen {
			preAct = preAct[:expectedOutputLen]
		}
	}

	// Wrap slices into tensors
	gradOutputTensor := NewTensorFromSlice(gradOutput, len(gradOutput))
	inputTensor := NewTensorFromSlice(input, len(input))
	preActTensor := NewTensorFromSlice(preAct, len(preAct))
	weightsTensor := NewTensorFromSlice(config.Kernel, inputSize*outputSize)

	// Call generic implementation
	gradInputT, gradWeightsT, gradBiasT := DenseBackward(gradOutputTensor, inputTensor, preActTensor, weightsTensor, inputSize, outputSize, batchSize, config.Activation)

	return gradInputT.Data, gradWeightsT.Data, gradBiasT.Data
}

