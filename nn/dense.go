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

// denseForwardCPU performs forward pass for dense layer
// input: [batchSize * inputSize]
// weights: [inputSize * outputSize]
// output: [batchSize * outputSize]
func denseForwardCPU(input []float32, config *LayerConfig, batchSize int) ([]float32, []float32) {
	inputSize := config.InputHeight   // Reused field
	outputSize := config.OutputHeight // Reused field
	weights := config.Kernel
	bias := config.Bias

	preAct := make([]float32, batchSize*outputSize)
	postAct := make([]float32, batchSize*outputSize)

	// Matrix multiplication: output = input @ weights + bias
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			sum := float32(0)
			for i := 0; i < inputSize; i++ {
				inputIdx := b*inputSize + i
				weightIdx := i*outputSize + o
				sum += input[inputIdx] * weights[weightIdx]
			}
			sum += bias[o]

			outIdx := b*outputSize + o
			preAct[outIdx] = sum
			postAct[outIdx] = activateCPU(sum, config.Activation)
		}
	}

	return preAct, postAct
}

// denseBackwardCPU performs backward pass for dense layer
func denseBackwardCPU(gradOutput, input, preAct []float32, config *LayerConfig, batchSize int) ([]float32, []float32, []float32) {
	inputSize := config.InputHeight
	outputSize := config.OutputHeight
	weights := config.Kernel

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

	gradInput := make([]float32, batchSize*inputSize)
	gradWeights := make([]float32, inputSize*outputSize)
	gradBias := make([]float32, outputSize)

	// Apply activation derivative
	gradPreAct := make([]float32, len(gradOutput))
	for i := 0; i < len(gradOutput); i++ {
		derivative := activateDerivativeCPU(preAct[i], config.Activation)
		gradPreAct[i] = gradOutput[i] * derivative
	}

	// Compute gradients
	for b := 0; b < batchSize; b++ {
		for o := 0; o < outputSize; o++ {
			outIdx := b*outputSize + o
			grad := gradPreAct[outIdx]

			// Gradient w.r.t bias
			gradBias[o] += grad

			// Gradient w.r.t weights and input
			for i := 0; i < inputSize; i++ {
				inputIdx := b*inputSize + i
				weightIdx := i*outputSize + o

				// Gradient w.r.t weights
				gradWeights[weightIdx] += input[inputIdx] * grad

				// Gradient w.r.t input
				gradInput[inputIdx] += weights[weightIdx] * grad
			}
		}
	}

	return gradInput, gradWeights, gradBias
}
