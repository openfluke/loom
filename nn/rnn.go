package nn

import (
	"math"
	"math/rand"
)

// InitRNNLayer initializes a Recurrent Neural Network layer with Xavier/Glorot initialization
// inputSize: size of input features
// hiddenSize: size of hidden state
// batchSize: batch size for processing
// seqLength: length of input sequences
func InitRNNLayer(inputSize, hiddenSize, batchSize, seqLength int) LayerConfig {
	config := LayerConfig{
		Type:         LayerRNN,
		Activation:   ActivationTanh, // RNN typically uses tanh
		RNNInputSize: inputSize,
		HiddenSize:   hiddenSize,
		SeqLength:    seqLength,
	}

	// Xavier/Glorot initialization for input-to-hidden weights
	// WeightIH: [hiddenSize x inputSize]
	stdIH := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	config.WeightIH = make([]float32, hiddenSize*inputSize)
	for i := range config.WeightIH {
		config.WeightIH[i] = float32(rand.NormFloat64() * stdIH)
	}

	// Xavier/Glorot initialization for hidden-to-hidden weights
	// WeightHH: [hiddenSize x hiddenSize]
	stdHH := math.Sqrt(2.0 / float64(hiddenSize+hiddenSize))
	config.WeightHH = make([]float32, hiddenSize*hiddenSize)
	for i := range config.WeightHH {
		config.WeightHH[i] = float32(rand.NormFloat64() * stdHH)
	}

	// Bias initialization (zeros)
	config.BiasH = make([]float32, hiddenSize)

	return config
}

// =============================================================================
// Generic RNN Implementation
// =============================================================================

// RNNForward performs forward pass for RNN layer with any numeric type.
// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
func RNNForward[T Numeric](
	input, weightIH, weightHH, biasH *Tensor[T],
	batchSize, seqLength, inputSize, hiddenSize int,
) (output, hiddenStates *Tensor[T]) {
	output = NewTensor[T](batchSize * seqLength * hiddenSize)
	hiddenStates = NewTensor[T](batchSize * (seqLength + 1) * hiddenSize)

	for t := 0; t < seqLength; t++ {
		for b := 0; b < batchSize; b++ {
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize

			for h := 0; h < hiddenSize; h++ {
				sum := float64(biasH.Data[h])

				// W_ih @ x_t
				for i := 0; i < inputSize; i++ {
					sum += float64(weightIH.Data[h*inputSize+i]) * float64(input.Data[inputIdx+i])
				}

				// W_hh @ h_{t-1}
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					sum += float64(weightHH.Data[h*hiddenSize+hPrev]) * float64(hiddenStates.Data[prevHiddenIdx+hPrev])
				}

				// Apply tanh activation
				hiddenStates.Data[currHiddenIdx+h] = T(math.Tanh(sum))
			}

			// Copy to output
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			for h := 0; h < hiddenSize; h++ {
				output.Data[outputIdx+h] = hiddenStates.Data[currHiddenIdx+h]
			}
		}
	}

	return output, hiddenStates
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// rnnForwardCPU performs forward pass for RNN layer
func rnnForwardCPU(config *LayerConfig, input []float32, batchSize, seqLength, inputSize, hiddenSize int) ([]float32, []float32) {
	inputT := NewTensorFromSlice(input, len(input))
	weightIHT := NewTensorFromSlice(config.WeightIH, len(config.WeightIH))
	weightHHT := NewTensorFromSlice(config.WeightHH, len(config.WeightHH))
	biasHT := NewTensorFromSlice(config.BiasH, len(config.BiasH))

	output, hiddenStates := RNNForward(inputT, weightIHT, weightHHT, biasHT, batchSize, seqLength, inputSize, hiddenSize)
	return output.Data, hiddenStates.Data
}

// rnnBackwardCPU performs backward pass for RNN layer using BPTT
// gradOutput: gradient from next layer [batchSize, seqLength, hiddenSize]
// input: original input [batchSize, seqLength, inputSize]
// hiddenStates: all hidden states from forward pass [batchSize, seqLength+1, hiddenSize]
// Returns: (gradInput, gradWeightIH, gradWeightHH, gradBiasH)
func rnnBackwardCPU(config *LayerConfig, gradOutput, input, hiddenStates []float32,
	batchSize, seqLength, inputSize, hiddenSize int) ([]float32, []float32, []float32, []float32) {

	// Gradients to return
	gradInput := make([]float32, batchSize*seqLength*inputSize)
	gradWeightIH := make([]float32, hiddenSize*inputSize)
	gradWeightHH := make([]float32, hiddenSize*hiddenSize)
	gradBiasH := make([]float32, hiddenSize)

	// Gradient of hidden state (accumulates across time)
	// [batchSize, hiddenSize]
	gradHidden := make([]float32, batchSize*hiddenSize)

	// Backpropagate through time (from last timestep to first)
	for t := seqLength - 1; t >= 0; t-- {
		for b := 0; b < batchSize; b++ {
			// Add gradient from output
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			gradHiddenIdx := b * hiddenSize

			for h := 0; h < hiddenSize; h++ {
				gradHidden[gradHiddenIdx+h] += gradOutput[outputIdx+h]
			}

			// Current and previous hidden states
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize

			// Gradient through tanh activation
			// d_tanh(x)/dx = 1 - tanh²(x)
			for h := 0; h < hiddenSize; h++ {
				hVal := hiddenStates[currHiddenIdx+h]
				tanhDeriv := 1.0 - hVal*hVal // 1 - tanh²(x)
				gradPreActivation := gradHidden[gradHiddenIdx+h] * tanhDeriv

				// Accumulate bias gradient
				gradBiasH[h] += gradPreActivation

				// Gradient w.r.t. input: W_ih^T @ grad
				for i := 0; i < inputSize; i++ {
					gradInput[inputIdx+i] += config.WeightIH[h*inputSize+i] * gradPreActivation
					// Accumulate weight gradient: grad ⊗ x_t
					gradWeightIH[h*inputSize+i] += gradPreActivation * input[inputIdx+i]
				}

				// Gradient w.r.t. previous hidden state: W_hh^T @ grad
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					gradHidden[b*hiddenSize+hPrev] += config.WeightHH[h*hiddenSize+hPrev] * gradPreActivation
					// Accumulate weight gradient: grad ⊗ h_{t-1}
					gradWeightHH[h*hiddenSize+hPrev] += gradPreActivation * hiddenStates[prevHiddenIdx+hPrev]
				}
			}

			// Reset gradHidden for this batch (it was propagated to previous timestep)
			// Actually, we need to keep accumulating for the batch, so we clear after processing all timesteps
		}
	}

	return gradInput, gradWeightIH, gradWeightHH, gradBiasH
}
