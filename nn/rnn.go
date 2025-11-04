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

// rnnForwardCPU performs forward pass for RNN layer
// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
// Returns: (output, hidden_states_all_timesteps) for backward pass
func rnnForwardCPU(config *LayerConfig, input []float32, batchSize, seqLength, inputSize, hiddenSize int) ([]float32, []float32) {
	// Output: [batchSize, seqLength, hiddenSize]
	output := make([]float32, batchSize*seqLength*hiddenSize)

	// Store all hidden states (including initial h_0=0) for backward pass
	// [batchSize, seqLength+1, hiddenSize]
	hiddenStates := make([]float32, batchSize*(seqLength+1)*hiddenSize)
	// h_0 is already zeros (implicit initialization)

	// Process each timestep
	for t := 0; t < seqLength; t++ {
		for b := 0; b < batchSize; b++ {
			// Get previous hidden state h_{t-1}
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize

			// Current hidden state h_t
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize

			// Input at timestep t: x_t
			inputIdx := b*seqLength*inputSize + t*inputSize

			// Compute h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
			for h := 0; h < hiddenSize; h++ {
				sum := config.BiasH[h]

				// W_ih @ x_t
				for i := 0; i < inputSize; i++ {
					sum += config.WeightIH[h*inputSize+i] * input[inputIdx+i]
				}

				// W_hh @ h_{t-1}
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					sum += config.WeightHH[h*hiddenSize+hPrev] * hiddenStates[prevHiddenIdx+hPrev]
				}

				// Store pre-activation (will be used for backward)
				preActivation := sum

				// Apply tanh activation
				hiddenStates[currHiddenIdx+h] = float32(math.Tanh(float64(preActivation)))
			}

			// Copy current hidden state to output
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			for h := 0; h < hiddenSize; h++ {
				output[outputIdx+h] = hiddenStates[currHiddenIdx+h]
			}
		}
	}

	return output, hiddenStates
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
