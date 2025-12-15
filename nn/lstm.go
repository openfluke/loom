package nn

import (
	"math"
	"math/rand"
)

// InitLSTMLayer initializes an LSTM layer with Xavier/Glorot initialization
// LSTM has 4 gates: input (i), forget (f), cell/candidate (g), output (o)
// inputSize: size of input features
// hiddenSize: size of hidden state and cell state
// batchSize: batch size for processing
// seqLength: length of input sequences
func InitLSTMLayer(inputSize, hiddenSize, batchSize, seqLength int) LayerConfig {
	config := LayerConfig{
		Type:         LayerLSTM,
		Activation:   ActivationTanh, // Cell state uses tanh
		RNNInputSize: inputSize,
		HiddenSize:   hiddenSize,
		SeqLength:    seqLength,
	}

	// Xavier/Glorot initialization standard deviations
	stdIH := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	stdHH := math.Sqrt(2.0 / float64(hiddenSize+hiddenSize))

	// Initialize weights and biases for each gate

	// Input gate (i)
	config.WeightIH_i = make([]float32, hiddenSize*inputSize)
	config.WeightHH_i = make([]float32, hiddenSize*hiddenSize)
	config.BiasH_i = make([]float32, hiddenSize)
	for i := range config.WeightIH_i {
		config.WeightIH_i[i] = float32(rand.NormFloat64() * stdIH)
	}
	for i := range config.WeightHH_i {
		config.WeightHH_i[i] = float32(rand.NormFloat64() * stdHH)
	}

	// Forget gate (f) - initialize bias to 1.0 to remember by default
	config.WeightIH_f = make([]float32, hiddenSize*inputSize)
	config.WeightHH_f = make([]float32, hiddenSize*hiddenSize)
	config.BiasH_f = make([]float32, hiddenSize)
	for i := range config.WeightIH_f {
		config.WeightIH_f[i] = float32(rand.NormFloat64() * stdIH)
	}
	for i := range config.WeightHH_f {
		config.WeightHH_f[i] = float32(rand.NormFloat64() * stdHH)
	}
	for i := range config.BiasH_f {
		config.BiasH_f[i] = 1.0 // Forget gate bias = 1.0
	}

	// Cell/Candidate gate (g)
	config.WeightIH_g = make([]float32, hiddenSize*inputSize)
	config.WeightHH_g = make([]float32, hiddenSize*hiddenSize)
	config.BiasH_g = make([]float32, hiddenSize)
	for i := range config.WeightIH_g {
		config.WeightIH_g[i] = float32(rand.NormFloat64() * stdIH)
	}
	for i := range config.WeightHH_g {
		config.WeightHH_g[i] = float32(rand.NormFloat64() * stdHH)
	}

	// Output gate (o)
	config.WeightIH_o = make([]float32, hiddenSize*inputSize)
	config.WeightHH_o = make([]float32, hiddenSize*hiddenSize)
	config.BiasH_o = make([]float32, hiddenSize)
	for i := range config.WeightIH_o {
		config.WeightIH_o[i] = float32(rand.NormFloat64() * stdIH)
	}
	for i := range config.WeightHH_o {
		config.WeightHH_o[i] = float32(rand.NormFloat64() * stdHH)
	}

	return config
}

// sigmoid implements the sigmoid activation function
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// lstmForwardCPU performs forward pass for LSTM layer
// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
// Returns: (output, all_states) where all_states contains hidden, cell, and gate values for backward
func lstmForwardCPU(config *LayerConfig, input []float32, batchSize, seqLength, inputSize, hiddenSize int) ([]float32, map[string][]float32) {
	// Output: [batchSize, seqLength, hiddenSize]
	output := make([]float32, batchSize*seqLength*hiddenSize)

	// Store all states for backward pass
	states := make(map[string][]float32)

	// Hidden states: [batchSize, seqLength+1, hiddenSize] (including h_0=0)
	states["hidden"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)

	// Cell states: [batchSize, seqLength+1, hiddenSize] (including c_0=0)
	states["cell"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)

	// Gate activations: [batchSize, seqLength, hiddenSize]
	states["i_gate"] = make([]float32, batchSize*seqLength*hiddenSize) // input gate
	states["f_gate"] = make([]float32, batchSize*seqLength*hiddenSize) // forget gate
	states["g_gate"] = make([]float32, batchSize*seqLength*hiddenSize) // cell candidate
	states["o_gate"] = make([]float32, batchSize*seqLength*hiddenSize) // output gate
	states["c_tanh"] = make([]float32, batchSize*seqLength*hiddenSize) // tanh(cell state)

	// Process each timestep
	for t := 0; t < seqLength; t++ {
		for b := 0; b < batchSize; b++ {
			// Previous states
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			prevCellIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize

			// Current states
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			currCellIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize

			// Input and gate indices
			inputIdx := b*seqLength*inputSize + t*inputSize
			gateIdx := b*seqLength*hiddenSize + t*hiddenSize

			// Compute all gates
			for h := 0; h < hiddenSize; h++ {
				// Input gate: i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
				i_sum := config.BiasH_i[h]
				for i := 0; i < inputSize; i++ {
					i_sum += config.WeightIH_i[h*inputSize+i] * input[inputIdx+i]
				}
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					i_sum += config.WeightHH_i[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev]
				}
				states["i_gate"][gateIdx+h] = sigmoid(i_sum)

				// Forget gate: f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)
				f_sum := config.BiasH_f[h]
				for i := 0; i < inputSize; i++ {
					f_sum += config.WeightIH_f[h*inputSize+i] * input[inputIdx+i]
				}
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					f_sum += config.WeightHH_f[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev]
				}
				states["f_gate"][gateIdx+h] = sigmoid(f_sum)

				// Cell candidate: g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
				g_sum := config.BiasH_g[h]
				for i := 0; i < inputSize; i++ {
					g_sum += config.WeightIH_g[h*inputSize+i] * input[inputIdx+i]
				}
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					g_sum += config.WeightHH_g[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev]
				}
				states["g_gate"][gateIdx+h] = float32(math.Tanh(float64(g_sum)))

				// Output gate: o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)
				o_sum := config.BiasH_o[h]
				for i := 0; i < inputSize; i++ {
					o_sum += config.WeightIH_o[h*inputSize+i] * input[inputIdx+i]
				}
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					o_sum += config.WeightHH_o[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev]
				}
				states["o_gate"][gateIdx+h] = sigmoid(o_sum)

				// Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
				states["cell"][currCellIdx+h] = states["f_gate"][gateIdx+h]*states["cell"][prevCellIdx+h] +
					states["i_gate"][gateIdx+h]*states["g_gate"][gateIdx+h]

				// Hidden state: h_t = o_t ⊙ tanh(c_t)
				c_tanh := float32(math.Tanh(float64(states["cell"][currCellIdx+h])))
				states["c_tanh"][gateIdx+h] = c_tanh
				states["hidden"][currHiddenIdx+h] = states["o_gate"][gateIdx+h] * c_tanh
			}

			// Copy hidden state to output
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			for h := 0; h < hiddenSize; h++ {
				output[outputIdx+h] = states["hidden"][currHiddenIdx+h]
			}
		}
	}

	return output, states
}

// lstmBackwardCPU performs backward pass for LSTM layer using BPTT
// Returns: (gradInput, gradWeights...) - one gradient tensor for each weight/bias
func lstmBackwardCPU(config *LayerConfig, gradOutput, input []float32, states map[string][]float32,
	batchSize, seqLength, inputSize, hiddenSize int) ([]float32, map[string][]float32) {

	// Gradient tensors
	gradInput := make([]float32, batchSize*seqLength*inputSize)

	grads := make(map[string][]float32)
	grads["WeightIH_i"] = make([]float32, hiddenSize*inputSize)
	grads["WeightHH_i"] = make([]float32, hiddenSize*hiddenSize)
	grads["BiasH_i"] = make([]float32, hiddenSize)

	grads["WeightIH_f"] = make([]float32, hiddenSize*inputSize)
	grads["WeightHH_f"] = make([]float32, hiddenSize*hiddenSize)
	grads["BiasH_f"] = make([]float32, hiddenSize)

	grads["WeightIH_g"] = make([]float32, hiddenSize*inputSize)
	grads["WeightHH_g"] = make([]float32, hiddenSize*hiddenSize)
	grads["BiasH_g"] = make([]float32, hiddenSize)

	grads["WeightIH_o"] = make([]float32, hiddenSize*inputSize)
	grads["WeightHH_o"] = make([]float32, hiddenSize*hiddenSize)
	grads["BiasH_o"] = make([]float32, hiddenSize)

	// Gradient accumulators for hidden and cell states
	gradHidden := make([]float32, batchSize*hiddenSize)
	gradCell := make([]float32, batchSize*hiddenSize)

	// Backpropagate through time
	for t := seqLength - 1; t >= 0; t-- {
		for b := 0; b < batchSize; b++ {
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			gradHiddenIdx := b * hiddenSize
			gateIdx := b*seqLength*hiddenSize + t*hiddenSize

			prevCellIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize

			// Add gradient from output
			for h := 0; h < hiddenSize; h++ {
				gradHidden[gradHiddenIdx+h] += gradOutput[outputIdx+h]
			}

			// Backward through LSTM cell
			for h := 0; h < hiddenSize; h++ {
				// Gradient w.r.t. hidden state at this timestep
				dh := gradHidden[gradHiddenIdx+h]

				// Gradient through h_t = o_t ⊙ tanh(c_t)
				o_gate := states["o_gate"][gateIdx+h]
				c_tanh := states["c_tanh"][gateIdx+h]

				// Gradient w.r.t. o_gate
				do := dh * c_tanh
				// Gradient w.r.t. cell state via tanh
				dc_from_h := dh * o_gate * (1.0 - c_tanh*c_tanh) // tanh derivative

				// Add to cell gradient
				gradCell[gradHiddenIdx+h] += dc_from_h

				// Gradient through c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
				dc := gradCell[gradHiddenIdx+h]

				i_gate := states["i_gate"][gateIdx+h]
				f_gate := states["f_gate"][gateIdx+h]
				g_gate := states["g_gate"][gateIdx+h]
				prevCell := states["cell"][prevCellIdx+h]

				// Gradients w.r.t. gates
				df := dc * prevCell
				di := dc * g_gate
				dg := dc * i_gate

				// Propagate gradient to previous cell state
				gradCell[gradHiddenIdx+h] = dc * f_gate

				// Gradient through gate activations (sigmoid and tanh derivatives)
				// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
				// tanh'(x) = 1 - tanh²(x)
				di_pre := di * i_gate * (1.0 - i_gate) // sigmoid derivative
				df_pre := df * f_gate * (1.0 - f_gate) // sigmoid derivative
				dg_pre := dg * (1.0 - g_gate*g_gate)   // tanh derivative
				do_pre := do * o_gate * (1.0 - o_gate) // sigmoid derivative

				// Accumulate bias gradients
				grads["BiasH_i"][h] += di_pre
				grads["BiasH_f"][h] += df_pre
				grads["BiasH_g"][h] += dg_pre
				grads["BiasH_o"][h] += do_pre

				// Accumulate weight gradients and propagate to inputs/hidden
				for i := 0; i < inputSize; i++ {
					x := input[inputIdx+i]
					gradInput[inputIdx+i] += config.WeightIH_i[h*inputSize+i]*di_pre +
						config.WeightIH_f[h*inputSize+i]*df_pre +
						config.WeightIH_g[h*inputSize+i]*dg_pre +
						config.WeightIH_o[h*inputSize+i]*do_pre

					grads["WeightIH_i"][h*inputSize+i] += di_pre * x
					grads["WeightIH_f"][h*inputSize+i] += df_pre * x
					grads["WeightIH_g"][h*inputSize+i] += dg_pre * x
					grads["WeightIH_o"][h*inputSize+i] += do_pre * x
				}

				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					hPrevVal := states["hidden"][prevHiddenIdx+hPrev]
					gradHidden[b*hiddenSize+hPrev] += config.WeightHH_i[h*hiddenSize+hPrev]*di_pre +
						config.WeightHH_f[h*hiddenSize+hPrev]*df_pre +
						config.WeightHH_g[h*hiddenSize+hPrev]*dg_pre +
						config.WeightHH_o[h*hiddenSize+hPrev]*do_pre

					grads["WeightHH_i"][h*hiddenSize+hPrev] += di_pre * hPrevVal
					grads["WeightHH_f"][h*hiddenSize+hPrev] += df_pre * hPrevVal
					grads["WeightHH_g"][h*hiddenSize+hPrev] += dg_pre * hPrevVal
					grads["WeightHH_o"][h*hiddenSize+hPrev] += do_pre * hPrevVal
				}
			}
		}
	}

	return gradInput, grads
}
