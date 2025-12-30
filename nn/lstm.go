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

// =============================================================================
// Generic LSTM Implementation
// =============================================================================

// LSTMWeights holds all LSTM gate weights for type-generic operations.
type LSTMWeights[T Numeric] struct {
	WeightIH_i, WeightHH_i, BiasH_i *Tensor[T] // Input gate
	WeightIH_f, WeightHH_f, BiasH_f *Tensor[T] // Forget gate
	WeightIH_g, WeightHH_g, BiasH_g *Tensor[T] // Cell candidate gate
	WeightIH_o, WeightHH_o, BiasH_o *Tensor[T] // Output gate
}

// LSTMForward performs LSTM forward pass for any numeric type.
// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
func LSTMForward[T Numeric](
	input *Tensor[T],
	weights *LSTMWeights[T],
	batchSize, seqLength, inputSize, hiddenSize int,
) (output *Tensor[T], hidden, cell *Tensor[T], allGates map[string]*Tensor[T]) {
	output = NewTensor[T](batchSize * seqLength * hiddenSize)
	hidden = NewTensor[T](batchSize * (seqLength + 1) * hiddenSize)
	cell = NewTensor[T](batchSize * (seqLength + 1) * hiddenSize)
	
	// Create map to store ALL intermediate gate values needed for backward
	allGates = make(map[string]*Tensor[T])
	allGates["i_gate"] = NewTensor[T](batchSize * seqLength * hiddenSize)
	allGates["f_gate"] = NewTensor[T](batchSize * seqLength * hiddenSize)
	allGates["g_gate"] = NewTensor[T](batchSize * seqLength * hiddenSize)
	allGates["o_gate"] = NewTensor[T](batchSize * seqLength * hiddenSize)
	allGates["c_tanh"] = NewTensor[T](batchSize * seqLength * hiddenSize)

	for t := 0; t < seqLength; t++ {
		for b := 0; b < batchSize; b++ {
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			prevCellIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			currCellIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize
			gateStoreIdx := b*seqLength*hiddenSize + t*hiddenSize

			for h := 0; h < hiddenSize; h++ {
				// Input gate
				i_sum := float64(weights.BiasH_i.Data[h])
				// Forget gate
				f_sum := float64(weights.BiasH_f.Data[h])
				// Cell candidate gate
				g_sum := float64(weights.BiasH_g.Data[h])
				// Output gate
				o_sum := float64(weights.BiasH_o.Data[h])

				for i := 0; i < inputSize; i++ {
					x := float64(input.Data[inputIdx+i])
					i_sum += float64(weights.WeightIH_i.Data[h*inputSize+i]) * x
					f_sum += float64(weights.WeightIH_f.Data[h*inputSize+i]) * x
					g_sum += float64(weights.WeightIH_g.Data[h*inputSize+i]) * x
					o_sum += float64(weights.WeightIH_o.Data[h*inputSize+i]) * x
				}

				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					hVal := float64(hidden.Data[prevHiddenIdx+hPrev])
					i_sum += float64(weights.WeightHH_i.Data[h*hiddenSize+hPrev]) * hVal
					f_sum += float64(weights.WeightHH_f.Data[h*hiddenSize+hPrev]) * hVal
					g_sum += float64(weights.WeightHH_g.Data[h*hiddenSize+hPrev]) * hVal
					o_sum += float64(weights.WeightHH_o.Data[h*hiddenSize+hPrev]) * hVal
				}

				// Apply activations
				i_gate := 1.0 / (1.0 + math.Exp(-i_sum)) // sigmoid
				f_gate := 1.0 / (1.0 + math.Exp(-f_sum)) // sigmoid
				g_gate := math.Tanh(g_sum)               // tanh
				o_gate := 1.0 / (1.0 + math.Exp(-o_sum)) // sigmoid
				
				// Store gates
				allGates["i_gate"].Data[gateStoreIdx+h] = T(i_gate)
				allGates["f_gate"].Data[gateStoreIdx+h] = T(f_gate)
				allGates["g_gate"].Data[gateStoreIdx+h] = T(g_gate)
				allGates["o_gate"].Data[gateStoreIdx+h] = T(o_gate)

				// Cell state: c_t = f_t * c_{t-1} + i_t * g_t
				prevC := float64(cell.Data[prevCellIdx+h])
				newC := f_gate*prevC + i_gate*g_gate
				cell.Data[currCellIdx+h] = T(newC)

				// Hidden state: h_t = o_t * tanh(c_t)
				c_tn := math.Tanh(newC)
				newH := o_gate * c_tn
				hidden.Data[currHiddenIdx+h] = T(newH)
				allGates["c_tanh"].Data[gateStoreIdx+h] = T(c_tn)
			}

			// Copy to output
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			for h := 0; h < hiddenSize; h++ {
				output.Data[outputIdx+h] = hidden.Data[currHiddenIdx+h]
			}
		}
	}

	return output, hidden, cell, allGates
}

// LSTMBackward performs backward pass for LSTM layer using BPTT with any numeric type.
func LSTMBackward[T Numeric](
	gradOutput, input *Tensor[T],
	states map[string]*Tensor[T], // hidden, cell, i_gate, f_gate, g_gate, o_gate, c_tanh
	weights *LSTMWeights[T],
	batchSize, seqLength, inputSize, hiddenSize int,
) (gradInput *Tensor[T], gradWeights *LSTMWeights[T]) {
	
	// Check inputs
	if gradOutput == nil || input == nil {
		return nil, nil // panic safe
	}

	// Initialize Gradients
	gradInput = NewTensor[T](batchSize * seqLength * inputSize)
	gradWeights = &LSTMWeights[T]{
		WeightIH_i: NewTensor[T](hiddenSize * inputSize), WeightHH_i: NewTensor[T](hiddenSize * hiddenSize), BiasH_i: NewTensor[T](hiddenSize),
		WeightIH_f: NewTensor[T](hiddenSize * inputSize), WeightHH_f: NewTensor[T](hiddenSize * hiddenSize), BiasH_f: NewTensor[T](hiddenSize),
		WeightIH_g: NewTensor[T](hiddenSize * inputSize), WeightHH_g: NewTensor[T](hiddenSize * hiddenSize), BiasH_g: NewTensor[T](hiddenSize),
		WeightIH_o: NewTensor[T](hiddenSize * inputSize), WeightHH_o: NewTensor[T](hiddenSize * hiddenSize), BiasH_o: NewTensor[T](hiddenSize),
	}

	// Gradient accumulators for hidden and cell states (float64 for precision)
	gradHidden := make([]float64, batchSize*hiddenSize)
	gradCell := make([]float64, batchSize*hiddenSize)
	
	// Unpack states for easier access
	tensorHidden := states["hidden"]
	tensorCell := states["cell"]
	tensorI := states["i_gate"]
	tensorF := states["f_gate"]
	tensorG := states["g_gate"]
	tensorO := states["o_gate"]
	tensorCTanh := states["c_tanh"]

	// Backpropagate through time
	for t := seqLength - 1; t >= 0; t-- {
		nextGradHidden := make([]float64, batchSize*hiddenSize)
		nextGradCell := make([]float64, batchSize*hiddenSize) // Cell gradient flows directly, but we need to track accumulation properly
		// Actually cell gradient accumulates from two sources:
		// 1. Through tanh(c_t) from h_t
		// 2. From next timestep (t+1) c_{t+1} = f_{t+1} * c_t ... -> dc_t += dc_{t+1} * f_{t+1}
		// The loop processes t descending. gradCell currently holds dc_{t} (accumulated from t+1).
		// We compute derivatives at step t, then update gradCell for step t-1.
		
		for b := 0; b < batchSize; b++ {
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			gradHiddenIdx := b * hiddenSize
			gateIdx := b*seqLength*hiddenSize + t*hiddenSize

			prevCellIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize
			
			// Add gradient from output to current hidden gradient
			for h := 0; h < hiddenSize; h++ {
				gradHidden[gradHiddenIdx+h] += float64(gradOutput.Data[outputIdx+h])
			}

			for h := 0; h < hiddenSize; h++ {
				dh := gradHidden[gradHiddenIdx+h]
				
				// Retrieve gate values
				o_gate := float64(tensorO.Data[gateIdx+h])
				c_tanh := float64(tensorCTanh.Data[gateIdx+h])
				i_gate := float64(tensorI.Data[gateIdx+h])
				f_gate := float64(tensorF.Data[gateIdx+h])
				g_gate := float64(tensorG.Data[gateIdx+h])
				prevC := float64(tensorCell.Data[prevCellIdx+h])

				// Gradient flow to cell state through output gate
				// h_t = o_t * tanh(c_t)
				// dc_from_h = dh * o_t * (1 - tanh^2(c_t))
				dc_from_h := dh * o_gate * (1.0 - c_tanh*c_tanh)
				gradCell[gradHiddenIdx+h] += dc_from_h
				
				dc := gradCell[gradHiddenIdx+h]
				
				// Gradient w.r.t Output Gate
				// do = dh * tanh(c_t)
				do := dh * c_tanh
				do_pre := do * o_gate * (1.0 - o_gate) // sigmoid derivative
				
				// Gradient w.r.t Forget Gate
				// c_t = f_t * c_{t-1} + ...
				// df = dc * c_{t-1}
				df := dc * prevC
				df_pre := df * f_gate * (1.0 - f_gate)
				
				// Gradient w.r.t Input Gate
				// c_t = ... + i_t * g_t
				// di = dc * g_t
				di := dc * g_gate
				di_pre := di * i_gate * (1.0 - i_gate)
				
				// Gradient w.r.t Cell Candidate (g)
				// dg = dc * i_t
				dg := dc * i_gate
				dg_pre := dg * (1.0 - g_gate*g_gate) // tanh derivative
				
				// Propagate gradient to previous cell state (t-1)
				// dc_{t-1} = dc_t * f_t
				// We store this in nextGradCell to act as gradCell for next iteration
				// Or since cell state logic is strictly recurrent, we can just update gradCell directly?
				// Yes, dc_{t-1} is simply dc * f_t.
				// However, since `gradCell` is used inside this `h` loop, updating it in-place for `h` is fine.
				// BUT we iterate `h` inside `b`. `gradCell` is sized [batch * hidden].
				// So `gradCell` at index `b*hidden + h` stores `dc_t[h]`.
				// We want to replace it with `dc_{t-1}[h]` for next time step.
				// So we can write to `gradCell`? 
				// Wait, `gradCell` accumulates contributions.
				// Is there any cross-h dependency? No element-wise operations for cell state.
				// So we can update `gradCell` in place? 
				// "gradCell[gradHiddenIdx+h] = dc * f_gate" at end of loop?
				// Yes.
				
				// Accumulate Bias Gradients
				gradWeights.BiasH_i.Data[h] += T(di_pre)
				gradWeights.BiasH_f.Data[h] += T(df_pre)
				gradWeights.BiasH_g.Data[h] += T(dg_pre)
				gradWeights.BiasH_o.Data[h] += T(do_pre)
				
				// Accumulate Weight Gradients & Propagate to Input
				for i := 0; i < inputSize; i++ {
					x := float64(input.Data[inputIdx+i])
					
					// Weights
					gradWeights.WeightIH_i.Data[h*inputSize+i] += T(di_pre * x)
					gradWeights.WeightIH_f.Data[h*inputSize+i] += T(df_pre * x)
					gradWeights.WeightIH_g.Data[h*inputSize+i] += T(dg_pre * x)
					gradWeights.WeightIH_o.Data[h*inputSize+i] += T(do_pre * x)
					
					// Input Gradient
					gradInput.Data[inputIdx+i] += T(
						float64(weights.WeightIH_i.Data[h*inputSize+i]) * di_pre +
						float64(weights.WeightIH_f.Data[h*inputSize+i]) * df_pre +
						float64(weights.WeightIH_g.Data[h*inputSize+i]) * dg_pre +
						float64(weights.WeightIH_o.Data[h*inputSize+i]) * do_pre,
					)
				}
				
				// Accumulate Recurrent Weight Gradients & Propagate to Previous Hidden
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					hPrevVal := float64(tensorHidden.Data[prevHiddenIdx+hPrev])
					
					// Weights
					gradWeights.WeightHH_i.Data[h*hiddenSize+hPrev] += T(di_pre * hPrevVal)
					gradWeights.WeightHH_f.Data[h*hiddenSize+hPrev] += T(df_pre * hPrevVal)
					gradWeights.WeightHH_g.Data[h*hiddenSize+hPrev] += T(dg_pre * hPrevVal)
					gradWeights.WeightHH_o.Data[h*hiddenSize+hPrev] += T(do_pre * hPrevVal)
					
					// Previous Hidden
					// Accumulate into nextGradHidden
					nextGradHidden[b*hiddenSize+hPrev] += 
						float64(weights.WeightHH_i.Data[h*hiddenSize+hPrev]) * di_pre +
						float64(weights.WeightHH_f.Data[h*hiddenSize+hPrev]) * df_pre +
						float64(weights.WeightHH_g.Data[h*hiddenSize+hPrev]) * dg_pre +
						float64(weights.WeightHH_o.Data[h*hiddenSize+hPrev]) * do_pre
				}
				
				// Update cell gradient for next step (t-1)
				// We can do this safely here because h's are independent
				// Use nextGradCell to be explicit
				nextGradCell[gradHiddenIdx+h] = dc * f_gate
			}
		}
		
		// Update gradient buffers for next timestep
		gradHidden = nextGradHidden
		gradCell = nextGradCell
	}

	return gradInput, gradWeights
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// sigmoid implements the sigmoid activation function
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// lstmForwardCPU performs forward pass for LSTM layer
// Input shape: [batchSize, seqLength, inputSize]
// Output shape: [batchSize, seqLength, hiddenSize]
// Returns: (output, all_states) where all_states contains hidden, cell, and gate values for backward
func lstmForwardCPU(config *LayerConfig, input []float32, batchSize, seqLength, inputSize, hiddenSize int) ([]float32, map[string][]float32) {
	// ... (implementation wrapping Generic version)
	
	inputT := NewTensorFromSlice(input, len(input))
	weights := &LSTMWeights[float32]{
		WeightIH_i: NewTensorFromSlice(config.WeightIH_i, len(config.WeightIH_i)),
		WeightHH_i: NewTensorFromSlice(config.WeightHH_i, len(config.WeightHH_i)),
		BiasH_i:    NewTensorFromSlice(config.BiasH_i, len(config.BiasH_i)),
		
		WeightIH_f: NewTensorFromSlice(config.WeightIH_f, len(config.WeightIH_f)),
		WeightHH_f: NewTensorFromSlice(config.WeightHH_f, len(config.WeightHH_f)),
		BiasH_f:    NewTensorFromSlice(config.BiasH_f, len(config.BiasH_f)),
		
		WeightIH_g: NewTensorFromSlice(config.WeightIH_g, len(config.WeightIH_g)),
		WeightHH_g: NewTensorFromSlice(config.WeightHH_g, len(config.WeightHH_g)),
		BiasH_g:    NewTensorFromSlice(config.BiasH_g, len(config.BiasH_g)),
		
		WeightIH_o: NewTensorFromSlice(config.WeightIH_o, len(config.WeightIH_o)),
		WeightHH_o: NewTensorFromSlice(config.WeightHH_o, len(config.WeightHH_o)),
		BiasH_o:    NewTensorFromSlice(config.BiasH_o, len(config.BiasH_o)),
	}
	
	_, _, _, gates := LSTMForward(inputT, weights, batchSize, seqLength, inputSize, hiddenSize)
	
	// Convert gates map back to float32 map
	states := make(map[string][]float32)
	for k, v := range gates {
		states[k] = v.Data
	}
	// Note: original lstmForwardCPU returned "hidden" and "cell" in the map too, plus gates.
	// We need to ensure complete compatibility.
	// The generic LSTMForward returns (hidden, cell) separately.
	// Original code expects them in map.
	
	// Let's keep original lstmForwardCPU logic intact rather than wrapping, to avoid breaking legacy interfaces if they differ slightly
	// But to avoid duplication, we should ideally wrap.
	// RE-READ: Generic LSTMForward returns "hidden" and "cell" tensors separately.
	// Original returns "states" map including "hidden" and "cell".
	
	// Re-implementing original logic safely inline to avoid breaking changes is preferred if I am replacing the file content.
	// For "Generic Implementation", I will leave the original function at bottom.
	
	// Wait, I am overwriting the file. I needs to include the original code!
	// I will copy the original code for lstmForwardCPU and lstmBackwardCPU.
	
	// Placeholder: re-paste original code from previous view_file.
	
	return lstmForwardCPU_Original(config, input, batchSize, seqLength, inputSize, hiddenSize)
}

func lstmForwardCPU_Original(config *LayerConfig, input []float32, batchSize, seqLength, inputSize, hiddenSize int) ([]float32, map[string][]float32) {
	output := make([]float32, batchSize*seqLength*hiddenSize)
	states := make(map[string][]float32)
	states["hidden"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
	states["cell"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
	states["i_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
	states["f_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
	states["g_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
	states["o_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
	states["c_tanh"] = make([]float32, batchSize*seqLength*hiddenSize)

	for t := 0; t < seqLength; t++ {
		for b := 0; b < batchSize; b++ {
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			prevCellIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			currCellIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize
			gateIdx := b*seqLength*hiddenSize + t*hiddenSize

			for h := 0; h < hiddenSize; h++ {
				i_sum := config.BiasH_i[h]
				for i := 0; i < inputSize; i++ { i_sum += config.WeightIH_i[h*inputSize+i] * input[inputIdx+i] }
				for hPrev := 0; hPrev < hiddenSize; hPrev++ { i_sum += config.WeightHH_i[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev] }
				states["i_gate"][gateIdx+h] = sigmoid(i_sum)

				f_sum := config.BiasH_f[h]
				for i := 0; i < inputSize; i++ { f_sum += config.WeightIH_f[h*inputSize+i] * input[inputIdx+i] }
				for hPrev := 0; hPrev < hiddenSize; hPrev++ { f_sum += config.WeightHH_f[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev] }
				states["f_gate"][gateIdx+h] = sigmoid(f_sum)

				g_sum := config.BiasH_g[h]
				for i := 0; i < inputSize; i++ { g_sum += config.WeightIH_g[h*inputSize+i] * input[inputIdx+i] }
				for hPrev := 0; hPrev < hiddenSize; hPrev++ { g_sum += config.WeightHH_g[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev] }
				states["g_gate"][gateIdx+h] = float32(math.Tanh(float64(g_sum)))

				o_sum := config.BiasH_o[h]
				for i := 0; i < inputSize; i++ { o_sum += config.WeightIH_o[h*inputSize+i] * input[inputIdx+i] }
				for hPrev := 0; hPrev < hiddenSize; hPrev++ { o_sum += config.WeightHH_o[h*hiddenSize+hPrev] * states["hidden"][prevHiddenIdx+hPrev] }
				states["o_gate"][gateIdx+h] = sigmoid(o_sum)

				states["cell"][currCellIdx+h] = states["f_gate"][gateIdx+h]*states["cell"][prevCellIdx+h] + states["i_gate"][gateIdx+h]*states["g_gate"][gateIdx+h]
				c_tanh := float32(math.Tanh(float64(states["cell"][currCellIdx+h])))
				states["c_tanh"][gateIdx+h] = c_tanh
				states["hidden"][currHiddenIdx+h] = states["o_gate"][gateIdx+h] * c_tanh
			}
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			for h := 0; h < hiddenSize; h++ { output[outputIdx+h] = states["hidden"][currHiddenIdx+h] }
		}
	}
	return output, states
}

// lstmBackwardCPU performs backward pass for LSTM layer using BPTT
// Returns: (gradInput, gradWeights...) - one gradient tensor for each weight/bias
func lstmBackwardCPU(config *LayerConfig, gradOutput, input []float32, states map[string][]float32,
	batchSize, seqLength, inputSize, hiddenSize int) ([]float32, map[string][]float32) {
	
	// Use Generic Implementation logic or keep original?
	// For safety, let's keep original logic to ensure exact compatibility if needed.
	// But it's massive.
	// Implementation matches generic one (it was the reference).
	// Just stub it out? No, existing code uses it.
	
	// Wrap generic implementation!
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))
	inputT := NewTensorFromSlice(input, len(input))
	
	// Convert states to tensors
	genericStates := make(map[string]*Tensor[float32])
	for k, v := range states {
		genericStates[k] = NewTensorFromSlice(v, len(v))
	}
	
	weights := &LSTMWeights[float32]{
		WeightIH_i: NewTensorFromSlice(config.WeightIH_i, len(config.WeightIH_i)),
		WeightHH_i: NewTensorFromSlice(config.WeightHH_i, len(config.WeightHH_i)),
		BiasH_i:    NewTensorFromSlice(config.BiasH_i, len(config.BiasH_i)),
		
		WeightIH_f: NewTensorFromSlice(config.WeightIH_f, len(config.WeightIH_f)),
		WeightHH_f: NewTensorFromSlice(config.WeightHH_f, len(config.WeightHH_f)),
		BiasH_f:    NewTensorFromSlice(config.BiasH_f, len(config.BiasH_f)),
		
		WeightIH_g: NewTensorFromSlice(config.WeightIH_g, len(config.WeightIH_g)),
		WeightHH_g: NewTensorFromSlice(config.WeightHH_g, len(config.WeightHH_g)),
		BiasH_g:    NewTensorFromSlice(config.BiasH_g, len(config.BiasH_g)),
		
		WeightIH_o: NewTensorFromSlice(config.WeightIH_o, len(config.WeightIH_o)),
		WeightHH_o: NewTensorFromSlice(config.WeightHH_o, len(config.WeightHH_o)),
		BiasH_o:    NewTensorFromSlice(config.BiasH_o, len(config.BiasH_o)),
	}
	
	gradInputT, gradWeightsT := LSTMBackward(gradOutputT, inputT, genericStates, weights, batchSize, seqLength, inputSize, hiddenSize)
	
	// Pack results back to map
	grads := make(map[string][]float32)
	grads["WeightIH_i"] = gradWeightsT.WeightIH_i.Data
	grads["WeightHH_i"] = gradWeightsT.WeightHH_i.Data
	grads["BiasH_i"]    = gradWeightsT.BiasH_i.Data
	
	grads["WeightIH_f"] = gradWeightsT.WeightIH_f.Data
	grads["WeightHH_f"] = gradWeightsT.WeightHH_f.Data
	grads["BiasH_f"]    = gradWeightsT.BiasH_f.Data
	
	grads["WeightIH_g"] = gradWeightsT.WeightIH_g.Data
	grads["WeightHH_g"] = gradWeightsT.WeightHH_g.Data
	grads["BiasH_g"]    = gradWeightsT.BiasH_g.Data
	
	grads["WeightIH_o"] = gradWeightsT.WeightIH_o.Data
	grads["WeightHH_o"] = gradWeightsT.WeightHH_o.Data
	grads["BiasH_o"]    = gradWeightsT.BiasH_o.Data
	
	return gradInputT.Data, grads
}
