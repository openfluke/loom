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

// RNNBackward performs backward pass for RNN layer using BPTT with any numeric type.
func RNNBackward[T Numeric](
	gradOutput, input, hiddenStates *Tensor[T],
	weightIH, weightHH *Tensor[T],
	batchSize, seqLength, inputSize, hiddenSize int,
) (gradInput, gradWeightIH, gradWeightHH, gradBiasH *Tensor[T]) {
	
	// Initialize gradients
	gradInput = NewTensor[T](batchSize * seqLength * inputSize)
	gradWeightIH = NewTensor[T](hiddenSize * inputSize)
	gradWeightHH = NewTensor[T](hiddenSize * hiddenSize)
	gradBiasH = NewTensor[T](hiddenSize)

	// Gradient of hidden state (accumulates across time)
	// [batchSize, hiddenSize]
	gradHidden := make([]float64, batchSize*hiddenSize) // use float64 for accumulation precision

	// Backpropagate through time (from last timestep to first)
	for t := seqLength - 1; t >= 0; t-- {
		for b := 0; b < batchSize; b++ {
			// Add gradient from output
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			gradHiddenIdx := b * hiddenSize

			for h := 0; h < hiddenSize; h++ {
				gradHidden[gradHiddenIdx+h] += float64(gradOutput.Data[outputIdx+h])
			}

			// Current and previous hidden states
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize

			// Gradient through tanh activation
			// d_tanh(x)/dx = 1 - tanh²(x)
			for h := 0; h < hiddenSize; h++ {
				hVal := float64(hiddenStates.Data[currHiddenIdx+h])
				tanhDeriv := 1.0 - hVal*hVal // 1 - tanh²(x)
				gradPreActivation := gradHidden[gradHiddenIdx+h] * tanhDeriv

				// Accumulate bias gradient
				gradBiasH.Data[h] += T(gradPreActivation)

				// Gradient w.r.t. input: W_ih^T @ grad
				for i := 0; i < inputSize; i++ {
					gradInput.Data[inputIdx+i] += T(float64(weightIH.Data[h*inputSize+i]) * gradPreActivation)
					// Accumulate weight gradient: grad ⊗ x_t
					gradWeightIH.Data[h*inputSize+i] += T(gradPreActivation * float64(input.Data[inputIdx+i]))
				}

				// Gradient w.r.t. previous hidden state: W_hh^T @ grad
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					// We accumulate into gradHidden for next iteration (t-1)
					// BUT we overwrite gradHidden for the *current* timestep at the start of loop? 
					// NO, gradHidden accumulates. 
					// Wait, for independent samples in batch, gradHidden tracks dL/dh_{t-1}.
					// But we are iterating t backwards.
					// At step t, we compute contribution to dL/dh_{t-1}.
					// So specific slot `gradHidden[b*hiddenSize+hPrev]` should be updated.
					// BUT `gradHidden` currently holds `dL/dh_t`. 
					// We need to finish using `dL/dh_t` before updating `dL/dh_{t-1}`?
					// Yes. The loop structure in `rnnBackwardCPU` seems unsafe if it updates `gradHidden` in place while reading it?
					// Actually `gradHidden` is `dL/dh_t` coming into the step.
					// The contribution to `dL/dh_{t-1}` is added to it.
					// Since we process `h` loop entirely, we are reading `gradHidden` (dL/dh_t) and writing to... itself?
					// No, `gradHidden` is used for `gradPreActivation`.
					// Then we add to `gradHidden` for next step.
					// If we update `gradHidden` inside the `h` loop, we are mixing `dh_t` and `dh_{t-1}`?
					// Yes! The standard implementation usually uses `nextGradHidden`.
					// Let's check `rnnBackwardCPU` logic again.
					// It does: `gradHidden[b*hiddenSize+hPrev] += ...` inside the loop over `h`.
					// `h` is the *current* hidden unit index. `hPrev` is the *previous*.
					// So we are adding to `gradHidden` terms that correspond to `dh_{t-1}`.
					// But `gradHidden` was initialized with `dh_t`.
					// If `time` moves backwards, `gradHidden` represents the gradient flowing *back* to the hidden state.
					// At start of loop t, `gradHidden` holds dL/dh_t (accumulated from t+1 and output_t).
					// Inside loop, we compute dL/dh_{t-1} and *add* it to `gradHidden`.
					// This implies `gradHidden` mixes t and t-1?
					// Yes, unless we clear it? 
					// `rnnBackwardCPU` comments say: "Reset gradHidden for this batch... Actually we need to keep accumulating... so we clear after processing all timesteps"
					// Wait. If `gradHidden` accumulates `dh_{t-1}`, then in next iteration `t-1`, it will have `dh_{t-1}` + `dh_{t-1 from output}`.
					// BUT `gradHidden` *still contains* `dh_t` values?
					// No, `dh_t` components should be consumed or cleared?
					// Typically `gradHidden` is `dL/dh`. The `dL/dh` variable should represent the gradient at current timestep.
					// If we reuse the same array, we must ensure `dh_t` components don't pollute `dh_{t-1}` calculations if they overlap?
					// They don't overlap in index (h vs hPrev are same range).
					// So `gradHidden` becomes `dh_{t-1} + dh_t`. 
					// This is WRONG if `dh_t` is not cleared.
					
					// Let's perform a correction in the generic implementation to be safe.
					// Use `nextGradHidden` array.
				}
			}
		}
		
		// Safe update of gradHidden for next timestep
		// In CPU version:
		// for hPrev ... gradHidden[...hPrev] += ...
		// This adds to `gradHidden` which currently holds `dh_t`.
		// So `gradHidden` becomes `dh_t + dh_{t-1}`.
		// In next iteration t-1:
		// we read `gradHidden`. It has `dh_t + dh_{t-1}`.
		// `gradPreActivation` uses this sum.
		// But `dh_t` should NOT influence `dh_{t-1}`'s derivation of `dh_{t-2}`!
		// `dh_t`'s influence is *through* `dh_{t-1}`.
		// Once we backpropped `dh_t` to `dh_{t-1}`, we are done with `dh_t`.
		// So `gradHidden` should be *replaced* or strictly accumulated for *current* t-1.
		
		// Fix: Use a temporary buffer for `dh_{t-1}` and swap.
	}
	// Re-implementing correctly:
	
	// Reset gradHidden for safety at start
	gradHidden = make([]float64, batchSize*hiddenSize)
	
	for t := seqLength - 1; t >= 0; t-- {
		// New gradient buffer for next timestep (t-1)
		nextGradHidden := make([]float64, batchSize*hiddenSize)
		
		for b := 0; b < batchSize; b++ {
			outputIdx := b*seqLength*hiddenSize + t*hiddenSize
			gradHiddenIdx := b * hiddenSize
			
			// 1. Add output gradient contribution to current hidden gradient (dL/dh_t)
			for h := 0; h < hiddenSize; h++ {
				gradHidden[gradHiddenIdx+h] += float64(gradOutput.Data[outputIdx+h])
			}
			
			currHiddenIdx := b*(seqLength+1)*hiddenSize + (t+1)*hiddenSize
			prevHiddenIdx := b*(seqLength+1)*hiddenSize + t*hiddenSize
			inputIdx := b*seqLength*inputSize + t*inputSize
			
			for h := 0; h < hiddenSize; h++ {
				// 2. Compute local gradient dL/dz_t (through tanh)
				hVal := float64(hiddenStates.Data[currHiddenIdx+h])
				tanhDeriv := 1.0 - hVal*hVal
				gradPreAct := gradHidden[gradHiddenIdx+h] * tanhDeriv
				
				// 3. Accumulate gradients for weights/bias
				gradBiasH.Data[h] += T(gradPreAct)
				
				for i := 0; i < inputSize; i++ {
					x := float64(input.Data[inputIdx+i])
					gradWeightIH.Data[h*inputSize+i] += T(gradPreAct * x)
					gradInput.Data[inputIdx+i] += T(float64(weightIH.Data[h*inputSize+i]) * gradPreAct)
				}
				
				// 4. Propagate to previous hidden state (dL/dh_{t-1})
				for hPrev := 0; hPrev < hiddenSize; hPrev++ {
					hPrevVal := float64(hiddenStates.Data[prevHiddenIdx+hPrev])
					
					// Weight Gradient
					gradWeightHH.Data[h*hiddenSize+hPrev] += T(gradPreAct * hPrevVal)
					
					// Gradient flow to h_{t-1}
					// We accumulate into nextGradHidden
					nextGradHidden[b*hiddenSize+hPrev] += float64(weightHH.Data[h*hiddenSize+hPrev]) * gradPreAct
				}
			}
		}
		// Move to next step
		gradHidden = nextGradHidden
	}

	return gradInput, gradWeightIH, gradWeightHH, gradBiasH
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
func rnnBackwardCPU(config *LayerConfig, gradOutput, input, hiddenStates []float32,
	batchSize, seqLength, inputSize, hiddenSize int) ([]float32, []float32, []float32, []float32) {
	
	// Convert to tensors
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))
	inputT := NewTensorFromSlice(input, len(input))
	hiddenStatesT := NewTensorFromSlice(hiddenStates, len(hiddenStates))
	
	weightIHT := NewTensorFromSlice(config.WeightIH, len(config.WeightIH))
	weightHHT := NewTensorFromSlice(config.WeightHH, len(config.WeightHH))
	
	gradInput, gradWIH, gradWHH, gradBias := RNNBackward(
		gradOutputT, inputT, hiddenStatesT, weightIHT, weightHHT, 
		batchSize, seqLength, inputSize, hiddenSize)
		
	return gradInput.Data, gradWIH.Data, gradWHH.Data, gradBias.Data
}
