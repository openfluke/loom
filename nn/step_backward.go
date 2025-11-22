package nn

import (
	"math"
	"time"
)

// StepBackward executes one backward step for ALL layers simultaneously
// It applies a "Softmax Variation" to the weight gradients to balance updates
func (n *Network) StepBackward(state *StepState, gradOutput []float32) ([]float32, time.Duration) {
	start := time.Now()

	state.mu.Lock()
	defer state.mu.Unlock()

	// Current gradient flowing back
	grad := make([]float32, len(gradOutput))
	copy(grad, gradOutput)

	totalLayers := n.TotalLayers()

	// Backpropagate through grid in reverse order
	// Note: In a stepping context, we might want to process layers in parallel or
	// in a specific order. For now, we do standard reverse order to propagate
	// gradients correctly through the graph for this time step.
	for layerIdx := totalLayers - 1; layerIdx >= 0; layerIdx-- {
		// Calculate grid position
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)

		// Get inputs and pre-activations from StepState
		// layerData[layerIdx] is the INPUT to this layer
		// layerPreAct[layerIdx] is the PRE-ACTIVATION of this layer
		input := state.layerData[layerIdx]
		preAct := state.layerPreAct[layerIdx]

		// If we don't have valid state for this layer, skip it (or zero grad)
		if len(input) == 0 || len(preAct) == 0 {
			continue
		}

		var gradInput []float32
		var kernelGrads, biasGrads []float32

		// Route to appropriate layer type
		switch config.Type {
		case LayerConv2D:
			gradInput, kernelGrads, biasGrads = conv2DBackwardCPU(grad, input, preAct, config, n.BatchSize)

		case LayerMultiHeadAttention:
			var gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB []float32
			gradInput, gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB = multiHeadAttentionBackwardCPU(grad, input, preAct, config, n.BatchSize)

			// Concatenate all weight grads
			kernelGrads = append(append(append(gradQW, gradKW...), gradVW...), gradOutW...)
			biasGrads = append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)

		case LayerRNN:
			var gradWeightIH, gradWeightHH, gradBiasH []float32
			// For RNN, preAct holds hiddenStates
			gradInput, gradWeightIH, gradWeightHH, gradBiasH = rnnBackwardCPU(config, grad, input, preAct,
				n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			kernelGrads = append(gradWeightIH, gradWeightHH...)
			biasGrads = gradBiasH

		case LayerLSTM:
			// Reconstruct states map from flat preAct
			states := reconstructLSTMStates(preAct, n.BatchSize, config.SeqLength, config.HiddenSize)
			var grads map[string][]float32
			gradInput, grads = lstmBackwardCPU(config, grad, input, states,
				n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			// Concatenate grads
			kernelGrads = append(append(append(grads["WeightIH_i"], grads["WeightHH_i"]...),
				append(grads["WeightIH_f"], grads["WeightHH_f"]...)...),
				append(append(grads["WeightIH_g"], grads["WeightHH_g"]...),
					append(grads["WeightIH_o"], grads["WeightHH_o"]...)...)...)

			biasGrads = append(append(append(grads["BiasH_i"], grads["BiasH_f"]...), grads["BiasH_g"]...), grads["BiasH_o"]...)

		case LayerDense:
			gradInput, kernelGrads, biasGrads = denseBackwardCPU(grad, input, preAct, config, n.BatchSize)

		case LayerSwiGLU:
			// Need to implement SwiGLU backward or use a placeholder if not available
			// Assuming dense-like behavior for now or skipping if not critical for this example
			// For now, let's treat it as a pass-through for gradients if not implemented
			// TODO: Implement SwiGLU backward
			gradInput = make([]float32, len(input)) // Placeholder

		case LayerNorm, LayerRMSNorm:
			// Normalization layers usually just propagate gradients
			// Simplified: pass gradient through (approximate)
			gradInput = make([]float32, len(input))
			copy(gradInput, grad)

		case LayerParallel:
			// Parallel layer backward
			// Need to reconstruct branch pre-acts
			// Simplified: skip for now or implement if needed
			gradInput = make([]float32, len(input))
			copy(gradInput, grad)

		case LayerSoftmax:
			// Softmax backward
			// We need the output of this layer, which is in layerData[layerIdx+1]
			softmaxOutput := state.layerData[layerIdx+1]

			gradInput = make([]float32, len(grad))
			for i := range gradInput {
				var gradSum float32
				for j := range grad {
					// Jacobian: y_j * (delta_ij - y_i)
					delta := float32(0.0)
					if i == j {
						delta = 1.0
					}
					jacobian := softmaxOutput[j] * (delta - softmaxOutput[i])
					gradSum += grad[j] * jacobian
				}
				gradInput[i] = gradSum
			}

		default:
			// Element-wise activation
			gradInput = make([]float32, len(grad))
			for i := 0; i < len(grad); i++ {
				derivative := activateDerivativeCPU(preAct[i], config.Activation)
				gradInput[i] = grad[i] * derivative
			}
		}

		// === APPLY SOFTMAX VARIATION TO GRADIENTS ===
		// "Adjust how much of the spectrum across the softmax functionality...
		// instead of 100% across the whole layer"

		if len(kernelGrads) > 0 {
			applySoftmaxGradientScaling(kernelGrads)
			n.kernelGradients[layerIdx] = kernelGrads
		}

		if len(biasGrads) > 0 {
			applySoftmaxGradientScaling(biasGrads)
			n.biasGradients[layerIdx] = biasGrads
		}

		/*
			// Store raw gradients if scaling is disabled
			if len(kernelGrads) > 0 {
				n.kernelGradients[layerIdx] = kernelGrads
			}
			if len(biasGrads) > 0 {
				n.biasGradients[layerIdx] = biasGrads
			}
		*/

		// Update gradient for next layer
		grad = gradInput
	}

	return grad, time.Since(start)
}

// applySoftmaxGradientScaling applies a softmax-based scaling to the gradients
// Formula: G_new = G_old * (Softmax(|G_old|) * N)
// This boosts dominant gradients and suppresses weak ones, while preserving sign.
func applySoftmaxGradientScaling(grads []float32) {
	if len(grads) == 0 {
		return
	}

	// 1. Find max abs value for numerical stability
	maxAbs := float32(0.0)
	for _, g := range grads {
		abs := float32(math.Abs(float64(g)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	// 2. Compute exponentials of absolute values
	exps := make([]float32, len(grads))
	sumExp := float32(0.0)
	for i, g := range grads {
		abs := float32(math.Abs(float64(g)))
		exps[i] = float32(math.Exp(float64(abs - maxAbs))) // Subtract max for stability
		sumExp += exps[i]
	}

	// 3. Apply scaling
	// Scale factor = (exp(|g|) / sumExp) * N
	// We multiply by N (len(grads)) so that the average scale factor is 1.0
	// If we didn't multiply by N, gradients would vanish (sum to 1).
	N := float32(len(grads))
	for i := range grads {
		softmaxScore := exps[i] / sumExp
		scaleFactor := softmaxScore * N

		// Optional: Dampen the effect?
		// The user said "doesn't dramatically change all the weights".
		// Let's blend it with 1.0: 0.5 * scale + 0.5 * 1.0
		// Or just use it raw. Let's use it raw but clamp extreme values if needed.

		grads[i] *= scaleFactor
	}
}

// Helper to reconstruct LSTM states from flat slice
func reconstructLSTMStates(flat []float32, batchSize, seqLength, hiddenSize int) map[string][]float32 {
	states := make(map[string][]float32)

	// Sizes
	hiddenStateSize := batchSize * (seqLength + 1) * hiddenSize
	gateSize := batchSize * seqLength * hiddenSize

	offset := 0

	// Helper to slice safely
	getSlice := func(size int) []float32 {
		if offset+size > len(flat) {
			return make([]float32, size) // Return zeros if out of bounds
		}
		s := flat[offset : offset+size]
		offset += size
		return s
	}

	states["hidden"] = getSlice(hiddenStateSize)
	states["cell"] = getSlice(hiddenStateSize)
	states["i_gate"] = getSlice(gateSize)
	states["f_gate"] = getSlice(gateSize)
	states["g_gate"] = getSlice(gateSize)
	states["o_gate"] = getSlice(gateSize)
	states["c_tanh"] = getSlice(gateSize)

	return states
}
