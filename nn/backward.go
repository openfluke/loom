package nn

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// =============================================================================
// Generic Backward Pass Implementation
// =============================================================================

// GenericBackwardPass computes gradients via backpropagation for any numeric type.
// Returns:
// - Gradient with respect to input
// - Slice of kernel gradients per layer (type depends on layer)
// - Slice of bias gradients per layer (type depends on layer)
// - Duration
func GenericBackwardPass[T Numeric](
	n *Network,
	gradOutput *Tensor[T],
	activations []*Tensor[T],
	backwardContext []any,
) (*Tensor[T], []any, []any, time.Duration) {
	start := time.Now()

	totalLayers := n.TotalLayers()
	if len(activations) <= totalLayers {
		// Needs input + output of each layer, so size is totalLayers+1
		return NewTensor[T](len(gradOutput.Data)), nil, nil, time.Since(start)
	}

	// Gradients for activations [0...totalLayers]
	// grads[i] is gradient w.r.t. activations[i] (output of layer i-1, input to layer i)
	grads := make([]*Tensor[T], totalLayers+1)

	// Initialize output gradient
	grads[totalLayers] = gradOutput.Clone()

	// Storage for weight gradients
	kernelGrads := make([]any, totalLayers)
	biasGrads := make([]any, totalLayers)

	// Backpropagate through grid in reverse order
	for layerIdx := totalLayers - 1; layerIdx >= 0; layerIdx-- {
		// Calculate grid position for this layer
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)

		// Inputs and Context
		input := activations[layerIdx] // Input to this layer
		context := backwardContext[layerIdx]
		gradOut := grads[layerIdx+1] // Gradient w.r.t. output of this layer

		if gradOut == nil {
			// Should not happen if graph is connected
			gradOut = NewTensor[T](len(input.Data)) // Zero gradient
		}

		if config.IsDisabled {
			// Pass gradient through to input
			accumulateGradient(grads, layerIdx, gradOut)
			continue
		}

		switch config.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
			// DenseBackward needs preActivation (stored in context)
			preAct, ok := context.(*Tensor[T])
			if !ok {
				// Fallback if context missing
				preAct = NewTensor[T](len(gradOut.Data))
			}

			gInput, gWeights, gBias := DenseBackward(gradOut, input, preAct, weights, config.InputHeight, config.OutputHeight, n.BatchSize, config.Activation)

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gWeights
			biasGrads[layerIdx] = gBias

		case LayerConv2D:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
			preAct, _ := context.(*Tensor[T])

			gInput, gKernel, gBias := Conv2DBackward(gradOut, input, preAct, weights,
				config.InputHeight, config.InputWidth, config.InputChannels,
				config.KernelSize, config.Stride, config.Padding, config.Filters,
				config.OutputHeight, config.OutputWidth, n.BatchSize, config.Activation)

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gKernel
			biasGrads[layerIdx] = gBias

		case LayerRNN:
			// RNNBackward
			wIH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH, len(config.WeightIH)))
			wHH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH, len(config.WeightHH)))
			hiddenStates, _ := context.(*Tensor[T])

			gInput, gWIH, gWHH, gBiasH := RNNBackward(gradOut, input, hiddenStates, wIH, wHH, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			accumulateGradient(grads, layerIdx, gInput)
			// Return slice of weight grads
			kernelGrads[layerIdx] = []*Tensor[T]{gWIH, gWHH}
			biasGrads[layerIdx] = gBiasH

		case LayerLSTM:
			// LSTMBackward
			states, _ := context.(map[string]*Tensor[T])
			weights := &LSTMWeights[T]{
				WeightIH_i: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH_i, len(config.WeightIH_i))),
				WeightHH_i: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH_i, len(config.WeightHH_i))),
				BiasH_i:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.BiasH_i, len(config.BiasH_i))),
				WeightIH_f: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH_f, len(config.WeightIH_f))),
				WeightHH_f: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH_f, len(config.WeightHH_f))),
				BiasH_f:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.BiasH_f, len(config.BiasH_f))),
				WeightIH_g: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH_g, len(config.WeightIH_g))),
				WeightHH_g: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH_g, len(config.WeightHH_g))),
				BiasH_g:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.BiasH_g, len(config.BiasH_g))),
				WeightIH_o: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH_o, len(config.WeightIH_o))),
				WeightHH_o: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH_o, len(config.WeightHH_o))),
				BiasH_o:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.BiasH_o, len(config.BiasH_o))),
			}

			gInput, gWeights := LSTMBackward(gradOut, input, states, weights, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			accumulateGradient(grads, layerIdx, gInput)
			// Return all weight grads map
			kernelGrads[layerIdx] = gWeights // Map
			biasGrads[layerIdx] = nil        // Biases inside map

		case LayerSoftmax:
			// SoftmaxBackward
			// Generic SoftmaxBackward signature: (gradOutput, output, rows, cols)
			softmaxOut, _ := context.(*Tensor[T])
			gInput := SoftmaxBackward(gradOut, softmaxOut, config.SoftmaxRows, config.SoftmaxCols)
			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = nil
			biasGrads[layerIdx] = nil

		case LayerNorm:
			// LayerNormBackward
			// Signature: (input, residual, gradOutput, gamma, beta, normSize, batchSize, epsilon)
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
			beta := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Beta, len(config.Beta)))

			gInput, gGamma, gBeta := LayerNormBackward(input, nil, gradOut, gamma, beta, config.NormSize, n.BatchSize, float64(config.Epsilon))

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gGamma
			biasGrads[layerIdx] = gBeta

		case LayerRMSNorm:
			// RMSNormBackward
			// Signature: (input, residual, gradOutput, gamma, normSize, batchSize, epsilon)
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
			gInput, gGamma := RMSNormBackward(input, nil, gradOut, gamma, config.NormSize, n.BatchSize, float64(config.Epsilon))

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gGamma
			biasGrads[layerIdx] = nil

		case LayerSwiGLU:
			// SwiGLUBackward
			gateW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.GateWeights, len(config.GateWeights)))
			upW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.UpWeights, len(config.UpWeights)))
			downW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.DownWeights, len(config.DownWeights)))

			gInput, gGateW, gUpW, gDownW, gGateB, gUpB, gDownB := SwiGLUBackward(
				gradOut, input,
				gateW, upW, downW,
				ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.GateBias, len(config.GateBias))),
				ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.UpBias, len(config.UpBias))),
				ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.DownBias, len(config.DownBias))),
				config.InputHeight, config.OutputHeight, n.BatchSize)

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = []*Tensor[T]{gGateW, gUpW, gDownW}
			biasGrads[layerIdx] = []*Tensor[T]{gGateB, gUpB, gDownB}

		case LayerMultiHeadAttention:
			// MultiHeadAttentionBackward
			weights := &AttentionWeights[T]{
				QWeights:     ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.QWeights, len(config.QWeights))),
				QBias:        ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.QBias, len(config.QBias))),
				KWeights:     ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.KWeights, len(config.KWeights))),
				KBias:        ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.KBias, len(config.KBias))),
				VWeights:     ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.VWeights, len(config.VWeights))),
				VBias:        ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.VBias, len(config.VBias))),
				OutputWeight: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.OutputWeight, len(config.OutputWeight))),
				OutputBias:   ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.OutputBias, len(config.OutputBias))),
				DModel:       config.DModel, NumHeads: config.NumHeads, NumKVHeads: config.NumKVHeads, HeadDim: config.HeadDim,
			}
			gInput, gWeights := MultiHeadAttentionBackward(gradOut, input, weights)

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gWeights // Struct
			biasGrads[layerIdx] = nil

		case LayerResidual:
			// ResidualBackward
			gInput, gSkip := ResidualBackward(gradOut)
			accumulateGradient(grads, layerIdx, gInput)
			// Apply skip gradient to previous layer if dimensions match
			if layerIdx > 0 {
				prevInput := activations[layerIdx-1]
				if prevInput != nil && len(prevInput.Data) == len(input.Data) {
					accumulateGradient(grads, layerIdx-1, gSkip)
				}
			}
			kernelGrads[layerIdx] = nil
			biasGrads[layerIdx] = nil

		case LayerParallel:
			// ParallelBackward
			branchIntermediates, _ := context.([]*Tensor[T])

			// Convert config.ParallelBranches ([]LayerConfig) to []*LayerConfig
			branches := make([]*LayerConfig, len(config.ParallelBranches))
			for i := range config.ParallelBranches {
				branches[i] = &config.ParallelBranches[i]
			}

			gInput, _ := ParallelBackward(gradOut, input, branches, branchIntermediates, config.CombineMode)
			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = nil
			biasGrads[layerIdx] = nil

		case LayerEmbedding:
			// EmbeddingBackward
			// context is nil for embedding backward (input activation used as tokens)
			// Input has shape [seqLen] (token IDs)
			// gradOut has shape [seqLen, embeddingDim]
			gWeights := EmbeddingBackward(gradOut, input, config.VocabSize, config.EmbeddingDim)

			// No gradient w.r.t. input (discrete tokens)
			accumulateGradient(grads, layerIdx, nil)
			kernelGrads[layerIdx] = gWeights
			biasGrads[layerIdx] = nil

		case LayerConv1D:
			// Conv1DBackward
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
			preAct, _ := context.(*Tensor[T])

			seqLen := config.InputHeight
			if seqLen <= 0 {
				seqLen = len(input.Data) / (config.Conv1DInChannels * n.BatchSize)
			}

			gInput, gKernel, gBias := Conv1DBackward(gradOut, input, preAct, weights,
				seqLen, config.Conv1DInChannels,
				config.Conv1DKernelSize, config.Conv1DStride, config.Conv1DPadding,
				config.Conv1DFilters, n.BatchSize, config.Activation)

			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gKernel
			biasGrads[layerIdx] = gBias

		default:
			// Unsupported or activation only
			accumulateGradient(grads, layerIdx, gradOut)
		}
	}

	return grads[0], kernelGrads, biasGrads, time.Since(start)
}

func accumulateGradient[T Numeric](grads []*Tensor[T], index int, g *Tensor[T]) {
	if g == nil {
		return
	}
	if grads[index] == nil {
		grads[index] = g.Clone()
	} else {
		// Add to existing gradient
		for i := range grads[index].Data {
			grads[index].Data[i] += g.Data[i]
		}
	}
}

// BackwardCPU computes gradients via backpropagation on CPU through the grid
// gradOutput: gradient flowing back from the loss (same size as network output)
// Returns: gradient with respect to the input
func (n *Network) BackwardCPU(gradOutput []float32) ([]float32, time.Duration) {
	// GPU auto-routing: if GPU mode is enabled and weights are mounted, use GPU
	if n.GPU && n.gpuMounted {
		start := time.Now()
		output, err := n.backwardGPU(gradOutput)
		if err == nil {
			return output, time.Since(start)
		}
		// Fall back to CPU on GPU error
	}

	start := time.Now()

	if len(n.activations) == 0 || len(n.activations[0]) == 0 {
		// No forward pass has been done
		return make([]float32, len(gradOutput)), time.Since(start)
	}

	// Current gradient
	grad := make([]float32, len(gradOutput))
	copy(grad, gradOutput)

	totalLayers := n.TotalLayers()

	// Backpropagate through grid in reverse order
	for layerIdx := totalLayers - 1; layerIdx >= 0; layerIdx-- {
		// Calculate grid position for this layer
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)
		preAct := n.preActivations[layerIdx]

		// Route to appropriate layer type
		if config.Type == LayerConv2D {
			// Conv2D backward
			input := n.activations[layerIdx]
			gradInput, gradKernel, gradBias := conv2DBackwardCPU(grad, input, preAct, config, n.BatchSize)

			// Store gradients for weight updates
			n.kernelGradients[layerIdx] = gradKernel
			n.biasGradients[layerIdx] = gradBias

			// Update gradient for next layer
			grad = gradInput
		} else if config.Type == LayerMultiHeadAttention {
			// Multi-Head Attention backward
			input := n.activations[layerIdx]
			gradInput, gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB := multiHeadAttentionBackwardCPU(grad, input, preAct, config, n.BatchSize)

			// Store weight gradients (Q, K, V, Output weights)
			allWeightGrads := append(append(append(gradQW, gradKW...), gradVW...), gradOutW...)
			n.kernelGradients[layerIdx] = allWeightGrads

			// Store bias gradients (Q, K, V, Output biases)
			allBiasGrads := append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)
			n.biasGradients[layerIdx] = allBiasGrads

			// Update gradient for next layer
			grad = gradInput
		} else if config.Type == LayerRNN {
			// RNN backward
			input := n.activations[layerIdx]
			hiddenStates := preAct // stored from forward pass

			gradInput, gradWeightIH, gradWeightHH, gradBiasH := rnnBackwardCPU(config, grad, input, hiddenStates,
				n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			// Store gradients concatenated
			allGrads := append(append(gradWeightIH, gradWeightHH...), gradBiasH...)
			n.kernelGradients[layerIdx] = allGrads

			grad = gradInput
		} else if config.Type == LayerLSTM {
			// LSTM backward
			input := n.activations[layerIdx]

			// Unflatten the states from preActivations
			batchSize := n.BatchSize
			seqLength := config.SeqLength
			hiddenSize := config.HiddenSize

			states := make(map[string][]float32)
			states["hidden"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
			states["cell"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
			states["i_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["f_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["g_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["o_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["c_tanh"] = make([]float32, batchSize*seqLength*hiddenSize)

			offset := 0
			for _, key := range []string{"hidden", "cell", "i_gate", "f_gate", "g_gate", "o_gate", "c_tanh"} {
				copy(states[key], preAct[offset:offset+len(states[key])])
				offset += len(states[key])
			}

			gradInput, grads := lstmBackwardCPU(config, grad, input, states,
				batchSize, seqLength, config.RNNInputSize, hiddenSize)

			// Store all gradients concatenated
			allGrads := append(append(append(grads["WeightIH_i"], grads["WeightHH_i"]...), grads["BiasH_i"]...),
				append(append(grads["WeightIH_f"], grads["WeightHH_f"]...), grads["BiasH_f"]...)...)
			allGrads = append(allGrads, append(append(grads["WeightIH_g"], grads["WeightHH_g"]...), grads["BiasH_g"]...)...)
			allGrads = append(allGrads, append(append(grads["WeightIH_o"], grads["WeightHH_o"]...), grads["BiasH_o"]...)...)
			n.kernelGradients[layerIdx] = allGrads

			grad = gradInput
		} else if config.Type == LayerDense {
			// Dense layer backward with proper weight gradients
			input := n.activations[layerIdx]
			gradInput, gradWeights, gradBias := denseBackwardCPU(grad, input, preAct, config, n.BatchSize)

			// Store gradients
			n.kernelGradients[layerIdx] = gradWeights
			n.biasGradients[layerIdx] = gradBias

			// Update gradient for next layer
			grad = gradInput
		} else if config.Type == LayerParallel {
			// Parallel layer backward
			input := n.activations[layerIdx]

			// Unflatten branch pre-activations from storage
			preActData := n.preActivations[layerIdx]
			numBranches := int(preActData[0])
			branchPreActs := make([][]float32, numBranches)
			offset := 1
			for i := 0; i < numBranches; i++ {
				size := int(preActData[offset])
				offset++
				branchPreActs[i] = preActData[offset : offset+size]
				offset += size
			}

			gradInput, kernelGrads, biasGrads, err := parallelBackwardCPU(input, grad, branchPreActs, config, n.BatchSize, "normal")
			if err != nil {
				// Log error but continue backprop
				fmt.Printf("Warning: parallel backward error at layer %d: %v\n", layerIdx, err)
				// Pass gradient through unchanged
			} else {
				grad = gradInput
				// Store gradients for weight updates
				// Flatten all branch gradients
				totalKernelGrads := 0
				totalBiasGrads := 0
				for i := range kernelGrads {
					totalKernelGrads += len(kernelGrads[i])
					totalBiasGrads += len(biasGrads[i])
				}
				n.kernelGradients[layerIdx] = make([]float32, totalKernelGrads)
				n.biasGradients[layerIdx] = make([]float32, totalBiasGrads)
				kOffset := 0
				bOffset := 0
				for i := range kernelGrads {
					copy(n.kernelGradients[layerIdx][kOffset:], kernelGrads[i])
					kOffset += len(kernelGrads[i])
					copy(n.biasGradients[layerIdx][bOffset:], biasGrads[i])
					bOffset += len(biasGrads[i])
				}
			}
		} else if config.Type == LayerEmbedding {
			// EmbeddingBackward
			// Input has shape [seqLen] (token IDs)
			// grad has shape [seqLen, embeddingDim]
			input := n.activations[layerIdx]
			gWeights := embeddingBackwardCPU(grad, input, config)

			// Store gradients
			n.kernelGradients[layerIdx] = gWeights
			n.biasGradients[layerIdx] = nil

			// No gradient w.r.t. input (discrete tokens)
			// But we need to maintain gradient size for previous layer if one exists
			// Usually embedding is first layer, so grad is ignored
			grad = make([]float32, len(input))

		} else if config.Type == LayerConv1D {
			// Conv1DBackward
			input := n.activations[layerIdx]

			gradInput, gradKernel, gradBias := conv1DBackwardCPU(grad, input, preAct, config, n.BatchSize)

			// Store gradients
			n.kernelGradients[layerIdx] = gradKernel
			n.biasGradients[layerIdx] = gradBias

			// Update gradient for next layer
			grad = gradInput

		} else if config.Type == LayerSoftmax {
			// Softmax layer backward - NOT element-wise!
			// Get the softmax output (which is stored in activations)
			softmaxOutput := n.activations[layerIdx+1] // output of this layer

			// Handle different softmax variants
			if config.SoftmaxRows > 0 {
				// Grid Softmax: per-row softmax gradient
				rows := config.SoftmaxRows
				cols := config.SoftmaxCols
				gradInput := make([]float32, len(grad))

				for row := 0; row < rows; row++ {
					start := row * cols
					end := start + cols

					// For each position in this row
					for i := start; i < end; i++ {
						var gradSum float32
						// Jacobian-vector product: dL/dx_i = Σ_j (dL/dy_j * dy_j/dx_i)
						// where dy_j/dx_i = y_i * (δ_ij - y_i) for softmax
						for j := start; j < end; j++ {
							jacobian := softmaxOutput[j] * (kroneckerFloat(i, j) - softmaxOutput[i])
							gradSum += grad[j] * jacobian
						}
						gradInput[i] = gradSum
					}
				}
				grad = gradInput
			} else {
				// Standard softmax: single softmax over all values
				gradInput := make([]float32, len(grad))

				for i := range gradInput {
					var gradSum float32
					for j := range grad {
						jacobian := softmaxOutput[j] * (kroneckerFloat(i, j) - softmaxOutput[i])
						gradSum += grad[j] * jacobian
					}
					gradInput[i] = gradSum
				}
				grad = gradInput
			}
		} else if config.Type == LayerNorm {
			// LayerNorm backward
			input := n.activations[layerIdx]
			// residual? Not stored in activations usually, unless residual layer.
			// For standalone LayerNorm, residual is nil.

			gradInput, gradGamma, gradBeta := layerNormBackwardCPU(input, nil, grad, config, n.BatchSize)

			n.kernelGradients[layerIdx] = gradGamma
			n.biasGradients[layerIdx] = gradBeta

			grad = gradInput
		} else {
		}

		// Notify observer for this layer (normal mode)
		if config.Observer != nil {
			notifyObserver(config, "normal", "backward", layerIdx, nil, grad, 0)
		}
	}

	return grad, time.Since(start)
}

// BackwardGPU computes gradients via backpropagation on GPU
// Note: This requires storing activations from forward pass
// For now, this is a simplified version that applies derivatives
func (n *Network) BackwardGPU(gradOutput []float32) ([]float32, time.Duration, error) {
	if n.deviceInfo == nil {
		return nil, 0, fmt.Errorf("GPU not initialized, call InitGPU first")
	}

	if len(n.preActivations) == 0 {
		return nil, 0, fmt.Errorf("no forward pass data available for backward pass")
	}

	// Check for specialized layer types and use dedicated GPU paths
	hasConv2D := false
	hasMHA := false
	for i := 0; i < n.TotalLayers(); i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		remainder := i % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)
		if config.Type == LayerConv2D {
			hasConv2D = true
		}
		if config.Type == LayerMultiHeadAttention {
			hasMHA = true
		}
	}

	if hasConv2D {
		return n.backwardGPUConv2D(gradOutput)
	}

	if hasMHA {
		return n.backwardGPUMultiHeadAttention(gradOutput)
	}

	start := time.Now()

	dev := n.deviceInfo.Device
	q := n.deviceInfo.Queue
	wgx := n.deviceInfo.WorkgroupX

	N := len(gradOutput)
	bytes := uint64(N * 4)
	totalLayers := n.TotalLayers()

	// Build pipelines for backward pass (derivative computations) - cached after first call
	var pipelines []*wgpu.ComputePipeline
	var bgls []*wgpu.BindGroupLayout

	if n.deviceInfo.backwardPipelines == nil {
		// First time - create and cache pipelines
		pipelines = make([]*wgpu.ComputePipeline, 5)
		bgls = make([]*wgpu.BindGroupLayout, 5)

		for act := 0; act < 5; act++ {
			shader := generateBackwardShader(wgx, act, N)
			module, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
				Label:          fmt.Sprintf("nn_bwd_shader_%d", act),
				WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
			})
			if err != nil {
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, fmt.Errorf("CreateShaderModule %d: %w", act, err)
			}

			bgl, err := dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
				Label: fmt.Sprintf("nn_bwd_bgl_%d", act),
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // grad_in
					{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}}, // pre_activation
					{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},         // grad_out
				},
			})
			if err != nil {
				module.Release()
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, err
			}
			bgls[act] = bgl

			pl, err := dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
				Label:            fmt.Sprintf("nn_bwd_pl_%d", act),
				BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
			})
			if err != nil {
				module.Release()
				bgl.Release()
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, err
			}

			pipeline, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
				Label:  fmt.Sprintf("nn_bwd_pipeline_%d", act),
				Layout: pl,
				Compute: wgpu.ProgrammableStageDescriptor{
					Module:     module,
					EntryPoint: "main",
				},
			})
			if err != nil {
				pl.Release()
				bgl.Release()
				module.Release()
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, err
			}

			pipelines[act] = pipeline
			pl.Release()
			module.Release()
		}

		// Cache the pipelines
		n.deviceInfo.backwardPipelines = pipelines
		n.deviceInfo.backwardBGLs = bgls
	} else {
		// Use cached pipelines
		pipelines = n.deviceInfo.backwardPipelines
		bgls = n.deviceInfo.backwardBGLs
	}

	// Create buffers for gradient computation
	bufGradA, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nn_bwd_grad_A",
		Size:  bytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, 0, err
	}
	defer bufGradA.Release()

	bufGradB, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nn_bwd_grad_B",
		Size:  bytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, 0, err
	}
	defer bufGradB.Release()

	// Create buffers for pre-activations
	preActBuffers := make([]*wgpu.Buffer, totalLayers)
	for i := 0; i < totalLayers; i++ {
		buf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: fmt.Sprintf("nn_bwd_preact_%d", i),
			Size:  bytes,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			for j := 0; j < i; j++ {
				preActBuffers[j].Release()
			}
			return nil, 0, err
		}
		preActBuffers[i] = buf

		// Upload pre-activation data from CPU
		preActData := n.preActivations[i]
		q.WriteBuffer(buf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&preActData[0])), int(bytes)))
	}

	defer func() {
		for _, buf := range preActBuffers {
			buf.Release()
		}
	}()

	readback, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nn_bwd_RB",
		Size:  bytes,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, 0, err
	}
	defer readback.Release()

	// Write initial gradient
	q.WriteBuffer(bufGradA, 0, unsafe.Slice((*byte)(unsafe.Pointer(&gradOutput[0])), int(bytes)))
	pollDevice(dev, 100)

	gx := uint32((N + int(wgx) - 1) / int(wgx))
	if gx == 0 {
		gx = 1
	}

	// Create single command encoder for all layers
	enc, err := dev.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "nn_bwd_enc_all",
	})
	if err != nil {
		return nil, 0, fmt.Errorf("create encoder: %w", err)
	}

	// Track bind groups to release after submission
	bindGroupsToRelease := make([]*wgpu.BindGroup, 0, totalLayers)

	// Backpropagate through grid in reverse order
	for layerIdx := totalLayers - 1; layerIdx >= 0; layerIdx-- {
		// Calculate grid position for this layer
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		activation := int(n.GetActivation(row, col, layer))
		pipeline := pipelines[activation]

		// Determine buffer indices (ping-pong)
		reverseIdx := totalLayers - 1 - layerIdx
		var gradIn, gradOut *wgpu.Buffer
		if reverseIdx%2 == 0 {
			gradIn = bufGradA
			gradOut = bufGradB
		} else {
			gradIn = bufGradB
			gradOut = bufGradA
		}

		preActBuf := preActBuffers[layerIdx]

		bgl := bgls[activation]
		bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("nn_bwd_bg_%d", layerIdx),
			Layout: bgl,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: gradIn, Offset: 0, Size: gradIn.GetSize()},
				{Binding: 1, Buffer: preActBuf, Offset: 0, Size: preActBuf.GetSize()},
				{Binding: 2, Buffer: gradOut, Offset: 0, Size: gradOut.GetSize()},
			},
		})
		if err != nil {
			enc.Release()
			for _, bgr := range bindGroupsToRelease {
				bgr.Release()
			}
			return nil, 0, fmt.Errorf("create backward bindgroup %d: %w", layerIdx, err)
		}
		bindGroupsToRelease = append(bindGroupsToRelease, bg)

		pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("nn_bwd_pass_%d", layerIdx),
		})
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, 1, 1)
		pass.End()
	}

	// Submit all layers at once
	cb, err := enc.Finish(nil)
	if err != nil {
		enc.Release()
		for _, bg := range bindGroupsToRelease {
			bg.Release()
		}
		return nil, 0, fmt.Errorf("finish command buffer: %w", err)
	}
	enc.Release()
	q.Submit(cb)
	cb.Release()

	// Release bind groups
	for _, bg := range bindGroupsToRelease {
		bg.Release()
	}
	pollDevice(dev, 1000)

	// Determine final gradient buffer
	var finalGrad *wgpu.Buffer
	if totalLayers%2 == 0 {
		finalGrad = bufGradA
	} else {
		finalGrad = bufGradB
	}

	// Copy back results
	enc2, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, 0, err
	}
	enc2.CopyBufferToBuffer(finalGrad, 0, readback, 0, bytes)
	cb2, err := enc2.Finish(nil)
	if err != nil {
		enc2.Release()
		return nil, 0, err
	}
	enc2.Release()

	q.Submit(cb2)
	cb2.Release()
	pollDevice(dev, 1000)

	// Map and read results
	done := false
	readback.MapAsync(wgpu.MapModeRead, 0, bytes, func(wgpu.BufferMapAsyncStatus) { done = true })
	for i := 0; i < 1000 && !done; i++ {
		dev.Poll(true, nil)
		time.Sleep(100 * time.Microsecond)
	}

	if !done {
		return nil, 0, fmt.Errorf("timeout mapping readback buffer")
	}

	view := readback.GetMappedRange(0, uint(bytes))
	gradInput := make([]float32, N)
	copy(gradInput, unsafe.Slice((*float32)(unsafe.Pointer(&view[0])), N))
	readback.Unmap()

	return gradInput, time.Since(start), nil
}

// backwardGPUConv2D executes backward pass for networks containing Conv2D layers on GPU
// This uses specialized pipelines for Conv2D layers
func (n *Network) backwardGPUConv2D(gradOutput []float32) ([]float32, time.Duration, error) {
	start := time.Now()

	// For now, use CPU for backward pass
	// Full GPU implementation requires separate shaders for:
	// - grad_input computation (transposed convolution)
	// - grad_kernel computation (correlation)
	// - grad_bias computation (reduction)
	gradInput, _ := n.BackwardCPU(gradOutput)

	return gradInput, time.Since(start), nil
}

// kroneckerFloat returns 1.0 if i==j, else 0.0 (for float32 computations)
func kroneckerFloat(i, j int) float32 {
	if i == j {
		return 1.0
	}
	return 0.0
}
