package nn

import (
	"fmt"
	"math"
	"time"
)

// =============================================================================
// Generic Backward Pass Support
// =============================================================================

// GenericBackwardResult holds the results of a generic backward pass.
type GenericBackwardResult[T Numeric] struct {
	GradInput     *Tensor[T]
	KernelGrads   *Tensor[T]
	BiasGrads     *Tensor[T]
}

// StepBackwardGeneric executes backward pass for GenericStepState.
// This is a placeholder for the generic backward pass implementation.
// StepBackwardGeneric executes backward pass for GenericStepState.
// Returns input gradient, kernel gradients, bias gradients, and duration.
func StepBackwardGeneric[T Numeric](
	n *Network,
	state *GenericStepState[T],
	gradOutput *Tensor[T],
) (*Tensor[T], []any, []any, time.Duration) {
	start := time.Now()

	state.mu.Lock()
	defer state.mu.Unlock()

	totalLayers := n.TotalLayers()
	if len(state.LayerData) <= totalLayers {
		return NewTensor[T](len(gradOutput.Data)), nil, nil, time.Since(start)
	}

	// Gradients for activations [0...totalLayers]
	grads := make([]*Tensor[T], totalLayers+1)
	
	// Initialize output gradient
	grads[totalLayers] = gradOutput.Clone()
	
	// Storage for weight gradients
	kernelGrads := make([]any, totalLayers)
	biasGrads := make([]any, totalLayers)

	// Backpropagate through grid in reverse order
	for layerIdx := totalLayers - 1; layerIdx >= 0; layerIdx-- {
		// Calculate grid position
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)

		// Get inputs and context
		input := state.LayerData[layerIdx]
		context := state.BackwardContext[layerIdx]
		gradOut := grads[layerIdx+1]

		if gradOut == nil {
			gradOut = NewTensor[T](len(input.Data))
		}

		if config.IsDisabled {
			accumulateGradient(grads, layerIdx, gradOut)
			continue
		}

		switch config.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
			preAct, _ := context.(*Tensor[T])
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
			wIH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH, len(config.WeightIH)))
			wHH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH, len(config.WeightHH)))
			hiddenStates, _ := context.(*Tensor[T])
			gInput, gWIH, gWHH, gBiasH := RNNBackward(gradOut, input, hiddenStates, wIH, wHH, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
			
			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = []*Tensor[T]{gWIH, gWHH}
			biasGrads[layerIdx] = gBiasH

		case LayerLSTM:
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
			kernelGrads[layerIdx] = gWeights
			biasGrads[layerIdx] = nil

		case LayerSoftmax:
			softmaxOut, _ := context.(*Tensor[T])
			gInput := SoftmaxBackward(gradOut, softmaxOut, config.SoftmaxRows, config.SoftmaxCols)
			accumulateGradient(grads, layerIdx, gInput)

		case LayerNorm:
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
			beta := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Beta, len(config.Beta)))
			gInput, gGamma, gBeta := LayerNormBackward(input, nil, gradOut, gamma, beta, config.NormSize, n.BatchSize, float64(config.Epsilon))
			
			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gGamma
			biasGrads[layerIdx] = gBeta

		case LayerRMSNorm:
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
			gInput, gGamma := RMSNormBackward(input, nil, gradOut, gamma, config.NormSize, n.BatchSize, float64(config.Epsilon))
			
			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gGamma

		case LayerSwiGLU:
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
			weights := &AttentionWeights[T]{
				QWeights: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.QWeights, len(config.QWeights))),
				QBias:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.QBias, len(config.QBias))),
				KWeights: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.KWeights, len(config.KWeights))),
				KBias:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.KBias, len(config.KBias))),
				VWeights: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.VWeights, len(config.VWeights))),
				VBias:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.VBias, len(config.VBias))),
				OutputWeight: ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.OutputWeight, len(config.OutputWeight))),
				OutputBias:   ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.OutputBias, len(config.OutputBias))),
				DModel: config.DModel, NumHeads: config.NumHeads, NumKVHeads: config.NumKVHeads, HeadDim: config.HeadDim,
			}
			gInput, gWeights := MultiHeadAttentionBackward(gradOut, input, weights)
			
			accumulateGradient(grads, layerIdx, gInput)
			kernelGrads[layerIdx] = gWeights

		case LayerResidual:
			gInput, gSkip := ResidualBackward(gradOut)
			accumulateGradient(grads, layerIdx, gInput)
			if layerIdx > 0 {
				accumulateGradient(grads, layerIdx-1, gSkip)
			}

		case LayerParallel:
			branchIntermediates, _ := context.([]*Tensor[T])
			branches := make([]*LayerConfig, len(config.ParallelBranches))
			for i := range config.ParallelBranches {
				branches[i] = &config.ParallelBranches[i]
			}
			gInput, _ := ParallelBackward(gradOut, input, branches, branchIntermediates, config.CombineMode)
			accumulateGradient(grads, layerIdx, gInput)

		default:
			accumulateGradient(grads, layerIdx, gradOut)
		}
		
		// Gradient Scaling / Attention (Optional, matching float32 logic if desired)
		// applySoftmaxGradientScalingGeneric(kernelGrads[layerIdx], biasGrads[layerIdx])
	}
	
	state.StepCount++
	return grads[0], kernelGrads, biasGrads, time.Since(start)
}

// =============================================================================
// Original float32 Implementation
// =============================================================================

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
	for layerIdx := totalLayers - 1; layerIdx >= 0; layerIdx-- {
		// Calculate grid position
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)

		// Get inputs and pre-activations from StepState
		input := state.layerData[layerIdx]
		preAct := state.layerPreAct[layerIdx]

		// Skip if no state (e.g., first step)
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

			// Flatten all weight gradients into one slice for storage
			kernelGrads = append(append(append(gradQW, gradKW...), gradVW...), gradOutW...)
			biasGrads = append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)

		case LayerRNN:
			var gradWeightIH, gradWeightHH, gradBiasH []float32
			gradInput, gradWeightIH, gradWeightHH, gradBiasH = rnnBackwardCPU(config, grad, input, preAct,
				n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			kernelGrads = append(gradWeightIH, gradWeightHH...)
			biasGrads = gradBiasH

		case LayerLSTM:
			states := reconstructLSTMStates(preAct, n.BatchSize, config.SeqLength, config.HiddenSize)
			var grads map[string][]float32
			gradInput, grads = lstmBackwardCPU(config, grad, input, states,
				n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

			// Flatten LSTM gradients
			kernelGrads = append(append(append(grads["WeightIH_i"], grads["WeightHH_i"]...),
				append(grads["WeightIH_f"], grads["WeightHH_f"]...)...),
				append(append(grads["WeightIH_g"], grads["WeightHH_g"]...),
					append(grads["WeightIH_o"], grads["WeightHH_o"]...)...)...)

			biasGrads = append(append(append(grads["BiasH_i"], grads["BiasH_f"]...), grads["BiasH_g"]...), grads["BiasH_o"]...)

		case LayerDense:
			gradInput, kernelGrads, biasGrads = denseBackwardCPU(grad, input, preAct, config, n.BatchSize)

		case LayerSwiGLU:
			// Placeholder: SwiGLU backward (Dense-like approximation if specific function missing)
			// Ideally call SwiGLUBackwardCPU(grad, input, preAct, config, n.BatchSize)
			// For now, we pass gradient through (Identity) to prevent crash if unimplemented
			gradInput = make([]float32, len(input))
			copy(gradInput, grad)

		case LayerNorm, LayerRMSNorm:
			// Normalization backward
			// Placeholder: Identity gradient
			gradInput = make([]float32, len(input))
			copy(gradInput, grad)

		case LayerParallel:
			// Unpack the flattened pre-activations from StepState
			// Format: [numBranches, size1, data1..., size2, data2...]
			if len(preAct) > 0 {
				numBranches := int(preAct[0])
				branchPreActs := make([][]float32, numBranches)
				offset := 1
				for i := 0; i < numBranches; i++ {
					if offset >= len(preAct) {
						break
					}
					size := int(preAct[offset])
					offset++
					if offset+size <= len(preAct) {
						branchPreActs[i] = preAct[offset : offset+size]
						offset += size
					}
				}

				var nestedKernelGrads, nestedBiasGrads [][]float32
				var err error

				// Delegate to parallelBackwardCPU (handles all combine modes)
				gradInput, nestedKernelGrads, nestedBiasGrads, err = parallelBackwardCPU(input, grad, branchPreActs, config, n.BatchSize, "step")

				if err != nil {
					fmt.Printf("Parallel Backward Error: %v\n", err)
					gradInput = make([]float32, len(input))
				} else {
					// Flatten nested gradients into single slices for storage
					totalK, totalB := 0, 0
					for _, g := range nestedKernelGrads {
						totalK += len(g)
					}
					for _, g := range nestedBiasGrads {
						totalB += len(g)
					}

					kernelGrads = make([]float32, totalK)
					biasGrads = make([]float32, totalB)

					kOff, bOff := 0, 0
					for i := range nestedKernelGrads {
						copy(kernelGrads[kOff:], nestedKernelGrads[i])
						kOff += len(nestedKernelGrads[i])
						copy(biasGrads[bOff:], nestedBiasGrads[i])
						bOff += len(nestedBiasGrads[i])
					}
				}
			} else {
				// No pre-act data? Zero grad.
				gradInput = make([]float32, len(input))
			}

		case LayerSoftmax:
			softmaxOutput := state.layerData[layerIdx+1]
			gradInput = make([]float32, len(grad))

			// Handle Grid Softmax vs Standard
			if config.SoftmaxRows > 0 && config.SoftmaxCols > 0 {
				rows := config.SoftmaxRows
				cols := config.SoftmaxCols
				for r := 0; r < rows; r++ {
					start := r * cols
					end := start + cols
					for i := start; i < end; i++ {
						var sum float32
						for j := start; j < end; j++ {
							delta := float32(0.0)
							if i == j {
								delta = 1.0
							}
							sum += grad[j] * softmaxOutput[j] * (delta - softmaxOutput[i])
						}
						gradInput[i] = sum
					}
				}
			} else {
				for i := range gradInput {
					var sum float32
					for j := range grad {
						delta := float32(0.0)
						if i == j {
							delta = 1.0
						}
						sum += grad[j] * softmaxOutput[j] * (delta - softmaxOutput[i])
					}
					gradInput[i] = sum
				}
			}

		default:
			// Activation-only layer (Element-wise)
			gradInput = make([]float32, len(grad))
			for i := 0; i < len(grad); i++ {
				derivative := activateDerivativeCPU(preAct[i], config.Activation)
				gradInput[i] = grad[i] * derivative
			}
		}

		// === Gradient Attention / Scaling ===
		if len(kernelGrads) > 0 {
			applySoftmaxGradientScaling(kernelGrads)
			n.kernelGradients[layerIdx] = kernelGrads
		}
		if len(biasGrads) > 0 {
			applySoftmaxGradientScaling(biasGrads)
			n.biasGradients[layerIdx] = biasGrads
		}

		// Notify observer if present (step mode)
		if config.Observer != nil {
			notifyObserver(config, "step", "backward", layerIdx, nil, gradInput, state.stepCount)
		}

		// Pass gradient to next layer
		grad = gradInput
	}

	return grad, time.Since(start)
}

// applySoftmaxGradientScaling applies a softmax-based scaling to the gradients
// Formula: G_new = G_old * (Softmax(|G_old|) * N)
func applySoftmaxGradientScaling(grads []float32) {
	if len(grads) == 0 {
		return
	}

	maxAbs := float32(0.0)
	for _, g := range grads {
		abs := float32(math.Abs(float64(g)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	exps := make([]float32, len(grads))
	sumExp := float32(0.0)
	for i, g := range grads {
		abs := float32(math.Abs(float64(g)))
		exps[i] = float32(math.Exp(float64(abs - maxAbs)))
		sumExp += exps[i]
	}

	N := float32(len(grads))
	for i := range grads {
		softmaxScore := exps[i] / sumExp
		scaleFactor := softmaxScore * N
		grads[i] *= scaleFactor
	}
}

// Helper to reconstruct LSTM states from flat slice
func reconstructLSTMStates(flat []float32, batchSize, seqLength, hiddenSize int) map[string][]float32 {
	states := make(map[string][]float32)
	hSize := batchSize * (seqLength + 1) * hiddenSize
	gSize := batchSize * seqLength * hiddenSize

	off := 0
	slice := func(sz int) []float32 {
		if off+sz > len(flat) {
			return make([]float32, sz)
		}
		s := flat[off : off+sz]
		off += sz
		return s
	}

	states["hidden"] = slice(hSize)
	states["cell"] = slice(hSize)
	states["i_gate"] = slice(gSize)
	states["f_gate"] = slice(gSize)
	states["g_gate"] = slice(gSize)
	states["o_gate"] = slice(gSize)
	states["c_tanh"] = slice(gSize)

	return states
}
