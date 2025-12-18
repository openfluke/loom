package nn

import (
	"fmt"
	"sync"
	"time"
)

// StepState holds the current state of each layer for stepping execution
type StepState struct {
	// Current data at each layer (input/output buffers)
	layerData [][]float32

	// Pre-activation values for each layer (for backprop)
	layerPreAct [][]float32

	// Residual connections tracking
	residuals [][]float32

	// Lock for thread-safe stepping
	mu sync.RWMutex

	// Step counter
	stepCount uint64
}

// Helper to calculate output size recursively
func getLayerOutputSize(config *LayerConfig, batchSize int) int {
	if config.Type == LayerConv2D {
		return config.Filters * config.OutputHeight * config.OutputWidth * batchSize
	} else if config.Type == LayerDense {
		return config.OutputHeight
	} else if config.Type == LayerRNN || config.Type == LayerLSTM {
		seqLen := config.SeqLength
		if seqLen <= 0 {
			seqLen = 1 // Default to 1 if not set
		}
		return batchSize * seqLen * config.HiddenSize
	} else if config.Type == LayerMultiHeadAttention {
		// MHA: prefer DModel with SeqLength, fallback to DModel alone, then OutputHeight
		if config.DModel > 0 {
			seqLen := config.SeqLength
			if seqLen <= 0 {
				seqLen = 1 // Default to 1 if not set
			}
			return batchSize * seqLen * config.DModel
		}
		if config.OutputHeight > 0 {
			return config.OutputHeight
		}
		return -1
	} else if config.Type == LayerSwiGLU {
		// SwiGLU: prefer OutputHeight, fallback to InputHeight
		if config.OutputHeight > 0 {
			return config.OutputHeight
		}
		return config.InputHeight
	} else if config.Type == LayerNorm || config.Type == LayerRMSNorm {
		// Norm layers: prefer NormSize, fallback to OutputHeight, then InputHeight
		if config.NormSize > 0 {
			return config.NormSize
		}
		if config.OutputHeight > 0 {
			return config.OutputHeight
		}
		if config.InputHeight > 0 {
			return config.InputHeight
		}
		return -1
	} else if config.Type == LayerParallel {
		// Calculate based on combine mode
		totalSize := 0

		if config.CombineMode == "add" || config.CombineMode == "avg" || config.CombineMode == "average" {
			// Output size is same as first branch (all branches must match)
			if len(config.ParallelBranches) > 0 {
				totalSize = getLayerOutputSize(&config.ParallelBranches[0], batchSize)
			}
		} else {
			// Concat or Grid Scatter: Sum of all branch outputs
			for i := range config.ParallelBranches {
				totalSize += getLayerOutputSize(&config.ParallelBranches[i], batchSize)
			}
		}
		return totalSize
	}

	// Default: Try to use OutputHeight if available
	if config.OutputHeight > 0 {
		return config.OutputHeight
	}
	if config.InputHeight > 0 {
		return config.InputHeight
	}

	// Unknown (should be handled by caller using input size)
	return -1
}

// InitStepState initializes the stepping state for the network
func (n *Network) InitStepState(inputSize int) *StepState {
	totalLayers := n.TotalLayers()

	state := &StepState{
		layerData:   make([][]float32, totalLayers+1), // +1 for input layer
		layerPreAct: make([][]float32, totalLayers),
		residuals:   make([][]float32, totalLayers),
		stepCount:   0,
	}

	// Initialize layer 0 (input) with zeros
	state.layerData[0] = make([]float32, inputSize)

	// Initialize all other layers based on their expected output sizes
	layerIdx := 0
	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				config := n.GetLayer(row, col, layer)

				// Estimate output size based on layer type
				outputSize := getLayerOutputSize(config, n.BatchSize)

				// If unknown (e.g. simple activation layer), assumes size preserves input
				if outputSize == -1 {
					outputSize = len(state.layerData[layerIdx]) // Use previous layer's size
					if outputSize == 0 {
						outputSize = inputSize
					} // Fallback
				}

				state.layerData[layerIdx+1] = make([]float32, outputSize)
				state.layerPreAct[layerIdx] = make([]float32, outputSize)
				state.residuals[layerIdx] = make([]float32, outputSize)

				layerIdx++
			}
		}
	}

	return state
}

// SetInput sets the input data for the network (layer 0)
func (state *StepState) SetInput(input []float32) {
	state.mu.Lock()
	defer state.mu.Unlock()

	if len(state.layerData[0]) != len(input) {
		state.layerData[0] = make([]float32, len(input))
	}
	copy(state.layerData[0], input)
}

// GetOutput retrieves the output from the final layer
func (state *StepState) GetOutput() []float32 {
	state.mu.RLock()
	defer state.mu.RUnlock()

	finalLayer := len(state.layerData) - 1
	output := make([]float32, len(state.layerData[finalLayer]))
	copy(output, state.layerData[finalLayer])
	return output
}

// GetLayerOutput retrieves the current output of a specific layer
func (state *StepState) GetLayerOutput(layerIdx int) []float32 {
	state.mu.RLock()
	defer state.mu.RUnlock()

	if layerIdx < 0 || layerIdx >= len(state.layerData) {
		return nil
	}

	output := make([]float32, len(state.layerData[layerIdx]))
	copy(output, state.layerData[layerIdx])
	return output
}

// GetLayerData returns the internal layer data (for debugging)
func (state *StepState) GetLayerData() [][]float32 {
	state.mu.RLock()
	defer state.mu.RUnlock()
	return state.layerData
}

// StepForward executes one step for ALL layers simultaneously
// Each layer processes its current input and updates its output
func (n *Network) StepForward(state *StepState) time.Duration {
	start := time.Now()

	state.mu.Lock()
	defer state.mu.Unlock()

	totalLayers := n.TotalLayers()

	// Create temporary storage for new outputs (double buffering)
	newOutputs := make([][]float32, totalLayers+1)
	newOutputs[0] = state.layerData[0] // Input stays the same

	// Process each layer independently using its current input
	layerIdx := 0
	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				config := n.GetLayer(row, col, layer)

				// Get current input for this layer
				input := state.layerData[layerIdx]

				// Process based on layer type
				var preAct, postAct []float32

				switch config.Type {
				case LayerConv2D:
					preAct, postAct = conv2DForwardCPU(input, config, n.BatchSize)

				case LayerMultiHeadAttention:
					preAct, postAct = MultiHeadAttentionForwardCPU(input, config, n.BatchSize)

					// Add residual if available
					if state.residuals[layerIdx] != nil && len(state.residuals[layerIdx]) == len(postAct) {
						for i := range postAct {
							postAct[i] += state.residuals[layerIdx][i]
						}
					}

					// Store as new residual
					state.residuals[layerIdx] = make([]float32, len(postAct))
					copy(state.residuals[layerIdx], postAct)

				case LayerRNN:
					output, hiddenStates := rnnForwardCPU(config, input, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
					preAct = hiddenStates
					postAct = output

				case LayerLSTM:
					output, states := lstmForwardCPU(config, input, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

					// Flatten states
					totalStateSize := len(states["hidden"]) + len(states["cell"]) +
						len(states["i_gate"]) + len(states["f_gate"]) +
						len(states["g_gate"]) + len(states["o_gate"]) + len(states["c_tanh"])
					flatStates := make([]float32, totalStateSize)
					offset := 0
					for _, key := range []string{"hidden", "cell", "i_gate", "f_gate", "g_gate", "o_gate", "c_tanh"} {
						copy(flatStates[offset:], states[key])
						offset += len(states[key])
					}
					preAct = flatStates
					postAct = output

				case LayerSoftmax:
					probs, err := ForwardSoftmaxCPU(input, config)
					if err != nil {
						probs = softmaxStandard(input, 1.0)
					}
					preAct = make([]float32, len(input))
					copy(preAct, input)
					postAct = probs

				case LayerDense:
					preAct, postAct = denseForwardCPU(input, config, n.BatchSize)

				case LayerSwiGLU:
					preAct, postAct = SwiGLUForwardCPU(input, config, n.BatchSize)

					// Add residual if available
					if state.residuals[layerIdx] != nil && len(state.residuals[layerIdx]) == len(postAct) {
						for i := range postAct {
							postAct[i] += state.residuals[layerIdx][i]
						}
					}

					// Store as new residual
					state.residuals[layerIdx] = make([]float32, len(postAct))
					copy(state.residuals[layerIdx], postAct)

				case LayerNorm:
					postAct = layerNormForwardCPU(input, state.residuals[layerIdx], config, n.BatchSize)
					preAct = make([]float32, len(input))
					copy(preAct, input)

					// Update residual
					if state.residuals[layerIdx] != nil {
						state.residuals[layerIdx] = make([]float32, len(postAct))
						copy(state.residuals[layerIdx], postAct)
					}

				case LayerRMSNorm:
					postAct = rmsNormForwardCPU(input, nil, config, n.BatchSize)
					preAct = make([]float32, len(input))
					copy(preAct, input)

					// Save as residual
					state.residuals[layerIdx] = make([]float32, len(input))
					copy(state.residuals[layerIdx], input)

				case LayerParallel:
					output, branchPreActs, err := parallelForwardCPU(input, config, n.BatchSize, "step")
					if err != nil {
						fmt.Printf("Parallel layer error: %v\n", err)
						output = input
						branchPreActs = nil
					}

					// Flatten branch pre-acts
					totalPreActSize := 0
					for _, preActBranch := range branchPreActs {
						totalPreActSize += len(preActBranch)
					}
					metadataSize := 1 + len(branchPreActs)
					preAct = make([]float32, totalPreActSize+metadataSize)
					preAct[0] = float32(len(branchPreActs))
					offset := 1
					for i, preActBranch := range branchPreActs {
						preAct[offset] = float32(len(preActBranch))
						offset++
						copy(preAct[offset:], preActBranch)
						offset += len(preActBranch)
						_ = i
					}
					postAct = output

				default:
					// Element-wise activation
					preAct = make([]float32, len(input))
					copy(preAct, input)

					postAct = make([]float32, len(input))
					for i := 0; i < len(input); i++ {
						postAct[i] = activateCPU(input[i], config.Activation)
					}
				}

				// Notify observer if present (step mode)
				if config.Observer != nil {
					notifyObserver(config, "step", "forward", layerIdx, input, postAct, state.stepCount)
				}

				// Store results
				state.layerPreAct[layerIdx] = preAct
				newOutputs[layerIdx+1] = postAct

				layerIdx++
			}
		}
	}

	// Swap buffers - update all layer outputs atomically
	for i := 1; i < len(state.layerData); i++ {
		state.layerData[i] = newOutputs[i]
	}

	state.stepCount++

	return time.Since(start)
}

// StepForwardSingle executes one step for a SINGLE layer
// This allows even finer-grained control over propagation
func (n *Network) StepForwardSingle(state *StepState, layerIdx int) time.Duration {
	start := time.Now()

	state.mu.Lock()
	defer state.mu.Unlock()

	if layerIdx < 0 || layerIdx >= n.TotalLayers() {
		return time.Since(start)
	}

	// Calculate grid position
	row := layerIdx / (n.GridCols * n.LayersPerCell)
	remainder := layerIdx % (n.GridCols * n.LayersPerCell)
	col := remainder / n.LayersPerCell
	layer := remainder % n.LayersPerCell

	config := n.GetLayer(row, col, layer)
	input := state.layerData[layerIdx]

	var preAct, postAct []float32

	// Same processing as StepForward but for single layer
	switch config.Type {
	case LayerConv2D:
		preAct, postAct = conv2DForwardCPU(input, config, n.BatchSize)

	case LayerMultiHeadAttention:
		preAct, postAct = MultiHeadAttentionForwardCPU(input, config, n.BatchSize)
		if state.residuals[layerIdx] != nil && len(state.residuals[layerIdx]) == len(postAct) {
			for i := range postAct {
				postAct[i] += state.residuals[layerIdx][i]
			}
		}
		state.residuals[layerIdx] = make([]float32, len(postAct))
		copy(state.residuals[layerIdx], postAct)

	case LayerDense:
		preAct, postAct = denseForwardCPU(input, config, n.BatchSize)

	default:
		preAct = make([]float32, len(input))
		copy(preAct, input)
		postAct = make([]float32, len(input))
		for i := 0; i < len(input); i++ {
			postAct[i] = activateCPU(input[i], config.Activation)
		}
	}

	// Notify observer if present (step mode)
	if config.Observer != nil {
		notifyObserver(config, "step", "forward", layerIdx, input, postAct, state.stepCount)
	}

	// Update this layer's state
	state.layerPreAct[layerIdx] = preAct
	state.layerData[layerIdx+1] = postAct

	return time.Since(start)
}

// GetStepCount returns the current step count
func (state *StepState) GetStepCount() uint64 {
	state.mu.RLock()
	defer state.mu.RUnlock()
	return state.stepCount
}
