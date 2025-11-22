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
				// This is a simplified estimation - you may need to adjust
				outputSize := inputSize
				if config.Type == LayerConv2D {
					outputSize = config.OutChannels * config.OutputHeight * config.OutputWidth * n.BatchSize
				} else if config.Type == LayerDense {
					outputSize = config.OutputSize
				} else if config.Type == LayerRNN || config.Type == LayerLSTM {
					outputSize = n.BatchSize * config.SeqLength * config.HiddenSize
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
					output, branchPreActs, err := parallelForwardCPU(input, config, n.BatchSize)
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
