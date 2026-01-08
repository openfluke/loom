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

// =============================================================================
// Generic StepState Implementation
// =============================================================================

// GenericStepState holds the current state of each layer for any numeric type.
type GenericStepState[T Numeric] struct {
	// Current data at each layer (input/output buffers)
	LayerData []*Tensor[T]

	// Backward context for each layer (intermediate states needed for backprop)
	// Can be *Tensor[T], map[string]*Tensor[T], or other types depending on layer
	BackwardContext []any

	// Residual connections tracking
	Residuals []*Tensor[T]

	// Lock for thread-safe stepping
	mu sync.RWMutex

	// Step counter
	StepCount uint64
}

// NewGenericStepState creates a new generic step state for the given network.
func NewGenericStepState[T Numeric](totalLayers, inputSize int) *GenericStepState[T] {
	state := &GenericStepState[T]{
		LayerData:       make([]*Tensor[T], totalLayers+1),
		BackwardContext: make([]any, totalLayers),
		Residuals:       make([]*Tensor[T], totalLayers),
		StepCount:       0,
	}

	// Initialize layer 0 (input) with zeros
	state.LayerData[0] = NewTensor[T](inputSize)

	return state
}

// SetInput sets the input data for the network (layer 0)
func (state *GenericStepState[T]) SetInput(input *Tensor[T]) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.LayerData[0] = input.Clone()
}

// GetOutput retrieves the output from the final layer
func (state *GenericStepState[T]) GetOutput() *Tensor[T] {
	state.mu.RLock()
	defer state.mu.RUnlock()
	finalLayer := len(state.LayerData) - 1
	if state.LayerData[finalLayer] == nil {
		return nil
	}
	return state.LayerData[finalLayer].Clone()
}

// GetLayerOutput retrieves the current output of a specific layer
func (state *GenericStepState[T]) GetLayerOutput(layerIdx int) *Tensor[T] {
	state.mu.RLock()
	defer state.mu.RUnlock()
	if layerIdx < 0 || layerIdx >= len(state.LayerData) || state.LayerData[layerIdx] == nil {
		return nil
	}
	return state.LayerData[layerIdx].Clone()
}

// StepForwardGeneric executes one step for ALL layers using generic tensors.
// Network weights are converted from float32 to T during execution.
func StepForwardGeneric[T Numeric](
	n *Network,
	state *GenericStepState[T],
	backend Backend[T],
) time.Duration {
	start := time.Now()

	state.mu.Lock()
	defer state.mu.Unlock()

	totalLayers := n.TotalLayers()

	// Create temporary storage for new outputs (double buffering)
	newOutputs := make([]*Tensor[T], totalLayers+1)
	newOutputs[0] = state.LayerData[0] // Input stays the same

	// Process each layer
	layerIdx := 0
	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				config := n.GetLayer(row, col, layer)
				input := state.LayerData[layerIdx]

				if input == nil {
					// Initialize with zeros of correct size if not present (first step)
					size := 1
					switch config.Type {
					case LayerDense, LayerSwiGLU:
						size = config.InputHeight
					case LayerConv2D:
						size = config.InputHeight * config.InputWidth * config.InputChannels
					case LayerRNN, LayerLSTM:
						size = config.RNNInputSize
					case LayerMultiHeadAttention:
						size = config.DModel
					case LayerNorm, LayerRMSNorm:
						size = config.NormSize
					case LayerResidual, LayerSoftmax:
						if config.InputHeight > 0 {
							size = config.InputHeight
						} else if layerIdx > 0 && layerIdx-1 < len(n.Layers) {
							// Infer size from previous layer's output
							size = n.Layers[layerIdx-1].OutputHeight
						}
					}
					if size <= 0 {
						size = 1
					}
					input = NewTensor[T](size * n.BatchSize)
				}
				
				var postAct *Tensor[T]
				var context any

				if config.IsDisabled {
					postAct = input.Clone()
					context = nil
				} else {
					switch config.Type {
					case LayerDense:
						weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
						bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Bias, len(config.Bias)))
						pre, post := DenseForward(input, weights, bias, config.InputHeight, config.OutputHeight, n.BatchSize, config.Activation)
						postAct = post
						context = pre

					case LayerConv2D:
						weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
						bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Bias, len(config.Bias)))
						pre, post := Conv2DForward(input, weights, bias,
							config.InputHeight, config.InputWidth, config.InputChannels,
							config.KernelSize, config.Stride, config.Padding, config.Filters,
							config.OutputHeight, config.OutputWidth, n.BatchSize, config.Activation)
						postAct = post
						context = pre

					case LayerRNN:
						wIH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH, len(config.WeightIH)))
						wHH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH, len(config.WeightHH)))
						biasH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.BiasH, len(config.BiasH)))
						output, hiddenStates := RNNForward(input, wIH, wHH, biasH, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
						postAct = output
						context = hiddenStates

					case LayerLSTM:
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
						output, hidden, cell, allGates := LSTMForward(input, weights, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
						postAct = output
						states := map[string]*Tensor[T]{
							"hidden": hidden,
							"cell":   cell,
						}
						for k, v := range allGates {
							states[k] = v
						}
						context = states

					case LayerSoftmax:
						postAct = ApplySoftmax(input, float64(config.Temperature))
						context = postAct // Softmax output needed for backward

					case LayerNorm:
						gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
						beta := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Beta, len(config.Beta)))
						var residual *Tensor[T]
						if state.Residuals[layerIdx] != nil {
							residual = state.Residuals[layerIdx]
						}
						normSize := config.NormSize
						if normSize <= 0 {
							normSize = len(input.Data)
						}
						postAct = LayerNormForward(input, residual, gamma, beta, normSize, n.BatchSize, float64(config.Epsilon))
						
						// Update residual tracking?
						// In GenericForwardPass, LayerResidual is explicit.
						context = nil

					case LayerRMSNorm:
						gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
						normSize := config.NormSize
						if normSize <= 0 {
							normSize = len(input.Data)
						}
						postAct = RMSNormForward(input, nil, gamma, normSize, float64(config.Epsilon))
						context = nil

					case LayerSwiGLU:
						gateW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.GateWeights, len(config.GateWeights)))
						upW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.UpWeights, len(config.UpWeights)))
						downW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.DownWeights, len(config.DownWeights)))
						gateBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.GateBias, len(config.GateBias)))
						upBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.UpBias, len(config.UpBias)))
						downBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.DownBias, len(config.DownBias)))
						postAct = SwiGLUForward(input, gateW, upW, downW, gateBias, upBias, downBias, config.InputHeight, config.OutputHeight, n.BatchSize)
						context = nil

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
						postAct = MultiHeadAttentionForward(input, weights, 10000.0)
						// Store output as residual if this layer acts as residual source?
						// In float32 StepForward, MHA adds residual if present, then stores output as *new* residual.
						if state.Residuals[layerIdx] != nil && len(state.Residuals[layerIdx].Data) == len(postAct.Data) {
							// Add residual
							for i := range postAct.Data {
								postAct.Data[i] += state.Residuals[layerIdx].Data[i]
							}
						}
						// Store as new residual
						state.Residuals[layerIdx] = postAct.Clone()
						context = nil

					case LayerParallel:
						branches := make([]*LayerConfig, len(config.ParallelBranches))
						for i := range config.ParallelBranches {
							branches[i] = &config.ParallelBranches[i]
						}
						output, intermediates, err := ParallelForward(input, branches, n.BatchSize, config.CombineMode)
						if err != nil {
							output = input.Clone()
						}
						postAct = output
						context = intermediates

					case LayerResidual:
						// Skip connection from previous layer
						var skipInput *Tensor[T]
						if layerIdx > 0 {
							skipInput = state.LayerData[layerIdx-1]
						}
						postAct = ResidualForward(input, skipInput)
						context = nil

					case LayerConv1D:
						kernel := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Conv1DKernel, len(config.Conv1DKernel)))
						bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Conv1DBias, len(config.Conv1DBias)))
						seqLen := config.InputHeight
						if seqLen <= 0 {
							seqLen = len(input.Data) / (config.Conv1DInChannels * n.BatchSize)
						}
						pre, post := Conv1DForward(input, kernel, bias,
							seqLen, config.Conv1DInChannels,
							config.Conv1DKernelSize, config.Conv1DStride, config.Conv1DPadding,
							config.Conv1DFilters, n.BatchSize, config.Activation)
						postAct = post
						context = pre

					default:
						// Apply activation using backend
						postAct = backend.Activate(input, config.Activation)
						context = nil
					}
				}

				newOutputs[layerIdx+1] = postAct
				state.BackwardContext[layerIdx] = context
				layerIdx++
			}
		}
	}

	// Swap buffers
	for i := 1; i < len(state.LayerData); i++ {
		state.LayerData[i] = newOutputs[i]
	}

	state.StepCount++
	return time.Since(start)
}

// =============================================================================
// Original float32 Implementation
// =============================================================================


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
		} else if config.CombineMode == "filter" {
			// Filter mode: Output size is same as first branch (after stitching)
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
	} else if config.Type == LayerSequential {
		if len(config.ParallelBranches) == 0 {
			return -1
		}
		// Return size of the last layer in sequence
		return getLayerOutputSize(&config.ParallelBranches[len(config.ParallelBranches)-1], batchSize)
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

				case LayerConv1D:
					preAct, postAct = conv1DForwardCPU(input, config, n.BatchSize)

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

				case LayerSequential:
					// Sequential layer
					output, _, err := sequentialForwardCPU(input, config.ParallelBranches, n.BatchSize)
					if err != nil {
						fmt.Printf("Sequential layer error: %v\n", err)
						output = input
					}
					// Store intermediates? (Limited support in StepForward for introspection inside Seq)
					preAct = make([]float32, len(output)) // Dummy preAct
					postAct = output

				case LayerResidual:
					var skipInput []float32
					if layerIdx > 0 {
						skipInput = state.layerData[layerIdx-1]
					}
					postAct = ResidualForwardCPU(input, skipInput)
					preAct = make([]float32, len(postAct)) // Dummy preAct
					copy(preAct, postAct)

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
