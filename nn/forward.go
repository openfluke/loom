package nn

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// =============================================================================
// Generic Forward Pass Implementation
// =============================================================================

// GenericForwardPass executes network forward pass for any numeric type.
// Uses the Backend interface for computation.
// Returns:
// - Final output tensor
// - List of output activations for each layer (activations[0] = input)
// - List of backward contexts for each layer (intermediate states needed for backprop)
// - Duration of forward pass
func GenericForwardPass[T Numeric](
	n *Network,
	input *Tensor[T],
	backend Backend[T],
) (*Tensor[T], []*Tensor[T], []any, time.Duration) {
	start := time.Now()

	totalLayers := n.TotalLayers()
	activations := make([]*Tensor[T], totalLayers+1)
	backwardContext := make([]any, totalLayers)

	// Store input
	activations[0] = input.Clone()
	data := input.Clone()

	layerIdx := 0

	// Forward through grid
	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				config := n.GetLayer(row, col, layer)

				// Context for this layer
				var context any

				if config.IsDisabled {
					// Pass through
					activations[layerIdx+1] = data.Clone()
					context = nil
				} else {
					switch config.Type {
					case LayerDense:
						weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
						bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Bias, len(config.Bias)))
						pre, post := DenseForward(data, weights, bias, config.InputHeight, config.OutputHeight, n.BatchSize, config.Activation)
						data = post
						activations[layerIdx+1] = post
						context = pre // Store pre-activation

					case LayerConv2D:
						weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
						bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Bias, len(config.Bias)))
						pre, post := Conv2DForward(data, weights, bias,
							config.InputHeight, config.InputWidth, config.InputChannels,
							config.KernelSize, config.Stride, config.Padding, config.Filters,
							config.OutputHeight, config.OutputWidth, n.BatchSize, config.Activation)
						data = post
						activations[layerIdx+1] = post
						context = pre // Store pre-activation

					case LayerRNN:
						wIH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightIH, len(config.WeightIH)))
						wHH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.WeightHH, len(config.WeightHH)))
						biasH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.BiasH, len(config.BiasH)))
						output, hiddenStates := RNNForward(data, wIH, wHH, biasH, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
						data = output
						activations[layerIdx+1] = output
						context = hiddenStates // Store hidden states

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
						output, hidden, cell, allGates := LSTMForward(data, weights, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)
						data = output
						activations[layerIdx+1] = output
						// Pack context for LSTMBackward
						states := map[string]*Tensor[T]{
							"hidden": hidden,
							"cell":   cell,
						}
						for k, v := range allGates {
							states[k] = v
						}
						context = states

					case LayerSoftmax:
						output := ApplySoftmax(data, float64(config.Temperature))
						data = output
						activations[layerIdx+1] = output
						context = output // Softmax needs its output for backward jacobian

					case LayerNorm:
						gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
						beta := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Beta, len(config.Beta)))
						normSize := config.NormSize
						if normSize <= 0 {
							normSize = len(data.Data)
						}
						output := LayerNormForward(data, nil, gamma, beta, normSize, n.BatchSize, float64(config.Epsilon))
						data = output
						activations[layerIdx+1] = output
						context = nil // LayerNorm backward only needs input, which is activations[layerIdx]

					case LayerRMSNorm:
						gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Gamma, len(config.Gamma)))
						normSize := config.NormSize
						if normSize <= 0 {
							normSize = len(data.Data)
						}
						output := RMSNormForward(data, nil, gamma, normSize, float64(config.Epsilon))
						data = output
						activations[layerIdx+1] = output
						context = nil // RMSNorm backward needs input (activations[layerIdx])

					case LayerSwiGLU:
						gateW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.GateWeights, len(config.GateWeights)))
						upW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.UpWeights, len(config.UpWeights)))
						downW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.DownWeights, len(config.DownWeights)))
						gateBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.GateBias, len(config.GateBias)))
						upBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.UpBias, len(config.UpBias)))
						downBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.DownBias, len(config.DownBias)))
						output := SwiGLUForward(data, gateW, upW, downW, gateBias, upBias, downBias, config.InputHeight, config.OutputHeight, n.BatchSize)
						data = output
						activations[layerIdx+1] = output
						context = nil // SwiGLU backward recomputes intermediates or needs caching. Current implementation recomputes.

					case LayerMultiHeadAttention:
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
						output := MultiHeadAttentionForward(data, weights, 10000.0)
						data = output
						activations[layerIdx+1] = output
						context = nil // Backward recomputes simply or doesn't need context stored here

					case LayerParallel:
						// Convert config.ParallelBranches ([]LayerConfig) to []*LayerConfig
						branches := make([]*LayerConfig, len(config.ParallelBranches))
						for i := range config.ParallelBranches {
							branches[i] = &config.ParallelBranches[i]
						}
						output, intermediates, err := ParallelForward(data, branches, n.BatchSize, config.CombineMode)
						if err != nil {
							// On error, identity
							output = data.Clone()
						}
						data = output
						activations[layerIdx+1] = output
						context = intermediates

						activations[layerIdx+1] = output
						context = intermediates

					case LayerSequential:
						layers := make([]*LayerConfig, len(config.ParallelBranches))
						for i := range config.ParallelBranches {
							layers[i] = &config.ParallelBranches[i]
						}
						output, intermediates, err := SequentialForward[T](data, layers, n.BatchSize)
						if err != nil {
							// On error, identity
							output = data.Clone()
						}
						data = output
						activations[layerIdx+1] = output
						context = intermediates

					case LayerEmbedding:
						weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.EmbeddingWeights, len(config.EmbeddingWeights)))
						output := EmbeddingForward(data, weights, config.VocabSize, config.EmbeddingDim)
						data = output
						activations[layerIdx+1] = output
						context = nil // Embedding backward needs token IDs (stored in activations[layerIdx])

					case LayerConv1D:
						kernel := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Kernel, len(config.Kernel)))
						bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(config.Bias, len(config.Bias)))
						seqLen := config.InputHeight // Using InputHeight for sequence length
						if seqLen <= 0 {
							seqLen = len(data.Data) / (config.Conv1DInChannels * n.BatchSize)
						}
						pre, post := Conv1DForward(data, kernel, bias,
							seqLen, config.Conv1DInChannels,
							config.Conv1DKernelSize, config.Conv1DStride, config.Conv1DPadding,
							config.Conv1DFilters, n.BatchSize, config.Activation)
						data = post
						activations[layerIdx+1] = post
						context = pre // Store pre-activation

					case LayerResidual:
						// Residual connects current input to something?
						// Current design: Residual layer takes input, assumes it's added to *previous* layer input.
						// Wait, standard residual is: y = x + f(x).
						// But if LayerResidual is a separate layer in grid, it implies:
						// Layer N: Dense -> output O
						// Layer N+1: Residual -> output O + Input_of_N?
						// Or Input_to_N is not available here easily.
						// Looking at `examples` test:
						// SetLayer(..., 0, Dense)
						// SetLayer(..., 1, Residual)
						// This implies Residual adds activations[layerIdx] + activations[layerIdx-1]?
						// Let's assume ResidualForward adds input + skipInput.
						// We need skipInput.

						// GenericForwardPass loop structure:
						// activations[layerIdx] is input to CURRENT layer.
						// activations[layerIdx+1] will be output.
						// Skip connection usually skips 1 layer or block.
						// If we assume skip from layerIdx-1 (input to previous layer):
						var skipInput *Tensor[T]
						if layerIdx > 0 {
							skipInput = activations[layerIdx-1]
						}
						// If layerIdx=0, skip is nil

						output := ResidualForward(data, skipInput)
						data = output
						activations[layerIdx+1] = output
						context = nil

					default:
						// Apply activation using backend
						data = backend.Activate(data, config.Activation)
						activations[layerIdx+1] = data.Clone()
						context = nil
					}
				}

				backwardContext[layerIdx] = context
				layerIdx++
			}
		}
	}

	return data, activations, backwardContext, time.Since(start)
}

// =============================================================================
// Original float32 Implementation
// =============================================================================

// ForwardCPU executes the grid network on CPU and stores intermediate activations for backprop
func (n *Network) ForwardCPU(input []float32) ([]float32, time.Duration) {
	// GPU auto-routing: if GPU mode is enabled and weights are mounted, use GPU
	if n.GPU && n.gpuMounted {
		start := time.Now()
		output, err := n.forwardGPU(input)
		if err == nil {
			return output, time.Since(start)
		}
		// Fall back to CPU on GPU error
	}

	start := time.Now()

	// Store input
	n.activations[0] = make([]float32, len(input))
	copy(n.activations[0], input)

	data := make([]float32, len(input))
	copy(data, input)

	layerIdx := 0

	// Track residual inputs for BERT-style skip connections
	var residualInput []float32

	// Forward through grid: iterate through rows, then columns, then layers per cell
	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				// Get layer configuration for this grid position
				config := n.GetLayer(row, col, layer)

				// Route to appropriate layer type
				if config.IsDisabled {
					// BYPASS: Pass input directly to output (Identity)
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)
					// Data remains unchanged for next layer

				} else if config.Type == LayerConv2D {
					// Conv2D layer
					preAct, postAct := conv2DForwardCPU(data, config, n.BatchSize)

					// Store pre-activation values (before activation function)
					n.preActivations[layerIdx] = preAct

					// Use post-activation for next layer
					data = postAct
				} else if config.Type == LayerMultiHeadAttention {
					// Multi-Head Attention layer with residual connection
					// In Pre-LN: x = Attention(Norm(x)) + x
					// The norm happens before this layer, so we need to add the pre-norm input

					preAct, postAct := MultiHeadAttentionForwardCPU(data, config, n.BatchSize)

					// Add residual connection if we have stored residual
					if residualInput != nil && len(residualInput) == len(postAct) {
						for i := range postAct {
							postAct[i] += residualInput[i]
						}
					}

					// Store pre-activation values
					n.preActivations[layerIdx] = preAct

					// Store output as residual for next sub-block
					residualInput = make([]float32, len(postAct))
					copy(residualInput, postAct)

					// Use post-activation for next layer
					data = postAct
				} else if config.Type == LayerRNN {
					// RNN layer
					output, hiddenStates := rnnForwardCPU(config, data, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

					// Store hidden states for backward pass (stored as pre-activation)
					n.preActivations[layerIdx] = hiddenStates

					// Use RNN output for next layer
					data = output
				} else if config.Type == LayerLSTM {
					// LSTM layer
					output, states := lstmForwardCPU(config, data, n.BatchSize, config.SeqLength, config.RNNInputSize, config.HiddenSize)

					// Store all LSTM states for backward pass
					// We'll flatten the states map into a single slice for storage
					// Format: [hidden, cell, i_gate, f_gate, g_gate, o_gate, c_tanh]
					totalStateSize := len(states["hidden"]) + len(states["cell"]) +
						len(states["i_gate"]) + len(states["f_gate"]) +
						len(states["g_gate"]) + len(states["o_gate"]) + len(states["c_tanh"])
					flatStates := make([]float32, totalStateSize)
					offset := 0
					for _, key := range []string{"hidden", "cell", "i_gate", "f_gate", "g_gate", "o_gate", "c_tanh"} {
						copy(flatStates[offset:], states[key])
						offset += len(states[key])
					}
					n.preActivations[layerIdx] = flatStates

					// Use LSTM output for next layer
					data = output
				} else if config.Type == LayerSoftmax {
					// Softmax layer
					probs, err := ForwardSoftmaxCPU(data, config)
					if err != nil {
						// If error, fall back to standard softmax
						probs = softmaxStandard(data, 1.0)
					}

					// Store input as pre-activation (logits)
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

					// Use probabilities for next layer
					data = probs
				} else if config.Type == LayerDense {
					// Dense/Fully-Connected layer with weight matrix
					preAct, postAct := denseForwardCPU(data, config, n.BatchSize)

					// Store pre-activation values
					n.preActivations[layerIdx] = preAct

					// Use post-activation for next layer
					data = postAct
				} else if config.Type == LayerSwiGLU {
					// SwiGLU gated activation layer with residual connection
					// In Pre-LN: x = SwiGLU(Norm(x)) + x
					// The norm happens before this layer, so we need to add the pre-norm input

					preAct, postAct := SwiGLUForwardCPU(data, config, n.BatchSize)

					// Add residual connection if we have stored residual
					if residualInput != nil && len(residualInput) == len(postAct) {
						for i := range postAct {
							postAct[i] += residualInput[i]
						}
					}

					// Store pre-activation values
					n.preActivations[layerIdx] = preAct

					// Store output as residual for next sub-block
					residualInput = make([]float32, len(postAct))
					copy(residualInput, postAct)

					// Use post-activation for next layer
					data = postAct
				} else if config.Type == LayerNorm {
					// Layer Normalization with residual connection
					normalized := layerNormForwardCPU(data, residualInput, config, n.BatchSize)

					// Store pre-normalization values
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

					// After LayerNorm, store output as potential residual for next block
					if residualInput != nil {
						residualInput = make([]float32, len(normalized))
						copy(residualInput, normalized)
					}

					// Use normalized output for next layer
					data = normalized
				} else if config.Type == LayerRMSNorm {
					// RMS Normalization (just normalize, no residual addition here)
					normalized := rmsNormForwardCPU(data, nil, config, n.BatchSize)

					// Store pre-normalization values
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

					// Save input as residual for next Attention or SwiGLU
					residualInput = make([]float32, len(data))
					copy(residualInput, data)

					// Use normalized output for next layer
					data = normalized
				} else if config.Type == LayerParallel {
					// Parallel layer - run multiple sub-layers and combine outputs
					output, branchPreActs, err := parallelForwardCPU(data, config, n.BatchSize, "normal")
					if err != nil {
						fmt.Printf("Parallel layer error: %v\n", err)
						// On error, pass through unchanged
						output = data
						branchPreActs = nil
					}

					// Store branch pre-activations (needed for backward)
					// Flatten all branch pre-acts into single slice
					totalPreActSize := 0
					for _, preAct := range branchPreActs {
						totalPreActSize += len(preAct)
					}
					// Metadata: 1 for numBranches + 1 per branch for size = 1 + len(branchPreActs)
					metadataSize := 1 + len(branchPreActs)
					n.preActivations[layerIdx] = make([]float32, totalPreActSize+metadataSize)
					// Store metadata: number of branches and their sizes
					n.preActivations[layerIdx][0] = float32(len(branchPreActs))
					offset := 1
					for i, preAct := range branchPreActs {
						n.preActivations[layerIdx][offset] = float32(len(preAct))
						offset++
						copy(n.preActivations[layerIdx][offset:], preAct)
						offset += len(preAct)
						_ = i // suppress unused warning
					}

					// Use parallel output for next layer
					data = output

					// Observer for the parallel layer itself (conceptual)
					if config.Observer != nil {
						notifyObserver(config, "normal", "forward", layerIdx, data, n.activations[layerIdx+1], 0)
					}
					if config.Observer != nil {
						notifyObserver(config, "normal", "forward", layerIdx, data, n.activations[layerIdx+1], 0)
					}
				} else if config.Type == LayerSequential {
					// Sequential layer
					output, _, err := sequentialForwardCPU(data, config.ParallelBranches, n.BatchSize)
					if err != nil {
						fmt.Printf("Sequential layer error: %v\n", err)
						output = data
					}

					// Store intermediates (simplified)
					// We can flatten intermediates similar to Parallel layer if we wanted,
					// but for now let's just use the output as "pre-activation" effectively since we don't fully support backward yet.
					n.preActivations[layerIdx] = make([]float32, 1) // Dummy

					data = output

				} else if config.Type == LayerEmbedding {
					// Embedding lookup layer
					output := embeddingForwardCPU(data, config)

					// Store token IDs as pre-activation (needed for backward)
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

					// Use embedding output for next layer
					data = output
				} else if config.Type == LayerConv1D {
					// Conv1D layer
					preAct, postAct := conv1DForwardCPU(data, config, n.BatchSize)

					// Store pre-activation values
					n.preActivations[layerIdx] = preAct

					// Use post-activation for next layer
					data = postAct
				} else {
					// Default: element-wise activation only
					// Store pre-activation values
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

					// Apply activation in-place
					for i := 0; i < len(data); i++ {
						data[i] = activateCPU(data[i], config.Activation)
					}
				}

				// Store post-activation values
				n.activations[layerIdx+1] = make([]float32, len(data))
				copy(n.activations[layerIdx+1], data)

				// For non-parallel layers, we need to notify the observer here
				// (Internal notifications were removed from dense.go etc.)
				if config.Type != LayerParallel && config.Observer != nil {
					notifyObserver(config, "normal", "forward", layerIdx, data, n.activations[layerIdx+1], 0)
				}

				layerIdx++
			}
		}
	}

	return data, time.Since(start)
}

// ForwardGPU executes the network on GPU
func (n *Network) ForwardGPU(input []float32) ([]float32, time.Duration, error) {
	if n.deviceInfo == nil {
		return nil, 0, fmt.Errorf("GPU not initialized, call InitGPU first")
	}

	// Check for specialized layer types and use dedicated GPU paths
	hasConv2D := false
	for i := 0; i < n.TotalLayers(); i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		remainder := i % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		config := n.GetLayer(row, col, layer)
		if config.Type == LayerConv2D {
			hasConv2D = true
			break
		}
	}

	if hasConv2D {
		return n.forwardGPUConv2D(input)
	}

	// fmt.Println("DEBUG: Starting ForwardGPU") // DEBUG
	start := time.Now()

	dev := n.deviceInfo.Device
	q := n.deviceInfo.Queue
	wgx := n.deviceInfo.WorkgroupX

	N := len(input)
	bytes := uint64(N * 4)
	totalLayers := n.TotalLayers()

	// Build pipelines for each unique activation type (cached after first call)
	var pipelines []*wgpu.ComputePipeline
	var bgls []*wgpu.BindGroupLayout

	if n.deviceInfo.forwardPipelines == nil {
		// First time - create and cache pipelines
		pipelines = make([]*wgpu.ComputePipeline, 5)
		bgls = make([]*wgpu.BindGroupLayout, 5)

		for act := 0; act < 5; act++ {
			shader := generateForwardShader(wgx, act, N)
			module, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
				Label:          fmt.Sprintf("nn_fwd_shader_%d", act),
				WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shader},
			})
			if err != nil {
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, fmt.Errorf("CreateShaderModule %d: %w", act, err)
			}

			bgl, err := dev.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
				Label: fmt.Sprintf("nn_fwd_bgl_%d", act),
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
					{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
				},
			})
			if err != nil {
				module.Release()
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, err
			}
			bgls[act] = bgl

			pl, err := dev.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
				Label:            fmt.Sprintf("nn_fwd_pl_%d", act),
				BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
			})
			if err != nil {
				module.Release()
				bgl.Release()
				cleanupPipelines(pipelines[:act], bgls[:act])
				return nil, 0, err
			}

			pipeline, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
				Label:  fmt.Sprintf("nn_fwd_pipeline_%d", act),
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
		n.deviceInfo.forwardPipelines = pipelines
		n.deviceInfo.forwardBGLs = bgls
	} else {
		// Use cached pipelines
		pipelines = n.deviceInfo.forwardPipelines
		bgls = n.deviceInfo.forwardBGLs
	}

	// Create buffers
	bufA, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nn_fwd_A",
		Size:  bytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, 0, err
	}
	defer bufA.Release()

	bufB, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nn_fwd_B",
		Size:  bytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, 0, err
	}
	defer bufB.Release()

	readback, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nn_fwd_RB",
		Size:  bytes,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, 0, err
	}
	defer readback.Release()

	// Create bind groups for each layer
	bindGroups := make([]*wgpu.BindGroup, totalLayers)
	for layerIdx := 0; layerIdx < totalLayers; layerIdx++ {
		// Calculate grid position for this layer
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		activation := int(n.GetActivation(row, col, layer))
		bgl := bgls[activation]

		var inBuf, outBuf *wgpu.Buffer
		if layerIdx%2 == 0 {
			inBuf = bufA
			outBuf = bufB
		} else {
			inBuf = bufB
			outBuf = bufA
		}

		bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Label:  fmt.Sprintf("nn_fwd_bg_%d", layerIdx),
			Layout: bgl,
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
				{Binding: 1, Buffer: outBuf, Offset: 0, Size: outBuf.GetSize()},
			},
		})
		if err != nil {
			cleanupBindGroups(bindGroups[:layerIdx])
			return nil, 0, fmt.Errorf("create bindgroup %d: %w", layerIdx, err)
		}
		bindGroups[layerIdx] = bg
	}

	defer cleanupBindGroups(bindGroups)

	// Write input data
	q.WriteBuffer(bufA, 0, unsafe.Slice((*byte)(unsafe.Pointer(&input[0])), int(bytes)))
	pollDevice(dev, 100)

	gx := uint32((N + int(wgx) - 1) / int(wgx))
	if gx == 0 {
		gx = 1
	}

	// Create single command encoder for all layers
	enc, err := dev.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
		Label: "nn_fwd_enc_all",
	})
	if err != nil {
		return nil, 0, fmt.Errorf("create encoder: %w", err)
	}

	// Execute all layers in a single command buffer
	for layerIdx := 0; layerIdx < totalLayers; layerIdx++ {
		// Calculate grid position for this layer
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		activation := int(n.GetActivation(row, col, layer))
		pipeline := pipelines[activation]
		bg := bindGroups[layerIdx]

		pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("nn_fwd_pass_%d", layerIdx),
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
		return nil, 0, fmt.Errorf("finish command buffer: %w", err)
	}
	enc.Release()
	q.Submit(cb)
	cb.Release()
	pollDevice(dev, 1000)

	// Determine final output buffer
	var finalOut *wgpu.Buffer
	if totalLayers%2 == 0 {
		finalOut = bufA
	} else {
		finalOut = bufB
	}

	// Copy back results
	enc2, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, 0, err
	}
	enc2.CopyBufferToBuffer(finalOut, 0, readback, 0, bytes)
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
	output := make([]float32, N)
	copy(output, unsafe.Slice((*float32)(unsafe.Pointer(&view[0])), N))
	readback.Unmap()

	return output, time.Since(start), nil
}

// forwardGPUConv2D executes networks containing Conv2D layers on GPU
// This uses specialized pipelines for Conv2D layers
func (n *Network) forwardGPUConv2D(input []float32) ([]float32, time.Duration, error) {
	start := time.Now()

	dev := n.deviceInfo.Device
	q := n.deviceInfo.Queue

	// Store input
	n.activations[0] = make([]float32, len(input))
	copy(n.activations[0], input)

	data := input
	layerIdx := 0

	// Forward through grid
	for row := 0; row < n.GridRows; row++ {
		for col := 0; col < n.GridCols; col++ {
			for layer := 0; layer < n.LayersPerCell; layer++ {
				config := n.GetLayer(row, col, layer)

				if config.Type == LayerConv2D {
					// GPU Conv2D
					output, err := conv2DForwardGPU(dev, q, data, config, n.BatchSize)
					if err != nil {
						// Fall back to CPU on error
						cpuOut, _ := n.ForwardCPU(input)
						return cpuOut, time.Since(start), nil
					}

					// Store pre-activation (output before activation)
					n.preActivations[layerIdx] = make([]float32, len(output))
					copy(n.preActivations[layerIdx], output)

					// Apply activation
					postAct := make([]float32, len(output))
					for i := 0; i < len(output); i++ {
						postAct[i] = activateCPU(output[i], config.Activation)
					}

					data = postAct
				} else {
					// Fallback to CPU for other layer types
					cpuOut, _ := n.ForwardCPU(input)
					return cpuOut, time.Since(start), nil
				}

				// Store post-activation
				n.activations[layerIdx+1] = make([]float32, len(data))
				copy(n.activations[layerIdx+1], data)

				layerIdx++
			}
		}
	}

	return data, time.Since(start), nil
}

// Helper functions for GPU resource cleanup

func cleanupPipelines(pipelines []*wgpu.ComputePipeline, bgls []*wgpu.BindGroupLayout) {
	for _, p := range pipelines {
		if p != nil {
			p.Release()
		}
	}
	for _, bgl := range bgls {
		if bgl != nil {
			bgl.Release()
		}
	}
}

func cleanupBindGroups(bindGroups []*wgpu.BindGroup) {
	for _, bg := range bindGroups {
		if bg != nil {
			bg.Release()
		}
	}
}

func pollDevice(dev *wgpu.Device, maxIter int) {
	for i := 0; i < maxIter; i++ {
		if dev.Poll(true, nil) {
			break
		}
		time.Sleep(100 * time.Microsecond)
	}
}
