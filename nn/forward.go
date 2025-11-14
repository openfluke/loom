package nn

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// ForwardCPU executes the grid network on CPU and stores intermediate activations for backprop
func (n *Network) ForwardCPU(input []float32) ([]float32, time.Duration) {
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
				if config.Type == LayerConv2D {
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
					output, branchPreActs, err := parallelForwardCPU(data, config, n.BatchSize)
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
				} else {
					// Default: element-wise activation only
					// Store pre-activation values
					n.preActivations[layerIdx] = make([]float32, len(data))
					copy(n.preActivations[layerIdx], data)

					// Apply activation in-place
					for i := 0; i < len(data); i++ {
						data[i] = activateCPU(data[i], config.Activation)
					}
				} // Store post-activation values
				n.activations[layerIdx+1] = make([]float32, len(data))
				copy(n.activations[layerIdx+1], data)
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
		return n.forwardGPUConv2D(input)
	}

	if hasMHA {
		return n.forwardGPUMultiHeadAttention(input)
	}

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
