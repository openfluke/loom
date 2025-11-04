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
					// Multi-Head Attention layer
					preAct, postAct := multiHeadAttentionForwardCPU(data, config, n.BatchSize)

					// Store pre-activation values
					n.preActivations[layerIdx] = preAct

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
				} else {
					// Dense layer (element-wise activation)
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

	// Check for Conv2D layers and use specialized path
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

	start := time.Now()

	dev := n.deviceInfo.Device
	q := n.deviceInfo.Queue
	wgx := n.deviceInfo.WorkgroupX

	N := len(input)
	bytes := uint64(N * 4)
	totalLayers := n.TotalLayers()

	// Build pipelines for each unique activation type
	pipelines := make([]*wgpu.ComputePipeline, 5)
	bgls := make([]*wgpu.BindGroupLayout, 5)

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

	defer cleanupPipelines(pipelines, bgls)

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

	// Execute layers
	for layerIdx := 0; layerIdx < totalLayers; layerIdx++ {
		// Calculate grid position for this layer
		row := layerIdx / (n.GridCols * n.LayersPerCell)
		remainder := layerIdx % (n.GridCols * n.LayersPerCell)
		col := remainder / n.LayersPerCell
		layer := remainder % n.LayersPerCell

		activation := int(n.GetActivation(row, col, layer))
		pipeline := pipelines[activation]
		bg := bindGroups[layerIdx]

		enc, err := dev.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
			Label: fmt.Sprintf("nn_fwd_enc_%d", layerIdx),
		})
		if err != nil {
			return nil, 0, fmt.Errorf("layer %d create encoder: %w", layerIdx, err)
		}

		pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("nn_fwd_pass_%d", layerIdx),
		})
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, 1, 1)
		pass.End()

		cb, err := enc.Finish(nil)
		if err != nil {
			enc.Release()
			return nil, 0, fmt.Errorf("layer %d finish: %w", layerIdx, err)
		}

		enc.Release()
		q.Submit(cb)
		cb.Release()
		pollDevice(dev, 1000)
	}

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

	// For now, fall back to CPU for Conv2D layers
	// Full implementation requires creating specialized pipelines with 4 bindings
	// (input, kernel, bias, output) and handling variable buffer sizes
	output, _ := n.ForwardCPU(input)

	return output, time.Since(start), nil
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
