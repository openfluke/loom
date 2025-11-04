package nn

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// BackwardCPU computes gradients via backpropagation on CPU through the grid
// gradOutput: gradient flowing back from the loss (same size as network output)
// Returns: gradient with respect to the input
func (n *Network) BackwardCPU(gradOutput []float32) ([]float32, time.Duration) {
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

			// Store gradients (we'll need to expand gradient storage for MHA)
			// For now, store concatenated gradients in kernelGradients
			allGrads := append(append(append(append(gradQW, gradKW...), gradVW...), gradOutW...), append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)...)
			n.kernelGradients[layerIdx] = allGrads

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
		} else {
			// Dense layer backward
			// Compute gradient with respect to pre-activation
			// grad_pre = grad_post * activation_derivative(pre_activation)
			for i := 0; i < len(grad); i++ {
				derivative := activateDerivativeCPU(preAct[i], config.Activation)
				grad[i] = grad[i] * derivative
			}
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
		return n.backwardGPUConv2D(gradOutput)
	}

	start := time.Now()

	dev := n.deviceInfo.Device
	q := n.deviceInfo.Queue
	wgx := n.deviceInfo.WorkgroupX

	N := len(gradOutput)
	bytes := uint64(N * 4)
	totalLayers := n.TotalLayers()

	// Build pipelines for backward pass (derivative computations)
	pipelines := make([]*wgpu.ComputePipeline, 5)
	bgls := make([]*wgpu.BindGroupLayout, 5)

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

	defer cleanupPipelines(pipelines, bgls)

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
			return nil, 0, fmt.Errorf("create backward bindgroup %d: %w", layerIdx, err)
		}

		enc, err := dev.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{
			Label: fmt.Sprintf("nn_bwd_enc_%d", layerIdx),
		})
		if err != nil {
			bg.Release()
			return nil, 0, fmt.Errorf("layer %d create encoder: %w", layerIdx, err)
		}

		pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{
			Label: fmt.Sprintf("nn_bwd_pass_%d", layerIdx),
		})
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, 1, 1)
		pass.End()

		cb, err := enc.Finish(nil)
		if err != nil {
			enc.Release()
			bg.Release()
			return nil, 0, fmt.Errorf("layer %d finish: %w", layerIdx, err)
		}

		enc.Release()
		q.Submit(cb)
		cb.Release()
		bg.Release()
		pollDevice(dev, 1000)
	}

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

	// For now, fall back to CPU for Conv2D layers
	// Full implementation requires creating specialized pipelines with multiple bindings
	// for gradients, kernels, inputs, and handling variable buffer sizes
	gradInput, _ := n.BackwardCPU(gradOutput)

	return gradInput, time.Since(start), nil
}
