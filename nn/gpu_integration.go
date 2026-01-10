package nn

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/webgpu/wgpu"
)

// WeightsToGPU uploads network weights to GPU and initializes GPU layers.
// After calling this, ForwardCPU will automatically use GPU acceleration when n.GPU = true.
// Returns an error if GPU initialization fails.
func (n *Network) WeightsToGPU() error {
	if n.gpuMounted {
		return nil // Already mounted
	}

	// Get GPU context
	ctx, err := gpu.GetContext()
	if err != nil {
		return fmt.Errorf("failed to get GPU context: %w", err)
	}
	n.gpuCtx = ctx

	// Build GPU layers from CPU layer configs
	layers := make([]gpu.GPULayer, 0, len(n.Layers))
	outputSize := 0

	for i, l := range n.Layers {
		if l.IsDisabled {
			continue
		}

		var gpuLayer gpu.GPULayer
		var layerOutputSize int
		var buildErr error

		gpuLayer, layerOutputSize, buildErr = n.buildGPULayer(&l, outputSize, i)
		if buildErr != nil {
			// Clean up already-built layers
			for _, built := range layers {
				built.Cleanup()
			}
			return fmt.Errorf("layer %d: %w", i, buildErr)
		}

		if gpuLayer != nil {
			// Set BatchSize if supported
			if dense, ok := gpuLayer.(*gpu.DenseLayer); ok {
				dense.BatchSize = n.BatchSize
			}
			if conv2d, ok := gpuLayer.(*gpu.Conv2DLayer); ok {
				conv2d.BatchSize = n.BatchSize
			}

			layers = append(layers, gpuLayer)
			if layerOutputSize > 0 {
				outputSize = layerOutputSize
			}
		}
	}

	if len(layers) == 0 {
		return fmt.Errorf("no GPU-compatible layers found")
	}

	// Allocate buffers for all layers
	for i, l := range layers {
		label := fmt.Sprintf("L%d", i)
		if err := l.AllocateBuffers(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("allocate buffers layer %d: %w", i, err)
		}
		if err := l.AllocateBackwardBuffers(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("allocate backward buffers layer %d: %w", i, err)
		}
	}

	// Compile shaders and create bind groups
	for i, l := range layers {
		label := fmt.Sprintf("L%d", i)
		if err := l.Compile(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("compile layer %d: %w", i, err)
		}
		if err := l.CompileBackward(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("compile backward layer %d: %w", i, err)
		}
		if err := l.CreateBindGroup(ctx, label); err != nil {
			n.cleanupGPULayers(layers)
			return fmt.Errorf("create bind group layer %d: %w", i, err)
		}
	}

	// Upload weights to GPU
	for _, l := range layers {
		l.UploadWeights(ctx)
	}

	// Create gradient application bind groups (init pipeline first)
	// This ensures we don't do it lazily later (which would fail to create cached bind groups)
	if err := n.EnsureGradientPipeline(ctx); err == nil {
		for _, l := range layers {
			if denseLayer, ok := l.(*gpu.DenseLayer); ok {
				createGradientBindGroups(ctx, n.gpuGradPipeline, n.gpuGradParams, denseLayer)
			}
		}
	} else {
		fmt.Printf("Warning: Failed to initialize gradient pipeline: %v\n", err)
	}

	// Wait for GPU to finish
	pollGPU(ctx)

	n.gpuLayers = layers
	n.gpuMounted = true
	n.gpuOutputSize = outputSize

	return nil
}

// WeightsToCPU downloads current GPU weights back to CPU layer configs.
// Call this before serialization if weights were modified on GPU during training.
func (n *Network) WeightsToCPU() error {
	if !n.gpuMounted {
		return nil // Nothing to download
	}

	layers, ok := n.gpuLayers.([]gpu.GPULayer)
	if !ok || len(layers) == 0 {
		return nil
	}

	ctx, ok := n.gpuCtx.(*gpu.Context)
	if !ok || ctx == nil {
		return fmt.Errorf("GPU context not available")
	}

	// Download weights from each GPU layer
	// Note: Currently weights are only read on GPU, so this is a no-op for most cases
	// If GPU training is implemented, this would download updated weights

	for i, gpuL := range layers {
		weights, biases, err := gpuL.DownloadWeights(ctx)
		if err != nil {
			return fmt.Errorf("download weights layer %d: %w", i, err)
		}

		// Find corresponding CPU layer and update weights
		if i < len(n.Layers) {
			if len(weights) > 0 {
				n.Layers[i].Kernel = weights
			}
			if len(biases) > 0 {
				n.Layers[i].Bias = biases
			}
		}
	}

	return nil
}

// ReleaseGPUWeights releases all GPU resources while preserving CPU weights.
// The network can continue to run on CPU after this call.
func (n *Network) ReleaseGPUWeights() {
	if !n.gpuMounted {
		return
	}

	if layers, ok := n.gpuLayers.([]gpu.GPULayer); ok {
		n.cleanupGPULayers(layers)
	}

	n.gpuLayers = nil
	n.gpuCtx = nil
	n.gpuMounted = false
	n.gpuOutputSize = 0
}

// IsGPUMounted returns true if weights are currently loaded on GPU
func (n *Network) IsGPUMounted() bool {
	return n.gpuMounted
}

// cleanupGPULayers releases all GPU layer resources
func (n *Network) cleanupGPULayers(layers []gpu.GPULayer) {
	for _, l := range layers {
		if l != nil {
			l.Cleanup()
		}
	}
}

// forwardGPU runs the forward pass on GPU
// Returns the output tensor
func (n *Network) forwardGPU(input []float32) ([]float32, error) {
	if !n.gpuMounted {
		return nil, fmt.Errorf("GPU not mounted, call WeightsToGPU first")
	}

	layers, ok := n.gpuLayers.([]gpu.GPULayer)
	if !ok || len(layers) == 0 {
		return nil, fmt.Errorf("no GPU layers available")
	}

	ctx, ok := n.gpuCtx.(*gpu.Context)
	if !ok || ctx == nil {
		return nil, fmt.Errorf("GPU context not available")
	}

	// Upload input to first layer
	firstLayer := layers[0]
	inputBuf := firstLayer.GetInputBuffer()

	// Handle embedding layer specially (expects uint32 token IDs)
	if _, isEmbed := firstLayer.(*gpu.EmbeddingLayer); isEmbed {
		inputU32 := make([]uint32, len(input))
		for k, v := range input {
			inputU32[k] = uint32(v)
		}
		ctx.Queue.WriteBuffer(inputBuf, 0, wgpu.ToBytes(inputU32))
	} else {
		ctx.Queue.WriteBuffer(inputBuf, 0, wgpu.ToBytes(input))
	}

	// Create command encoder and dispatch all layers
	cmdEnc, err := ctx.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("create command encoder: %w", err)
	}

	for i, l := range layers {
		pass := cmdEnc.BeginComputePass(nil)
		l.Dispatch(pass)
		pass.End()

		if i < len(layers)-1 {
			// Copy output to next layer's input
			next := layers[i+1]
			cmdEnc.CopyBufferToBuffer(l.GetOutputBuffer(), 0, next.GetInputBuffer(), 0, l.GetOutputBuffer().GetSize())
		} else {
			// Copy final output to staging
			cmdEnc.CopyBufferToBuffer(l.GetOutputBuffer(), 0, l.GetStagingBuffer(), 0, l.GetOutputBuffer().GetSize())
		}
	}

	cmd, err := cmdEnc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("command encoder finish: %w", err)
	}
	ctx.Queue.Submit(cmd)
	pollGPU(ctx)

	// Read output from staging buffer
	lastLayer := layers[len(layers)-1]
	batch := n.BatchSize
	if batch <= 0 {
		batch = 1
	}
	output, err := readStagingBuffer(ctx, lastLayer.GetStagingBuffer(), n.gpuOutputSize*batch)
	if err != nil {
		return nil, fmt.Errorf("read output: %w", err)
	}

	return output, nil
}

// backwardGPU runs the backward pass on GPU
// dOutput is the gradient of loss with respect to output
// Returns the gradient of loss with respect to input
func (n *Network) backwardGPU(dOutput []float32) ([]float32, error) {
	if !n.gpuMounted {
		return nil, fmt.Errorf("GPU not mounted, call WeightsToGPU first")
	}

	layers, ok := n.gpuLayers.([]gpu.GPULayer)
	if !ok || len(layers) == 0 {
		return nil, fmt.Errorf("no GPU layers available")
	}

	ctx, ok := n.gpuCtx.(*gpu.Context)
	if !ok || ctx == nil {
		return nil, fmt.Errorf("GPU context not available")
	}

	// Create dOutput buffer and upload
	dOutBuffer, err := ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "dOutput",
		Size:  uint64(len(dOutput) * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create dOutput buffer: %w", err)
	}
	defer dOutBuffer.Destroy()
	ctx.Queue.WriteBuffer(dOutBuffer, 0, wgpu.ToBytes(dOutput))

	// Create backward bind groups - link each layer to the next's input gradient
	for i, l := range layers {
		label := fmt.Sprintf("L%d", i)
		var dOutRef *wgpu.Buffer
		if i == len(layers)-1 {
			dOutRef = dOutBuffer
		} else {
			dOutRef = layers[i+1].GetInputGradientBuffer()
		}
		if err := l.CreateBackwardBindGroup(ctx, label, dOutRef); err != nil {
			return nil, fmt.Errorf("create backward bind group layer %d: %w", i, err)
		}
	}

	// Zero gradient buffers
	for _, l := range layers {
		l.ZeroGradients(ctx)
	}

	// Dispatch backward passes in reverse order
	cmdEnc, err := ctx.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("create backward command encoder: %w", err)
	}

	for i := len(layers) - 1; i >= 0; i-- {
		layers[i].DispatchBackward(cmdEnc)
	}

	cmd, err := cmdEnc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("backward command encoder finish: %w", err)
	}
	ctx.Queue.Submit(cmd)
	pollGPU(ctx)

	// Download gradients and store them in Network's gradient arrays
	for i, l := range layers {
		kernelGrad, biasGrad, _, err := l.DownloadGradients(ctx)
		if err != nil {
			// Non-fatal, just log
			continue
		}

		// Store in network gradient arrays
		if i < len(n.kernelGradients) {
			if kernelGrad != nil {
				n.kernelGradients[i] = kernelGrad
			}
		}
		if i < len(n.biasGradients) {
			if biasGrad != nil {
				n.biasGradients[i] = biasGrad
			}
		}
	}

	return nil, nil // Input gradient could be returned if needed
}

// BackwardGPUNew runs backward pass on GPU if mounted, otherwise falls back to CPU
// This is the public API for GPU backward pass using the new integration
func (n *Network) BackwardGPUNew(dOutput []float32) ([]float32, time.Duration, error) {
	if !n.GPU || !n.gpuMounted {
		// Fall back to CPU
		dInput, cpuTime := n.BackwardCPU(dOutput)
		return dInput, cpuTime, nil
	}

	start := time.Now()
	dInput, err := n.backwardGPU(dOutput)
	return dInput, time.Since(start), err
}

// buildGPULayer constructs a GPU layer from an nn.LayerConfig
func (n *Network) buildGPULayer(l *LayerConfig, prevOutputSize int, idx int) (gpu.GPULayer, int, error) {
	inputSize := prevOutputSize
	if idx == 0 && inputSize == 0 {
		// First layer, use configured input size
		inputSize = l.InputHeight
		if inputSize == 0 {
			inputSize = n.InputSize
		}
	}

	switch l.Type {
	case LayerDense:
		actCode := gpu.ActNone
		switch l.Activation {
		case ActivationSigmoid:
			actCode = gpu.ActSigmoid
		case ActivationTanh:
			actCode = gpu.ActTanh
		case ActivationLeakyReLU:
			actCode = gpu.ActLeakyReLU
		case ActivationScaledReLU:
			actCode = gpu.ActReLU
		}
		spec := gpu.DenseLayerSpec{
			InputSize:  l.InputHeight,
			OutputSize: l.OutputHeight,
			Activation: actCode,
			Weights:    l.Kernel,
			Biases:     l.Bias,
		}
		if spec.InputSize == 0 || spec.OutputSize == 0 {
			return nil, inputSize, nil // Skip invalid layer
		}
		return &gpu.DenseLayer{Spec: spec}, l.OutputHeight, nil

	case LayerNorm:
		pixelSize := l.NormSize
		if pixelSize == 0 {
			pixelSize = l.OutputHeight
		}
		batchSize := 1
		if inputSize > pixelSize && pixelSize > 0 {
			batchSize = inputSize / pixelSize
		}
		spec := gpu.LayerNormSpec{
			NormSize:  pixelSize,
			BatchSize: batchSize,
			Epsilon:   l.Epsilon,
			Gamma:     l.Gamma,
			Beta:      l.Beta,
		}
		return &gpu.LayerNormLayer{Spec: spec}, pixelSize * batchSize, nil

	case LayerRMSNorm:
		pixelSize := l.NormSize
		if pixelSize == 0 {
			pixelSize = inputSize
		}
		batchSize := 1
		if inputSize > pixelSize && pixelSize > 0 {
			batchSize = inputSize / pixelSize
		}
		spec := gpu.RMSNormSpec{
			NormSize:  pixelSize,
			BatchSize: batchSize,
			Epsilon:   l.Epsilon,
			Gamma:     l.Gamma,
		}
		return &gpu.RMSNormLayer{Spec: spec}, pixelSize * batchSize, nil

	case LayerSoftmax:
		temp := l.Temperature
		if temp <= 0 {
			temp = 1.0
		}
		spec := gpu.SoftmaxSpec{
			Size:        inputSize,
			BatchSize:   1,
			Temperature: temp,
		}
		return &gpu.SoftmaxLayer{Spec: spec}, inputSize, nil

	case LayerEmbedding:
		spec := gpu.EmbeddingSpec{
			VocabSize:    l.VocabSize,
			EmbeddingDim: l.EmbeddingDim,
			SeqLength:    inputSize,
			Weights:      l.EmbeddingWeights,
		}
		return &gpu.EmbeddingLayer{Spec: spec}, l.EmbeddingDim * inputSize, nil

	case LayerConv1D:
		seqLen := inputSize / l.Conv1DInChannels
		if seqLen == 0 {
			seqLen = 32 // Default sequence length
		}
		stride := l.Conv1DStride
		if stride < 1 {
			stride = 1
		}
		outLen := (seqLen+2*l.Conv1DPadding-l.Conv1DKernelSize)/stride + 1
		if outLen <= 0 {
			outLen = 1
		}

		// Ensure weights and bias are properly sized
		// Use l.Kernel and l.Bias (which is where InitConv1DLayer stores the weights)
		weightSize := l.Conv1DFilters * l.Conv1DInChannels * l.Conv1DKernelSize
		weights := l.Kernel
		if len(weights) != weightSize {
			weights = make([]float32, weightSize)
			// Initialize with small random values
			for i := range weights {
				weights[i] = float32(i%100) * 0.01
			}
		}

		bias := l.Bias
		if len(bias) != l.Conv1DFilters {
			bias = make([]float32, l.Conv1DFilters)
		}

		spec := gpu.Conv1DSpec{
			SeqLen:      seqLen,
			InChannels:  l.Conv1DInChannels,
			OutChannels: l.Conv1DFilters,
			KernelSize:  l.Conv1DKernelSize,
			Stride:      stride,
			Padding:     l.Conv1DPadding,
			Weights:     weights,
			Bias:        bias,
			Activation:  "relu",
		}
		return &gpu.Conv1DLayer{Spec: spec}, outLen * l.Conv1DFilters, nil

	case LayerConv2D:
		// Default parameters
		inChannels := l.InputChannels
		if inChannels <= 0 {
			inChannels = 8
		}
		filters := l.Filters
		if filters <= 0 {
			filters = 8
		}
		kernelSize := l.KernelSize
		if kernelSize <= 0 {
			kernelSize = 3
		}
		stride := l.Stride
		if stride < 1 {
			stride = 1
		}

		w := 32
		if inChannels > 0 && inputSize > 0 {
			val := float64(inputSize / inChannels)
			if val > 0 {
				w = int(math.Sqrt(val))
			}
		}
		if w <= 0 {
			w = 16
		}

		outH := (w+2*l.Padding-kernelSize)/stride + 1
		outW := (w+2*l.Padding-kernelSize)/stride + 1
		if outH <= 0 {
			outH = w
		}
		if outW <= 0 {
			outW = w
		}

		// Ensure weights and bias are properly sized
		weightSize := filters * inChannels * kernelSize * kernelSize
		if weightSize < 1 {
			weightSize = 1
		}
		weights := l.Kernel
		if len(weights) != weightSize {
			weights = make([]float32, weightSize)
			for i := range weights {
				weights[i] = float32(i%100) * 0.01
			}
		}

		bias := l.Bias
		if len(bias) != filters {
			bias = make([]float32, filters)
		}

		spec := gpu.Conv2DSpec{
			InputWidth:  w,
			InputHeight: w,
			InChannels:  inChannels,
			OutChannels: filters,
			KernelSize:  kernelSize,
			Stride:      stride,
			Padding:     l.Padding,
			Weights:     weights,
			Bias:        bias,
			Activation:  "relu",
		}
		return &gpu.Conv2DLayer{Spec: spec}, outH * outW * filters, nil

	case LayerMultiHeadAttention:
		headDim := 0
		if l.NumHeads > 0 {
			headDim = l.DModel / l.NumHeads
		}
		seqLen := 100
		if l.DModel > 0 && inputSize > 0 {
			seqLen = inputSize / l.DModel
		}
		spec := gpu.MHASpec{
			DModel:   l.DModel,
			NumHeads: l.NumHeads,
			HeadDim:  headDim,
			SeqLen:   seqLen,
			QWeights: l.QWeights,
			KWeights: l.KWeights,
			VWeights: l.VWeights,
			OWeights: l.OutputWeight,
		}
		return &gpu.MHALayer{Spec: spec}, l.DModel * seqLen, nil

	case LayerRNN:
		// Default parameters
		rnnInputSize := l.RNNInputSize
		if rnnInputSize <= 0 {
			rnnInputSize = 64
		}
		hiddenSize := l.HiddenSize
		if hiddenSize <= 0 {
			hiddenSize = 64
		}

		seqLen := 32
		if rnnInputSize > 0 && inputSize > 0 {
			seqLen = inputSize / rnnInputSize
		}
		if seqLen <= 0 {
			seqLen = 32
		}

		spec := gpu.RNNSpec{
			InputSize:  rnnInputSize,
			HiddenSize: hiddenSize,
			SeqLen:     seqLen,
			WeightIH:   l.WeightIH,
			WeightHH:   l.WeightHH,
			BiasH:      l.BiasH,
		}
		return &gpu.RNNLayer{Spec: spec}, hiddenSize * seqLen, nil

	case LayerLSTM:
		// Default parameters
		rnnInputSize := l.RNNInputSize
		if rnnInputSize <= 0 {
			rnnInputSize = 64
		}
		hiddenSize := l.HiddenSize
		if hiddenSize <= 0 {
			hiddenSize = 64
		}

		seqLen := 32
		if rnnInputSize > 0 && inputSize > 0 {
			seqLen = inputSize / rnnInputSize
		}
		if seqLen <= 0 {
			seqLen = 32
		}

		spec := gpu.LSTMSpec{
			InputSize:  rnnInputSize,
			HiddenSize: hiddenSize,
			SeqLen:     seqLen,
			WeightIH_i: l.WeightIH_i,
			WeightIH_f: l.WeightIH_f,
			WeightIH_g: l.WeightIH_g,
			WeightIH_o: l.WeightIH_o,
			WeightHH_i: l.WeightHH_i,
			WeightHH_f: l.WeightHH_f,
			WeightHH_g: l.WeightHH_g,
			WeightHH_o: l.WeightHH_o,
			BiasH_i:    l.BiasH_i,
			BiasH_f:    l.BiasH_f,
			BiasH_g:    l.BiasH_g,
			BiasH_o:    l.BiasH_o,
		}
		return &gpu.LSTMLayer{Spec: spec}, l.HiddenSize * seqLen, nil

	case LayerSwiGLU:
		interSize := 0
		if inputSize > 0 && len(l.GateWeights) > 0 {
			interSize = len(l.GateWeights) / inputSize
		}
		spec := gpu.SwiGLUSpec{
			InputSize:        inputSize,
			IntermediateSize: interSize,
			SeqLen:           1,
			GateWeights:      l.GateWeights,
			UpWeights:        l.UpWeights,
			DownWeights:      l.DownWeights,
		}
		return &gpu.SwiGLULayer{Spec: spec}, inputSize, nil

	default:
		// Layer type not supported on GPU
		return nil, inputSize, nil
	}
}

// pollGPU waits for GPU operations to complete
func pollGPU(ctx *gpu.Context) {
	done := make(chan struct{})
	go func() {
		ctx.Device.Poll(true, nil)
		close(done)
	}()

	select {
	case <-done:
		return
	case <-time.After(2 * time.Second):
		// Timeout, continue anyway
	}
}

// readStagingBuffer reads float32 data from a staging buffer
func readStagingBuffer(ctx *gpu.Context, buf *wgpu.Buffer, size int) ([]float32, error) {
	done := make(chan struct{})
	var mapErr error

	buf.MapAsync(wgpu.MapModeRead, 0, buf.GetSize(), func(status wgpu.BufferMapAsyncStatus) {
		if status != wgpu.BufferMapAsyncStatusSuccess {
			mapErr = fmt.Errorf("map status: %d", status)
		}
		close(done)
	})

	// Poll until mapped
	timeout := time.After(2 * time.Second)
Loop:
	for {
		ctx.Device.Poll(false, nil)
		select {
		case <-done:
			break Loop
		case <-timeout:
			return nil, fmt.Errorf("map timeout")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	if mapErr != nil {
		return nil, mapErr
	}

	data := buf.GetMappedRange(0, uint(size*4))
	defer buf.Unmap()

	if data == nil {
		return nil, fmt.Errorf("mapped range nil")
	}

	out := make([]float32, size)
	copy(out, wgpu.FromBytes[float32](data))

	return out, nil
}
