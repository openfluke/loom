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
	//fmt.Printf("WeightsToGPU Called. Mounted: %v\n", n.gpuMounted)

	if gpu.Debug {
		gpu.Log("WeightsToGPU Start. n=%p", n)
		gpu.Log("gpuMounted=%v", n.gpuMounted)
	}

	if n.gpuMounted {
		return nil // Already mounted
	}

	// Get GPU context
	// Get GPU context
	if gpu.Debug {
		gpu.Log("Getting GPU context...")
	}
	ctx, err := gpu.GetContext()
	if err != nil {
		return fmt.Errorf("failed to get GPU context: %w", err)
	}
	n.gpuCtx = ctx
	if gpu.Debug {
		gpu.Log("GPU context obtained.")
	}

	// Build GPU layers from CPU layer configs
	if gpu.Debug {
		gpu.Log("Building GPU layers...")
	}
	layers := make([]gpu.GPULayer, 0, len(n.Layers))
	outputSize := 0

	for i, l := range n.Layers {
		if l.IsDisabled {
			continue
		}

		var gpuLayer gpu.GPULayer
		var layerOutputSize int

		var buildErr error

		if gpu.Debug {
			gpu.Log("Building layer %d type %d", i, l.Type)
		}
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
			if conv1d, ok := gpuLayer.(*gpu.Conv1DLayer); ok {
				conv1d.BatchSize = n.BatchSize
			}
			if rnn, ok := gpuLayer.(*gpu.RNNLayer); ok {
				rnn.BatchSize = n.BatchSize
			}
			if lstm, ok := gpuLayer.(*gpu.LSTMLayer); ok {
				lstm.BatchSize = n.BatchSize
			}
			if mha, ok := gpuLayer.(*gpu.MHALayer); ok {
				mha.BatchSize = n.BatchSize
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
			if lstmLayer, ok := l.(*gpu.LSTMLayer); ok {
				createGradientBindGroupsForLSTM(ctx, n.gpuGradPipeline, n.gpuGradParams, lstmLayer)
			}
			if conv1dLayer, ok := l.(*gpu.Conv1DLayer); ok {
				createGradientBindGroupsForConv1D(ctx, n.gpuGradPipeline, n.gpuGradParams, conv1dLayer)
			}
		}
	} else {
		fmt.Printf("Warning: Failed to initialize gradient pipeline: %v\n", err)
	}

	// Initialize Residual Buffers/Pipeline
	// Find max buffer size (max sequence length * hidden size)
	maxSize := 0
	for _, l := range layers {
		if mha, ok := l.(*gpu.MHALayer); ok {
			size := int(mha.GetInputBuffer().GetSize() / 4)
			if size > maxSize {
				maxSize = size
			}
		} else if swiglu, ok := l.(*gpu.SwiGLULayer); ok {
			size := int(swiglu.GetInputBuffer().GetSize() / 4)
			if size > maxSize {
				maxSize = size
			}
		}
	}
	if maxSize > 0 {
		n.gpuResidualBuffer, err = ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "Network_ResidualBuf",
			Size:  uint64(maxSize * 4),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
		if err != nil {
			return fmt.Errorf("create residual buffer: %w", err)
		}

		n.gpuResidualAdder, err = gpu.NewInPlaceResidual(ctx, maxSize)
		if err != nil {
			return fmt.Errorf("create residual adder: %w", err)
		}
	}

	// Wait for GPU to finish
	pollGPU(ctx)

	n.gpuLayers = layers
	n.gpuMounted = true
	n.gpuOutputSize = outputSize

	if gpu.Debug {
		gpu.Log("WeightsToGPU Complete! gpuMounted=%v layers=%d", n.gpuMounted, len(layers))
	}

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

	// Fix for Windows ARM64 / Adreno:
	// Ensure all GPU commands are finished before destroying resources.
	// This prevents "CommandBuffer cannot be destroyed because is still in use" panic.
	if ctx, ok := n.gpuCtx.(*gpu.Context); ok && ctx != nil {
		pollGPU(ctx)
	}

	if layers, ok := n.gpuLayers.([]gpu.GPULayer); ok {
		n.cleanupGPULayers(layers)
	}

	if n.gpuResidualBuffer != nil {
		n.gpuResidualBuffer.Destroy()
		n.gpuResidualBuffer = nil
	}
	// InPlaceResidual (pipeline cleanup?)
	if resAdder, ok := n.gpuResidualAdder.(*gpu.InPlaceResidual); ok {
		resAdder.Cleanup()
	}
	n.gpuResidualAdder = nil

	n.gpuLayers = nil
	n.gpuCtx = nil
	n.gpuMounted = false
	n.gpuOutputSize = 0
}

// SetGPU enables or disables GPU acceleration for the network
func (n *Network) SetGPU(enabled bool) {
	n.GPU = enabled
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
	if gpu.Debug {
		gpu.Log("forwardGPU Start. Input len: %d", len(input))
	}
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
		// Pad to buffer capacity (important if compiled for MAX_SEQ_LEN)
		capFloats := int(inputBuf.GetSize() / 4)
		if len(input) > capFloats {
			return nil, fmt.Errorf("GPU input too large: %d floats > buffer capacity %d floats", len(input), capFloats)
		}
		padded := make([]uint32, capFloats)
		for k := 0; k < len(input); k++ {
			padded[k] = uint32(input[k])
		}
		ctx.Queue.WriteBuffer(inputBuf, 0, wgpu.ToBytes(padded))
	} else {

		// Pad to buffer capacity so unused tokens are zeroed
		capFloats := int(inputBuf.GetSize() / 4)
		if len(input) > capFloats {
			return nil, fmt.Errorf("GPU input too large: %d floats > buffer capacity %d floats", len(input), capFloats)
		}
		if len(input) == capFloats {
			ctx.Queue.WriteBuffer(inputBuf, 0, wgpu.ToBytes(input))
		} else {
			padded := make([]float32, capFloats)
			copy(padded, input)
			ctx.Queue.WriteBuffer(inputBuf, 0, wgpu.ToBytes(padded))
		}
	}

	// Create command encoder and dispatch all layers
	cmdEnc, err := ctx.Device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("create command encoder: %w", err)
	}

	// Track temporary bind groups to release after submission
	tempBindGroups := make([]*wgpu.BindGroup, 0, len(layers))
	defer func() {
		for _, bg := range tempBindGroups {
			bg.Release()
		}
	}()

	var pass *wgpu.ComputePassEncoder
	resAdder, _ := n.gpuResidualAdder.(*gpu.InPlaceResidual)

	// Helper to ensure active pass
	ensurePass := func() {
		if pass == nil {
			pass = cmdEnc.BeginComputePass(nil)
		}
	}
	// Helper to end active pass
	endPass := func() {
		if pass != nil {
			pass.End()
			pass = nil
		}
	}

	// 1. Determine Sequence Length (seqTokens)
	// If First Layer is Embedding, input is TokenIDs (floats), so seqTokens = len(input)
	// If First Layer is Dense/other, input is flattened vectors, so seqTokens = len(input) / InputDim
	seqTokens := 1
	if _, ok := layers[0].(*gpu.EmbeddingLayer); ok {
		seqTokens = len(input)
	} else {
		// Fallback for non-embedding input (assuming D_MODEL of first layer logic?)
		// We need to know InputSize. buildGPULayer uses n.InputSize.
		// Approximating:
		if n.InputSize > 0 {
			seqTokens = len(input) / n.InputSize
		}
	}
	if seqTokens < 1 {
		seqTokens = 1
	}

	for i, l := range layers {
		// 1. Pre-Layer Actions (Residual Capture, MHA Setup)
		isMHA := false
		var mhaLayer *gpu.MHALayer
		if mha, ok := l.(*gpu.MHALayer); ok {
			isMHA = true
			mhaLayer = mha
			// Fix A: Correct Sequence Length (Tokens vs Floats)
			mha.SetActualSeqLen(ctx, seqTokens)
		}

		// Residual Start: Capture input of RMSNorm/LayerNorm
		isNorm := false
		if _, ok := l.(*gpu.RMSNormLayer); ok {
			isNorm = true
		}
		if _, ok := l.(*gpu.LayerNormLayer); ok {
			isNorm = true
		}

		if isNorm && n.gpuResidualBuffer != nil {
			endPass()
			// Copy this layer's input (which is the input to the block) to residual buffer
			cmdEnc.CopyBufferToBuffer(l.GetInputBuffer(), 0, n.gpuResidualBuffer, 0, l.GetInputBuffer().GetSize())
		}

		// 2. Dispatch Layer
		if isMHA {
			endPass() // MHA uses DispatchFull
			mhaLayer.DispatchFull(cmdEnc)
		} else {
			ensurePass()
			l.Dispatch(pass)
		}

		// 3. Post-Layer Actions (Residual Add)
		// Apply residual after MHA or SwiGLU (end of block)
		isSwiGLU := false
		if _, ok := l.(*gpu.SwiGLULayer); ok {
			isSwiGLU = true
		}

		if (isMHA || isSwiGLU) && n.gpuResidualBuffer != nil && resAdder != nil {
			// Add residual: Output = Output + ResidualBuf
			ensurePass() // Must be in pass
			bg, err := resAdder.Dispatch(ctx, pass, l.GetOutputBuffer(), n.gpuResidualBuffer)
			if err != nil {
				return nil, fmt.Errorf("residual bind group: %w", err)
			}
			tempBindGroups = append(tempBindGroups, bg)
		}

		// 4. Data Flow (Copy Output to Next Layer)
		if i < len(layers)-1 {
			next := layers[i+1]
			endPass() // Copy requires encoder level
			cmdEnc.CopyBufferToBuffer(l.GetOutputBuffer(), 0, next.GetInputBuffer(), 0, l.GetOutputBuffer().GetSize())
		} else {
			// Final output to staging
			endPass()
			cmdEnc.CopyBufferToBuffer(l.GetOutputBuffer(), 0, l.GetStagingBuffer(), 0, l.GetOutputBuffer().GetSize())
		}
	}
	endPass() // Ensure closed if last layer continued pass

	cmd, err := cmdEnc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("command encoder finish: %w", err)
	}
	ctx.Queue.Submit(cmd)
	pollGPU(ctx)

	// Read output from staging buffer
	lastLayer := layers[len(layers)-1]

	// IMPORTANT: staging buffer is already allocated to the compiled shape.
	// Do NOT multiply by n.BatchSize here (it changes during generation).
	outFloats := int(lastLayer.GetStagingBuffer().GetSize() / 4)
	output, err := readStagingBuffer(ctx, lastLayer.GetStagingBuffer(), outFloats)
	if err != nil {
		return nil, fmt.Errorf("read output: %w", err)
	}

	// Optional: slice down to the "real" output length implied by input embeddings.
	// For transformer blocks, output length should match input length (seqLen * hidden).
	// But we must compute expected size carefully.
	if seqTokens > 0 {
		// We don't easily know "lastLayer.OutputDim" here without casting.
		// But usually Output matches Input for transformers.
		// Let's assume full buffer read is safe, or slice if we are sure.
		// Safest fix: slice to (seqTokens * n.gpuOutputSize) if known?
		// n.gpuOutputSize was set in WeightsToGPU.
		expected := seqTokens * n.gpuOutputSize
		if expected > 0 && expected <= len(output) {
			return output[:expected], nil
		}
	}
	return output, nil
}

// backwardGPU runs the backward pass on GPU
// dOutput is the gradient of loss with respect to output
// Returns the gradient of loss with respect to input
func (n *Network) backwardGPU(dOutput []float32) ([]float32, error) {
	if gpu.Debug {
		gpu.Log("backwardGPU Start. dOutput len: %d", len(dOutput))
	}
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

// buildGPULayer constructs a GPU layer from an nn.LayerConfig
func (n *Network) buildGPULayer(l *LayerConfig, prevOutputSize int, idx int) (gpu.GPULayer, int, error) {
	if gpu.Debug {
		gpu.Log("buildGPULayer idx=%d type=%d", idx, l.Type)
	}
	inputSize := prevOutputSize
	if idx == 0 && inputSize == 0 {
		// First layer, use configured input size
		inputSize = l.InputHeight
		if inputSize == 0 {
			inputSize = n.InputSize
		}
	}

	// Resolve a stable seqLen/batch for transformer-ish layers.
	// Priority:
	//  1) l.SeqLength (you set this before mounting)
	//  2) n.BatchSize (mount-time BatchSize)
	//  3) fallback derived guess
	resolveSeqLen := func(derived int) int {
		if l.SeqLength > 0 {
			return l.SeqLength
		}
		if n.BatchSize > 0 {
			return n.BatchSize
		}
		return max1(derived)
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

		derived := 1
		if inputSize > 0 && pixelSize > 0 {
			derived = inputSize / pixelSize
		}
		batchSize := resolveSeqLen(derived)

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

		derived := 1
		if inputSize > 0 && pixelSize > 0 {
			derived = inputSize / pixelSize
		}
		batchSize := resolveSeqLen(derived)

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
			BatchSize:   resolveSeqLen(1),
			Temperature: temp,
		}
		return &gpu.SoftmaxLayer{Spec: spec}, inputSize, nil

	case LayerEmbedding:
		seqLen := resolveSeqLen(inputSize)
		spec := gpu.EmbeddingSpec{
			VocabSize:    l.VocabSize,
			EmbeddingDim: l.EmbeddingDim,
			SeqLength:    seqLen,
			Weights:      l.EmbeddingWeights,
		}
		return &gpu.EmbeddingLayer{Spec: spec}, l.EmbeddingDim * seqLen, nil

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

		// Determine Input Dimensions
		inH := 32
		inW := 32

		if l.InputHeight > 0 && l.InputWidth > 0 {
			inH = l.InputHeight
			inW = l.InputWidth
		} else if inChannels > 0 && inputSize > 0 {
			// Fallback: assume square input from flattened size
			val := float64(inputSize / inChannels)
			if val > 0 {
				side := int(math.Sqrt(val))
				inH = side
				inW = side
			}
		}

		outH := (inH+2*l.Padding-kernelSize)/stride + 1
		outW := (inW+2*l.Padding-kernelSize)/stride + 1
		if outH <= 0 {
			outH = 1
		}
		if outW <= 0 {
			outW = 1
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
			InputWidth:  inW,
			InputHeight: inH,
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

		derived := 1
		if inputSize > 0 && l.DModel > 0 {
			derived = inputSize / l.DModel
		}
		seqLen := resolveSeqLen(derived)

		spec := gpu.MHASpec{
			DModel:       l.DModel,
			NumHeads:     l.NumHeads,
			NumKVHeads:   l.NumKVHeads,
			HeadDim:      headDim,
			SeqLen:       seqLen,
			QWeights:     l.QWeights,
			KWeights:     l.KWeights,
			VWeights:     l.VWeights,
			OWeights:     l.OutputWeight,
			QBias:        l.QBias,
			KBias:        l.KBias,
			VBias:        l.VBias,
			OBias:        l.OutputBias,
			RoPEFreqBase: l.RoPEFreqBase,
		}
		return &gpu.MHALayer{Spec: spec}, l.DModel * seqLen, nil

	case LayerSwiGLU:

		derived := 1
		if l.InputHeight > 0 && inputSize > 0 {
			derived = inputSize / l.InputHeight
		}
		seqLen := resolveSeqLen(derived)

		// Ensure we have weights
		if len(l.GateWeights) == 0 {
			// Should not happen if loaded correctly, but handle gracefully?
			// For now assume loaded.
		}

		spec := gpu.SwiGLUSpec{
			InputSize:        l.InputHeight,
			IntermediateSize: l.OutputHeight, // SwiGLU OutputHeight stored intermediate dim?
			// Wait, in load_transformer: OutputHeight: config.IntermediateSize.
			// But SwiGLU output dimension is usually SAME as input dimension (down projection back to hidden).
			// Let's check load_transformer.go again.
			// It says: OutputHeight: config.IntermediateSize.
			// And DownWeights: [IntermediateSize * InputSize].
			// So l.OutputHeight holds the intermediate size.
			// The final output size is InputHeight (HiddenSize).
			// So Spec.IntermediateSize = l.OutputHeight.

			SeqLen:      seqLen,
			GateWeights: l.GateWeights,
			UpWeights:   l.UpWeights,
			DownWeights: l.DownWeights,
			GateBias:    l.GateBias,
			UpBias:      l.UpBias,
			DownBias:    l.DownBias,
		}

		// The output size of SwiGLU block is back to InputHeight (HiddenSize)
		// because of the down projection.
		// However, buildGPULayer returns `layerOutputSize`.
		// If I return l.OutputHeight, next layer thinks input is IntermediateSize.
		// But SwiGLU (as a block) returns HiddenSize.
		// Let's verify SwiGLU architecture in load_transformer.
		// DownWeights transposes from Intermediate back to Hidden.
		// So the output of the layer is HiddenSize.
		// But l.OutputHeight in Config was set to IntermediateSize?
		// load_transformer.go line 154: OutputHeight: config.IntermediateSize
		// This might be a semantic mismatch in Config usage.
		// For GPU layer build, I should return the ACTUAL output size of the block.
		// Which is l.InputHeight * seqLen.

		return &gpu.SwiGLULayer{Spec: spec}, l.InputHeight * seqLen, nil

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

	case LayerParallel:
		branches := make([]gpu.GPULayer, len(l.ParallelBranches))
		totalOutputSize := 0
		firstBranchSize := 0

		for i := range l.ParallelBranches {
			branchConf := &l.ParallelBranches[i]
			// Pass idx=0 for branches as they start with the parallel layer's input
			bl, outSize, err := n.buildGPULayer(branchConf, inputSize, 0)
			if err != nil {
				return nil, 0, err
			}
			branches[i] = bl

			if i == 0 {
				firstBranchSize = outSize
			}
			totalOutputSize += outSize
		}

		// Calculate final output size
		finalSize := totalOutputSize
		mode := l.CombineMode
		if mode == "add" || mode == "sum" || mode == "avg" || mode == "average" {
			finalSize = firstBranchSize
		}

		// Create ParallelLayer
		return gpu.NewParallelLayer(branches, l.CombineMode, n.BatchSize), finalSize, nil

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

	default:
		// Layer type not supported on GPU
		return nil, inputSize, nil
	}
}

// pollGPU waits for GPU operations to complete
func pollGPU(ctx *gpu.Context) {
	// For reliable synchronization, especially on tiled/mobile GPUs (ARM64),
	// we must wait until the device is truly idle before destroying resources.
	// wgpu.MaintainWait (true) blocks until all submitted work is done.
	ctx.Device.Poll(true, nil)
}

// readStagingBuffer reads float32 data from a staging buffer
func readStagingBuffer(ctx *gpu.Context, buf *wgpu.Buffer, size int) ([]float32, error) {
	done := make(chan struct{})
	var mapErr error

	if size <= 0 {
		return nil, fmt.Errorf("readStagingBuffer: invalid size %d", size)
	}

	bufFloats := int(buf.GetSize() / 4)
	if size > bufFloats {
		return nil, fmt.Errorf("readStagingBuffer: requested %d floats (%d bytes) but buffer holds %d floats (%d bytes)",
			size, size*4, bufFloats, buf.GetSize())
	}

	wantBytes := uint64(size) * 4
	buf.MapAsync(wgpu.MapModeRead, 0, wantBytes, func(status wgpu.BufferMapAsyncStatus) {
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

	data := buf.GetMappedRange(0, uint(wantBytes))
	defer buf.Unmap()

	if data == nil {
		return nil, fmt.Errorf("mapped range nil")
	}

	out := make([]float32, size)
	copy(out, wgpu.FromBytes[float32](data)[:size])

	return out, nil
}

func max1(v int) int {
	if v <= 0 {
		return 1
	}
	return v
}
