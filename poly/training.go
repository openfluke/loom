package poly

import (
	"fmt"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

// TrainingMode selects which execution path is used for forward and backward passes.
type TrainingMode int

const (
	TrainingModeCPUNormal TrainingMode = iota // serial CPU, no tiling
	TrainingModeCPUSC                         // CPU tiling, single-core
	TrainingModeCPUMC                         // CPU tiling, multi-core parallel
	TrainingModeGPUNormal                     // GPU global-memory path
	TrainingModeGPUSC                         // GPU tiled SC (workgroup 64)
	TrainingModeGPUMC                         // GPU tiled MC (workgroup 256)
)

func (m TrainingMode) String() string {
	switch m {
	case TrainingModeCPUNormal:
		return "CPU-Normal"
	case TrainingModeCPUSC:
		return "CPU-SC-Tiled"
	case TrainingModeCPUMC:
		return "CPU-MC-Tiled"
	case TrainingModeGPUNormal:
		return "GPU-Normal"
	case TrainingModeGPUSC:
		return "GPU-SC-Tiled"
	case TrainingModeGPUMC:
		return "GPU-MC-Tiled"
	default:
		return "Unknown"
	}
}

// IsGPU reports whether the mode requires GPU execution.
func (m TrainingMode) IsGPU() bool { return m >= TrainingModeGPUNormal }

// TrainingConfig holds configuration for training in the Volumetric Grid.
type TrainingConfig struct {
	Epochs       int
	LearningRate float32
	LossType     string       // "mse" or "cross_entropy"
	GradientClip float32      // Max gradient norm (0 = no clipping)
	Verbose      bool
	UseGPU       bool         // Deprecated: use Mode instead
	Mode         TrainingMode // Execution path; overrides UseGPU when non-zero
	DeviceID     int
	TrackPerf    bool
}

// TrainingBatch represents a single training batch for the Poly engine.
type TrainingBatch[T Numeric] struct {
	Input  *Tensor[T]
	Target *Tensor[T]
}

// TrainingResult contains training statistics for the Poly engine.
type TrainingResult struct {
	FinalLoss   float64
	TotalTime   time.Duration
	LossHistory []float64
	EpochTimes  []time.Duration
}

// DefaultTrainingConfig returns sensible defaults for the Bedrock architecture.
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		Epochs:       10,
		LearningRate: 0.01,
		LossType:     "mse",
		Verbose:      true,
	}
}

// resolveMode returns the effective TrainingMode, honouring legacy UseGPU flag.
func resolveMode(config *TrainingConfig) TrainingMode {
	if config.Mode != 0 {
		return config.Mode
	}
	if config.UseGPU {
		return TrainingModeGPUNormal
	}
	return TrainingModeCPUNormal
}

// configureNetworkForMode sets tiling/GPU flags on all layers and syncs state.
func configureNetworkForMode(n *VolumetricNetwork, mode TrainingMode) error {
	switch mode {
	case TrainingModeCPUNormal:
		for i := range n.Layers {
			n.Layers[i].UseTiling = false
			n.Layers[i].EnableMultiCoreTiling = false
		}
		n.SyncToCPU()
	case TrainingModeCPUSC:
		for i := range n.Layers {
			n.Layers[i].UseTiling = true
			n.Layers[i].EnableMultiCoreTiling = false
		}
		n.SyncToCPU()
	case TrainingModeCPUMC:
		n.EnableMultiCoreTiling = true
		for i := range n.Layers {
			n.Layers[i].UseTiling = true
			n.Layers[i].EnableMultiCoreTiling = true
		}
		n.SyncToCPU()
	case TrainingModeGPUNormal, TrainingModeGPUSC, TrainingModeGPUMC:
		if n.GPUContext == nil {
			if err := n.InitWGPU(); err != nil {
				return fmt.Errorf("failed to initialize GPU: %w", err)
			}
		}
		if err := n.SyncToGPU(); err != nil {
			return fmt.Errorf("failed to sync weights to GPU: %w", err)
		}
	}
	return nil
}

// SyncWeightsFromGPU reads GPU float32 weight buffers back into each layer's
// WeightStore.Master. Call this after GPU training before serializing.
func SyncWeightsFromGPU(n *VolumetricNetwork) error {
	ctx := n.GPUContext
	if ctx == nil {
		return fmt.Errorf("no GPU context")
	}
	for i := range n.Layers {
		l := &n.Layers[i]
		if l.WeightStore == nil {
			continue
		}
		wBuf, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		if !ok || wBuf == nil {
			continue
		}
		data, err := ctx.ReadBuffer(wBuf)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		if len(data) != len(l.WeightStore.Master) {
			return fmt.Errorf("layer %d: weight size mismatch GPU=%d CPU=%d", i, len(data), len(l.WeightStore.Master))
		}
		copy(l.WeightStore.Master, data)
		// Clear stale Versions and GPU cache tags (other than the FP32 buffer we just read)
		l.WeightStore.Versions = make(map[DType]any)
	}
	return nil
}

// Train executes the training loop on a VolumetricNetwork.
func Train[T Numeric](n *VolumetricNetwork, batches []TrainingBatch[T], config *TrainingConfig) (*TrainingResult, error) {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	mode := resolveMode(config)
	if err := configureNetworkForMode(n, mode); err != nil {
		return nil, err
	}

	// Determine CNN3 tile sizes for GPU tiled modes.
	cnn3TileSize := 0
	if n.GPUContext != nil && (mode == TrainingModeGPUSC || mode == TrainingModeGPUMC) {
		scTile, mcTile := CNN3GPUTileSizes(n.GPUContext)
		if mode == TrainingModeGPUSC {
			cnn3TileSize = scTile
		} else {
			cnn3TileSize = mcTile
		}
	}

	result := &TrainingResult{
		LossHistory: make([]float64, 0, config.Epochs),
		EpochTimes:  make([]time.Duration, 0, config.Epochs),
	}

	totalStart := time.Now()
	numBatches := len(batches)

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochStart := time.Now()
		epochLoss := 0.0

		for _, batch := range batches {
			if mode.IsGPU() {
				loss, err := trainBatchGPU(n, batch, config, cnn3TileSize)
				if err != nil {
					return nil, err
				}
				epochLoss += loss
			} else {
				epochLoss += trainBatchCPU(n, batch, config)
			}
		}

		epochDuration := time.Since(epochStart)
		avgLoss := epochLoss / float64(numBatches)

		result.LossHistory = append(result.LossHistory, avgLoss)
		result.EpochTimes = append(result.EpochTimes, epochDuration)

		if config.Verbose {
			elapsed := time.Since(totalStart)
			avgEpochTime := elapsed / time.Duration(epoch+1)
			remainingEpochs := config.Epochs - (epoch + 1)
			eta := avgEpochTime * time.Duration(remainingEpochs)

			samplesPerSec := 0.0
			if elapsed > 0 {
				samplesPerSec = float64(int64(epoch+1)*int64(numBatches)) / elapsed.Seconds()
			}

			fmt.Printf("Epoch %d/%d - Loss: %.6f | Time: %v | Samples/s: %.2f | ETA: %v\n",
				epoch+1, config.Epochs, avgLoss, epochDuration, samplesPerSec, eta.Round(time.Second))
		}
	}

	result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	result.TotalTime = time.Since(totalStart)
	return result, nil
}

func trainBatchCPU[T Numeric](n *VolumetricNetwork, batch TrainingBatch[T], config *TrainingConfig) float64 {
	// 1. Forward Pass
	histIn := make([]*Tensor[T], len(n.Layers))
	histPre := make([]*Tensor[T], len(n.Layers))
	curr := batch.Input

	for idx := range n.Layers {
		l := &n.Layers[idx]
		if l.IsDisabled { continue }
		histIn[idx] = curr
		pre, post := DispatchLayer(l, curr, nil)
		histPre[idx] = pre
		curr = post
	}

	// 2. Compute Loss Gradient
	gradOut := ComputeLossGradient(curr, batch.Target, config.LossType)
	lossVal := CalculateLoss(curr, batch.Target, config.LossType)
	
	// 3. Backward Pass
	_, layerGradients, _ := BackwardPolymorphic(n, gradOut, histIn, histPre)

	// 4. Update Weights
	for idx := range n.Layers {
		l := &n.Layers[idx]
		if layerGradients[idx][1] != nil {
			gW := ConvertTensor[T, float32](layerGradients[idx][1])
			ApplyRecursiveGradients(l, gW, config.LearningRate)
		}
	}
	return lossVal
}

// trainBatchGPU runs one training batch on the GPU.
// cnn3TileSize == 0  → GPU Normal (global-memory DispatchForwardLayer / DispatchBackwardLayer).
// cnn3TileSize  > 0  → GPU Tiled (DispatchCNN3Tiled / DispatchCNN3TiledBackwardDX+DW for CNN3 layers).
func trainBatchGPU[T Numeric](n *VolumetricNetwork, batch TrainingBatch[T], config *TrainingConfig, cnn3TileSize int) (float64, error) {
	ctx := n.GPUContext
	if ctx == nil {
		return 0, fmt.Errorf("GPU context is nil")
	}
	batchSize := batch.Input.Shape[0]

	// 1. Upload input and target to GPU (queue-level writes, safe before BeginFrame)
	inData := ConvertTensor[T, float32](batch.Input).Data
	inBuf := ctx.GetActivationBuffer("batch_input", uint64(len(inData)*4), wgpu.BufferUsageStorage)
	if inBuf == nil {
		return 0, fmt.Errorf("failed to get inBuf")
	}
	ctx.Queue.WriteBuffer(inBuf, 0, wgpu.ToBytes(inData))

	targetData := ConvertTensor[T, float32](batch.Target).Data
	targetBuf := ctx.GetActivationBuffer("batch_target", uint64(len(targetData)*4), wgpu.BufferUsageStorage)
	if targetBuf == nil {
		return 0, fmt.Errorf("failed to get targetBuf")
	}
	ctx.Queue.WriteBuffer(targetBuf, 0, wgpu.ToBytes(targetData))

	// 2. Begin batched GPU frame: entire forward + grad + backward in ONE submission
	if err := ctx.BeginFrame(); err != nil {
		return 0, fmt.Errorf("failed to begin GPU frame: %w", err)
	}

	// 3. Forward pass (all recorded into shared encoder)
	histInBuf := make([]*wgpu.Buffer, len(n.Layers))
	histPreBuf := make([]*wgpu.Buffer, len(n.Layers))
	curBuf := inBuf

	for i := range n.Layers {
		l := &n.Layers[i]
		if l.IsDisabled {
			continue
		}
		histInBuf[i] = curBuf

		outSize := l.OutputHeight
		if l.Type == LayerCNN2 || l.Type == LayerCNN3 {
			d := l.OutputDepth
			if d == 0 {
				d = 1
			}
			h := l.OutputHeight
			if h == 0 {
				h = 1
			}
			w := l.OutputWidth
			if w == 0 {
				w = 1
			}
			outSize = d * h * w * l.Filters
		} else if l.Type == LayerCNN1 {
			outSize = l.OutputHeight * l.Filters
		}
		if outSize == 0 {
			outSize = l.InputHeight
		}

		preBuf := ctx.GetActivationBuffer(fmt.Sprintf("pre_%d", i), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
		if preBuf == nil {
			ctx.FlushFrame()
			return 0, fmt.Errorf("failed to get preBuf for layer %d", i)
		}

		var fwdErr error
		if cnn3TileSize > 0 && l.Type == LayerCNN3 {
			kernelVol := l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize
			scale := l.WeightStore.Scale
			if scale == 0 {
				scale = 1.0
			}
			wBuf, _ := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
			fwdErr = ctx.DispatchCNN3Tiled(cnn3TileSize, kernelVol, batchSize,
				l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth,
				l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth,
				l.KernelSize, l.KernelSize, l.KernelSize,
				l.Stride, l.Stride, l.Stride,
				l.Padding, l.Padding, l.Padding,
				scale, curBuf, wBuf, preBuf)
		} else {
			fwdErr = ctx.DispatchForwardLayer(l, batchSize, curBuf, preBuf)
		}
		if fwdErr != nil {
			ctx.FlushFrame()
			return 0, fwdErr
		}

		if l.Activation != ActivationLinear {
			postBuf := ctx.GetActivationBuffer(fmt.Sprintf("post_%d", i), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
			if postBuf == nil {
				ctx.FlushFrame()
				return 0, fmt.Errorf("failed to get postBuf for layer %d", i)
			}
			if err := ctx.DispatchActivation(outSize*batchSize, l.Activation, preBuf, postBuf); err != nil {
				ctx.FlushFrame()
				return 0, err
			}
			curBuf = postBuf
		} else {
			curBuf = preBuf
		}
		histPreBuf[i] = preBuf
	}

	// 4. GPU MSE gradient + partial loss (no CPU readback needed to continue)
	totalOutput := len(targetData)
	numWG := (totalOutput + 255) / 256
	gradBuf := ctx.GetActivationBuffer("grad_out", uint64(totalOutput*4), wgpu.BufferUsageStorage)
	partialsBuf := ctx.GetActivationBuffer("loss_partials", uint64(numWG*4), wgpu.BufferUsageStorage)
	if gradBuf == nil || partialsBuf == nil {
		ctx.FlushFrame()
		return 0, fmt.Errorf("failed to allocate loss buffers")
	}
	if err := ctx.DispatchMSEGradPartialLoss(totalOutput, curBuf, targetBuf, gradBuf, partialsBuf); err != nil {
		ctx.FlushFrame()
		return 0, err
	}

	// 5. Backward pass + weight updates (inside same frame)
	// Queue.WriteBuffer for DW zero-clearing is safe inside BeginFrame:
	// all writes are guaranteed to complete before the encoder submit.
	curGradBuf := gradBuf
	for i := len(n.Layers) - 1; i >= 0; i-- {
		l := &n.Layers[i]
		if l.IsDisabled {
			continue
		}

		inSize := l.InputHeight
		if l.Type == LayerCNN2 || l.Type == LayerCNN3 {
			d := l.InputDepth
			if d == 0 {
				d = 1
			}
			h := l.InputHeight
			if h == 0 {
				h = 1
			}
			w := l.InputWidth
			if w == 0 {
				w = 1
			}
			inSize = d * h * w * l.InputChannels
		} else if l.Type == LayerCNN1 {
			inSize = l.InputHeight * l.InputChannels
		}
		if inSize == 0 {
			inSize = l.OutputHeight
		}

		outSize := l.OutputHeight
		if l.Type == LayerCNN2 || l.Type == LayerCNN3 {
			d := l.OutputDepth
			if d == 0 {
				d = 1
			}
			h := l.OutputHeight
			if h == 0 {
				h = 1
			}
			w := l.OutputWidth
			if w == 0 {
				w = 1
			}
			outSize = d * h * w * l.Filters
		} else if l.Type == LayerCNN1 {
			outSize = l.OutputHeight * l.Filters
		}
		if outSize == 0 {
			outSize = l.InputHeight
		}

		dxBuf := ctx.GetActivationBuffer(fmt.Sprintf("dx_%d", i), uint64(inSize*batchSize*4), wgpu.BufferUsageStorage)
		if dxBuf == nil {
			ctx.FlushFrame()
			return 0, fmt.Errorf("failed to get dxBuf for layer %d", i)
		}

		wSize := len(l.WeightStore.Master)
		dwBuf := ctx.GetActivationBuffer(fmt.Sprintf("dw_%d", i), uint64(wSize*4), wgpu.BufferUsageStorage)
		if dwBuf == nil {
			ctx.FlushFrame()
			return 0, fmt.Errorf("failed to get dwBuf for layer %d", i)
		}
		// Zero DW buffer before accumulation (WriteBuffer is a queue-level op, safe inside BeginFrame)
		ctx.Queue.WriteBuffer(dwBuf, 0, make([]byte, wSize*4))

		var gradPreBuf *wgpu.Buffer
		if l.Activation != ActivationLinear {
			gradPreBuf = ctx.GetActivationBuffer(fmt.Sprintf("grad_pre_%d", i), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
			if gradPreBuf == nil {
				ctx.FlushFrame()
				return 0, fmt.Errorf("failed to get gradPreBuf for layer %d", i)
			}
			if err := ctx.DispatchActivationBackward(outSize*batchSize, l.Activation, curGradBuf, histPreBuf[i], gradPreBuf); err != nil {
				ctx.FlushFrame()
				return 0, err
			}
		} else {
			gradPreBuf = curGradBuf
		}

		var bwdErr error
		if cnn3TileSize > 0 && l.Type == LayerCNN3 {
			kernelVol := l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize
			wBuf, _ := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
			if err := ctx.DispatchCNN3TiledBackwardDX(cnn3TileSize, kernelVol, batchSize,
				l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth,
				l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth,
				l.KernelSize, l.KernelSize, l.KernelSize,
				l.Stride, l.Stride, l.Stride,
				l.Padding, l.Padding, l.Padding,
				l.Activation, gradPreBuf, wBuf, histPreBuf[i], dxBuf); err != nil {
				ctx.FlushFrame()
				return 0, err
			}
			bwdErr = ctx.DispatchCNN3TiledBackwardDW(cnn3TileSize, batchSize,
				l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth,
				l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth,
				l.KernelSize, l.KernelSize, l.KernelSize,
				l.Stride, l.Stride, l.Stride,
				l.Padding, l.Padding, l.Padding,
				l.Activation, gradPreBuf, histInBuf[i], histPreBuf[i], dwBuf)
		} else {
			bwdErr = ctx.DispatchBackwardLayer(l, batchSize, gradPreBuf, histInBuf[i], histPreBuf[i], dxBuf, dwBuf)
		}
		if bwdErr != nil {
			ctx.FlushFrame()
			return 0, bwdErr
		}

		if l.WeightStore != nil {
			wBuf, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
			if ok && wBuf != nil {
				if err := ctx.DispatchApplyGradients(wSize, config.LearningRate, wBuf, dwBuf); err != nil {
					ctx.FlushFrame()
					return 0, err
				}
			}
		}

		curGradBuf = dxBuf
	}

	// 6. Submit entire forward + backward in ONE GPU call
	ctx.FlushFrame()

	// 7. Read back only the tiny partial loss sums (numWG * 4 bytes vs full output readback)
	partials, err := ctx.ReadBuffer(partialsBuf)
	if err != nil {
		return 0, err
	}
	lossVal := 0.0
	nPartials := numWG
	if nPartials > len(partials) {
		nPartials = len(partials)
	}
	for _, p := range partials[:nPartials] {
		lossVal += float64(p)
	}
	return lossVal, nil
}

// CalculateLoss computes the loss between output and target.
func CalculateLoss[T Numeric](output, target *Tensor[T], lossType string) float64 {
	if len(output.Data) != len(target.Data) {
		return 0
	}
	sum := 0.0
	switch lossType {
	case "mse":
		for i := range output.Data {
			diff := float64(output.Data[i] - target.Data[i])
			sum += diff * diff
		}
		return sum / float64(len(output.Data))
	default:
		return 0
	}
}

// ComputeLossGradient computes the gradient of the loss with respect to the output.
func ComputeLossGradient[T Numeric](output, target *Tensor[T], lossType string) *Tensor[T] {
	grad := NewTensor[T](output.Shape...)
	switch lossType {
	case "mse":
		// Gradient of MSE: 2/N * (output - target)
		scale := T(2.0 / float32(len(output.Data)))
		for i := range output.Data {
			grad.Data[i] = (output.Data[i] - target.Data[i]) * scale
		}
	}
	return grad
}
// ApplyRecursiveGradients traverses the layer hierarchy and updates weights in all nested WeightStores.
func ApplyRecursiveGradients(layer *VolumetricLayer, gradWeights *Tensor[float32], lr float32) {
	if layer == nil || gradWeights == nil {
		return
	}

	// 1. Update local weights if they exist
	if layer.WeightStore != nil {
		layer.WeightStore.ApplyGradients(gradWeights, lr)
	}

	// 2. Recursively update Parallel branches
	if layer.Type == LayerParallel && len(layer.ParallelBranches) > 0 && len(gradWeights.Nested) > 0 {
		for i := range layer.ParallelBranches {
			if i < len(gradWeights.Nested) {
				ApplyRecursiveGradients(&layer.ParallelBranches[i], gradWeights.Nested[i], lr)
			}
		}
	}

	// 3. Recursively update Sequential stages
	if layer.Type == LayerSequential && len(layer.SequentialLayers) > 0 && len(gradWeights.Nested) > 0 {
		for i := range layer.SequentialLayers {
			if i < len(gradWeights.Nested) {
				// Sequential intermediates are containers: [0]=bPre, [1]=bInput
				// But gradWeights for Sequential is just a tree of weights.
				// Wait, SequentialBackward returns &Tensor[T]{Nested: branchGradWeights}
				// So gradWeights.Nested[i] IS the gradWeights for sub-layer i.
				ApplyRecursiveGradients(&layer.SequentialLayers[i], gradWeights.Nested[i], lr)
			}
		}
	}
}
