package poly

import (
	"fmt"
	"math"
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
	LossType     string  // "mse", "crossentropy", or "multi_head_softmax_ce"
	GradientClip float32 // Max gradient norm (0 = no clipping)
	Verbose      bool
	UseGPU       bool         // Deprecated: use Mode instead
	Mode         TrainingMode // Execution path; overrides UseGPU when non-zero
	DeviceID     int
	TrackPerf    bool
	// SoftmaxCEHeads is the class count per contiguous slice of the output (logits), e.g. []int{4,17,20}.
	// When LossType is "multi_head_softmax_ce", each row of the batch is split into these heads; inside
	// each head we apply softmax and cross-entropy vs the matching one-hot slice of the target.
	SoftmaxCEHeads []int
	// DisableCPUTrainingFallback, when true, GPU training returns an error instead of silently running
	// the optimizer on CPU for networks that still lack a full GPU backward path for some layer types.
	DisableCPUTrainingFallback bool
}

// TrainingBatch represents a single training batch for the Poly engine.
type TrainingBatch[T Numeric] struct {
	Input  *Tensor[T]
	Target *Tensor[T]
	// MultiHeadMask is optional. When LossType is multi_head_softmax_ce, each row contributes
	// three mask values {GPS,EACS,Sent} at indices [b*3+0], [b*3+1], [b*3+2] (use 0 or 1).
	// Inactive heads skip CE and receive zero dL/dlogit on that slice (CPU path; GPU uses CPU fallback for the batch).
	MultiHeadMask *Tensor[T]
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
	return TrainingModeCPUMC
}

// ConfigureNetworkForMode sets tiling/GPU flags on all layers and syncs state.
func ConfigureNetworkForMode(n *VolumetricNetwork, mode TrainingMode) error {
	switch mode {
	case TrainingModeCPUNormal, TrainingModeCPUSC, TrainingModeCPUMC:
		// CPU paths use multi-core tiled implementations only (no separate naive / serial-tiled stacks).
		n.EnableMultiCoreTiling = true
		n.RefreshRuntimeTileSizes()
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
		n.RefreshRuntimeTileSizes()
		// Reset bind group cache so new weight buffers get fresh bind groups.
		// Stale cached bind groups (keyed by old buffer pointers) can cause
		// DispatchApplyGradients to write to a previous network's weight buffer
		// instead of this network's, leaving weights unchanged after training.
		n.GPUContext.ResetCache()
		if err := n.SyncToGPU(); err != nil {
			return fmt.Errorf("failed to sync weights to GPU: %w", err)
		}
		// Training needs a Float32 master buffer on GPU for every layer so that
		// backward kernels compute gradients at full precision against the master weights.
		// This is intentionally separate from SyncToGPU so that inference-only paths
		// do not waste VRAM uploading an extra Float32 copy alongside quantized buffers.
		if err := ensureGPUFloat32Weights(n); err != nil {
			return fmt.Errorf("failed to ensure GPU float32 weights: %w", err)
		}
	}
	return nil
}

// ensureGPUFloat32Weights guarantees that the high-precision (FP32) accumulation
// state on the GPU is synchronized with the secondary Master body on the CPU.
// In the current architecture, this ensures backward kernels have an exact-precision
// buffer (the secondary Source of Truth) for gradient computation.
func ensureGPUFloat32Weights(n *VolumetricNetwork) error {
	if n.GPUContext == nil {
		return nil
	}
	for i := range n.Layers {
		if err := ensureGPUFloat32WeightsLayer(&n.Layers[i]); err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
	}
	return nil
}

func ensureGPUFloat32WeightsLayer(l *VolumetricLayer) error {
	ctx := l.Network.GPUContext
	if ctx == nil {
		return nil
	}
	if l.WeightStore != nil && len(l.WeightStore.Master) > 0 {
		if existingAny, hasF32 := l.WeightStore.GPUWeights[DTypeFloat32]; hasF32 {
			if wBuf, ok := existingAny.(*wgpu.Buffer); ok && wBuf != nil {
				ctx.Queue.WriteBuffer(wBuf, 0, wgpu.ToBytes(l.WeightStore.Master))
			}
		} else {
			buf, err := ctx.CreatePersistentBuffer(l.WeightStore.Master, "Training FP32 Master")
			if err != nil {
				return err
			}
			l.WeightStore.GPUWeights[DTypeFloat32] = buf
		}
	}
	for i := range l.ParallelBranches {
		if err := ensureGPUFloat32WeightsLayer(&l.ParallelBranches[i]); err != nil {
			return err
		}
	}
	for i := range l.SequentialLayers {
		if err := ensureGPUFloat32WeightsLayer(&l.SequentialLayers[i]); err != nil {
			return err
		}
	}
	return nil
}

// SyncWeightsFromGPU reads high-precision GPU weight buffers back into the
// secondary Master storage on the CPU. Call this after GPU training before
// serializing to ensure persistence matches the trained GPU state.
func SyncWeightsFromGPU(n *VolumetricNetwork) error {
	if n.GPUContext == nil {
		return fmt.Errorf("no GPU context")
	}
	for i := range n.Layers {
		if err := syncWeightsFromGPULayer(&n.Layers[i]); err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
	}
	return nil
}

func syncWeightsFromGPULayer(l *VolumetricLayer) error {
	ctx := l.Network.GPUContext
	if ctx == nil {
		return nil
	}
	if l.WeightStore != nil {
		wBuf, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
		if ok && wBuf != nil {
			data, err := ctx.ReadBuffer(wBuf)
			if err != nil {
				return err
			}
			if len(data) != len(l.WeightStore.Master) {
				return fmt.Errorf("weight size mismatch GPU=%d CPU=%d", len(data), len(l.WeightStore.Master))
			}
			copy(l.WeightStore.Master, data)
			l.WeightStore.Versions = make(map[DType]any)
			l.WeightStore.CPUPacked = make(map[DType]any)
		}
	}
	for i := range l.ParallelBranches {
		if err := syncWeightsFromGPULayer(&l.ParallelBranches[i]); err != nil {
			return err
		}
	}
	for i := range l.SequentialLayers {
		if err := syncWeightsFromGPULayer(&l.SequentialLayers[i]); err != nil {
			return err
		}
	}
	return nil
}

// Train executes the training loop on a VolumetricNetwork.
func Train[T Numeric](n *VolumetricNetwork, batches []TrainingBatch[T], config *TrainingConfig) (*TrainingResult, error) {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	mode := resolveMode(config)
	if err := ConfigureNetworkForMode(n, mode); err != nil {
		return nil, err
	}

	// GPU tile sizes are now per-layer per-dtype; passed via mode to trainBatchGPU.

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
				loss, err := trainBatchGPU(n, batch, config, mode)
				if err != nil {
					return nil, err
				}
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					if config.Verbose {
						fmt.Printf("⚠️ Fatal NaN/Inf loss in epoch %d. Skipping epoch.\n", epoch+1)
					}
					break
				}
				epochLoss += loss
			} else {
				// CPU Path: Calculate then Validate then Apply
				loss, layerGradients := executeBatchCPU(n, batch, config)
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					if config.Verbose {
						fmt.Printf("⚠️ Fatal NaN/Inf loss in epoch %d. Skipping epoch.\n", epoch+1)
					}
					break
				}

				// Defensive Validation
				hasInvalid := false
				for idx := range layerGradients {
					if layerGradients[idx][1] != nil && layerGradients[idx][1].HasInvalid() {
						hasInvalid = true
						break
					}
				}

				if hasInvalid {
					if config.Verbose {
						fmt.Printf("⚠️ Fatal NaN/Inf gradients in epoch %d. Skipping batch updates.\n", epoch+1)
					}
					continue
				}

				// Safe Update
				applyGradientsCPU(n, layerGradients, config)
				epochLoss += loss
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

func executeBatchCPU[T Numeric](n *VolumetricNetwork, batch TrainingBatch[T], config *TrainingConfig) (float64, [][2]*Tensor[T]) {
	// 1. Forward Pass
	histIn := make([]*Tensor[T], len(n.Layers))
	histPre := make([]*Tensor[T], len(n.Layers))
	curr := batch.Input

	for idx := range n.Layers {
		l := &n.Layers[idx]
		if l.IsDisabled {
			continue
		}
		histIn[idx] = curr
		pre, post := DispatchLayer(l, curr, nil)
		histPre[idx] = pre
		curr = post
	}

	// 2. Compute Loss Gradient
	var gradOut *Tensor[T]
	var lossVal float64
	if config.LossType == "multi_head_softmax_ce" {
		if batch.MultiHeadMask != nil {
			gradOut = computeLossGradientMultiHeadSoftmaxCEMasked(curr, batch.Target, config.SoftmaxCEHeads, batch.MultiHeadMask)
			lossVal = calculateLossMultiHeadSoftmaxCEMasked(curr, batch.Target, config.SoftmaxCEHeads, batch.MultiHeadMask)
		} else {
			gradOut = computeLossGradientMultiHeadSoftmaxCE(curr, batch.Target, config.SoftmaxCEHeads)
			lossVal = calculateLossMultiHeadSoftmaxCE(curr, batch.Target, config.SoftmaxCEHeads)
		}
	} else {
		gradOut = ComputeLossGradient(curr, batch.Target, config.LossType)
		lossVal = CalculateLoss(curr, batch.Target, config.LossType)
	}

	// 3. Backward Pass
	_, layerGradients, _ := BackwardPolymorphic(n, gradOut, histIn, histPre)
	return lossVal, layerGradients
}

func applyGradientsCPU[T Numeric](n *VolumetricNetwork, layerGradients [][2]*Tensor[T], config *TrainingConfig) {
	// 4. Update Weights
	for idx := range n.Layers {
		l := &n.Layers[idx]
		if layerGradients[idx][1] != nil {
			gW := ConvertTensor[T, float32](layerGradients[idx][1])
			ApplyRecursiveGradients(l, gW, config.LearningRate, config.GradientClip)
		}
	}
}

// trainBatchGPU runs one training batch on the GPU.
// mode == TrainingModeGPUNormal → global-memory dispatch.
// mode == TrainingModeGPUSC/MC  → tiled dispatch; tile size is read per-layer from l.GetGPUSCTileSize / GetGPUMCTileSize.
func trainBatchGPU[T Numeric](n *VolumetricNetwork, batch TrainingBatch[T], config *TrainingConfig, mode TrainingMode) (float64, error) {
	ctx := n.GPUContext
	if ctx == nil {
		return 0, fmt.Errorf("GPU context is nil")
	}

	// Some layer types still have incomplete or unstable GPU backward paths.
	// For those, keep the user's requested GPU mode for network residency/state,
	// but execute the actual gradient step on CPU and re-upload weights so the
	// next epoch still runs against the current parameters.
	if gpuTrainingNeedsCPUFallback(n) {
		if config.DisableCPUTrainingFallback {
			return 0, fmt.Errorf("GPU training: this network still uses a CPU optimizer fallback for at least one layer type; use CPU mode, remove/replace those layers, or clear DisableCPUTrainingFallback")
		}
		loss, layerGradients := executeBatchCPU(n, batch, config)
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return loss, nil
		}
		applyGradientsCPU(n, layerGradients, config)
		if err := n.SyncToGPU(); err != nil {
			return 0, fmt.Errorf("cpu-fallback sync-to-gpu failed: %w", err)
		}
		if err := ensureGPUFloat32Weights(n); err != nil {
			return 0, fmt.Errorf("cpu-fallback ensure float32 weights failed: %w", err)
		}
		return loss, nil
	}
	batchSize := batch.Input.Shape[0]

	// 1. Upload input and target to GPU (queue-level writes, safe before BeginFrame)
	inData := ConvertTensor[T, float32](batch.Input).Data
	inBuf := ctx.GetActivationBuffer("batch_input", uint64(len(inData)*4), wgpu.BufferUsageStorage)
	if inBuf == nil {
		return 0, fmt.Errorf("failed to get inBuf")
	}

	// SPECIAL CASE: If the first layer is an Embedding layer, we must upload the indices as uint32.
	// trainBatchGPU converts everything to float32 for general layer compatibility, but
	// Embedding layers read their input bindings as array<u32> (token indices).
	if len(n.Layers) > 0 && n.Layers[0].Type == LayerEmbedding {
		ids := make([]uint32, len(inData))
		for i, v := range inData {
			ids[i] = uint32(v)
		}
		ctx.Queue.WriteBuffer(inBuf, 0, wgpu.ToBytes(ids))
	} else {
		ctx.Queue.WriteBuffer(inBuf, 0, wgpu.ToBytes(inData))
	}

	targetData := ConvertTensor[T, float32](batch.Target).Data
	targetBuf := ctx.GetActivationBuffer("batch_target", uint64(len(targetData)*4), wgpu.BufferUsageStorage)
	if targetBuf == nil {
		return 0, fmt.Errorf("failed to get targetBuf")
	}
	ctx.Queue.WriteBuffer(targetBuf, 0, wgpu.ToBytes(targetData))

	var maskBuf *wgpu.Buffer
	if batch.MultiHeadMask != nil && config.LossType == "multi_head_softmax_ce" {
		need := batchSize * 3
		mf := ConvertTensor[T, float32](batch.MultiHeadMask).Data
		if len(mf) < need {
			return 0, fmt.Errorf("MultiHeadMask: need %d floats (batch*3), got %d", need, len(mf))
		}
		maskBuf = ctx.GetActivationBuffer("multihead_ce_mask", uint64(need*4), wgpu.BufferUsageStorage)
		if maskBuf == nil {
			return 0, fmt.Errorf("failed to get multi-head mask buffer")
		}
		ctx.Queue.WriteBuffer(maskBuf, 0, wgpu.ToBytes(mf[:need]))
	}

	// 2. Begin batched GPU frame: entire forward + grad + backward in ONE submission
	if err := ctx.BeginFrame(); err != nil {
		return 0, fmt.Errorf("failed to begin GPU frame: %w", err)
	}

	// 3. Forward pass (all recorded into shared encoder)
	histInBuf := make([]*wgpu.Buffer, len(n.Layers))
	histPreBuf := make([]*wgpu.Buffer, len(n.Layers))
	exactDWBatches := make([]*wgpu.Buffer, len(n.Layers))
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
		} else if l.Type == LayerMultiHeadAttention {
			sl := l.SeqLength
			if sl <= 0 {
				sl = 1
			}
			outSize = sl * l.DModel
		}
		if outSize == 0 {
			outSize = l.InputHeight
		}
		if l.Type == LayerParallel {
			sum := 0
			for bi := range l.ParallelBranches {
				br := &l.ParallelBranches[bi]
				if br.Type == LayerSequential && len(br.SequentialLayers) > 0 {
					last := &br.SequentialLayers[len(br.SequentialLayers)-1]
					sum += gpuTrainLayerOutputSize(last)
				}
			}
			if sum > 0 {
				outSize = sum
			}
		}

		preBuf := ctx.GetActivationBuffer(fmt.Sprintf("pre_%d", i), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
		if preBuf == nil {
			ctx.FlushFrame()
			return 0, fmt.Errorf("failed to get preBuf for layer %d", i)
		}

		var fwdErr error
		layerTileSize := 0
		if mode == TrainingModeGPUSC {
			layerTileSize = l.GetGPUSCTileSize(l.DType)
		} else if mode == TrainingModeGPUMC {
			layerTileSize = l.GetGPUMCTileSize(l.DType)
		}
		if l.Type == LayerParallel {
			fwdErr = gpuParallelConcatForward(ctx, mode, batchSize, i, l, curBuf, preBuf)
		} else {
			scale := float32(1.0)
			var wBuf *wgpu.Buffer
			if l.WeightStore != nil {
				if l.WeightStore.Scale != 0 {
					scale = l.WeightStore.Scale
				}
				// Use the DType-native forward buffer (PTQ-simulated), falling back to FP32.
				wBuf = GetGPUWeightBuffer(l)
			}
			if l.Type == LayerCNN1 && isCNN1NativeGPUQuantDType(l.DType) {
				scale = cnn1PackedGPUScale(l)
			}
			if layerTileSize > 0 && l.Type == LayerCNN1 {
				kernelVol := l.InputChannels * l.KernelSize
				if isCNN1NativeGPUQuantDType(l.DType) {
					fwdErr = ctx.DispatchCNN1PackedTiled(l.DType, layerTileSize, kernelVol, batchSize,
						l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight,
						l.KernelSize, l.Stride, l.Padding,
						scale, curBuf, wBuf, preBuf)
				} else {
					fwdErr = ctx.DispatchCNN1Tiled(layerTileSize, kernelVol, batchSize,
						l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight,
						l.KernelSize, l.Stride, l.Padding,
						scale, curBuf, wBuf, preBuf)
				}
			} else if layerTileSize > 0 && l.Type == LayerCNN3 {
				kernelVol := l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize
				fwdErr = ctx.DispatchCNN3Tiled(layerTileSize, kernelVol, batchSize,
					l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth,
					l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth,
					l.KernelSize, l.KernelSize, l.KernelSize,
					l.Stride, l.Stride, l.Stride,
					l.Padding, l.Padding, l.Padding,
					scale, curBuf, wBuf, preBuf)
			} else {
				fwdErr = ctx.DispatchForwardLayer(l, batchSize, curBuf, preBuf)
			}
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

	// 4. GPU gradient + partial loss (no CPU readback needed to continue)
	totalOutput := len(targetData)
	var numWG int
	if config.LossType == "multi_head_softmax_ce" {
		numWG = (batchSize + 255) / 256
	} else {
		numWG = (totalOutput + 255) / 256
	}
	gradBuf := ctx.GetActivationBuffer("grad_out", uint64(totalOutput*4), wgpu.BufferUsageStorage)
	partialsBuf := ctx.GetActivationBuffer("loss_partials", uint64(numWG*4), wgpu.BufferUsageStorage)
	if gradBuf == nil || partialsBuf == nil {
		ctx.FlushFrame()
		return 0, fmt.Errorf("failed to allocate loss buffers")
	}

	var lossErr error
	switch config.LossType {
	case "multi_head_softmax_ce":
		heads := config.SoftmaxCEHeads
		if len(heads) != 3 {
			ctx.FlushFrame()
			return 0, fmt.Errorf("multi_head_softmax_ce requires exactly 3 heads (got %d)", len(heads))
		}
		row := heads[0] + heads[1] + heads[2]
		if row <= 0 || totalOutput != batchSize*row {
			ctx.FlushFrame()
			return 0, fmt.Errorf("multi_head_softmax_ce: batch*row=%d*%d vs totalOutput=%d", batchSize, row, totalOutput)
		}
		if maskBuf != nil {
			lossErr = ctx.DispatchMultiHeadSoftmaxCEGradPartialLossMasked(batchSize, row, heads[0], heads[1], heads[2], curBuf, targetBuf, gradBuf, partialsBuf, maskBuf)
		} else {
			lossErr = ctx.DispatchMultiHeadSoftmaxCEGradPartialLoss(batchSize, row, heads[0], heads[1], heads[2], curBuf, targetBuf, gradBuf, partialsBuf)
		}
	case "crossentropy":
		lossErr = ctx.DispatchCEGradPartialLoss(totalOutput, curBuf, targetBuf, gradBuf, partialsBuf)
	default:
		lossErr = ctx.DispatchMSEGradPartialLoss(totalOutput, curBuf, targetBuf, gradBuf, partialsBuf)
	}
	if lossErr != nil {
		ctx.FlushFrame()
		return 0, lossErr
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
		} else if l.Type == LayerMultiHeadAttention {
			sl := l.SeqLength
			if sl <= 0 {
				sl = 1
			}
			inSize = sl * l.DModel
		}
		if inSize == 0 {
			inSize = l.OutputHeight
		}
		if l.Type == LayerParallel && l.InputHeight > 0 {
			inSize = l.InputHeight
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
		} else if l.Type == LayerMultiHeadAttention {
			sl := l.SeqLength
			if sl <= 0 {
				sl = 1
			}
			outSize = sl * l.DModel
		}
		if outSize == 0 {
			outSize = l.InputHeight
		}
		if l.Type == LayerParallel {
			sum := 0
			for bi := range l.ParallelBranches {
				br := &l.ParallelBranches[bi]
				if br.Type == LayerSequential && len(br.SequentialLayers) > 0 {
					last := &br.SequentialLayers[len(br.SequentialLayers)-1]
					sum += gpuTrainLayerOutputSize(last)
				}
			}
			if sum > 0 {
				outSize = sum
			}
		}

		dxBuf := ctx.GetActivationBuffer(fmt.Sprintf("dx_%d", i), uint64(inSize*batchSize*4), wgpu.BufferUsageStorage)
		if dxBuf == nil {
			ctx.FlushFrame()
			return 0, fmt.Errorf("failed to get dxBuf for layer %d", i)
		}

		wSize := 1
		if l.WeightStore != nil {
			wSize = len(l.WeightStore.Master)
			if wSize <= 0 {
				wSize = 1
			}
		}
		dwBuf := ctx.GetActivationBuffer(fmt.Sprintf("dw_%d", i), uint64(wSize*4), wgpu.BufferUsageStorage)
		if dwBuf == nil {
			ctx.FlushFrame()
			return 0, fmt.Errorf("failed to get dwBuf for layer %d", i)
		}
		// Zero DW buffer before accumulation on GPU
		if err := ctx.DispatchFillZero(wSize, dwBuf); err != nil {
			ctx.FlushFrame()
			return 0, err
		}

		if l.Type == LayerParallel {
			var gradPreRoot *wgpu.Buffer
			if l.Activation != ActivationLinear {
				gradPreRoot = ctx.GetActivationBuffer(fmt.Sprintf("grad_pre_%d", i), uint64(outSize*batchSize*4), wgpu.BufferUsageStorage)
				if gradPreRoot == nil {
					ctx.FlushFrame()
					return 0, fmt.Errorf("failed to get gradPreBuf for parallel layer %d", i)
				}
				if err := ctx.DispatchActivationBackward(outSize*batchSize, l.Activation, curGradBuf, histPreBuf[i], gradPreRoot); err != nil {
					ctx.FlushFrame()
					return 0, err
				}
			} else {
				gradPreRoot = curGradBuf
			}
			if err := gpuParallelConcatBackward(ctx, mode, batchSize, i, l, gradPreRoot, dxBuf, config); err != nil {
				ctx.FlushFrame()
				return 0, err
			}
			curGradBuf = dxBuf
			continue
		}

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

		bwdTileSize := 0
		if mode == TrainingModeGPUSC {
			bwdTileSize = l.GetGPUSCTileSize(l.DType)
		} else if mode == TrainingModeGPUMC {
			bwdTileSize = l.GetGPUMCTileSize(l.DType)
		}
		var bwdWBuf *wgpu.Buffer
		if l.WeightStore != nil {
			if l.Type == LayerCNN1 && isCNN1NativeGPUQuantDType(l.DType) {
				bwdWBuf = GetGPUWeightBuffer(l)
			} else {
				// Backward pass uses the FP32 master buffer for full-precision gradient computation.
				// Fall back to the DType buffer if FP32 absent.
				if buf, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer); ok && buf != nil {
					bwdWBuf = buf
				} else {
					bwdWBuf = GetGPUWeightBuffer(l)
				}
			}
		}
		var bwdErr error
		if bwdTileSize > 0 && l.Type == LayerCNN1 {
			kernelVol := l.InputChannels * l.KernelSize
			if isCNN1NativeGPUQuantDType(l.DType) {
				if err := ctx.DispatchCNN1PackedBackwardDXTiled(l.DType, bwdTileSize, kernelVol, batchSize,
					l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight,
					l.KernelSize, l.Stride, l.Padding,
					l.Activation, cnn1PackedGPUScale(l), gradPreBuf, bwdWBuf, histPreBuf[i], dxBuf); err != nil {
					ctx.FlushFrame()
					return 0, err
				}
			} else {
				if err := ctx.DispatchCNN1TiledBackwardDX(bwdTileSize, kernelVol, batchSize,
					l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight,
					l.KernelSize, l.Stride, l.Padding,
					l.Activation, gradPreBuf, bwdWBuf, histPreBuf[i], dxBuf); err != nil {
					ctx.FlushFrame()
					return 0, err
				}
			}
			bwdErr = ctx.DispatchCNN1TiledBackwardDW(bwdTileSize, batchSize,
				l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight,
				l.KernelSize, l.Stride, l.Padding,
				l.Activation, gradPreBuf, histInBuf[i], histPreBuf[i], dwBuf)
		} else if bwdTileSize > 0 && l.Type == LayerCNN3 {
			kernelVol := l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize
			if err := ctx.DispatchCNN3TiledBackwardDX(bwdTileSize, kernelVol, batchSize,
				l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth,
				l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth,
				l.KernelSize, l.KernelSize, l.KernelSize,
				l.Stride, l.Stride, l.Stride,
				l.Padding, l.Padding, l.Padding,
				l.Activation, gradPreBuf, bwdWBuf, histPreBuf[i], dxBuf); err != nil {
				ctx.FlushFrame()
				return 0, err
			}
			bwdErr = ctx.DispatchCNN3TiledBackwardDW(bwdTileSize, batchSize,
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
			if l.Type == LayerCNN1 && isCNN1NativeGPUQuantDType(l.DType) {
				packedBuf, _ := l.WeightStore.GPUWeights[l.DType].(*wgpu.Buffer)
				masterBuf, _ := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
				if packedBuf != nil && masterBuf != nil {
					if err := ctx.DispatchCNN1PackedApplyGradients(
						l.DType,
						wSize,
						config.LearningRate,
						config.GradientClip,
						cnn1PackedGPUScale(l),
						packedBuf,
						dwBuf,
						masterBuf,
					); err != nil {
						ctx.FlushFrame()
						return 0, err
					}
				} else {
					exactDWBatches[i] = dwBuf
				}
				curGradBuf = dxBuf
				continue
			}
			wBuf, ok := l.WeightStore.GPUWeights[DTypeFloat32].(*wgpu.Buffer)
			if ok && wBuf != nil {
				if err := ctx.DispatchApplyGradients(wSize, config.LearningRate, config.GradientClip, wBuf, dwBuf); err != nil {
					ctx.FlushFrame()
					return 0, err
				}
				// Sync low-bit buffers if they are active
				if l.DType == DTypeInt8 {
					if native, ok := l.WeightStore.GPUWeights[DTypeInt8].(*wgpu.Buffer); ok && native != nil {
						ctx.DispatchQuantizeI8(wSize, l.WeightStore.Scale, wBuf, native)
					}
				} else if l.DType == DTypeInt4 {
					if native, ok := l.WeightStore.GPUWeights[DTypeInt4].(*wgpu.Buffer); ok && native != nil {
						ctx.DispatchQuantizeI4(wSize, l.WeightStore.Scale, wBuf, native)
					}
				} else if l.DType == DTypeFP4 {
					if native, ok := l.WeightStore.GPUWeights[DTypeFP4].(*wgpu.Buffer); ok && native != nil {
						ctx.DispatchQuantizeFP4(wSize, l.WeightStore.Scale, wBuf, native)
					}
				} else if l.DType == DTypeTernary {
					if native, ok := l.WeightStore.GPUWeights[DTypeTernary].(*wgpu.Buffer); ok && native != nil {
						ctx.DispatchQuantizeTernary(wSize, l.WeightStore.Scale, wBuf, native)
					}
				} else if l.DType == DTypeBinary {
					if native, ok := l.WeightStore.GPUWeights[DTypeBinary].(*wgpu.Buffer); ok && native != nil {
						ctx.DispatchQuantizeBinary(wSize, l.WeightStore.Scale, wBuf, native)
					}
				}
				// SwiGLU and MHA store their weights in SPLIT GPU buffers
				// Q/K/V/O for MHA) that are separate from the master buffer that ApplyGradients just
				// updated. Propagate the updated master sub-ranges back to the split buffers so the
				// NEXT epoch's forward pass reads the updated weights.
				if ctx.ActiveEncoder != nil {
					ctx.propagateSplitWeights(l, wBuf)
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
	if config.LossType == "multi_head_softmax_ce" {
		lossVal /= float64(batchSize)
	}

	exactUpdated := false
	for i, dwBuf := range exactDWBatches {
		if dwBuf == nil {
			continue
		}
		gradData, err := ctx.ReadBuffer(dwBuf)
		if err != nil {
			return 0, err
		}
		gradTensor := NewTensorFromSlice(gradData, len(gradData))
		ApplyRecursiveGradients(&n.Layers[i], gradTensor, config.LearningRate, config.GradientClip)
		exactUpdated = true
	}
	if exactUpdated {
		if err := n.SyncToGPU(); err != nil {
			return 0, fmt.Errorf("exact cnn1 sync-to-gpu failed: %w", err)
		}
		if err := ensureGPUFloat32Weights(n); err != nil {
			return 0, fmt.Errorf("exact cnn1 ensure float32 weights failed: %w", err)
		}
	}
	return lossVal, nil
}

func gpuTrainingNeedsCPUFallback(n *VolumetricNetwork) bool {
	for i := range n.Layers {
		l := &n.Layers[i]
		if l.IsDisabled {
			continue
		}
		switch l.Type {
		case LayerMultiHeadAttention, LayerSwiGLU:
			return true
		case LayerDense:
			// Dense Int4 training uses a special Q4 forward path with separate scale
			// buffers; the generic post-update GPU requantize path does not rebuild
			// that representation safely yet.
			if l.DType == DTypeInt4 {
				return true
			}
		case LayerRNN, LayerLSTM:
			// Recurrent layers use PTQ-simulated float32 buffers for low-bit GPU
			// forward passes, not packed INT buffers. The generic Int8/Int4 GPU
			// requantize kernels corrupt those buffers after an optimizer step.
			if l.DType == DTypeInt8 || l.DType == DTypeInt4 {
				return true
			}
		}
	}
	return false
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
	case "crossentropy":
		eps := 1e-10
		for i := range output.Data {
			p := float64(output.Data[i])
			y := float64(target.Data[i])
			sum -= y * math.Log(p+eps)
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
	case "crossentropy":
		// Gradient of CE wrt probabilities: -y / (p + eps)
		// Normalized by total elements to match engine's MSE scaling convention.
		eps := 1e-10
		scale := 1.0 / float64(len(output.Data))
		for i := range output.Data {
			p := float64(output.Data[i])
			y := float64(target.Data[i])
			grad.Data[i] = T(-(y / (p + eps)) * scale)
		}
	}
	return grad
}

// calculateLossMultiHeadSoftmaxCE is mean over batch of (sum of per-head softmax CE) for 3 contiguous heads.
func calculateLossMultiHeadSoftmaxCE[T Numeric](output, target *Tensor[T], heads []int) float64 {
	if len(heads) != 3 || len(output.Data) != len(target.Data) {
		return 0
	}
	h0, h1, h2 := heads[0], heads[1], heads[2]
	R := h0 + h1 + h2
	if R <= 0 || len(output.Data)%R != 0 {
		return 0
	}
	B := len(output.Data) / R
	od := ConvertTensor[T, float32](output).Data
	td := ConvertTensor[T, float32](target).Data
	var sum float64
	for b := 0; b < B; b++ {
		row := b * R
		sum += softmaxCEFloat(od, td, row, h0)
		sum += softmaxCEFloat(od, td, row+h0, h1)
		sum += softmaxCEFloat(od, td, row+h0+h1, h2)
	}
	return sum / float64(B)
}

func softmaxCEFloat(od, td []float32, base, C int) float64 {
	if C <= 0 {
		return 0
	}
	maxv := float64(od[base])
	for i := 1; i < C; i++ {
		v := float64(od[base+i])
		if v > maxv {
			maxv = v
		}
	}
	var sume float64
	for i := 0; i < C; i++ {
		sume += math.Exp(float64(od[base+i]) - maxv)
	}
	var ce float64
	for i := 0; i < C; i++ {
		p := math.Exp(float64(od[base+i])-maxv) / sume
		y := float64(td[base+i])
		ce -= y * math.Log(p+1e-7)
	}
	return ce
}

// computeLossGradientMultiHeadSoftmaxCE returns dL/dlogit with L = (1/B) * sum_b (CE_head0 + CE_head1 + CE_head2).
func computeLossGradientMultiHeadSoftmaxCE[T Numeric](output, target *Tensor[T], heads []int) *Tensor[T] {
	grad := NewTensor[T](output.Shape...)
	if len(heads) != 3 || len(output.Data) != len(target.Data) {
		return grad
	}
	h0, h1, h2 := heads[0], heads[1], heads[2]
	R := h0 + h1 + h2
	if R <= 0 || len(output.Data)%R != 0 {
		return grad
	}
	B := len(output.Data) / R
	od := ConvertTensor[T, float32](output).Data
	td := ConvertTensor[T, float32](target).Data
	invB := float32(1.0 / float64(B))
	for b := 0; b < B; b++ {
		row := b * R
		softmaxCEGradInto[T](od, td, grad, row, h0, invB)
		softmaxCEGradInto[T](od, td, grad, row+h0, h1, invB)
		softmaxCEGradInto[T](od, td, grad, row+h0+h1, h2, invB)
	}
	return grad
}

func multiHeadMaskActive[T Numeric](mask *Tensor[T], b, h, B int) bool {
	if mask == nil || len(mask.Data) < B*3 {
		return true
	}
	return float64(mask.Data[b*3+h]) > 0.5
}

// calculateLossMultiHeadSoftmaxCEMasked is like calculateLossMultiHeadSoftmaxCE but only sums CE for heads
// where mask[b*3+head] > 0.5. Inactive heads contribute 0 to the per-row loss.
func calculateLossMultiHeadSoftmaxCEMasked[T Numeric](output, target *Tensor[T], heads []int, mask *Tensor[T]) float64 {
	if len(heads) != 3 || len(output.Data) != len(target.Data) {
		return 0
	}
	h0, h1, h2 := heads[0], heads[1], heads[2]
	R := h0 + h1 + h2
	if R <= 0 || len(output.Data)%R != 0 {
		return 0
	}
	B := len(output.Data) / R
	od := ConvertTensor[T, float32](output).Data
	td := ConvertTensor[T, float32](target).Data
	var sum float64
	for b := 0; b < B; b++ {
		row := b * R
		var rowLoss float64
		if multiHeadMaskActive(mask, b, 0, B) {
			rowLoss += softmaxCEFloat(od, td, row, h0)
		}
		if multiHeadMaskActive(mask, b, 1, B) {
			rowLoss += softmaxCEFloat(od, td, row+h0, h1)
		}
		if multiHeadMaskActive(mask, b, 2, B) {
			rowLoss += softmaxCEFloat(od, td, row+h0+h1, h2)
		}
		sum += rowLoss
	}
	return sum / float64(B)
}

// computeLossGradientMultiHeadSoftmaxCEMasked applies softmax+CE gradients only on masked-in head slices.
func computeLossGradientMultiHeadSoftmaxCEMasked[T Numeric](output, target *Tensor[T], heads []int, mask *Tensor[T]) *Tensor[T] {
	grad := NewTensor[T](output.Shape...)
	if len(heads) != 3 || len(output.Data) != len(target.Data) {
		return grad
	}
	h0, h1, h2 := heads[0], heads[1], heads[2]
	R := h0 + h1 + h2
	if R <= 0 || len(output.Data)%R != 0 {
		return grad
	}
	B := len(output.Data) / R
	od := ConvertTensor[T, float32](output).Data
	td := ConvertTensor[T, float32](target).Data
	invB := float32(1.0 / float64(B))
	for b := 0; b < B; b++ {
		row := b * R
		if multiHeadMaskActive(mask, b, 0, B) {
			softmaxCEGradInto[T](od, td, grad, row, h0, invB)
		}
		if multiHeadMaskActive(mask, b, 1, B) {
			softmaxCEGradInto[T](od, td, grad, row+h0, h1, invB)
		}
		if multiHeadMaskActive(mask, b, 2, B) {
			softmaxCEGradInto[T](od, td, grad, row+h0+h1, h2, invB)
		}
	}
	return grad
}

func softmaxCEGradInto[T Numeric](od, td []float32, grad *Tensor[T], base, C int, invB float32) {
	if C <= 0 {
		return
	}
	maxv := od[base]
	for i := 1; i < C; i++ {
		if od[base+i] > maxv {
			maxv = od[base+i]
		}
	}
	var sume float32
	for i := 0; i < C; i++ {
		sume += float32(math.Exp(float64(od[base+i] - maxv)))
	}
	for i := 0; i < C; i++ {
		p := float32(math.Exp(float64(od[base+i]-maxv))) / sume
		y := td[base+i]
		grad.Data[base+i] = T((p - y) * invB)
	}
}

// ApplyRecursiveGradients traverses the layer hierarchy and updates weights in all nested WeightStores.
func ApplyRecursiveGradients(layer *VolumetricLayer, gradWeights *Tensor[float32], lr float32, clipVal float32) {
	if layer == nil || gradWeights == nil {
		return
	}

	// 1. Update local weights if they exist
	if layer.WeightStore != nil {
		usedNative := false
		if layer.Network != nil && layer.Network.UseExactDType {
			usedNative = layer.WeightStore.ApplyGradientsNative(layer.DType, gradWeights, lr, clipVal)
		}
		if !usedNative {
			layer.WeightStore.ApplyGradients(gradWeights, lr, clipVal)
		}

		// Re-quantize after gradient update when the active state still lives in Master.
		if !usedNative && layer.DType != DTypeFloat32 {
			layer.WeightStore.Morph(layer.DType)
		}
	}

	// 2. Recursively update Parallel branches
	if layer.Type == LayerParallel && len(layer.ParallelBranches) > 0 && len(gradWeights.Nested) > 0 {
		for i := range layer.ParallelBranches {
			if i < len(gradWeights.Nested) {
				ApplyRecursiveGradients(&layer.ParallelBranches[i], gradWeights.Nested[i], lr, clipVal)
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
				ApplyRecursiveGradients(&layer.SequentialLayers[i], gradWeights.Nested[i], lr, clipVal)
			}
		}
	}
}
