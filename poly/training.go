package poly

import (
	"fmt"
	"time"

	"github.com/openfluke/webgpu/wgpu"
)

// TrainingConfig holds configuration for training in the Volumetric Grid.
type TrainingConfig struct {
	Epochs       int
	LearningRate float32
	LossType     string  // "mse" or "cross_entropy"
	GradientClip float32 // Max gradient norm (0 = no clipping)
	Verbose      bool
	UseGPU       bool
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

// Train executes the training loop on a VolumetricNetwork.
func Train[T Numeric](n *VolumetricNetwork, batches []TrainingBatch[T], config *TrainingConfig) (*TrainingResult, error) {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	if config.UseGPU && n.GPUContext == nil {
		if err := n.InitWGPU(); err != nil {
			return nil, fmt.Errorf("failed to initialize GPU: %w", err)
		}
	}

	if config.UseGPU {
		if err := n.SyncToGPU(); err != nil {
			return nil, fmt.Errorf("failed to sync weights to GPU: %w", err)
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
			if config.UseGPU {
				loss, err := trainBatchWGPU(n, batch, config)
				if err != nil {
					return nil, err
				}
				epochLoss += loss
			} else {
				// CPU implementation (existing)
				loss := trainBatchCPU(n, batch, config)
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
				samplesPerSec = float64(int64(epoch+1) * int64(numBatches)) / elapsed.Seconds()
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

func trainBatchWGPU[T Numeric](n *VolumetricNetwork, batch TrainingBatch[T], config *TrainingConfig) (float64, error) {
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

		if err := ctx.DispatchForwardLayer(l, batchSize, curBuf, preBuf); err != nil {
			ctx.FlushFrame()
			return 0, err
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

		if err := ctx.DispatchBackwardLayer(l, batchSize, gradPreBuf, histInBuf[i], histPreBuf[i], dxBuf, dwBuf); err != nil {
			ctx.FlushFrame()
			return 0, err
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
