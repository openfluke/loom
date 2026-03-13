package poly

import (
	"fmt"
	"time"
)

// TrainingConfig holds configuration for training in the Volumetric Grid.
type TrainingConfig struct {
	Epochs       int
	LearningRate float32
	LossType     string  // "mse" or "cross_entropy"
	GradientClip float32 // Max gradient norm (0 = no clipping)
	Verbose      bool
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
// It automatically handles forward history capture, backward dispatch, and weight mutation.
func Train[T Numeric](n *VolumetricNetwork, batches []TrainingBatch[T], config *TrainingConfig) (*TrainingResult, error) {
	if config == nil {
		config = DefaultTrainingConfig()
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
			// 1. Forward Pass with History Capture
			histIn := make([]*Tensor[T], len(n.Layers))
			histPre := make([]*Tensor[T], len(n.Layers))
			curr := batch.Input

			for idx := range n.Layers {
				l := &n.Layers[idx]
				if l.IsDisabled {
					continue
				}
				histIn[idx] = curr
				// Forward pass one layer
				pre, post := DispatchLayer(l, curr, nil) // Updated line
				histPre[idx] = pre
				curr = post
			}

			// 2. Compute Loss Gradient
			gradOut := ComputeLossGradient(curr, batch.Target, config.LossType)
			lossVal := CalculateLoss(curr, batch.Target, config.LossType)
			epochLoss += lossVal
			
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
			
			// Format duration with better resolution
			durationStr := epochDuration.String()
			if epochDuration == 0 {
				durationStr = "< 1µs"
			} else if epochDuration < time.Millisecond {
				durationStr = fmt.Sprintf("%v", epochDuration)
			}

			// STABLE THROUGHPUT: Total samples processed so far / Total time elapsed
			totalSamples := int64(epoch+1) * int64(numBatches)
			samplesPerSec := 0.0
			if elapsed > 0 {
				samplesPerSec = float64(totalSamples) / elapsed.Seconds()
			}

			fmt.Printf("Epoch %d/%d - Loss: %.6f | Time: %s | Samples/s: %.2f | ETA: %v\n", 
				epoch+1, config.Epochs, avgLoss, durationStr, samplesPerSec, eta.Round(time.Second))
		}
	}

	result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	result.TotalTime = time.Since(totalStart)
	return result, nil
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
