package nn

import (
	"fmt"
	"math"
	"time"
)

// TrainingConfig holds configuration for training
type TrainingConfig struct {
	Epochs            int
	LearningRate      float32
	UseGPU            bool
	PrintEveryBatch   int     // Print progress every N batches (0 = only print epoch summary)
	GradientClip      float32 // Max gradient norm (0 = no clipping)
	LossType          string  // "mse" or "cross_entropy"
	Verbose           bool
	EvaluateEveryN    int         // Evaluate on validation set every N epochs (0 = no evaluation)
	ValidationInputs  [][]float32 // Optional: validation inputs for evaluation
	ValidationTargets []float64   // Optional: validation expected outputs
}

// TrainingBatch represents a single training batch
type TrainingBatch struct {
	Input  []float32
	Target []float32 // Flat target vector (one-hot or direct values)
}

// TrainingResult contains training statistics
type TrainingResult struct {
	FinalLoss     float64
	BestLoss      float64
	TotalTime     time.Duration
	AvgThroughput float64           // samples per second
	LossHistory   []float64         // loss per epoch
	EvalMetrics   *DeviationMetrics // Optional: evaluation metrics if validation set provided
}

// DefaultTrainingConfig returns sensible defaults
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		Epochs:          10,
		LearningRate:    0.05,
		UseGPU:          true,
		PrintEveryBatch: 0, // Only print epoch summary
		GradientClip:    0, // No clipping
		LossType:        "mse",
		Verbose:         true,
	}
}

// Train trains the network on provided batches using the Verified Logic
func (n *Network) Train(batches []TrainingBatch, config *TrainingConfig) (*TrainingResult, error) {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	// Initialize GPU if requested
	if config.UseGPU {
		n.GPU = true
		if !n.gpuMounted {
			if config.Verbose {
				fmt.Println("Initializing GPU (WeightsToGPU)...")
			}
			if err := n.WeightsToGPU(); err != nil {
				return nil, fmt.Errorf("WeightsToGPU failed: %w", err)
			}
			if config.Verbose {
				fmt.Println("✓ GPU initialized")
			}
		}
	} else {
		n.GPU = false
	}

	result := &TrainingResult{
		BestLoss:    math.MaxFloat64,
		LossHistory: make([]float64, 0, config.Epochs),
	}

	startTime := time.Now()
	numBatches := len(batches)

	name := "CPU"
	if n.GPU {
		name = "GPU"
	}

	if config.Verbose {
		fmt.Printf("\n=== Training Configuration ===\n")
		fmt.Printf("Epochs: %d\n", config.Epochs)
		fmt.Printf("Learning Rate: %.6f\n", config.LearningRate)
		fmt.Printf("Batches per Epoch: %d\n", numBatches)
		fmt.Printf("Backend: %s\n", name)
		fmt.Printf("Loss Function: %s\n", config.LossType)
		fmt.Println()
	}

	// Training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		totalLoss := float32(0.0)
		samplesProcessed := 0

		for b, batch := range batches {
			// Determine current batch size based on input length and input dim
			// n.InputSize might be reliable, or we calculate from expected BatchSize
			currentBatchSize := n.BatchSize
			if n.InputSize > 0 && len(batch.Input) > 0 {
				currentBatchSize = len(batch.Input) / n.InputSize
			}
			// Fallback: trust the helper's chunking if n.BatchSize is set
			if currentBatchSize <= 0 {
				currentBatchSize = 1
			}

			// 1. Forward Pass
			output, _ := n.Forward(batch.Input)

			// 2. Compute Gradients & Loss
			dOutput := make([]float32, len(output))

			// Infer output size per sample
			outputSize := len(output)
			if currentBatchSize > 0 {
				outputSize = len(output) / currentBatchSize
			}

			// Loop over samples in batch
			for i := 0; i < currentBatchSize; i++ {
				// Output corresponds to batch index i
				outStart := i * outputSize
				if outStart+outputSize > len(output) {
					break
				}
				sampleOut := output[outStart : outStart+outputSize]

				// Target for this sample
				tgtStart := i * outputSize

				if config.LossType == "cross_entropy" {
					// Cross Entropy Loss
					for j := 0; j < outputSize; j++ {
						if tgtStart+j >= len(batch.Target) {
							break
						}
						t := batch.Target[tgtStart+j]
						if t > 0 {
							val := sampleOut[j]
							if val < 1e-7 {
								val = 1e-7
							}
							totalLoss += -float32(math.Log(float64(val)))
						}

						// Gradient: (O - T) / N
						dOutput[outStart+j] = (sampleOut[j] - t) / float32(currentBatchSize)
					}

				} else {
					// MSE (Default)
					for j := 0; j < outputSize; j++ {
						if tgtStart+j >= len(batch.Target) {
							break
						}
						t := batch.Target[tgtStart+j]
						diff := sampleOut[j] - t
						totalLoss += diff * diff // Sum of squares

						// Gradient: (O - T) / N
						dOutput[outStart+j] = diff / float32(currentBatchSize)
					}
				}
			}

			// 3. Backward Pass
			n.Backward(dOutput)

			// 4. Update Weights
			n.ApplyGradients(config.LearningRate)

			samplesProcessed += currentBatchSize

			// Optional: Print batch progress
			if config.Verbose && config.PrintEveryBatch > 0 && b%config.PrintEveryBatch == 0 {
				fmt.Printf("\r  [%s] Epoch %d/%d - Batch %d/%d", name, epoch+1, config.Epochs, b+1, numBatches)
			}
		}

		// epoch summary
		avgLoss := float32(0)
		if samplesProcessed > 0 {
			avgLoss = totalLoss / float32(samplesProcessed)
		}
		result.LossHistory = append(result.LossHistory, float64(avgLoss))

		if float64(avgLoss) < result.BestLoss {
			result.BestLoss = float64(avgLoss)
		}

		if config.Verbose {
			fmt.Printf("  [%s] Epoch %d/%d - Loss: %.4f\n", name, epoch+1, config.Epochs, avgLoss)
		}
	}

	result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	result.TotalTime = time.Since(startTime)

	return result, nil
}

// TrainStandard is a helper for training with float32 inputs and targets (Regression, etc.)
// It automatically handles batch creation and flattening.
func (n *Network) TrainStandard(inputs, targets [][]float32, config *TrainingConfig) (*TrainingResult, error) {
	if len(inputs) != len(targets) {
		return nil, fmt.Errorf("input count %d does not match target count %d", len(inputs), len(targets))
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no training data provided")
	}

	batchSize := n.BatchSize
	if batchSize <= 0 {
		batchSize = 1
		n.BatchSize = 1
	}

	numSamples := len(inputs)
	numBatches := (numSamples + batchSize - 1) / batchSize
	batches := make([]TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		start := b * batchSize
		end := start + batchSize
		if end > numSamples {
			end = numSamples
		}

		currentSize := end - start
		inputDim := len(inputs[0])
		targetDim := len(targets[0])

		flatInput := make([]float32, currentSize*inputDim)
		flatTarget := make([]float32, currentSize*targetDim)

		for i := 0; i < currentSize; i++ {
			copy(flatInput[i*inputDim:], inputs[start+i])
			copy(flatTarget[i*targetDim:], targets[start+i])
		}

		batches[b] = TrainingBatch{
			Input:  flatInput,
			Target: flatTarget,
		}
	}

	return n.Train(batches, config)
}

// TrainLabels is a helper for training with float32 inputs and integer class labels (Classification)
// It automatically performs one-hot encoding of targets.
func (n *Network) TrainLabels(inputs [][]float32, labels []int, config *TrainingConfig) (*TrainingResult, error) {
	if len(inputs) != len(labels) {
		return nil, fmt.Errorf("input count %d does not match label count %d", len(inputs), len(labels))
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no training data provided")
	}

	batchSize := n.BatchSize
	if batchSize <= 0 {
		batchSize = 1
		n.BatchSize = 1
	}

	// Determine output size for one-hot encoding
	outputSize := 0
	// 1. Try to find last layer output size
	if len(n.Layers) > 0 {
		last := n.Layers[len(n.Layers)-1]
		outputSize = last.OutputHeight * last.OutputWidth
		if last.Filters > 0 {
			outputSize *= last.Filters
		}
	}
	// 2. Scan labels for max value if still 0
	if outputSize == 0 {
		maxLabel := 0
		for _, l := range labels {
			if l > maxLabel {
				maxLabel = l
			}
		}
		outputSize = maxLabel + 1
	}

	numSamples := len(inputs)
	numBatches := (numSamples + batchSize - 1) / batchSize
	batches := make([]TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		start := b * batchSize
		end := start + batchSize
		if end > numSamples {
			end = numSamples
		}

		currentSize := end - start
		inputDim := len(inputs[0])

		flatInput := make([]float32, currentSize*inputDim)
		flatTarget := make([]float32, currentSize*outputSize)

		for i := 0; i < currentSize; i++ {
			// Input
			copy(flatInput[i*inputDim:], inputs[start+i])

			// Target One-Hot
			label := labels[start+i]
			if label >= 0 && label < outputSize {
				flatTarget[i*outputSize+label] = 1.0
			}
		}

		batches[b] = TrainingBatch{
			Input:  flatInput,
			Target: flatTarget,
		}
	}

	return n.Train(batches, config)
}

// =============================================================================
// Helper Functions
// =============================================================================

// clipGradients clips gradients by global norm
func (n *Network) clipGradients(maxNorm float32) {
	kernelGrads := n.KernelGradients()
	biasGrads := n.BiasGradients()

	// Calculate global gradient norm
	totalNorm := float32(0.0)
	for _, grads := range kernelGrads {
		if grads != nil {
			for _, g := range grads {
				totalNorm += g * g
			}
		}
	}
	for _, grads := range biasGrads {
		if grads != nil {
			for _, g := range grads {
				totalNorm += g * g
			}
		}
	}
	totalNorm = float32(math.Sqrt(float64(totalNorm)))

	// Clip if necessary
	if totalNorm > maxNorm {
		scale := maxNorm / totalNorm

		// Scale kernel gradients
		for i := range kernelGrads {
			if kernelGrads[i] != nil {
				for j := range kernelGrads[i] {
					kernelGrads[i][j] *= scale
				}
			}
		}

		// Scale bias gradients
		for i := range biasGrads {
			if biasGrads[i] != nil {
				for j := range biasGrads[i] {
					biasGrads[i][j] *= scale
				}
			}
		}
	}
}

// printGradientStats prints gradient statistics for debugging
func (n *Network) printGradientStats() {
	kernelGrads := n.KernelGradients()
	biasGrads := n.BiasGradients()

	fmt.Printf("  Gradient Statistics:\n")
	fmt.Printf("    Total layers: %d, Kernel grads: %d, Bias grads: %d\n", n.TotalLayers(), len(kernelGrads), len(biasGrads))

	for layerIdx := 0; layerIdx < len(kernelGrads); layerIdx++ {
		if kernelGrads[layerIdx] != nil && len(kernelGrads[layerIdx]) > 0 {
			// Calculate gradient norm
			norm := float32(0)
			mean := float32(0)
			for _, g := range kernelGrads[layerIdx] {
				norm += g * g
				mean += g
			}
			norm = float32(math.Sqrt(float64(norm)))
			mean /= float32(len(kernelGrads[layerIdx]))

			fmt.Printf("    Layer %d: norm=%.6f, mean=%.6f, size=%d\n",
				layerIdx, norm, mean, len(kernelGrads[layerIdx]))
		}
	}
}

// calculateMSELoss computes Mean Squared Error loss
func calculateMSELoss(output, target []float32) float64 {
	sum := 0.0
	for i := 0; i < len(output) && i < len(target); i++ {
		diff := float64(output[i] - target[i])
		sum += diff * diff
	}
	return sum / float64(len(output))
}

// calculateMSEGradient computes gradient for MSE loss
func calculateMSEGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	scale := float32(2.0) / float32(len(output))
	for i := 0; i < len(output) && i < len(target); i++ {
		grad[i] = (output[i] - target[i]) * scale
	}
	return grad
}

// calculateCrossEntropyLoss computes cross-entropy loss
// Assumes output and target are probability distributions
func calculateCrossEntropyLoss(output, target []float32) float64 {
	sum := 0.0
	totalProb := 0.0
	epsilon := 1e-7 // Prevent log(0)
	for i := 0; i < len(output) && i < len(target); i++ {
		if target[i] > 0 {
			// Clamp output to prevent numerical issues
			pred := math.Max(float64(output[i]), epsilon)
			pred = math.Min(pred, 1.0-epsilon)
			sum -= float64(target[i]) * math.Log(pred)
			totalProb += float64(target[i])
		}
	}
	if totalProb == 0 {
		return 0
	}
	return sum / totalProb
}

// calculateCrossEntropyGradient computes gradient for cross-entropy loss
func calculateCrossEntropyGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	epsilon := float32(1e-7)

	// Calculate total probability mass for normalization (effective batch size)
	totalProb := float32(0)
	for _, t := range target {
		totalProb += t
	}
	if totalProb == 0 {
		return grad // No gradient if no target
	}
	scale := 1.0 / totalProb

	for i := 0; i < len(output) && i < len(target); i++ {
		if target[i] > 0 {
			// Prevent division by zero
			pred := output[i]
			if pred < epsilon {
				pred = epsilon
			}
			grad[i] = -(target[i] / pred) * scale
		}
	}
	return grad
}
