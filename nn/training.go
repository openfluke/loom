package nn

import (
	"fmt"
	"math"
	"time"
)

// TrainingConfig holds configuration for training
type TrainingConfig struct {
	Epochs          int
	LearningRate    float32
	UseGPU          bool
	PrintEveryBatch int     // Print progress every N batches (0 = only print epoch summary)
	GradientClip    float32 // Max gradient norm (0 = no clipping)
	LossType        string  // "mse" or "cross_entropy"
	Verbose         bool
}

// TrainingBatch represents a single training batch
type TrainingBatch struct {
	Input  []float32
	Target []float32
}

// TrainingResult contains training statistics
type TrainingResult struct {
	FinalLoss     float64
	BestLoss      float64
	TotalTime     time.Duration
	AvgThroughput float64   // samples per second
	LossHistory   []float64 // loss per epoch
}

// DefaultTrainingConfig returns sensible defaults
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		Epochs:          10,
		LearningRate:    0.001,
		UseGPU:          true,
		PrintEveryBatch: 0, // Only print epoch summary
		GradientClip:    0, // No clipping
		LossType:        "mse",
		Verbose:         true,
	}
}

// Train trains the network on provided batches
func (n *Network) Train(batches []TrainingBatch, config *TrainingConfig) (*TrainingResult, error) {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	// Initialize GPU if requested
	if config.UseGPU && n.deviceInfo == nil {
		if config.Verbose {
			fmt.Println("Initializing GPU...")
		}
		if err := n.InitGPU(); err != nil {
			return nil, fmt.Errorf("GPU initialization failed: %w", err)
		}
		if config.Verbose {
			fmt.Println("✓ GPU initialized")
		}
	}

	result := &TrainingResult{
		BestLoss:    math.MaxFloat64,
		LossHistory: make([]float64, 0, config.Epochs),
	}

	startTime := time.Now()
	totalSamples := len(batches) * n.BatchSize

	if config.Verbose {
		fmt.Printf("\n=== Training Configuration ===\n")
		fmt.Printf("Epochs: %d\n", config.Epochs)
		fmt.Printf("Learning Rate: %.6f\n", config.LearningRate)
		fmt.Printf("Batches per Epoch: %d\n", len(batches))
		fmt.Printf("Samples per Epoch: %d\n", totalSamples)
		fmt.Printf("Backend: %s\n", map[bool]string{true: "GPU", false: "CPU"}[config.UseGPU])
		fmt.Printf("Loss Function: %s\n", config.LossType)
		if config.GradientClip > 0 {
			fmt.Printf("Gradient Clipping: %.2f\n", config.GradientClip)
		}
		fmt.Println()
	}

	// Training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochStart := time.Now()
		totalLoss := 0.0

		for batchIdx, batch := range batches {
			// Forward pass
			var output []float32
			var err error

			if config.UseGPU {
				output, _, err = n.ForwardGPU(batch.Input)
			} else {
				output, _ = n.ForwardCPU(batch.Input)
			}

			if err != nil {
				return nil, fmt.Errorf("forward pass failed at epoch %d, batch %d: %w", epoch, batchIdx, err)
			}

			// Calculate loss
			loss := n.calculateLoss(output, batch.Target, config.LossType)
			totalLoss += loss

			// Backward pass
			gradOutput := n.calculateGradient(output, batch.Target, config.LossType)

			if config.UseGPU {
				_, _, err = n.BackwardGPU(gradOutput)
			} else {
				_, _ = n.BackwardCPU(gradOutput)
			}

			if err != nil {
				return nil, fmt.Errorf("backward pass failed at epoch %d, batch %d: %w", epoch, batchIdx, err)
			}

			// Gradient clipping if enabled
			if config.GradientClip > 0 {
				n.clipGradients(config.GradientClip)
			}

			// Update weights
			n.UpdateWeights(config.LearningRate)

			// Print batch progress if configured
			if config.Verbose && config.PrintEveryBatch > 0 && batchIdx%config.PrintEveryBatch == 0 {
				fmt.Printf("\rEpoch %d/%d - Batch %d/%d - Loss: %.6f",
					epoch+1, config.Epochs, batchIdx+1, len(batches), loss)
			}
		}

		// Epoch summary
		avgLoss := totalLoss / float64(len(batches))
		epochTime := time.Since(epochStart)
		result.LossHistory = append(result.LossHistory, avgLoss)

		if avgLoss < result.BestLoss {
			result.BestLoss = avgLoss
		}

		if config.Verbose {
			samplesPerSec := float64(totalSamples) / epochTime.Seconds()
			fmt.Printf("\rEpoch %d/%d - Avg Loss: %.6f (Best: %.6f) - Time: %v - Throughput: %.0f samples/sec\n",
				epoch+1, config.Epochs, avgLoss, result.BestLoss, epochTime, samplesPerSec)

			// Show gradient statistics every 5 epochs
			if (epoch+1)%5 == 0 {
				n.printGradientStats()
			}
		}
	}

	result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	result.TotalTime = time.Since(startTime)
	result.AvgThroughput = float64(totalSamples*config.Epochs) / result.TotalTime.Seconds()

	if config.Verbose {
		fmt.Printf("\n✓ Training Complete!\n")
		fmt.Printf("Final Loss: %.6f\n", result.FinalLoss)
		fmt.Printf("Best Loss: %.6f\n", result.BestLoss)
		fmt.Printf("Total Time: %v\n", result.TotalTime)
		fmt.Printf("Average Throughput: %.0f samples/sec\n\n", result.AvgThroughput)
	}

	return result, nil
}

// calculateLoss computes the loss between output and target
func (n *Network) calculateLoss(output, target []float32, lossType string) float64 {
	switch lossType {
	case "mse":
		return calculateMSELoss(output, target)
	case "cross_entropy":
		return calculateCrossEntropyLoss(output, target)
	default:
		return calculateMSELoss(output, target)
	}
}

// calculateGradient computes the gradient of the loss
func (n *Network) calculateGradient(output, target []float32, lossType string) []float32 {
	switch lossType {
	case "mse":
		return calculateMSEGradient(output, target)
	case "cross_entropy":
		return calculateCrossEntropyGradient(output, target)
	default:
		return calculateMSEGradient(output, target)
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
	epsilon := 1e-7 // Prevent log(0)
	for i := 0; i < len(output) && i < len(target); i++ {
		if target[i] > 0 {
			// Clamp output to prevent numerical issues
			pred := math.Max(float64(output[i]), epsilon)
			pred = math.Min(pred, 1.0-epsilon)
			sum -= float64(target[i]) * math.Log(pred)
		}
	}
	return sum / float64(len(output))
}

// calculateCrossEntropyGradient computes gradient for cross-entropy loss
func calculateCrossEntropyGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	epsilon := float32(1e-7)
	scale := float32(1.0) / float32(len(output))

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
