package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	nn "github.com/openfluke/loom/nn"
)

// TargetQueue handles the delay between input and output in the stepping network
type TargetQueue struct {
	targets [][]float32
	maxSize int
}

func NewTargetQueue(size int) *TargetQueue {
	return &TargetQueue{
		targets: make([][]float32, 0, size),
		maxSize: size,
	}
}

func (q *TargetQueue) Push(target []float32) {
	q.targets = append(q.targets, target)
}

func (q *TargetQueue) Pop() []float32 {
	if len(q.targets) == 0 {
		return nil
	}
	t := q.targets[0]
	q.targets = q.targets[1:]
	return t
}

func (q *TargetQueue) IsFull() bool {
	return len(q.targets) >= q.maxSize
}

func main() {
	fmt.Println("=== LOOM Stepping Neural Network v4: 3-Layer Dense Network ===")
	fmt.Println("3-Layer Network: Dense -> Dense -> Dense")
	fmt.Println("Task: Simple 3-Class Classification (same as v2/v3)")
	fmt.Println()

	// Architecture: Input(4) -> Dense(32) -> Dense(16) -> Dense(3)
	// Same as v2 for comparison

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 4,
				"output_height": 32,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_height": 32,
				"output_height": 16,
				"activation": "gelu"
			},
			{
				"type": "dense",
				"input_height": 16,
				"output_height": 3,
				"activation": "softmax"
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	// Initialize stepping state
	inputSize := 4
	state := net.InitStepState(inputSize)

	// Training Data (3 Classes) - same as v2 and v3
	trainingData := []struct {
		input  []float32
		target []float32
		label  string
	}{
		// Class 0: Low values
		{[]float32{0.1, 0.2, 0.1, 0.3}, []float32{1, 0, 0}, "Low"},
		{[]float32{0.2, 0.1, 0.3, 0.2}, []float32{1, 0, 0}, "Low"},

		// Class 1: High values
		{[]float32{0.8, 0.9, 0.8, 0.7}, []float32{0, 1, 0}, "High"},
		{[]float32{0.9, 0.8, 0.7, 0.9}, []float32{0, 1, 0}, "High"},

		// Class 2: Mixed
		{[]float32{0.1, 0.9, 0.1, 0.9}, []float32{0, 0, 1}, "Mix"},
		{[]float32{0.9, 0.1, 0.9, 0.1}, []float32{0, 0, 1}, "Mix"},
	}

	// Setup Training Loop
	totalSteps := 100000
	targetDelay := 3
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.02)
	minLearningRate := float32(0.001)
	decayRate := float32(0.99995)

	// MHA benefits from gradient clipping too
	gradientClipValue := float32(1.0)

	fmt.Printf("Training for %d steps\n", totalSteps)
	fmt.Printf("Target Delay: %d steps\n", targetDelay)
	fmt.Printf("LR: %.4f -> %.4f (decay: %.5f)\n", learningRate, minLearningRate, decayRate)
	fmt.Printf("Gradient Clipping: %.2f\n", gradientClipValue)
	fmt.Println()

	startTime := time.Now()
	stepCount := 0
	currentSampleIdx := 0

	fmt.Printf("%-6s %-10s %-25s %-10s\n", "Step", "Input", "Output (ArgMax)", "Loss")
	fmt.Println("──────────────────────────────────────────────────────────")

	for stepCount < totalSteps {
		// Rotate sample every 20 steps
		if stepCount%20 == 0 {
			currentSampleIdx = rand.Intn(len(trainingData))
		}
		sample := trainingData[currentSampleIdx]

		// Set Input
		state.SetInput(sample.input)

		// Step Forward
		net.StepForward(state)

		// Manage Target Queue
		targetQueue.Push(sample.target)

		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

			// Calculate Loss & Gradient
			loss := float32(0.0)
			gradOutput := make([]float32, len(output))

			for i := 0; i < len(output); i++ {
				p := output[i]
				if p < 1e-7 {
					p = 1e-7
				}
				if p > 1.0-1e-7 {
					p = 1.0 - 1e-7
				}

				if delayedTarget[i] > 0.5 {
					loss -= float32(math.Log(float64(p)))
				}

				gradOutput[i] = output[i] - delayedTarget[i]
			}

			// Gradient clipping
			gradNorm := float32(0.0)
			for _, g := range gradOutput {
				gradNorm += g * g
			}
			gradNorm = float32(math.Sqrt(float64(gradNorm)))

			if gradNorm > gradientClipValue {
				scale := gradientClipValue / gradNorm
				for i := range gradOutput {
					gradOutput[i] *= scale
				}
			}

			// Backward Pass
			net.StepBackward(state, gradOutput)

			// Update Weights
			net.ApplyGradients(learningRate)

			// Decay Learning Rate
			learningRate *= decayRate
			if learningRate < minLearningRate {
				learningRate = minLearningRate
			}

			// Logging
			if stepCount%500 == 0 {
				maxIdx := 0
				maxVal := output[0]
				for i := 1; i < len(output); i++ {
					if output[i] > maxVal {
						maxVal = output[i]
						maxIdx = i
					}
				}

				tMaxIdx := 0
				for i := 1; i < len(delayedTarget); i++ {
					if delayedTarget[i] > 0.5 {
						tMaxIdx = i
					}
				}

				mark := "✗"
				if maxIdx == tMaxIdx {
					mark = "✓"
				}

				fmt.Printf("%-6d %-10s Class %d (%.2f) [%s] Exp: %d  Loss: %.4f  LR: %.5f\n",
					stepCount, sample.label, maxIdx, maxVal, mark, tMaxIdx, loss, learningRate)
			}
		}

		stepCount++
	}

	totalTime := time.Since(startTime)
	fmt.Println()
	fmt.Println("=== Training Complete ===")
	fmt.Printf("Total Time: %v\n", totalTime)
	fmt.Printf("Speed: %.2f steps/sec\n", float64(totalSteps)/totalTime.Seconds())
	fmt.Println()

	// Final Evaluation
	fmt.Println("Evaluating on all samples (with settling time)...")
	correct := 0
	settlingSteps := 10

	for _, sample := range trainingData {
		state.SetInput(sample.input)
		// Settle
		for i := 0; i < settlingSteps; i++ {
			net.StepForward(state)
		}
		output := state.GetOutput()

		maxIdx := 0
		maxVal := output[0]
		for i := 1; i < len(output); i++ {
			if output[i] > maxVal {
				maxVal = output[i]
				maxIdx = i
			}
		}

		tMaxIdx := 0
		for i := 1; i < len(sample.target); i++ {
			if sample.target[i] > 0.5 {
				tMaxIdx = i
			}
		}

		mark := "✗"
		if maxIdx == tMaxIdx {
			correct++
			mark = "✓"
		}

		fmt.Printf("%s %s: Pred Class%d (%.2f) Exp Class%d\n", mark, sample.label, maxIdx, maxVal, tMaxIdx)
	}

	fmt.Printf("Final Accuracy: %d/%d\n", correct, len(trainingData))
}
