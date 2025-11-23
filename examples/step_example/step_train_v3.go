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
	fmt.Println("=== LOOM Stepping Neural Network v3: LSTM Middle Layer ===")
	fmt.Println("3-Layer Network: Dense -> LSTM -> Dense")
	fmt.Println()

	// 1. Define Network Architecture
	// 3 Layers: Input(4) -> Dense(8) -> LSTM(seq=1, hidden=12) -> Dense(3)
	// The LSTM layer processes sequences, adding temporal memory capability
	// Note: batchSize=1, seqLength=1 for stepping (one timestep at a time)

	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 4,
				"output_height": 8,
				"activation": "relu"
			},
			{
				"type": "lstm",
				"input_size": 8,
				"hidden_size": 12,
				"seq_length": 1,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 12,
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

	// 2. Create Training Data (3 Classes)
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

	// 3. Setup Continuous Training Loop
	totalSteps := 100000

	// Network Depth = 3 layers
	// Target delay for LSTM stepping
	targetDelay := 3

	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.015)
	minLearningRate := float32(0.001)
	decayRate := float32(0.99995)

	// Gradient clipping for LSTM stability
	gradientClipValue := float32(1.0)

	fmt.Printf("Training for %d steps (Max Speed)\n", totalSteps)
	fmt.Printf("Target Delay: %d steps (accounts for LSTM internal state)\n", targetDelay)
	fmt.Printf("LR Decay: %.4f per step (min %.4f)\n", decayRate, minLearningRate)
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

		// B. Set Input
		// For LSTM, we need to reshape input to [batch, seq, features]
		// Since seqLength=1, input is [1, 1, 4] but flattened to [4]
		state.SetInput(sample.input)

		// C. Step Forward
		net.StepForward(state)

		// D. Manage Target Queue
		targetQueue.Push(sample.target)

		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

			// F. Calculate Loss & Gradient
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

			// Apply gradient clipping to prevent LSTM gradient explosion
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

			// G. Backward Pass
			// LSTM backward pass uses BPTT (Backpropagation Through Time)
			// This is more complex than simple feedforward layers
			net.StepBackward(state, gradOutput)

			// H. Update Weights
			net.ApplyGradients(learningRate)

			// Decay Learning Rate
			learningRate *= decayRate
			if learningRate < minLearningRate {
				learningRate = minLearningRate
			}

			// I. Logging
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

	// LSTM settling time
	settlingSteps := 10

	for _, sample := range trainingData {
		state.SetInput(sample.input)
		// Settle - LSTM needs time to stabilize hidden and cell states
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

		fmt.Printf("%s %s: Pred %d (%.2f) Exp %d\n", mark, sample.label, maxIdx, maxVal, tMaxIdx)
	}

	fmt.Printf("Final Accuracy: %d/%d\n", correct, len(trainingData))

}
