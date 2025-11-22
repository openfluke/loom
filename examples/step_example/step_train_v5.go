package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

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
	fmt.Println("=== LOOM Stepping Network: Dense -> CNN -> Dense ===")
	fmt.Println("Architecture: [Input Projection] -> [Feature Extraction] -> [Classification]")
	fmt.Println("Task: 2D Pattern Classification (Horizontal, Vertical, Diagonal, Cross)")
	fmt.Println()

	// FIXED JSON: Added output_height/width to Conv2D
	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 16,
				"output_height": 16,
				"activation": "tanh" 
			},
			{
				"type": "conv2d",
				"input_height": 4,
				"input_width": 4,
				"input_channels": 1,
				"filters": 8,
				"kernel_size": 3,
				"stride": 1,
				"padding": 1,
				"output_height": 4,
				"output_width": 4,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_height": 128, 
				"output_height": 4,
				"activation": "linear"
			}
		]
	}`
	// Note on Layer 2 Output Dims: (4 + 2*1 - 3)/1 + 1 = 4.
	// Note on Layer 3 Input: 4 (H) * 4 (W) * 8 (Filters) = 128.

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	// Initialize stepping state
	inputSize := 16 // 4x4 image flattened
	state := net.InitStepState(inputSize)

	// --- Data Generation Helpers ---
	createHorizontalLine := func() []float32 {
		img := make([]float32, 16)
		for i := 4; i < 12; i++ {
			img[i] = 1.0
		}
		return img
	}
	createVerticalLine := func() []float32 {
		img := make([]float32, 16)
		for row := 0; row < 4; row++ {
			img[row*4+1] = 1.0
			img[row*4+2] = 1.0
		}
		return img
	}
	createDiagonalLine := func() []float32 {
		img := make([]float32, 16)
		for i := 0; i < 4; i++ {
			img[i*4+i] = 1.0
		}
		return img
	}
	createCrossPattern := func() []float32 {
		img := make([]float32, 16)
		for i := 4; i < 12; i++ {
			img[i] = 1.0
		}
		for row := 0; row < 4; row++ {
			img[row*4+1] = 1.0
			img[row*4+2] = 1.0
		}
		return img
	}
	addNoise := func(img []float32, amount float32) []float32 {
		noisy := make([]float32, len(img))
		copy(noisy, img)
		for i := range noisy {
			noisy[i] += (rand.Float32() - 0.5) * amount
			if noisy[i] < 0 {
				noisy[i] = 0
			}
			if noisy[i] > 1 {
				noisy[i] = 1
			}
		}
		return noisy
	}

	trainingData := []struct {
		input  []float32
		target []float32
		label  string
	}{
		{createHorizontalLine(), []float32{1, 0, 0, 0}, "Horizontal"},
		{addNoise(createHorizontalLine(), 0.2), []float32{1, 0, 0, 0}, "Horizontal"},

		{createVerticalLine(), []float32{0, 1, 0, 0}, "Vertical"},
		{addNoise(createVerticalLine(), 0.2), []float32{0, 1, 0, 0}, "Vertical"},

		{createDiagonalLine(), []float32{0, 0, 1, 0}, "Diagonal"},
		{addNoise(createDiagonalLine(), 0.2), []float32{0, 0, 1, 0}, "Diagonal"},

		{createCrossPattern(), []float32{0, 0, 0, 1}, "Cross"},
		{addNoise(createCrossPattern(), 0.2), []float32{0, 0, 0, 1}, "Cross"},
	}

	// --- Training Loop ---
	totalSteps := 80000
	targetDelay := 3
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.01)
	minLearningRate := float32(0.0001)
	decayRate := float32(0.99995)
	gradientClipValue := float32(1.0)

	fmt.Printf("Training for %d steps\n", totalSteps)
	fmt.Printf("Architecture Delay: %d steps\n", targetDelay)
	fmt.Println("────────────────────────────────────────────────────────────")

	//startTime := time.Now()
	stepCount := 0
	currentSampleIdx := 0

	for stepCount < totalSteps {
		if stepCount%20 == 0 {
			currentSampleIdx = rand.Intn(len(trainingData))
		}
		sample := trainingData[currentSampleIdx]

		// A. Set Input
		state.SetInput(sample.input)

		// B. Step Forward
		net.StepForward(state)

		// C. Manage Delays
		targetQueue.Push(sample.target)

		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

			// D. Loss & Gradients
			gradOutput := make([]float32, len(output))
			loss := float32(0.0)

			maxVal := output[0]
			for _, v := range output {
				if v > maxVal {
					maxVal = v
				}
			}
			sumExp := float32(0.0)
			exps := make([]float32, len(output))
			for i, v := range output {
				exps[i] = float32(math.Exp(float64(v - maxVal)))
				sumExp += exps[i]
			}

			for i := range output {
				probs := exps[i] / sumExp
				if delayedTarget[i] > 0.5 {
					loss -= float32(math.Log(float64(probs + 1e-7)))
				}
				gradOutput[i] = probs - delayedTarget[i]
			}

			// Gradient Clipping
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

			// E. Backward Pass
			net.StepBackward(state, gradOutput)

			// F. Update Weights
			net.ApplyGradients(learningRate)

			// Decay
			learningRate *= decayRate
			if learningRate < minLearningRate {
				learningRate = minLearningRate
			}

			if stepCount%1000 == 0 {
				predIdx := 0
				for i := 1; i < len(output); i++ {
					if output[i] > output[predIdx] {
						predIdx = i
					}
				}
				targetIdx := 0
				for i := 1; i < len(delayedTarget); i++ {
					if delayedTarget[i] > 0.5 {
						targetIdx = i
					}
				}

				mark := "✗"
				if predIdx == targetIdx {
					mark = "✓"
				}

				fmt.Printf("Step %-6d [%s] Loss: %.4f | Pred: %d Exp: %d | LR: %.5f\n",
					stepCount, mark, loss, predIdx, targetIdx, learningRate)
			}
		}
		stepCount++
	}

	// --- Final Evaluation ---
	fmt.Println("\n=== Final Evaluation (with Settling) ===")
	correct := 0
	settlingSteps := 5

	for _, sample := range trainingData {
		state.SetInput(sample.input)
		for i := 0; i < settlingSteps; i++ {
			net.StepForward(state)
		}

		output := state.GetOutput()
		predIdx := 0
		for i := 1; i < len(output); i++ {
			if output[i] > output[predIdx] {
				predIdx = i
			}
		}

		targetIdx := 0
		for i := 1; i < len(sample.target); i++ {
			if sample.target[i] > 0.5 {
				targetIdx = i
			}
		}

		mark := "✗"
		if predIdx == targetIdx {
			correct++
			mark = "✓"
		}
		fmt.Printf("%s %-10s: Pred %d Exp %d\n", mark, sample.label, predIdx, targetIdx)
	}

	fmt.Printf("Accuracy: %d/%d\n", correct, len(trainingData))
}
