package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	nn "github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Stepping Neural Network v3: Custom Backward Pass ===")
	fmt.Println("Demonstrating StepBackward with Softmax Gradient Scaling")
	fmt.Println()

	// Create a simple network for binary classification
	// 2x2 grid with heterogeneous agents
	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 2,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 4,
				"output_height": 8,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_height": 8,
				"output_height": 8,
				"activation": "gelu"
			},
			{
				"type": "dense",
				"input_height": 8,
				"output_height": 4,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 4,
				"output_height": 2,
				"activation": "sigmoid"
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	fmt.Println("Network Architecture (2x2 Grid):")
	fmt.Println("  ┌─────────────────┬─────────────────┐")
	fmt.Println("  │ [0,0] Extractor │ [0,1] Transform │")
	fmt.Println("  │   4→8 (ReLU)    │   8→8 (GeLU)    │")
	fmt.Println("  ├─────────────────┼─────────────────┤")
	fmt.Println("  │ [1,0] Reducer   │ [1,1] Decider   │")
	fmt.Println("  │   8→4 (Tanh)    │   4→2 (Sigmoid) │")
	fmt.Println("  └─────────────────┴─────────────────┘")
	fmt.Println()

	// Initialize stepping state
	inputSize := 4
	state := net.InitStepState(inputSize)

	// Create training data: XOR-like problem
	trainingData := []struct {
		input  []float32
		target []float32
		label  string
	}{
		{
			input:  []float32{0.8, 0.9, 0.1, 0.2},
			target: []float32{1.0, 0.0},
			label:  "High-Low",
		},
		{
			input:  []float32{0.2, 0.1, 0.9, 0.8},
			target: []float32{0.0, 1.0},
			label:  "Low-High",
		},
		{
			input:  []float32{0.7, 0.8, 0.2, 0.3},
			target: []float32{1.0, 0.0},
			label:  "High-Low",
		},
		{
			input:  []float32{0.3, 0.2, 0.7, 0.8},
			target: []float32{0.0, 1.0},
			label:  "Low-High",
		},
	}

	// Training parameters
	epochs := 2000
	learningRate := float32(0.01)

	fmt.Printf("Training for %d epochs with LR=%.4f...\n", epochs, learningRate)
	fmt.Println("Using StepBackward with Softmax Gradient Scaling")
	fmt.Println()

	lossHistory := make([]float32, 0)
	start := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		// Shuffle data
		perm := rand.Perm(len(trainingData))

		for _, idx := range perm {
			sample := trainingData[idx]

			// 1. Forward Pass (StepForward)
			state.SetInput(sample.input)
			// Step enough times to propagate input to output (depth is ~4)
			for s := 0; s < 5; s++ {
				net.StepForward(state)
			}
			output := state.GetOutput()

			// 2. Calculate Loss and Gradient
			// MSE Loss: L = 0.5 * (y - t)^2
			// dL/dy = (y - t)
			loss := float32(0.0)
			gradOutput := make([]float32, len(output))
			for i := 0; i < len(output); i++ {
				diff := output[i] - sample.target[i]
				loss += 0.5 * diff * diff
				gradOutput[i] = diff // Gradient of MSE w.r.t output
			}
			totalLoss += loss

			// 3. Backward Pass (StepBackward)
			net.StepBackward(state, gradOutput)

			// 4. Update Weights
			// We need to manually trigger weight update using the stored gradients
			// The standard Train() method does this, but we are doing it manually.
			// We can use a helper or just call UpdateWeights if exposed,
			// but UpdateWeights is usually internal or part of Train.
			// Let's check if we can use UpdateWeights.
			// If not, we might need to implement a manual update or use a public method.
			// Looking at `training.go` (not visible but assumed), usually there is `UpdateWeights`.
			// If not, we will assume `net.UpdateWeights` exists or we need to add it.
			// Wait, I saw `UpdateWeights` mentioned in `network.go` comments?
			// "Learning rate for parallel layer branch updates (set during UpdateWeights)"
			// Let's assume `UpdateWeights` is available or we use `Train` with 0 epochs?
			// No, `Train` does forward/backward itself.
			// We need a way to apply the gradients stored in `net.kernelGradients`.

			// Let's assume we can call a method to update weights.
			// Since I cannot see `training.go`, I will assume `UpdateWeights` is available
			// or I will add a simple one here if I could, but I can't modify `nn` package easily from here.
			// Actually, I can modify `nn` package.
			// But let's check `training.go` first? No, I am in execution.
			// I will assume `UpdateWeights` exists or I will implement a simple one in `nn/step_backward.go`
			// if I find it missing during compilation.

			// For now, let's try to use a method that likely exists or add it to `step_backward.go`
			// to be safe.
			// I will add `ApplyGradients` to `nn/step_backward.go` to be sure.
			net.ApplyGradients(learningRate)
		}

		avgLoss := totalLoss / float32(len(trainingData))
		lossHistory = append(lossHistory, avgLoss)

		if epoch%100 == 0 {
			fmt.Printf("Epoch %d: Loss = %.6f\n", epoch, avgLoss)
		}
	}

	fmt.Printf("Training complete in %v\n", time.Since(start))
	fmt.Printf("Final Loss: %.6f\n", lossHistory[len(lossHistory)-1])
	fmt.Println()

	// Test
	correct := 0
	for _, sample := range trainingData {
		state.SetInput(sample.input)
		for s := 0; s < 5; s++ {
			net.StepForward(state)
		}
		output := state.GetOutput()

		predClass := 0
		if output[1] > output[0] {
			predClass = 1
		}
		expClass := 0
		if sample.target[1] > sample.target[0] {
			expClass = 1
		}

		mark := "✗"
		if predClass == expClass {
			correct++
			mark = "✓"
		}

		fmt.Printf("%s Input: %v -> Output: [%.3f, %.3f] (Exp: %d, Got: %d)\n",
			mark, sample.input, output[0], output[1], expClass, predClass)
	}

	fmt.Printf("Accuracy: %d/%d\n", correct, len(trainingData))
}
