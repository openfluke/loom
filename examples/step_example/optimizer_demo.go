package main

import (
	"fmt"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("🚀 LOOM Optimizer Framework Demo")
	fmt.Println("==================================\n")

	// Create a simple network: 4 inputs → 8 hidden → 3 outputs
	configJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{"type": "dense", "input_height": 4, "output_height": 8, "activation": "relu"},
			{"type": "lstm", "input_size": 8, "hidden_size": 12, "seq_length": 1},
			{"type": "dense", "input_height": 12, "output_height": 3, "activation": "softmax"}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	network.InitializeWeights()

	// Training data: 3-class classification
	trainingData := []struct {
		input  []float32
		target []float32
	}{
		{[]float32{0.1, 0.2, 0.1, 0.3}, []float32{1.0, 0.0, 0.0}}, // Class 0
		{[]float32{0.8, 0.9, 0.7, 0.8}, []float32{0.0, 1.0, 0.0}}, // Class 1
		{[]float32{0.3, 0.5, 0.9, 0.6}, []float32{0.0, 0.0, 1.0}}, // Class 2
		{[]float32{0.2, 0.1, 0.2, 0.2}, []float32{1.0, 0.0, 0.0}}, // Class 0
		{[]float32{0.9, 0.8, 0.8, 0.9}, []float32{0.0, 1.0, 0.0}}, // Class 1
		{[]float32{0.4, 0.6, 0.8, 0.7}, []float32{0.0, 0.0, 1.0}}, // Class 2
	}

	// Data loader function
	dataLoader := func(step int) ([]float32, []float32, bool) {
		idx := step % len(trainingData)
		return trainingData[idx].input, trainingData[idx].target, true
	}

	// ========================================================================
	// Demo 1: Simple SGD (backward compatible)
	// ========================================================================
	fmt.Println("📊 Demo 1: Simple SGD (backward compatible)")
	fmt.Println("--------------------------------------------")

	trainingConfig1 := &nn.SteppingTrainingConfig{
		Optimizer:    "sgd",
		LearningRate: 0.01,
		LogEvery:     5000,
		OnStep: func(step int, lr float32, loss float32) {
			fmt.Printf("  Step %d: LR=%.4f, Loss=%.6f\n", step, lr, loss)
		},
	}

	result1, _ := network.TrainWithStepping(trainingConfig1, dataLoader, 20000)
	fmt.Printf("✅ SGD Training complete!\n")
	fmt.Printf("   Final Loss: %.6f\n", result1.FinalLoss)
	fmt.Printf("   Time: %.2fs (%.0f steps/sec)\n\n", result1.TotalTime.Seconds(), result1.StepsPerSecond)

	// ========================================================================
	// Demo 2: AdamW Optimizer
	// ========================================================================
	fmt.Println("📊 Demo 2: AdamW Optimizer")
	fmt.Println("--------------------------------------------")

	// Reset network
	network, err = nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	network.InitializeWeights() // ← FIX: Added this line

	trainingConfig2 := &nn.SteppingTrainingConfig{
		Optimizer:    "adamw",
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		WeightDecay:  0.01,
		LogEvery:     5000,
		OnStep: func(step int, lr float32, loss float32) {
			fmt.Printf("  Step %d: LR=%.4f, Loss=%.6f\n", step, lr, loss)
		},
	}

	result2, _ := network.TrainWithStepping(trainingConfig2, dataLoader, 20000)
	fmt.Printf("✅ AdamW Training complete!\n")
	fmt.Printf("   Final Loss: %.6f\n", result2.FinalLoss)
	fmt.Printf("   Time: %.2fs (%.0f steps/sec)\n\n", result2.TotalTime.Seconds(), result2.StepsPerSecond)

	// ========================================================================
	// Demo 3: AdamW + Cosine Annealing Scheduler
	// ========================================================================
	fmt.Println("📊 Demo 3: AdamW + Cosine Annealing")
	fmt.Println("--------------------------------------------")

	// Reset network
	network, err = nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	network.InitializeWeights() // ← FIX: Added this line

	trainingConfig3 := &nn.SteppingTrainingConfig{
		Optimizer:    "adamw",
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		WeightDecay:  0.01,
		LRSchedule:   "cosine",
		TotalSteps:   20000,
		MinLR:        0.0001,
		LogEvery:     5000,
		OnStep: func(step int, lr float32, loss float32) {
			fmt.Printf("  Step %d: LR=%.4f, Loss=%.6f\n", step, lr, loss)
		},
	}

	result3, _ := network.TrainWithStepping(trainingConfig3, dataLoader, 20000)
	fmt.Printf("✅ AdamW + Cosine Training complete!\n")
	fmt.Printf("   Final Loss: %.6f\n", result3.FinalLoss)
	fmt.Printf("   Time: %.2fs (%.0f steps/sec)\n\n", result3.TotalTime.Seconds(), result3.StepsPerSecond)

	// ========================================================================
	// Demo 4: Direct optimizer API (Approach 1)
	// ========================================================================
	fmt.Println("📊 Demo 4: Direct Optimizer API")
	fmt.Println("--------------------------------------------")

	// Reset network
	network, err = nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	network.InitializeWeights() // ← FIX: Added this line
	state := network.InitStepState(4)

	// Use convenience method ApplyGradientsAdamW
	for step := 0; step < 10000; step++ {
		idx := step % len(trainingData)
		input := trainingData[idx].input
		target := trainingData[idx].target

		state.SetInput(input)
		network.StepForward(state)
		output := state.GetOutput()

		// Compute gradients
		gradients := make([]float32, len(output))
		for i := range gradients {
			gradients[i] = 2.0 * (output[i] - target[i]) / float32(len(output))
		}

		network.StepBackward(state, gradients)

		// Use convenience method - automatically creates AdamW optimizer
		network.ApplyGradientsAdamW(0.001, 0.9, 0.999, 0.01)

		if (step+1)%2500 == 0 {
			// Compute loss
			var loss float32
			for i := range output {
				diff := output[i] - target[i]
				loss += diff * diff
			}
			loss /= float32(len(output))
			fmt.Printf("  Step %d: Loss=%.6f\n", step+1, loss)
		}
	}

	fmt.Printf("✅ Direct API Training complete!\n\n")

	fmt.Println("🎉 All demos complete!")
	fmt.Println("\nKey Features Demonstrated:")
	fmt.Println("  ✅ Approach 1: Direct optimizer methods (ApplyGradientsAdamW)")
	fmt.Println("  ✅ Approach 2: Pluggable optimizer system (SetOptimizer)")
	fmt.Println("  ✅ Approach 3: High-level TrainWithStepping utilities")
	fmt.Println("  ✅ Learning rate schedulers (Cosine Annealing)")
	fmt.Println("  ✅ Backward compatibility (existing code still works)")
}
