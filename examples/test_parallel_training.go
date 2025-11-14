package main

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Testing Parallel Layer Training ===\n")

	// Create network: Dense (2->4) -> Parallel[Dense 4->4, Dense 4->4] (add) -> Dense (4->1)
	net := nn.NewNetwork(2, 1, 1, 3)
	net.BatchSize = 1

	// Layer 0: Dense (2 -> 4)
	dense1 := nn.InitDenseLayer(2, 4, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, dense1)

	// Layer 1: Parallel with 2 branches (add mode)
	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(4, 4, nn.ActivationLeakyReLU),
			nn.InitDenseLayer(4, 4, nn.ActivationLeakyReLU),
		},
	}
	net.SetLayer(0, 0, 1, parallel)

	// Layer 2: Dense (4 -> 1) - output
	output := nn.InitDenseLayer(4, 1, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 2, output)

	fmt.Println("Network initialized")
	fmt.Printf("  Layer 0: Dense 2 -> 4 (LeakyReLU)\n")
	fmt.Printf("  Layer 1: Parallel with %d branches (combine: %s)\n", len(net.Layers[1].ParallelBranches), net.Layers[1].CombineMode)
	for i := range net.Layers[1].ParallelBranches {
		fmt.Printf("    Branch %d: Dense 4 -> 4 (LeakyReLU)\n", i)
	}
	fmt.Printf("  Layer 2: Dense 4 -> 1 (Sigmoid)\n\n")

	// Save initial branch weights
	branch0InitialWeights := make([]float32, len(net.Layers[1].ParallelBranches[0].Kernel))
	branch1InitialWeights := make([]float32, len(net.Layers[1].ParallelBranches[1].Kernel))
	copy(branch0InitialWeights, net.Layers[1].ParallelBranches[0].Kernel)
	copy(branch1InitialWeights, net.Layers[1].ParallelBranches[1].Kernel)

	fmt.Printf("Initial branch 0 weights (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
		branch0InitialWeights[0], branch0InitialWeights[1], branch0InitialWeights[2],
		branch0InitialWeights[3], branch0InitialWeights[4])
	fmt.Printf("Initial branch 1 weights (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n",
		branch1InitialWeights[0], branch1InitialWeights[1], branch1InitialWeights[2],
		branch1InitialWeights[3], branch1InitialWeights[4])

	// Simple XOR-like training data
	inputs := [][]float32{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	targets := [][]float32{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	// Training loop
	learningRate := float32(0.1)
	epochs := 100

	fmt.Printf("Training for %d epochs with learning rate %.3f...\n", epochs, learningRate)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		for i := 0; i < len(inputs); i++ {
			// Forward pass
			forwardOutput, _ := net.ForwardCPU(inputs[i])

			// Compute loss (MSE)
			loss := float32(0.0)
			for j := 0; j < len(forwardOutput); j++ {
				diff := forwardOutput[j] - targets[i][j]
				loss += diff * diff
			}
			totalLoss += loss

			// Compute gradient
			gradOutput := make([]float32, len(forwardOutput))
			for j := 0; j < len(forwardOutput); j++ {
				gradOutput[j] = 2.0 * (forwardOutput[j] - targets[i][j])
			}

			// Backward pass
			net.BackwardCPU(gradOutput)

			// Update weights
			net.UpdateWeights(learningRate)
		}

		if epoch%20 == 0 || epoch == epochs-1 {
			avgLoss := totalLoss / float32(len(inputs))
			fmt.Printf("  Epoch %3d: Loss = %.6f\n", epoch, avgLoss)
		}
	}

	fmt.Println("\nTraining complete!\n")

	// Check if branch weights changed
	branch0FinalWeights := net.Layers[1].ParallelBranches[0].Kernel
	branch1FinalWeights := net.Layers[1].ParallelBranches[1].Kernel

	fmt.Printf("Final branch 0 weights (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
		branch0FinalWeights[0], branch0FinalWeights[1], branch0FinalWeights[2],
		branch0FinalWeights[3], branch0FinalWeights[4])
	fmt.Printf("Final branch 1 weights (first 5): [%.4f, %.4f, %.4f, %.4f, %.4f]\n\n",
		branch1FinalWeights[0], branch1FinalWeights[1], branch1FinalWeights[2],
		branch1FinalWeights[3], branch1FinalWeights[4])

	// Calculate weight change magnitude
	branch0Change := float32(0.0)
	for i := 0; i < len(branch0InitialWeights); i++ {
		diff := branch0FinalWeights[i] - branch0InitialWeights[i]
		branch0Change += float32(math.Abs(float64(diff)))
	}

	branch1Change := float32(0.0)
	for i := 0; i < len(branch1InitialWeights); i++ {
		diff := branch1FinalWeights[i] - branch1InitialWeights[i]
		branch1Change += float32(math.Abs(float64(diff)))
	}

	fmt.Printf("Branch 0 total weight change: %.6f\n", branch0Change)
	fmt.Printf("Branch 1 total weight change: %.6f\n\n", branch1Change)

	if branch0Change > 0.01 && branch1Change > 0.01 {
		fmt.Println("✅ SUCCESS: Both parallel branches trained! Weights changed significantly.")
	} else {
		fmt.Println("❌ FAILURE: Parallel branches did not train. Weights are frozen.")
		if branch0Change <= 0.01 {
			fmt.Printf("   Branch 0 change (%.6f) is too small\n", branch0Change)
		}
		if branch1Change <= 0.01 {
			fmt.Printf("   Branch 1 change (%.6f) is too small\n", branch1Change)
		}
	}

	// Test predictions
	fmt.Println("\nFinal predictions:")
	for i := 0; i < len(inputs); i++ {
		forwardOutput, _ := net.ForwardCPU(inputs[i])
		fmt.Printf("  Input: [%.1f, %.1f] -> Output: %.4f (Target: %.1f)\n",
			inputs[i][0], inputs[i][1], forwardOutput[0], targets[i][0])
	}
}
