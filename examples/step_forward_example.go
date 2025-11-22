package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

// Rename main to avoid conflicts
func main() {
	// Initialize the network using the BuildNetworkFromJSON function
	networkConfig := `{
		"batch_size": 1,
		"grid_rows": 2,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 4,
				"output_size": 4,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_size": 4,
				"output_size": 4,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_size": 4,
				"output_size": 4,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_size": 4,
				"output_size": 4,
				"activation": "relu"
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(networkConfig)
	if err != nil {
		log.Fatalf("Failed to initialize network: %v", err)
	}

	network.InitializeWeights()

	// Example input and target output
	input := []float32{0.1, 0.2, 0.3, 0.4}
	target := []float32{0.9, 0.8, 0.7, 0.6}

	// Training parameters
	learningRate := float32(0.01)
	epochs := 10

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Epoch %d\n", epoch+1)

		// Step through the network layer by layer
		output, duration := network.StepThroughNetworkCPU(input)
		fmt.Printf("Step duration: %v\n", duration)

		// Calculate loss (mean squared error)
		loss := calculateLoss(output, target)
		fmt.Printf("Loss: %f\n", loss)

		// Backpropagation and weight update (simplified example)
		updateWeights(network, output, target, learningRate)

		// Timer-based loop for stepping through the network until it learns enough
		lossThreshold := float32(0.01) // Define the acceptable loss value
		maxCycles := 1000              // Safety mechanism to prevent infinite loops
		cycle := 0
		for cycle < maxCycles {
			output, duration := network.StepThroughNetworkCPU(input)
			loss := calculateLoss(output, target)
			fmt.Printf("Cycle %d duration: %v, loss: %f, output: %v\n", cycle+1, duration, loss, output)

			if loss < lossThreshold {
				fmt.Println("Loss threshold reached. Exiting timer loop.")
				break
			}
			cycle++
		}

		// Verify outputs by stepping multiple times
		for step := 0; step < 3; step++ {
			output, _ = network.StepThroughNetworkCPU(input)
			fmt.Printf("Step %d output: %v\n", step+1, output)
		}
	}

	// Replace simplified training logic with proper training process
	fmt.Println("Starting proper training process...")

	// Training configuration
	config := &nn.TrainingConfig{
		Epochs:          200,
		LearningRate:    0.01,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    1.0,
		LossType:        "mse",
		Verbose:         false,
	}

	// Generate training data
	fmt.Println("Generating training data...")
	numSamples := 100
	batches := make([]nn.TrainingBatch, numSamples)
	for i := 0; i < numSamples; i++ {
		var input []float32
		var target []float32

		if i%2 == 0 {
			// Pattern type 0: higher values in first half
			input = make([]float32, 4)
			for j := 0; j < 2; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			for j := 2; j < 4; j++ {
				input[j] = rand.Float32() * 0.3
			}
			target = []float32{1.0, 0.0}
		} else {
			// Pattern type 1: higher values in second half
			input = make([]float32, 4)
			for j := 0; j < 2; j++ {
				input[j] = rand.Float32() * 0.3
			}
			for j := 2; j < 4; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			target = []float32{0.0, 1.0}
		}

		batches[i] = nn.TrainingBatch{
			Input:  input,
			Target: target,
		}
	}
	fmt.Printf("Generated %d training samples\n", numSamples)

	// Train the network
	fmt.Println("Training the network...")
	result, err := network.Train(batches, config)
	if err != nil {
		panic(fmt.Sprintf("Training failed: %v", err))
	}

	fmt.Println("Training complete!")
	fmt.Printf("Initial Loss: %.6f\n", result.LossHistory[0])
	fmt.Printf("Final Loss: %.6f\n", result.FinalLoss)
	fmt.Printf("Improvement: %.6f (%.1f%%)\n",
		result.LossHistory[0]-result.FinalLoss,
		100*(result.LossHistory[0]-result.FinalLoss)/result.LossHistory[0])
}

func calculateLoss(output, target []float32) float32 {
	var loss float32
	for i := range output {
		diff := output[i] - target[i]
		loss += diff * diff
	}
	return loss / float32(len(output))
}

func updateWeights(network *nn.Network, output, target []float32, learningRate float32) {
	// Simplified weight update logic
	for i := range output {
		error := target[i] - output[i]
		for layerIdx := 0; layerIdx < network.TotalLayers(); layerIdx++ {
			// Example: Adjust weights based on error (placeholder logic)
			adjustment := error * learningRate * rand.Float32()
			activations := network.Activations()
			if activations[layerIdx] != nil {
				activations[layerIdx][0] += adjustment
			}
		}
	}
}
