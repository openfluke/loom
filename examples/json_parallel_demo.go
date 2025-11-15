package main

import (
	"fmt"
	"log"
	"math"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Parallel Layer Demo ===")
	fmt.Println("This demonstrates using multiple layer types in parallel within a single layer position.")
	fmt.Println()

	// Example 1: Parallel Dense + LSTM processing with concatenation
	example1JSON := `{
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 4,
				"output_size": 8,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 8,
						"output_size": 6,
						"activation": "relu"
					},
					{
						"type": "lstm",
						"input_size": 8,
						"hidden_size": 6,
						"seq_length": 1
					}
				]
			},
			{
				"type": "dense",
				"input_size": 12,
				"output_size": 2,
				"activation": "linear"
			}
		]
	}`

	fmt.Println("Example 1: Parallel [Dense + LSTM] with concat combine mode")
	fmt.Println("Architecture: Input(4) → Dense(8) → Parallel[Dense(6) + LSTM(6)] → Dense(2)")
	fmt.Println()

	net1, err := nn.BuildNetworkFromJSON(example1JSON)
	if err != nil {
		log.Fatalf("Failed to build network 1: %v", err)
	}

	// Initialize weights
	net1.InitializeWeights()

	// Create sample input
	batchSize := 2
	input1 := make([]float32, batchSize*4)
	for i := range input1 {
		input1[i] = float32(i % 3)
	}

	// Forward pass
	output1, _ := net1.ForwardCPU(input1)

	fmt.Printf("Input shape: [%d, %d]\n", batchSize, 4)
	fmt.Printf("Output shape: [%d, %d]\n", batchSize, 2)
	fmt.Printf("Output: %v\n", output1)
	fmt.Println()

	// Example 2: Parallel with add combine mode (same output sizes required)
	example2JSON := `{
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "parallel",
				"combine_mode": "add",
				"branches": [
					{
						"type": "dense",
						"input_size": 10,
						"output_size": 10,
						"activation": "relu"
					},
					{
						"type": "dense",
						"input_size": 10,
						"output_size": 10,
						"activation": "tanh"
					},
					{
						"type": "dense",
						"input_size": 10,
						"output_size": 10,
						"activation": "sigmoid"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 10,
				"output_size": 3,
				"activation": "linear"
			}
		]
	}`

	fmt.Println("Example 2: Parallel [Dense(relu) + Dense(tanh) + Dense(sigmoid)] with add combine mode")
	fmt.Println("Architecture: Parallel[Dense + Dense + Dense] → Dense(3)")
	fmt.Println()

	net2, err := nn.BuildNetworkFromJSON(example2JSON)
	if err != nil {
		log.Fatalf("Failed to build network 2: %v", err)
	}

	net2.InitializeWeights()

	input2 := make([]float32, batchSize*10)
	for i := range input2 {
		input2[i] = float32(i%5) * 0.1
	}

	output2, _ := net2.ForwardCPU(input2)

	fmt.Printf("Input shape: [%d, %d]\n", batchSize, 10)
	fmt.Printf("Output shape: [%d, %d]\n", batchSize, 3)
	fmt.Printf("Output: %v\n", output2)
	fmt.Println()

	// Example 3: Parallel with avg combine mode
	example3JSON := `{
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "parallel",
				"combine_mode": "avg",
				"branches": [
					{
						"type": "dense",
						"input_size": 5,
						"output_size": 8,
						"activation": "relu"
					},
					{
						"type": "dense",
						"input_size": 5,
						"output_size": 8,
						"activation": "gelu"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 1,
				"activation": "linear"
			}
		]
	}`

	fmt.Println("Example 3: Parallel [Dense(relu) + Dense(gelu)] with avg combine mode")
	fmt.Println("Architecture: Parallel[Dense + Dense] → Dense(1)")
	fmt.Println()

	net3, err := nn.BuildNetworkFromJSON(example3JSON)
	if err != nil {
		log.Fatalf("Failed to build network 3: %v", err)
	}

	net3.InitializeWeights()

	input3 := make([]float32, batchSize*5)
	for i := range input3 {
		input3[i] = float32(i) * 0.2
	}

	output3, _ := net3.ForwardCPU(input3)

	fmt.Printf("Input shape: [%d, %d]\n", batchSize, 5)
	fmt.Printf("Output shape: [%d, %d]\n", batchSize, 1)
	fmt.Printf("Output: %v\n", output3)
	fmt.Println()

	// Example 4: Training with parallel layers
	fmt.Println("Example 4: Training network with parallel layers")
	fmt.Println("Task: XOR-like pattern recognition")
	fmt.Println()

	trainingJSON := `{
		"batch_size": 4,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 2,
				"output_size": 8,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 8,
						"output_size": 4,
						"activation": "relu"
					},
					{
						"type": "dense",
						"input_size": 8,
						"output_size": 4,
						"activation": "tanh"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 1,
				"activation": "sigmoid"
			}
		]
	}`

	netTrain, err := nn.BuildNetworkFromJSON(trainingJSON)
	if err != nil {
		log.Fatalf("Failed to build training network: %v", err)
	}

	netTrain.InitializeWeights()

	// XOR training data
	trainData := []float32{
		0, 0, // Input 1
		0, 1, // Input 2
		1, 0, // Input 3
		1, 1, // Input 4
	}

	trainLabels := []float32{0, 1, 1, 0} // XOR outputs

	// Training loop
	learningRate := float32(0.1)
	epochs := 1000

	fmt.Printf("Training for %d epochs with learning rate %.3f\n", epochs, learningRate)

	initialLoss := 0.0
	finalLoss := 0.0

	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		output, _ := netTrain.ForwardCPU(trainData)

		// Compute loss (MSE)
		loss := float32(0.0)
		for i := range trainLabels {
			diff := output[i] - trainLabels[i]
			loss += diff * diff
		}
		loss /= float32(len(trainLabels))

		if epoch == 0 {
			initialLoss = float64(loss)
		}
		if epoch == epochs-1 {
			finalLoss = float64(loss)
		}

		// Compute gradients
		gradOutput := make([]float32, len(trainLabels))
		for i := range trainLabels {
			gradOutput[i] = 2 * (output[i] - trainLabels[i]) / float32(len(trainLabels))
		}

		// Backward pass
		netTrain.BackwardCPU(gradOutput)

		// Update weights
		netTrain.UpdateWeights(learningRate)

		if (epoch+1)%200 == 0 || epoch == 0 {
			fmt.Printf("Epoch %4d: Loss = %.6f\n", epoch+1, loss)
		}
	}

	fmt.Println()
	fmt.Printf("Initial Loss: %.6f\n", initialLoss)
	fmt.Printf("Final Loss:   %.6f\n", finalLoss)
	fmt.Printf("Improvement:  %.2f%%\n", (1.0-finalLoss/initialLoss)*100)
	fmt.Println()

	// Test final predictions
	fmt.Println("Final predictions:")
	finalOutput, _ := netTrain.ForwardCPU(trainData)
	for i := range trainLabels {
		fmt.Printf("Input: [%.0f, %.0f] → Predicted: %.4f, Expected: %.0f\n",
			trainData[i*2], trainData[i*2+1], finalOutput[i], trainLabels[i])
	}
	fmt.Println()

	// Example 5: Complex parallel with nested layers
	fmt.Println("Example 5: Complex multi-branch parallel architecture")
	fmt.Println()

	complexJSON := `{
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 16,
				"output_size": 16,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 16,
						"output_size": 8,
						"activation": "relu"
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "dense",
						"input_size": 16,
						"output_size": 8,
						"activation": "gelu"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 4,
				"activation": "linear"
			}
		]
	}`

	fmt.Println("Architecture: Dense(16) → Parallel[Dense(8) + LSTM(8) + Dense(8)] → Dense(4)")
	fmt.Println("Parallel combines 3 branches with concat: 8 + 8 + 8 = 24 features")
	fmt.Println()

	netComplex, err := nn.BuildNetworkFromJSON(complexJSON)
	if err != nil {
		log.Fatalf("Failed to build complex network: %v", err)
	}

	netComplex.InitializeWeights()

	inputComplex := make([]float32, batchSize*16)
	for i := range inputComplex {
		inputComplex[i] = float32(math.Sin(float64(i) * 0.1))
	}

	outputComplex, _ := netComplex.ForwardCPU(inputComplex)

	fmt.Printf("Input shape: [%d, %d]\n", batchSize, 16)
	fmt.Printf("Output shape: [%d, %d]\n", batchSize, 4)
	fmt.Printf("Output sample: %v\n", outputComplex[:4])
	fmt.Println()

	fmt.Println("=== Demo Complete ===")
	fmt.Println("Parallel layers allow multiple processing paths within a single layer position,")
	fmt.Println("enabling richer feature extraction and ensemble-like behavior within networks.")
}
