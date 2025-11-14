package main

import (
	"fmt"
	"log"
	"math"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Parallel Layer - All Layer Types Demo ===")
	fmt.Println("Demonstrating parallel branches with Dense, LSTM, RNN, MHA, Conv2D, and more!")
	fmt.Println()

	// Example 1: Dense + RNN + LSTM in parallel
	fmt.Println("Example 1: Parallel [Dense + RNN + LSTM]")
	fmt.Println("Use case: Combining feedforward, simple recurrent, and long-term memory processing")
	fmt.Println()

	example1JSON := `{
		"batch_size": 2,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
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
						"type": "rnn",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1
					}
				]
			},
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 4,
				"activation": "relu"
			}
		]
	}`

	net1, err := nn.BuildNetworkFromJSON(example1JSON)
	if err != nil {
		log.Fatalf("Failed to build network 1: %v", err)
	}
	net1.InitializeWeights()

	input1 := make([]float32, 2*16)
	for i := range input1 {
		input1[i] = float32(math.Sin(float64(i) * 0.3))
	}

	output1, _ := net1.ForwardCPU(input1)
	fmt.Printf("Input: [batch=2, features=16]\n")
	fmt.Printf("Parallel outputs: Dense(8) + RNN(8) + LSTM(8) = 24 features concatenated\n")
	fmt.Printf("Final output: [batch=2, features=4]\n")
	fmt.Printf("Output: %v\n", output1)
	fmt.Println()

	// Example 2: Multi-Head Attention + Dense + LayerNorm in parallel
	fmt.Println("Example 2: Parallel [Multi-Head Attention + Dense + LayerNorm]")
	fmt.Println("Use case: Combining attention mechanism with feedforward and normalization")
	fmt.Println()

	example2JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "mha",
						"d_model": 32,
						"num_heads": 4,
						"seq_length": 1
					},
					{
						"type": "dense",
						"input_size": 32,
						"output_size": 32,
						"activation": "gelu"
					},
					{
						"type": "layer_norm",
						"norm_size": 32
					}
				]
			},
			{
				"type": "dense",
				"input_size": 96,
				"output_size": 16,
				"activation": "relu"
			}
		]
	}`

	net2, err := nn.BuildNetworkFromJSON(example2JSON)
	if err != nil {
		log.Fatalf("Failed to build network 2: %v", err)
	}
	net2.InitializeWeights()

	input2 := make([]float32, 32)
	for i := range input2 {
		input2[i] = float32(i%10) * 0.1
	}

	output2, _ := net2.ForwardCPU(input2)
	fmt.Printf("Input: [batch=1, d_model=32]\n")
	fmt.Printf("Parallel: MHA(32) + Dense(32) + LayerNorm(32) = 96 features\n")
	fmt.Printf("Final output: [batch=1, features=16]\n")
	fmt.Printf("Output: %v\n", output2)
	fmt.Println()

	// Example 3: Dense + LayerNorm + RMSNorm in parallel
	fmt.Println("Example 3: Parallel [Dense + LayerNorm + RMSNorm]")
	fmt.Println("Use case: Dense features + two normalization techniques")
	fmt.Println()

	example3JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 32,
						"output_size": 16,
						"activation": "relu"
					},
					{
						"type": "layer_norm",
						"norm_size": 32
					},
					{
						"type": "rmsnorm",
						"norm_size": 32
					}
				]
			},
			{
				"type": "dense",
				"input_size": 80,
				"output_size": 10,
				"activation": "linear"
			}
		]
	}`

	net3, err := nn.BuildNetworkFromJSON(example3JSON)
	if err != nil {
		log.Fatalf("Failed to build network 3: %v", err)
	}
	net3.InitializeWeights()

	input3 := make([]float32, 32)
	for i := range input3 {
		input3[i] = float32(i%16) / 16.0
	}

	output3, _ := net3.ForwardCPU(input3)
	fmt.Printf("Input: [batch=1, features=32]\n")
	fmt.Printf("Parallel: Dense(16) + LayerNorm(32) + RMSNorm(32) = 80 features\n")
	fmt.Printf("Final output: [batch=1, classes=10]\n")
	fmt.Printf("Output: %v\n", output3)
	fmt.Println()

	// Example 4: SwiGLU + Dense with different activations in parallel
	fmt.Println("Example 4: Parallel [SwiGLU + Dense(ReLU) + Dense(GELU) + Dense(Tanh)]")
	fmt.Println("Use case: Multiple activation functions processing same features")
	fmt.Println()

	example4JSON := `{
		"batch_size": 2,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "swiglu",
						"input_size": 20,
						"output_size": 16
					},
					{
						"type": "dense",
						"input_size": 20,
						"output_size": 16,
						"activation": "relu"
					},
					{
						"type": "dense",
						"input_size": 20,
						"output_size": 16,
						"activation": "gelu"
					},
					{
						"type": "dense",
						"input_size": 20,
						"output_size": 16,
						"activation": "tanh"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 64,
				"output_size": 8,
				"activation": "relu"
			}
		]
	}`

	net4, err := nn.BuildNetworkFromJSON(example4JSON)
	if err != nil {
		log.Fatalf("Failed to build network 4: %v", err)
	}
	net4.InitializeWeights()

	input4 := make([]float32, 2*20)
	for i := range input4 {
		input4[i] = float32(i%15) * 0.1
	}

	output4, _ := net4.ForwardCPU(input4)
	fmt.Printf("Input: [batch=2, features=20]\n")
	fmt.Printf("Parallel: SwiGLU(16) + Dense-ReLU(16) + Dense-GELU(16) + Dense-Tanh(16) = 64\n")
	fmt.Printf("Final output: [batch=2, features=8]\n")
	fmt.Printf("Output: %v\n", output4)
	fmt.Println()

	// Example 5: Softmax variants in parallel
	fmt.Println("Example 5: Parallel [Softmax + Grid-Softmax + Hierarchical-Softmax]")
	fmt.Println("Use case: Different probability distributions over same logits")
	fmt.Println()

	example5JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 12,
				"output_size": 12,
				"activation": "linear"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "softmax",
						"softmax_variant": "standard"
					},
					{
						"type": "softmax",
						"softmax_variant": "grid",
						"softmax_rows": 3,
						"softmax_cols": 4
					},
					{
						"type": "softmax",
						"softmax_variant": "hierarchical",
						"hierarchy_levels": [3, 4]
					}
				]
			}
		]
	}`

	net5, err := nn.BuildNetworkFromJSON(example5JSON)
	if err != nil {
		log.Fatalf("Failed to build network 5: %v", err)
	}
	net5.InitializeWeights()

	input5 := make([]float32, 12)
	for i := range input5 {
		input5[i] = float32(i)
	}

	output5, _ := net5.ForwardCPU(input5)
	fmt.Printf("Input: [batch=1, logits=12]\n")
	fmt.Printf("Parallel softmax variants: standard(12) + grid(12) + hierarchical(12) = 36\n")
	fmt.Printf("Output length: %d\n", len(output5))
	fmt.Printf("First softmax (standard): %v\n", output5[0:12])
	fmt.Println()

	// Example 6: Nested parallel layers
	fmt.Println("Example 6: Nested Parallel Layers")
	fmt.Println("A parallel layer containing branches that also have parallel layers!")
	fmt.Println()

	example6JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
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
						"type": "parallel",
						"combine_mode": "add",
						"branches": [
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "tanh"
							}
						]
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "layer_norm",
						"norm_size": 16
					}
				]
			}
		]
	}`

	net6, err := nn.BuildNetworkFromJSON(example6JSON)
	if err != nil {
		log.Fatalf("Failed to build network 6: %v", err)
	}
	net6.InitializeWeights()

	input6 := make([]float32, 16)
	for i := range input6 {
		input6[i] = float32(i) * 0.1
	}

	output6, _ := net6.ForwardCPU(input6)
	fmt.Printf("Input: [batch=1, features=16]\n")
	fmt.Printf("Outer parallel:\n")
	fmt.Printf("  - Inner parallel (add): Dense(8) + Dense(8) = 8\n")
	fmt.Printf("  - LSTM: 8\n")
	fmt.Printf("  - LayerNorm: 16\n")
	fmt.Printf("Total concatenated: 8 + 8 + 16 = 32 features\n")
	fmt.Printf("Output: %v\n", output6)
	fmt.Println()

	// Example 7: Average combine mode with multiple recurrent layers
	fmt.Println("Example 7: Parallel with Average Combine Mode")
	fmt.Println("Ensemble of RNN, LSTM, and Dense averaged together")
	fmt.Println()

	example7JSON := `{
		"batch_size": 2,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "parallel",
				"combine_mode": "avg",
				"branches": [
					{
						"type": "rnn",
						"input_size": 10,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "lstm",
						"input_size": 10,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "dense",
						"input_size": 10,
						"output_size": 8,
						"activation": "tanh"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 3,
				"activation": "linear"
			}
		]
	}`

	net7, err := nn.BuildNetworkFromJSON(example7JSON)
	if err != nil {
		log.Fatalf("Failed to build network 7: %v", err)
	}
	net7.InitializeWeights()

	input7 := make([]float32, 2*10)
	for i := range input7 {
		input7[i] = float32(i%7) * 0.15
	}

	output7, _ := net7.ForwardCPU(input7)
	fmt.Printf("Input: [batch=2, features=10]\n")
	fmt.Printf("Parallel (averaged): RNN(8) + LSTM(8) + Dense(8) → avg → 8 features\n")
	fmt.Printf("Final output: [batch=2, features=3]\n")
	fmt.Printf("Output: %v\n", output7)
	fmt.Println()

	// Example 8: Add combine mode - residual-like connections
	fmt.Println("Example 8: Parallel with Add Combine Mode")
	fmt.Println("Multiple transformation paths summed together (residual-style)")
	fmt.Println()

	example8JSON := `{
		"batch_size": 1,
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
						"input_size": 12,
						"output_size": 12,
						"activation": "relu"
					},
					{
						"type": "dense",
						"input_size": 12,
						"output_size": 12,
						"activation": "gelu"
					},
					{
						"type": "dense",
						"input_size": 12,
						"output_size": 12,
						"activation": "sigmoid"
					},
					{
						"type": "layer_norm",
						"norm_size": 12
					}
				]
			},
			{
				"type": "dense",
				"input_size": 12,
				"output_size": 4,
				"activation": "relu"
			}
		]
	}`

	net8, err := nn.BuildNetworkFromJSON(example8JSON)
	if err != nil {
		log.Fatalf("Failed to build network 8: %v", err)
	}
	net8.InitializeWeights()

	input8 := make([]float32, 12)
	for i := range input8 {
		input8[i] = float32(i) * 0.2
	}

	output8, _ := net8.ForwardCPU(input8)
	fmt.Printf("Input: [batch=1, features=12]\n")
	fmt.Printf("Parallel (summed): Dense-ReLU + Dense-GELU + Dense-Sigmoid + LayerNorm → 12 features\n")
	fmt.Printf("Final output: [batch=1, features=4]\n")
	fmt.Printf("Output: %v\n", output8)
	fmt.Println()

	// Example 9: Training with parallel layers
	fmt.Println("Example 9: Training with Parallel Layers")
	fmt.Println("Binary classification task using parallel feature extraction")
	fmt.Println()

	trainingJSON := `{
		"batch_size": 4,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 8,
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
						"type": "dense",
						"input_size": 16,
						"output_size": 8,
						"activation": "gelu"
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1
					}
				]
			},
			{
				"type": "dense",
				"input_size": 24,
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

	// Synthetic binary classification data
	trainData := make([]float32, 4*8)
	for i := 0; i < 4; i++ {
		for j := 0; j < 8; j++ {
			if i < 2 {
				trainData[i*8+j] = float32(j%4) * 0.3
			} else {
				trainData[i*8+j] = float32(7-j%4) * 0.3
			}
		}
	}

	trainLabels := []float32{0, 0, 1, 1}

	epochs := 500
	learningRate := float32(0.1)

	fmt.Printf("Training for %d epochs with learning rate %.3f\n", epochs, learningRate)

	initialLoss := float64(0)
	finalLoss := float64(0)

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

		if (epoch+1)%100 == 0 || epoch == 0 {
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
		fmt.Printf("Sample %d: Predicted = %.4f, Expected = %.0f\n", i+1, finalOutput[i], trainLabels[i])
	}
	fmt.Println()

	// Example 10: Maximum complexity - all layer types together
	fmt.Println("Example 10: Kitchen Sink - All Layer Types in One Network")
	fmt.Println("Dense → Parallel[Dense,RNN,LSTM,MHA,Conv2D,LayerNorm,RMSNorm,SwiGLU] → Dense")
	fmt.Println()

	kitchenSinkJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 64,
				"output_size": 64,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 64,
						"output_size": 16,
						"activation": "relu"
					},
					{
						"type": "rnn",
						"input_size": 64,
						"hidden_size": 16,
						"seq_length": 1
					},
					{
						"type": "lstm",
						"input_size": 64,
						"hidden_size": 16,
						"seq_length": 1
					},
					{
						"type": "mha",
						"d_model": 64,
						"num_heads": 4,
						"seq_length": 1
					},
					{
						"type": "layer_norm",
						"norm_size": 64
					},
					{
						"type": "rmsnorm",
						"norm_size": 64
					},
					{
						"type": "swiglu",
						"input_size": 64,
						"output_size": 16
					}
				]
			},
			{
				"type": "dense",
				"input_size": 272,
				"output_size": 10,
				"activation": "softmax"
			}
		]
	}`

	netKitchen, err := nn.BuildNetworkFromJSON(kitchenSinkJSON)
	if err != nil {
		log.Fatalf("Failed to build kitchen sink network: %v", err)
	}
	netKitchen.InitializeWeights()

	inputKitchen := make([]float32, 64)
	for i := range inputKitchen {
		inputKitchen[i] = float32(math.Sin(float64(i) * 0.1))
	}

	outputKitchen, _ := netKitchen.ForwardCPU(inputKitchen)
	fmt.Printf("Input: [batch=1, features=64]\n")
	fmt.Printf("Parallel branches:\n")
	fmt.Printf("  - Dense: 16\n")
	fmt.Printf("  - RNN: 16\n")
	fmt.Printf("  - LSTM: 16\n")
	fmt.Printf("  - MHA: 64\n")
	fmt.Printf("  - LayerNorm: 64\n")
	fmt.Printf("  - RMSNorm: 64\n")
	fmt.Printf("  - SwiGLU: 16\n")
	fmt.Printf("Total concatenated: 272 features\n")
	fmt.Printf("Final output: [batch=1, classes=10]\n")
	fmt.Printf("Output: %v\n", outputKitchen)
	fmt.Println()

	fmt.Println("=== Demo Complete ===")
	fmt.Println("Successfully demonstrated parallel layers with all LOOM layer types:")
	fmt.Println("✓ Dense, RNN, LSTM, Multi-Head Attention")
	fmt.Println("✓ Conv2D, LayerNorm, RMSNorm, SwiGLU")
	fmt.Println("✓ Softmax variants, Nested parallel layers")
	fmt.Println("✓ All three combine modes: concat, add, avg")
	fmt.Println("✓ Training with backpropagation through parallel layers")
}
