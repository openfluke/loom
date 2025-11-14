package main

import (
	"fmt"
	"log"
	"math"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Nested Parallel Layers Demo ===")
	fmt.Println("Exploring the power of parallel layers within parallel layers!")
	fmt.Println()

	// Example 1: Simple 2-level nesting
	fmt.Println("Example 1: Two-Level Nested Parallel")
	fmt.Println("Outer parallel contains inner parallel branches")
	fmt.Println()

	example1JSON := `{
		"batch_size": 2,
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
							},
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "sigmoid"
							}
						]
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1
					}
				]
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
		input1[i] = float32(i%10) * 0.1
	}

	output1, _ := net1.ForwardCPU(input1)
	fmt.Printf("Input: [batch=2, features=16]\n")
	fmt.Printf("Structure:\n")
	fmt.Printf("  Outer Parallel:\n")
	fmt.Printf("    - Inner Parallel (add): Dense(8) + Dense(8) + Dense(8) → 8\n")
	fmt.Printf("    - LSTM: 8\n")
	fmt.Printf("  Total: 8 + 8 = 16 features\n")
	fmt.Printf("Output shape: [batch=2, features=16]\n")
	fmt.Printf("Output: %v\n", output1)
	fmt.Println()

	// Example 2: Three-level nesting
	fmt.Println("Example 2: Three-Level Nested Parallel")
	fmt.Println("Going deeper with parallel within parallel within parallel!")
	fmt.Println()

	example2JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 20,
				"output_size": 20,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "parallel",
						"combine_mode": "concat",
						"branches": [
							{
								"type": "parallel",
								"combine_mode": "avg",
								"branches": [
									{
										"type": "dense",
										"input_size": 20,
										"output_size": 8,
										"activation": "relu"
									},
									{
										"type": "dense",
										"input_size": 20,
										"output_size": 8,
										"activation": "gelu"
									}
								]
							},
							{
								"type": "dense",
								"input_size": 20,
								"output_size": 8,
								"activation": "tanh"
							}
						]
					},
					{
						"type": "rnn",
						"input_size": 20,
						"hidden_size": 8,
						"seq_length": 1
					}
				]
			}
		]
	}`

	net2, err := nn.BuildNetworkFromJSON(example2JSON)
	if err != nil {
		log.Fatalf("Failed to build network 2: %v", err)
	}
	net2.InitializeWeights()

	input2 := make([]float32, 20)
	for i := range input2 {
		input2[i] = float32(math.Sin(float64(i) * 0.2))
	}

	output2, _ := net2.ForwardCPU(input2)
	fmt.Printf("Input: [batch=1, features=20]\n")
	fmt.Printf("Structure:\n")
	fmt.Printf("  Level 1 Parallel (concat):\n")
	fmt.Printf("    Branch 1 - Level 2 Parallel (concat):\n")
	fmt.Printf("      Branch 1 - Level 3 Parallel (avg): Dense(8) + Dense(8) → 8\n")
	fmt.Printf("      Branch 2 - Dense: 8\n")
	fmt.Printf("      Total: 8 + 8 = 16\n")
	fmt.Printf("    Branch 2 - RNN: 8\n")
	fmt.Printf("  Total: 16 + 8 = 24 features\n")
	fmt.Printf("Output shape: [batch=1, features=24]\n")
	fmt.Printf("Output: %v\n", output2)
	fmt.Println()

	// Example 3: Multiple nested branches at same level
	fmt.Println("Example 3: Multiple Nested Branches")
	fmt.Println("Multiple inner parallel layers as siblings")
	fmt.Println()

	example3JSON := `{
		"batch_size": 2,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 24,
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
								"input_size": 24,
								"output_size": 12,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 24,
								"output_size": 12,
								"activation": "gelu"
							}
						]
					},
					{
						"type": "parallel",
						"combine_mode": "add",
						"branches": [
							{
								"type": "lstm",
								"input_size": 24,
								"hidden_size": 12,
								"seq_length": 1
							},
							{
								"type": "rnn",
								"input_size": 24,
								"hidden_size": 12,
								"seq_length": 1
							}
						]
					},
					{
						"type": "parallel",
						"combine_mode": "avg",
						"branches": [
							{
								"type": "layer_norm",
								"norm_size": 24
							},
							{
								"type": "rmsnorm",
								"norm_size": 24
							}
						]
					}
				]
			}
		]
	}`

	net3, err := nn.BuildNetworkFromJSON(example3JSON)
	if err != nil {
		log.Fatalf("Failed to build network 3: %v", err)
	}
	net3.InitializeWeights()

	input3 := make([]float32, 2*24)
	for i := range input3 {
		input3[i] = float32(i%15) * 0.1
	}

	output3, _ := net3.ForwardCPU(input3)
	fmt.Printf("Input: [batch=2, features=24]\n")
	fmt.Printf("Structure (3 parallel inner branches):\n")
	fmt.Printf("  Branch 1 (add): Dense(12) + Dense(12) → 12\n")
	fmt.Printf("  Branch 2 (add): LSTM(12) + RNN(12) → 12\n")
	fmt.Printf("  Branch 3 (avg): LayerNorm(24) + RMSNorm(24) → 24\n")
	fmt.Printf("Total: 12 + 12 + 24 = 48 features\n")
	fmt.Printf("Output shape: [batch=2, features=48]\n")
	fmt.Printf("Output: %v\n", output3)
	fmt.Println()

	// Example 4: Asymmetric nesting
	fmt.Println("Example 4: Asymmetric Nested Parallel")
	fmt.Println("Some branches nested, some not - mixed architecture")
	fmt.Println()

	example4JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 32,
				"output_size": 32,
				"activation": "relu"
			},
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
						"type": "parallel",
						"combine_mode": "concat",
						"branches": [
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 8,
								"activation": "tanh"
							},
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 8,
								"activation": "sigmoid"
							}
						]
					},
					{
						"type": "lstm",
						"input_size": 32,
						"hidden_size": 16,
						"seq_length": 1
					},
					{
						"type": "parallel",
						"combine_mode": "avg",
						"branches": [
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 16,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 16,
								"activation": "gelu"
							},
							{
								"type": "dense",
								"input_size": 32,
								"output_size": 16,
								"activation": "swish"
							}
						]
					}
				]
			}
		]
	}`

	net4, err := nn.BuildNetworkFromJSON(example4JSON)
	if err != nil {
		log.Fatalf("Failed to build network 4: %v", err)
	}
	net4.InitializeWeights()

	input4 := make([]float32, 32)
	for i := range input4 {
		input4[i] = float32(i) * 0.05
	}

	output4, _ := net4.ForwardCPU(input4)
	fmt.Printf("Input: [batch=1, features=32]\n")
	fmt.Printf("Structure (mixed nested/flat):\n")
	fmt.Printf("  Branch 1: Dense → 16\n")
	fmt.Printf("  Branch 2 (nested concat): Dense(8) + Dense(8) → 16\n")
	fmt.Printf("  Branch 3: LSTM → 16\n")
	fmt.Printf("  Branch 4 (nested avg): Dense + Dense + Dense → 16\n")
	fmt.Printf("Total: 16 + 16 + 16 + 16 = 64 features\n")
	fmt.Printf("Output shape: [batch=1, features=64]\n")
	fmt.Printf("Output: %v\n", output4)
	fmt.Println()

	// Example 5: Deep nesting with different combine modes
	fmt.Println("Example 5: Deep Nesting with Mixed Combine Modes")
	fmt.Println("Each level uses different combine strategy")
	fmt.Println()

	example5JSON := `{
		"batch_size": 2,
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
								"type": "parallel",
								"combine_mode": "avg",
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
									},
									{
										"type": "dense",
										"input_size": 16,
										"output_size": 8,
										"activation": "sigmoid"
									}
								]
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
						"input_size": 16,
						"output_size": 8,
						"activation": "relu"
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

	input5 := make([]float32, 2*16)
	for i := range input5 {
		input5[i] = float32(i%8) * 0.15
	}

	output5, _ := net5.ForwardCPU(input5)
	fmt.Printf("Input: [batch=2, features=16]\n")
	fmt.Printf("Structure (mixed combine modes):\n")
	fmt.Printf("  Level 1 (concat):\n")
	fmt.Printf("    Branch 1 - Level 2 (add):\n")
	fmt.Printf("      Branch 1 - Level 3 (avg): Dense + Dense + Dense → 8\n")
	fmt.Printf("      Branch 2 - Dense → 8\n")
	fmt.Printf("      Combined (add) → 8\n")
	fmt.Printf("    Branch 2 - Dense → 8\n")
	fmt.Printf("  Total (concat): 8 + 8 = 16 features\n")
	fmt.Printf("Output shape: [batch=2, features=16]\n")
	fmt.Printf("Output: %v\n", output5)
	fmt.Println()

	// Example 6: Training with nested parallel layers
	fmt.Println("Example 6: Training with Nested Parallel Layers")
	fmt.Println("Multi-class classification with hierarchical feature extraction")
	fmt.Println()

	trainingJSON := `{
		"batch_size": 4,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 10,
				"output_size": 20,
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
								"input_size": 20,
								"output_size": 12,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 20,
								"output_size": 12,
								"activation": "gelu"
							}
						]
					},
					{
						"type": "lstm",
						"input_size": 20,
						"hidden_size": 12,
						"seq_length": 1
					},
					{
						"type": "parallel",
						"combine_mode": "avg",
						"branches": [
							{
								"type": "dense",
								"input_size": 20,
								"output_size": 12,
								"activation": "tanh"
							},
							{
								"type": "dense",
								"input_size": 20,
								"output_size": 12,
								"activation": "sigmoid"
							}
						]
					}
				]
			},
			{
				"type": "dense",
				"input_size": 36,
				"output_size": 3,
				"activation": "sigmoid"
			}
		]
	}`

	netTrain, err := nn.BuildNetworkFromJSON(trainingJSON)
	if err != nil {
		log.Fatalf("Failed to build training network: %v", err)
	}
	netTrain.InitializeWeights()

	// Create synthetic 3-class classification data
	trainData := make([]float32, 4*10)
	for i := 0; i < 4; i++ {
		for j := 0; j < 10; j++ {
			if i == 0 {
				trainData[i*10+j] = float32(j) * 0.1
			} else if i == 1 {
				trainData[i*10+j] = float32(9-j) * 0.1
			} else if i == 2 {
				trainData[i*10+j] = float32((j+3)%10) * 0.1
			} else {
				trainData[i*10+j] = float32((j+7)%10) * 0.1
			}
		}
	}

	// One-hot encoded labels for 3 classes
	trainLabels := []float32{
		1, 0, 0, // Sample 1: class 0
		0, 1, 0, // Sample 2: class 1
		0, 0, 1, // Sample 3: class 2
		1, 0, 0, // Sample 4: class 0
	}

	epochs := 1000
	learningRate := float32(0.1)

	fmt.Printf("Training for %d epochs with learning rate %.3f\n", epochs, learningRate)
	fmt.Printf("Task: 3-class classification with nested parallel feature extraction\n")

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
	fmt.Println("Final predictions (3 outputs per sample):")
	finalOutput, _ := netTrain.ForwardCPU(trainData)
	for i := 0; i < 4; i++ {
		predicted := finalOutput[i*3 : i*3+3]
		expected := trainLabels[i*3 : i*3+3]

		// Find predicted class (argmax)
		maxIdx := 0
		maxVal := predicted[0]
		for j := 1; j < 3; j++ {
			if predicted[j] > maxVal {
				maxVal = predicted[j]
				maxIdx = j
			}
		}

		// Find expected class
		expIdx := 0
		for j := 0; j < 3; j++ {
			if expected[j] == 1.0 {
				expIdx = j
				break
			}
		}

		correct := "✓"
		if maxIdx != expIdx {
			correct = "✗"
		}

		fmt.Printf("Sample %d: [%.3f, %.3f, %.3f] → Class %d (expected %d) %s\n",
			i+1, predicted[0], predicted[1], predicted[2], maxIdx, expIdx, correct)
	}
	fmt.Println()

	// Example 7: Extreme nesting - 4 levels deep
	fmt.Println("Example 7: Extreme Nesting - 4 Levels Deep!")
	fmt.Println("Testing the limits of nested parallel layers")
	fmt.Println()

	example7JSON := `{
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
						"combine_mode": "concat",
						"branches": [
							{
								"type": "parallel",
								"combine_mode": "add",
								"branches": [
									{
										"type": "parallel",
										"combine_mode": "avg",
										"branches": [
											{
												"type": "dense",
												"input_size": 16,
												"output_size": 4,
												"activation": "relu"
											},
											{
												"type": "dense",
												"input_size": 16,
												"output_size": 4,
												"activation": "gelu"
											}
										]
									},
									{
										"type": "dense",
										"input_size": 16,
										"output_size": 4,
										"activation": "tanh"
									}
								]
							},
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 4,
								"activation": "sigmoid"
							}
						]
					},
					{
						"type": "dense",
						"input_size": 16,
						"output_size": 4,
						"activation": "relu"
					}
				]
			}
		]
	}`

	net7, err := nn.BuildNetworkFromJSON(example7JSON)
	if err != nil {
		log.Fatalf("Failed to build network 7: %v", err)
	}
	net7.InitializeWeights()

	input7 := make([]float32, 16)
	for i := range input7 {
		input7[i] = float32(math.Cos(float64(i) * 0.3))
	}

	output7, _ := net7.ForwardCPU(input7)
	fmt.Printf("Input: [batch=1, features=16]\n")
	fmt.Printf("Structure (4 levels deep!):\n")
	fmt.Printf("  L1 (concat):\n")
	fmt.Printf("    B1 - L2 (concat):\n")
	fmt.Printf("      B1 - L3 (add):\n")
	fmt.Printf("        B1 - L4 (avg): Dense(4) + Dense(4) → 4\n")
	fmt.Printf("        B2 - Dense(4)\n")
	fmt.Printf("        Combined → 4\n")
	fmt.Printf("      B2 - Dense(4)\n")
	fmt.Printf("      Combined → 4 + 4 = 8\n")
	fmt.Printf("    B2 - Dense(4)\n")
	fmt.Printf("  Total: 8 + 4 = 12 features\n")
	fmt.Printf("Output shape: [batch=1, features=12]\n")
	fmt.Printf("Output: %v\n", output7)
	fmt.Println()

	// Example 8: Binary tree structure
	fmt.Println("Example 8: Binary Tree Nested Structure")
	fmt.Println("Each branch splits into two more branches")
	fmt.Println()

	example8JSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 12,
				"output_size": 12,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "parallel",
						"combine_mode": "concat",
						"branches": [
							{
								"type": "dense",
								"input_size": 12,
								"output_size": 4,
								"activation": "relu"
							},
							{
								"type": "dense",
								"input_size": 12,
								"output_size": 4,
								"activation": "tanh"
							}
						]
					},
					{
						"type": "parallel",
						"combine_mode": "concat",
						"branches": [
							{
								"type": "dense",
								"input_size": 12,
								"output_size": 4,
								"activation": "gelu"
							},
							{
								"type": "dense",
								"input_size": 12,
								"output_size": 4,
								"activation": "sigmoid"
							}
						]
					}
				]
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
		input8[i] = float32(i) * 0.1
	}

	output8, _ := net8.ForwardCPU(input8)
	fmt.Printf("Input: [batch=1, features=12]\n")
	fmt.Printf("Binary tree structure:\n")
	fmt.Printf("        Root\n")
	fmt.Printf("       /    \\\n")
	fmt.Printf("    Left    Right\n")
	fmt.Printf("    / \\      / \\\n")
	fmt.Printf("   4   4    4   4\n")
	fmt.Printf("Total: 4 + 4 + 4 + 4 = 16 features\n")
	fmt.Printf("Output shape: [batch=1, features=16]\n")
	fmt.Printf("Output: %v\n", output8)
	fmt.Println()

	fmt.Println("=== Demo Complete ===")
	fmt.Println("Successfully demonstrated nested parallel layers with:")
	fmt.Println("✓ 2-level, 3-level, and 4-level deep nesting")
	fmt.Println("✓ Multiple nested branches as siblings")
	fmt.Println("✓ Asymmetric nesting (mixed nested/flat)")
	fmt.Println("✓ Different combine modes at each level (concat, add, avg)")
	fmt.Println("✓ Training with nested parallel architecture")
	fmt.Println("✓ Binary tree and hierarchical structures")
	fmt.Println("✓ All layer types working in nested configurations")
	fmt.Println()
	fmt.Println("Nested parallel layers enable:")
	fmt.Println("- Hierarchical feature extraction")
	fmt.Println("- Multi-scale processing")
	fmt.Println("- Ensemble methods within single network")
	fmt.Println("- Complex architectural patterns")
}
