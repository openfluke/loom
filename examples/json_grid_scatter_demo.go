package main

import (
	"fmt"
	"log"
	"math"

	nn "github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Grid Scatter Demo ===")
	fmt.Println("Doing things that make traditional neural networks cry...")
	fmt.Println()

	// Example 1: Spatial Feature Router
	// Different processing paths route their outputs to different spatial locations
	// Imagine: CNN features → top-left, LSTM features → top-right, Attention → bottom
	fmt.Println("Example 1: Spatial Feature Router")
	fmt.Println("Process input 4 different ways, route to 2x2 grid positions")
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
				"combine_mode": "grid_scatter",
				"grid_output_rows": 2,
				"grid_output_cols": 2,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0},
					{"branch_index": 2, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 3, "target_row": 1, "target_col": 1, "target_layer": 0}
				],
				"branches": [
					{
						"type": "dense",
						"input_size": 16,
						"output_size": 8,
						"activation": "relu",
						"comment": "High-level features → top-left"
					},
					{
						"type": "lstm",
						"input_size": 16,
						"hidden_size": 8,
						"seq_length": 1,
						"comment": "Temporal features → top-right"
					},
					{
						"type": "dense",
						"input_size": 16,
						"output_size": 8,
						"activation": "tanh",
						"comment": "Mid-level features → bottom-left"
					},
					{
						"type": "dense",
						"input_size": 16,
						"output_size": 8,
						"activation": "gelu",
						"comment": "Low-level features → bottom-right"
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
		input1[i] = float32(i%16) * 0.1
	}

	output1, _ := net1.ForwardCPU(input1)
	fmt.Printf("Input: [batch=2, features=16]\n")
	fmt.Printf("Grid Layout (2x2):\n")
	fmt.Printf("  ┌─────────────┬─────────────┐\n")
	fmt.Printf("  │ Dense(ReLU) │ LSTM(temp)  │  [0,0]: High-level\n")
	fmt.Printf("  │    8 feat   │   8 feat    │  [0,1]: Temporal\n")
	fmt.Printf("  ├─────────────┼─────────────┤\n")
	fmt.Printf("  │ Dense(Tanh) │ Dense(GELU) │  [1,0]: Mid-level\n")
	fmt.Printf("  │    8 feat   │   8 feat    │  [1,1]: Low-level\n")
	fmt.Printf("  └─────────────┴─────────────┘\n")
	fmt.Printf("Output shape: [batch=2, grid=2x2x1, features=8] = %d\n", len(output1))
	fmt.Printf("Output (first 32): %v\n", output1[:32])
	fmt.Println()

	// Example 2: Multi-Resolution Feature Pyramid
	// Process at different resolutions, place in grid layers (depth dimension)
	fmt.Println("Example 2: Multi-Resolution Feature Pyramid")
	fmt.Println("Different branches extract features at different resolutions")
	fmt.Println("Placed in grid LAYERS (depth), not just rows/cols!")
	fmt.Println()

	example2JSON := `{
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
				"combine_mode": "grid_scatter",
				"grid_output_rows": 1,
				"grid_output_cols": 1,
				"grid_output_layers": 3,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 0, "target_layer": 1},
					{"branch_index": 2, "target_row": 0, "target_col": 0, "target_layer": 2}
				],
				"branches": [
					{
						"type": "dense",
						"input_size": 32,
						"output_size": 16,
						"activation": "relu",
						"comment": "Coarse resolution"
					},
					{
						"type": "dense",
						"input_size": 32,
						"output_size": 16,
						"activation": "gelu",
						"comment": "Medium resolution"
					},
					{
						"type": "dense",
						"input_size": 32,
						"output_size": 16,
						"activation": "tanh",
						"comment": "Fine resolution"
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

	input2 := make([]float32, 32)
	for i := range input2 {
		input2[i] = float32(math.Sin(float64(i) * 0.2))
	}

	output2, _ := net2.ForwardCPU(input2)
	fmt.Printf("Input: [batch=1, features=32]\n")
	fmt.Printf("Grid Layout (1x1x3 - vertical layers!):\n")
	fmt.Printf("  Layer 0 (coarse):  Dense(ReLU) → 16 features\n")
	fmt.Printf("  Layer 1 (medium):  Dense(GELU) → 16 features\n")
	fmt.Printf("  Layer 2 (fine):    Dense(Tanh) → 16 features\n")
	fmt.Printf("Output shape: [batch=1, grid=1x1x3, features=16] = %d\n", len(output2))
	fmt.Printf("Layer 0 (coarse): %v\n", output2[0:16])
	fmt.Printf("Layer 1 (medium): %v\n", output2[16:32])
	fmt.Printf("Layer 2 (fine):   %v\n", output2[32:48])
	fmt.Println()

	// Example 3: THE INSANE ONE - Training with Dynamic Spatial Routing
	// Learn to route image patches to specialized grid positions
	// Different regions learn different transformations
	fmt.Println("Example 3: Dynamic Spatial Routing with Training")
	fmt.Println("Task: Process 4 'image patches', route to grid, classify")
	fmt.Println("Each patch gets specialized processing based on spatial position")
	fmt.Println()

	trainingJSON := `{
		"batch_size": 4,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 12,
				"output_size": 24,
				"activation": "relu",
				"comment": "Initial feature extraction"
			},
			{
				"type": "parallel",
				"combine_mode": "grid_scatter",
				"grid_output_rows": 2,
				"grid_output_cols": 2,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0},
					{"branch_index": 2, "target_row": 1, "target_col": 0, "target_layer": 0},
					{"branch_index": 3, "target_row": 1, "target_col": 1, "target_layer": 0}
				],
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
						],
						"comment": "Top-left specialist (ensemble)"
					},
					{
						"type": "lstm",
						"input_size": 24,
						"hidden_size": 12,
						"seq_length": 1,
						"comment": "Top-right specialist (temporal)"
					},
					{
						"type": "dense",
						"input_size": 24,
						"output_size": 12,
						"activation": "tanh",
						"comment": "Bottom-left specialist (bounded)"
					},
					{
						"type": "parallel",
						"combine_mode": "avg",
						"branches": [
							{
								"type": "dense",
								"input_size": 24,
								"output_size": 12,
								"activation": "sigmoid"
							},
							{
								"type": "dense",
								"input_size": 24,
								"output_size": 12,
								"activation": "swish"
							}
						],
						"comment": "Bottom-right specialist (smooth blend)"
					}
				]
			},
			{
				"type": "dense",
				"input_size": 48,
				"output_size": 2,
				"activation": "sigmoid",
				"comment": "Binary classification from spatial grid"
			}
		]
	}`

	netTrain, err := nn.BuildNetworkFromJSON(trainingJSON)
	if err != nil {
		log.Fatalf("Failed to build training network: %v", err)
	}
	netTrain.InitializeWeights()

	// Create bizarre synthetic data
	// Pattern: sum of first half > sum of second half = class 1, else class 0
	trainData := make([]float32, 4*12)
	trainLabels := []float32{
		0, 1, // Sample 1: more weight in second half
		1, 0, // Sample 2: more weight in first half
		1, 0, // Sample 3: more weight in first half
		0, 1, // Sample 4: more weight in second half
	}

	// Sample 1: [0.1, 0.1, ..., 0.8, 0.8, ...]
	for i := 0; i < 6; i++ {
		trainData[0*12+i] = 0.1
		trainData[0*12+6+i] = 0.8
	}
	// Sample 2: [0.9, 0.9, ..., 0.2, 0.2, ...]
	for i := 0; i < 6; i++ {
		trainData[1*12+i] = 0.9
		trainData[1*12+6+i] = 0.2
	}
	// Sample 3: [0.7, 0.7, ..., 0.3, 0.3, ...]
	for i := 0; i < 6; i++ {
		trainData[2*12+i] = 0.7
		trainData[2*12+6+i] = 0.3
	}
	// Sample 4: [0.2, 0.2, ..., 0.9, 0.9, ...]
	for i := 0; i < 6; i++ {
		trainData[3*12+i] = 0.2
		trainData[3*12+6+i] = 0.9
	}

	epochs := 1000
	learningRate := float32(0.1)

	fmt.Printf("Training for %d epochs with learning rate %.3f\n", epochs, learningRate)
	fmt.Printf("Architecture: Input → Dense → Grid Scatter (2x2) → Dense → Binary Output\n")
	fmt.Printf("Each grid position has specialized processing:\n")
	fmt.Printf("  [0,0]: Ensemble (add two dense layers)\n")
	fmt.Printf("  [0,1]: LSTM (temporal patterns)\n")
	fmt.Printf("  [1,0]: Tanh (bounded features)\n")
	fmt.Printf("  [1,1]: Blend (avg two dense layers)\n")
	fmt.Println()

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

		// Compute gradients (MSE derivative)
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
	fmt.Println("Final predictions (2 outputs per sample - binary classification):")
	finalOutput, _ := netTrain.ForwardCPU(trainData)
	for i := 0; i < 4; i++ {
		predicted := finalOutput[i*2 : i*2+2]
		expected := trainLabels[i*2 : i*2+2]

		// Predicted class
		predClass := 0
		if predicted[1] > predicted[0] {
			predClass = 1
		}

		// Expected class
		expClass := 0
		if expected[1] > expected[0] {
			expClass = 1
		}

		correct := "✓"
		if predClass != expClass {
			correct = "✗"
		}

		fmt.Printf("Sample %d: [%.3f, %.3f] → Class %d (expected %d) %s\n",
			i+1, predicted[0], predicted[1], predClass, expClass, correct)
	}
	fmt.Println()

	// Example 4: Nested Grid Scatter - MAXIMUM CHAOS
	fmt.Println("Example 4: Nested Grid Scatter - The Final Boss")
	fmt.Println("Grid scatter WITHIN grid scatter... because why not?")
	fmt.Println()

	example4JSON := `{
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
				"combine_mode": "grid_scatter",
				"grid_output_rows": 2,
				"grid_output_cols": 1,
				"grid_output_layers": 1,
				"grid_positions": [
					{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
					{"branch_index": 1, "target_row": 1, "target_col": 0, "target_layer": 0}
				],
				"branches": [
					{
						"type": "parallel",
						"combine_mode": "grid_scatter",
						"grid_output_rows": 1,
						"grid_output_cols": 2,
						"grid_output_layers": 1,
						"grid_positions": [
							{"branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0},
							{"branch_index": 1, "target_row": 0, "target_col": 1, "target_layer": 0}
						],
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
								"activation": "tanh"
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
			}
		]
	}`

	net4, err := nn.BuildNetworkFromJSON(example4JSON)
	if err != nil {
		log.Fatalf("Failed to build network 4: %v", err)
	}
	net4.InitializeWeights()

	input4 := make([]float32, 16)
	for i := range input4 {
		input4[i] = float32(i) * 0.05
	}

	output4, _ := net4.ForwardCPU(input4)
	fmt.Printf("Input: [batch=1, features=16]\n")
	fmt.Printf("Outer Grid (2x1):\n")
	fmt.Printf("  Row 0: INNER GRID SCATTER (1x2) → [Dense(4), Dense(4)] = 8\n")
	fmt.Printf("  Row 1: Dense(8) = 8\n")
	fmt.Printf("Output shape: [batch=1, features=16]\n")
	fmt.Printf("Output: %v\n", output4)
	fmt.Println()

	fmt.Println("=== Demo Complete ===")
	fmt.Println("Successfully demonstrated grid scatter mode:")
	fmt.Println("✓ Spatial feature routing (different paths → different grid positions)")
	fmt.Println("✓ Multi-resolution pyramids (using grid LAYERS/depth)")
	fmt.Println("✓ Training with specialized grid processors")
	fmt.Println("✓ Nested grid scatter (grid scatter within grid scatter)")
	fmt.Println()
	fmt.Println("Why this is impossible in traditional neural networks:")
	fmt.Println("• Traditional NNs are LINEAR CHAINS: input → hidden → hidden → output")
	fmt.Println("• Grid scatter creates SPATIAL GRAPHS with explicit topology")
	fmt.Println("• Each grid position can have specialized architecture")
	fmt.Println("• Gradients flow through 2D/3D spatial structure, not just sequential layers")
	fmt.Println("• You can route features to SPECIFIC LOCATIONS based on semantics")
	fmt.Println("• Nested grid scatter creates hierarchical spatial decomposition")
	fmt.Println()
	fmt.Println("Potential applications:")
	fmt.Println("→ Image segmentation: route patches to grid positions matching image layout")
	fmt.Println("→ Multi-agent systems: each agent occupies grid cell with specialized processing")
	fmt.Println("→ Hierarchical RL: high-level policy → grid → low-level controllers per position")
	fmt.Println("→ Neural architecture search: grid positions = architecture choices")
	fmt.Println("→ Ensemble learning: diverse models at different grid locations")
	fmt.Println("→ Spatially-aware feature fusion: explicit control over where features combine")
}
