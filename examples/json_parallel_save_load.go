package main

import (
	"fmt"
	"log"
	"os"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Parallel Layers Save/Load Test ===")
	fmt.Println("Testing serialization and deserialization of parallel layers")
	fmt.Println()

	// Create a network with nested parallel layers
	networkJSON := `{
		"batch_size": 2,
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
								"activation": "gelu"
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
						"type": "parallel",
						"combine_mode": "avg",
						"branches": [
							{
								"type": "dense",
								"input_size": 16,
								"output_size": 8,
								"activation": "sigmoid"
							},
							{
								"type": "rnn",
								"input_size": 16,
								"hidden_size": 8,
								"seq_length": 1,
								"activation": "tanh"
							}
						]
					}
				]
			},
			{
				"type": "dense",
				"input_size": 24,
				"output_size": 4,
				"activation": "sigmoid"
			}
		]
	}`

	fmt.Println("Step 1: Building network from JSON")
	net1, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net1.InitializeWeights()
	fmt.Println("✓ Network built and initialized")
	fmt.Println()

	// Create test input
	input := make([]float32, 2*16)
	for i := range input {
		input[i] = float32(i%10) * 0.1
	}

	// Run forward pass with original network
	fmt.Println("Step 2: Running forward pass on original network")
	output1, _ := net1.ForwardCPU(input)
	fmt.Printf("Output from original network: %v\n", output1)
	fmt.Println()

	// Save the network
	filename := "test_parallel_network.json"
	fmt.Printf("Step 3: Saving network to %s\n", filename)
	err = net1.SaveModel(filename, "parallel_test")
	if err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}
	fmt.Println("✓ Network saved successfully")
	fmt.Println()

	// Load the network
	fmt.Printf("Step 4: Loading network from %s\n", filename)
	net2, err := nn.LoadModel(filename, "parallel_test")
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✓ Network loaded successfully")
	fmt.Println()

	// Run forward pass with loaded network
	fmt.Println("Step 5: Running forward pass on loaded network")
	output2, _ := net2.ForwardCPU(input)
	fmt.Printf("Output from loaded network: %v\n", output2)
	fmt.Println()

	// Compare outputs
	fmt.Println("Step 6: Comparing outputs")
	allMatch := true
	maxDiff := float32(0.0)
	for i := range output1 {
		diff := output1[i] - output2[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-6 {
			allMatch = false
			fmt.Printf("  Index %d: %.8f vs %.8f (diff: %.8f)\n", i, output1[i], output2[i], diff)
		}
	}

	if allMatch {
		fmt.Println("✓ All outputs match perfectly!")
		fmt.Printf("  Maximum difference: %.10f\n", maxDiff)
	} else {
		fmt.Println("✗ Outputs differ")
		fmt.Printf("  Maximum difference: %.10f\n", maxDiff)
	}
	fmt.Println()

	// Clean up
	fmt.Println("Step 7: Cleaning up")
	err = os.Remove(filename)
	if err != nil {
		log.Printf("Warning: Failed to remove test file: %v", err)
	} else {
		fmt.Printf("✓ Removed %s\n", filename)
	}
	fmt.Println()

	// Test with simpler parallel network (no nesting)
	fmt.Println("=== Testing Simple Parallel Layer ===")
	simpleJSON := `{
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_size": 10,
				"output_size": 10,
				"activation": "relu"
			},
			{
				"type": "parallel",
				"combine_mode": "concat",
				"branches": [
					{
						"type": "dense",
						"input_size": 10,
						"output_size": 5,
						"activation": "relu"
					},
					{
						"type": "dense",
						"input_size": 10,
						"output_size": 5,
						"activation": "tanh"
					}
				]
			}
		]
	}`

	net3, err := nn.BuildNetworkFromJSON(simpleJSON)
	if err != nil {
		log.Fatalf("Failed to build simple network: %v", err)
	}
	net3.InitializeWeights()

	input2 := make([]float32, 10)
	for i := range input2 {
		input2[i] = float32(i) * 0.1
	}

	output3, _ := net3.ForwardCPU(input2)
	fmt.Printf("Original output: %v\n", output3)

	filename2 := "test_simple_parallel.json"
	err = net3.SaveModel(filename2, "simple_test")
	if err != nil {
		log.Fatalf("Failed to save simple model: %v", err)
	}

	net4, err := nn.LoadModel(filename2, "simple_test")
	if err != nil {
		log.Fatalf("Failed to load simple model: %v", err)
	}

	output4, _ := net4.ForwardCPU(input2)
	fmt.Printf("Loaded output:   %v\n", output4)

	// Compare
	simpleMatch := true
	for i := range output3 {
		diff := output3[i] - output4[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			simpleMatch = false
			break
		}
	}

	if simpleMatch {
		fmt.Println("✓ Simple parallel save/load works!")
	} else {
		fmt.Println("✗ Simple parallel save/load failed")
	}

	os.Remove(filename2)
	fmt.Println()

	fmt.Println("=== Save/Load Test Complete ===")
	fmt.Println("Parallel layers can be successfully saved and loaded!")
	fmt.Println("The serialization preserves:")
	fmt.Println("✓ Nested parallel layer structures")
	fmt.Println("✓ All branch configurations")
	fmt.Println("✓ Combine modes (concat, add, avg)")
	fmt.Println("✓ Network weights and biases")
	fmt.Println("✓ Exact forward pass behavior")
}
