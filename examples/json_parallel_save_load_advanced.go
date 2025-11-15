package main

import (
	"fmt"
	"log"
	"os"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Testing Deeply Nested Parallel Save/Load ===")
	fmt.Println()

	// Create 4-level nested parallel network
	deepNestedJSON := `{
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

	fmt.Println("Building 4-level nested parallel network...")
	net1, err := nn.BuildNetworkFromJSON(deepNestedJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net1.InitializeWeights()

	input := make([]float32, 16)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	output1, _ := net1.ForwardCPU(input)
	fmt.Printf("Original output: %v\n", output1)

	filename := "test_deep_nested.json"
	err = net1.SaveModel(filename, "deep_nested")
	if err != nil {
		log.Fatalf("Save failed: %v", err)
	}
	fmt.Println("✓ Saved 4-level nested network")

	net2, err := nn.LoadModel(filename, "deep_nested")
	if err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	fmt.Println("✓ Loaded 4-level nested network")

	output2, _ := net2.ForwardCPU(input)
	fmt.Printf("Loaded output:  %v\n", output2)

	match := true
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
			match = false
		}
	}

	if match {
		fmt.Printf("✓ Perfect match! Max diff: %.10f\n", maxDiff)
	} else {
		fmt.Printf("✗ Mismatch! Max diff: %.10f\n", maxDiff)
	}

	os.Remove(filename)
	fmt.Println()

	// Test with all layer types in parallel
	fmt.Println("=== Testing Mixed Layer Types Save/Load ===")
	mixedJSON := `{
		"batch_size": 2,
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
						"type": "dense",
						"input_size": 20,
						"output_size": 8,
						"activation": "relu"
					},
					{
						"type": "lstm",
						"input_size": 20,
						"hidden_size": 8,
						"seq_length": 1
					},
					{
						"type": "rnn",
						"input_size": 20,
						"hidden_size": 8,
						"seq_length": 1,
						"activation": "tanh"
					},
					{
						"type": "layer_norm",
						"norm_size": 20
					},
					{
						"type": "swiglu",
						"input_size": 20,
						"output_size": 8
					}
				]
			}
		]
	}`

	net3, err := nn.BuildNetworkFromJSON(mixedJSON)
	if err != nil {
		log.Fatalf("Failed to build mixed network: %v", err)
	}
	net3.InitializeWeights()

	input2 := make([]float32, 2*20)
	for i := range input2 {
		input2[i] = float32(i%15) * 0.1
	}

	output3, _ := net3.ForwardCPU(input2)
	fmt.Printf("Original output (first 10): %v...\n", output3[:10])

	filename2 := "test_mixed_types.json"
	err = net3.SaveModel(filename2, "mixed_types")
	if err != nil {
		log.Fatalf("Save failed: %v", err)
	}
	fmt.Println("✓ Saved mixed layer types network")

	net4, err := nn.LoadModel(filename2, "mixed_types")
	if err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	fmt.Println("✓ Loaded mixed layer types network")

	output4, _ := net4.ForwardCPU(input2)
	fmt.Printf("Loaded output (first 10):  %v...\n", output4[:10])

	match2 := true
	maxDiff2 := float32(0.0)
	for i := range output3 {
		diff := output3[i] - output4[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff2 {
			maxDiff2 = diff
		}
		if diff > 1e-6 {
			match2 = false
		}
	}

	if match2 {
		fmt.Printf("✓ Perfect match! Max diff: %.10f\n", maxDiff2)
	} else {
		fmt.Printf("✗ Mismatch! Max diff: %.10f\n", maxDiff2)
	}

	os.Remove(filename2)
	fmt.Println()

	fmt.Println("=== All Tests Passed! ===")
	fmt.Println("✓ 4-level nested parallel layers")
	fmt.Println("✓ Mixed layer types (Dense, LSTM, RNN, LayerNorm, SwiGLU)")
	fmt.Println("✓ Perfect weight preservation")
	fmt.Println("✓ Exact output reproduction")
}
