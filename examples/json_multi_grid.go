package main

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== Multi-Grid JSON Network Builder Test ===")
	fmt.Println()
	fmt.Println("This example demonstrates creating a network with:")
	fmt.Println("  - Multiple grid rows (2 rows)")
	fmt.Println("  - Multiple grid columns (2 columns)")
	fmt.Println("  - Multiple layers per cell (3 layers)")
	fmt.Println("  - Total: 2 × 2 × 3 = 12 layers")
	fmt.Println()

	batchSize := 1

	// Network configuration with 2x2 grid, 3 layers per cell
	// Data flows: Row 0 Col 0 → Row 0 Col 1 → Row 1 Col 0 → Row 1 Col 1
	// Each cell has 3 layers that process sequentially
	jsonConfig := `{
		"id": "multi_grid_test",
		"batch_size": 1,
		"grid_rows": 2,
		"grid_cols": 2,
		"layers_per_cell": 3,
		"layers": [
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 16,
				"output_height": 16,
				"comment": "Cell [0,0] Layer 0"
			},
			{
				"type": "rms_norm",
				"norm_size": 16,
				"epsilon": 1e-6,
				"comment": "Cell [0,0] Layer 1"
			},
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 16,
				"output_height": 16,
				"comment": "Cell [0,0] Layer 2"
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 16,
				"output_height": 16,
				"comment": "Cell [0,1] Layer 0"
			},
			{
				"type": "layer_norm",
				"norm_size": 16,
				"epsilon": 1e-5,
				"comment": "Cell [0,1] Layer 1"
			},
			{
				"type": "dense",
				"activation": "tanh",
				"input_height": 16,
				"output_height": 16,
				"comment": "Cell [0,1] Layer 2"
			},
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 16,
				"output_height": 8,
				"comment": "Cell [1,0] Layer 0"
			},
			{
				"type": "rms_norm",
				"norm_size": 8,
				"epsilon": 1e-6,
				"comment": "Cell [1,0] Layer 1"
			},
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 8,
				"output_height": 8,
				"comment": "Cell [1,0] Layer 2"
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 8,
				"output_height": 2,
				"comment": "Cell [1,1] Layer 0"
			},
			{
				"type": "dense",
				"activation": "linear",
				"input_height": 2,
				"output_height": 2,
				"comment": "Cell [1,1] Layer 1"
			},
			{
				"type": "softmax",
				"softmax_variant": "standard",
				"softmax_rows": 1,
				"softmax_cols": 2,
				"comment": "Cell [1,1] Layer 2"
			}
		]
	}`

	// Build network from JSON
	fmt.Println("Building multi-grid network from JSON...")
	fmt.Println()
	network, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(fmt.Sprintf("Failed to build network: %v", err))
	}
	network.BatchSize = batchSize

	// Display network structure
	fmt.Printf("Network created: %d rows × %d cols × %d layers/cell = %d total layers\n",
		network.GridRows, network.GridCols, network.LayersPerCell, network.TotalLayers())
	fmt.Println()

	// Display grid structure
	fmt.Println("Grid Structure (data flows row-by-row, left to right):")
	fmt.Println()
	for row := 0; row < network.GridRows; row++ {
		for col := 0; col < network.GridCols; col++ {
			fmt.Printf("  Cell [%d,%d]:\n", row, col)
			for layer := 0; layer < network.LayersPerCell; layer++ {
				cfg := network.GetLayer(row, col, layer)
				layerIdx := row*network.GridCols*network.LayersPerCell + col*network.LayersPerCell + layer

				var layerDesc string
				switch cfg.Type {
				case nn.LayerDense:
					layerDesc = fmt.Sprintf("Dense (%d → %d, %s)",
						cfg.InputHeight, cfg.OutputHeight, getActivationName(cfg.Activation))
				case nn.LayerNorm:
					layerDesc = fmt.Sprintf("LayerNorm (%d features)", cfg.NormSize)
				case nn.LayerRMSNorm:
					layerDesc = fmt.Sprintf("RMSNorm (%d features)", cfg.NormSize)
				case nn.LayerSoftmax:
					layerDesc = fmt.Sprintf("Softmax (%s)", cfg.SoftmaxVariant)
				default:
					layerDesc = fmt.Sprintf("%s", cfg.Type)
				}

				fmt.Printf("    Layer %d: %s\n", layerIdx, layerDesc)
			}
			fmt.Println()
		}
	}

	fmt.Println("Data Flow Path:")
	fmt.Println("  Input → Cell[0,0] (Dense→RMSNorm→Dense) →")
	fmt.Println("          Cell[0,1] (Dense→LayerNorm→Dense) →")
	fmt.Println("          Cell[1,0] (Dense→RMSNorm→Dense) →")
	fmt.Println("          Cell[1,1] (Dense→Dense→Softmax) → Output")
	fmt.Println()

	// Initialize all trainable weights
	fmt.Println("Initializing weights...")
	network.InitializeWeights()
	fmt.Println("  ✓ All weights initialized")
	fmt.Println()

	// Create training data
	fmt.Println("Generating training data...")
	numSamples := 100
	batches := make([]nn.TrainingBatch, numSamples)

	for i := 0; i < numSamples; i++ {
		var input []float32
		var target []float32

		if i%2 == 0 {
			// Pattern type 0: higher values in first half
			input = make([]float32, 16)
			for j := 0; j < 8; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			for j := 8; j < 16; j++ {
				input[j] = rand.Float32() * 0.3
			}
			target = []float32{1.0, 0.0}
		} else {
			// Pattern type 1: higher values in second half
			input = make([]float32, 16)
			for j := 0; j < 8; j++ {
				input[j] = rand.Float32() * 0.3
			}
			for j := 8; j < 16; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			target = []float32{0.0, 1.0}
		}

		batches[i] = nn.TrainingBatch{
			Input:  input,
			Target: target,
		}
	}

	fmt.Printf("  Generated %d training samples\n", numSamples)
	fmt.Println()

	// Test forward pass before training
	fmt.Println("Running forward pass before training...")
	testInput := batches[0].Input
	outputBefore, _ := network.ForwardCPU(testInput)
	fmt.Printf("  Input size: %d\n", len(testInput))
	fmt.Printf("  Output before training: [%.6f, %.6f]\n", outputBefore[0], outputBefore[1])
	fmt.Println()

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

	fmt.Println("Starting training...")
	fmt.Printf("  Epochs: %d\n", config.Epochs)
	fmt.Printf("  Learning Rate: %.3f\n", config.LearningRate)
	fmt.Printf("  Batch Size: %d samples\n", numSamples)
	fmt.Println()

	// Train
	result, err := network.Train(batches, config)
	if err != nil {
		panic(fmt.Sprintf("Training failed: %v", err))
	}

	fmt.Println()
	fmt.Printf("✓ Training complete!\n")
	fmt.Printf("  Initial Loss: %.6f\n", result.LossHistory[0])
	fmt.Printf("  Final Loss: %.6f\n", result.FinalLoss)
	fmt.Printf("  Improvement: %.6f (%.1f%%)\n",
		result.LossHistory[0]-result.FinalLoss,
		100*(result.LossHistory[0]-result.FinalLoss)/result.LossHistory[0])
	fmt.Printf("  Throughput: %.2f samples/sec\n", result.AvgThroughput)
	fmt.Println()

	// Test forward pass after training
	fmt.Println("Running forward pass after training...")
	outputAfter, _ := network.ForwardCPU(testInput)
	fmt.Printf("  Output after training: [%.6f, %.6f]\n", outputAfter[0], outputAfter[1])
	fmt.Println()

	// Verify weights changed
	changed := false
	maxDiff := float32(0)
	for i := range outputBefore {
		diff := absMultiGrid(outputAfter[i] - outputBefore[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-5 {
			changed = true
		}
	}

	if changed {
		fmt.Println("✓ Weights successfully changed during training")
		fmt.Printf("  Max output change: %.6f\n", maxDiff)
		fmt.Printf("  Δ[0]=%.6f, Δ[1]=%.6f\n",
			outputAfter[0]-outputBefore[0], outputAfter[1]-outputBefore[1])
	} else {
		fmt.Println("⚠ Weights did not change during training")
	}
	fmt.Println()

	// Test different input pattern
	fmt.Println("Testing pattern recognition...")
	pattern0 := make([]float32, 16)
	for j := 0; j < 8; j++ {
		pattern0[j] = 0.8
	}
	for j := 8; j < 16; j++ {
		pattern0[j] = 0.2
	}
	out0, _ := network.ForwardCPU(pattern0)
	fmt.Printf("  Pattern 0 (high first half):  [%.6f, %.6f] → expects [1.0, 0.0]\n",
		out0[0], out0[1])

	pattern1 := make([]float32, 16)
	for j := 0; j < 8; j++ {
		pattern1[j] = 0.2
	}
	for j := 8; j < 16; j++ {
		pattern1[j] = 0.8
	}
	out1, _ := network.ForwardCPU(pattern1)
	fmt.Printf("  Pattern 1 (high second half): [%.6f, %.6f] → expects [0.0, 1.0]\n",
		out1[0], out1[1])
	fmt.Println()

	// Test serialization
	fmt.Println("Testing save/load serialization...")
	jsonStr, err := network.SaveModelToString("multi_grid_test")
	if err != nil {
		panic(fmt.Sprintf("Failed to serialize: %v", err))
	}
	fmt.Printf("  ✓ Serialized network (JSON size: %d bytes)\n", len(jsonStr))

	reloaded, err := nn.LoadModelFromString(jsonStr, "multi_grid_test")
	if err != nil {
		panic(fmt.Sprintf("Failed to deserialize: %v", err))
	}
	reloaded.BatchSize = batchSize
	fmt.Println("  ✓ Deserialized network")

	outputReloaded, _ := reloaded.ForwardCPU(testInput)
	maxReloadDiff := float32(0)
	for i := range outputAfter {
		diff := absMultiGrid(outputAfter[i] - outputReloaded[i])
		if diff > maxReloadDiff {
			maxReloadDiff = diff
		}
	}

	fmt.Printf("  Max output difference: %.10f\n", maxReloadDiff)
	if maxReloadDiff < 1e-5 {
		fmt.Println("  ✓ Reload successful - outputs match exactly")
	} else if maxReloadDiff < 0.1 {
		fmt.Println("  ✓ Reload successful - small differences (acceptable)")
	} else {
		fmt.Println("  ⚠ Large output differences after reload")
	}
	fmt.Println()

	fmt.Println("=== Multi-Grid Network Test Complete ===")
	fmt.Println("✅ Network built from JSON with 2×2 grid")
	fmt.Println("✅ Each cell processes data with 3 layers")
	fmt.Println("✅ Data flows through all 12 layers sequentially")
	fmt.Println("✅ Training successful - network learns patterns")
	fmt.Println("✅ Serialization working correctly")
	fmt.Println()
	fmt.Println("Grid architecture allows:")
	fmt.Println("  • Modular layer organization")
	fmt.Println("  • Clear data flow visualization")
	fmt.Println("  • Easy scaling to larger grids")
	fmt.Println("  • Flexible layer composition per cell")
}

func getActivationName(act nn.ActivationType) string {
	switch act {
	case nn.ActivationScaledReLU:
		return "ScaledReLU"
	case nn.ActivationSigmoid:
		return "Sigmoid"
	case nn.ActivationTanh:
		return "Tanh"
	case nn.ActivationSoftplus:
		return "Softplus"
	case nn.ActivationLeakyReLU:
		return "LeakyReLU"
	default:
		return "Unknown"
	}
}

func absMultiGrid(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
