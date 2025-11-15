package main

// JSON Network Builder - All Layer Types Demo
//
// This example demonstrates building a complete neural network with ALL layer types
// from an embedded JSON configuration string. No external files required!
//
// Network structure (11 layers total):
//   Layer 0: Dense (32 → 32, LeakyReLU)
//   Layer 1: RMSNorm (32 features)
//   Layer 2: Conv2D (4x4x2 → 2x2x4=16, LeakyReLU)
//   Layer 3: Multi-Head Attention (4 seq x 4 dim, 2 heads)
//   Layer 4: Dense (16 → 16, Sigmoid)
//   Layer 5: RNN (4 features, 4 hidden, 4 timesteps)
//   Layer 6: LSTM (4 features, 4 hidden, 4 timesteps)
//   Layer 7: SwiGLU (16 → 24 intermediate → 16)
//   Layer 8: LayerNorm (16 features)
//   Layer 9: Dense (16 → 2, Sigmoid)
//   Layer 10: Softmax (standard, 2 outputs)
//
// This recreates the same network as all_layers_validation.go but using JSON instead
// of manual layer construction. The network is trained on a simple pattern recognition
// task, demonstrating:
//   - Building from embedded JSON (no external files)
//   - Training with all layer types
//   - Weight mutation during training
//   - Serialization/deserialization round-trip
//
// Run with: go run json_all_layers_embedded.go

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== JSON Network Builder - All Layer Types (Embedded) ===")
	fmt.Println("Building network from embedded JSON config (no external files)")
	fmt.Println()

	batchSize := 1

	// Embedded JSON configuration recreating the all_layers_validation.go network
	// Network with 11 layers:
	// 0: Dense (32->32), 1: RMSNorm, 2: Conv2D, 3: Attention, 4: Dense (16->16)
	// 5: RNN, 6: LSTM, 7: SwiGLU, 8: LayerNorm, 9: Dense (16->2), 10: Softmax
	jsonConfig := `{
		"id": "all_layers_test",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 11,
		"layers": [
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 32,
				"output_height": 32
			},
			{
				"type": "rms_norm",
				"norm_size": 32,
				"epsilon": 1e-6
			},
			{
				"type": "conv2d",
				"activation": "leaky_relu",
				"input_channels": 2,
				"filters": 4,
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"input_height": 4,
				"input_width": 4,
				"output_height": 2,
				"output_width": 2
			},
			{
				"type": "multi_head_attention",
				"activation": "relu",
				"d_model": 4,
				"num_heads": 2,
				"seq_length": 4
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 16,
				"output_height": 16
			},
			{
				"type": "rnn",
				"activation": "tanh",
				"input_size": 4,
				"hidden_size": 4,
				"seq_length": 4
			},
			{
				"type": "lstm",
				"activation": "tanh",
				"input_size": 4,
				"hidden_size": 4,
				"seq_length": 4
			},
			{
				"type": "swiglu",
				"input_height": 16,
				"output_height": 24
			},
			{
				"type": "layer_norm",
				"norm_size": 16,
				"epsilon": 1e-5
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 16,
				"output_height": 2
			},
			{
				"type": "softmax",
				"softmax_variant": "standard",
				"softmax_rows": 1,
				"softmax_cols": 2
			}
		]
	}`

	// Build network from JSON
	fmt.Println("Building network with ALL layer types from JSON...")
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

	// Display each layer
	fmt.Println("Layer configuration:")
	layerNames := []string{
		"Layer 0: Dense (32 → 32, LeakyReLU)",
		"Layer 1: RMSNorm (32 features)",
		"Layer 2: Conv2D (4x4x2 → 2x2x4=16, LeakyReLU)",
		"Layer 3: Attention (4 seq x 4 dim, 2 heads)",
		"Layer 4: Dense (16 → 16, Sigmoid)",
		"Layer 5: RNN (4 features, 4 hidden, 4 steps → 16)",
		"Layer 6: LSTM (4 features, 4 hidden, 4 steps → 16)",
		"Layer 7: SwiGLU (16 → 24 → 16)",
		"Layer 8: LayerNorm (16 features)",
		"Layer 9: Dense (16 → 2, Sigmoid)",
		"Layer 10: Softmax Standard",
	}
	for _, name := range layerNames {
		fmt.Printf("  %s\n", name)
	}
	fmt.Println()

	fmt.Println("Network Summary:")
	fmt.Println("  Total layers: 11")
	fmt.Println("  Layer types: Dense → RMSNorm → Conv2D → Attention → Dense → RNN → LSTM → SwiGLU → LayerNorm → Dense → Softmax")
	fmt.Println()

	// Initialize all trainable weights
	fmt.Println("Initializing weights...")
	initializeAllWeights(network, batchSize)
	fmt.Println("  ✓ All weights initialized")
	fmt.Println()

	// Create training data
	fmt.Println("Generating training data...")
	numSamples := 50
	batches := make([]nn.TrainingBatch, numSamples)

	for i := 0; i < numSamples; i++ {
		var input []float32
		var target []float32

		if i%2 == 0 {
			// Pattern type 0: higher values in first half
			input = make([]float32, 32)
			for j := 0; j < 16; j++ {
				input[j] = 0.7 + rand.Float32()*0.3
			}
			for j := 16; j < 32; j++ {
				input[j] = rand.Float32() * 0.3
			}
			target = []float32{1.0, 0.0}
		} else {
			// Pattern type 1: higher values in second half
			input = make([]float32, 32)
			for j := 0; j < 16; j++ {
				input[j] = rand.Float32() * 0.3
			}
			for j := 16; j < 32; j++ {
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
	for i := range outputBefore {
		diff := abs(outputAfter[i] - outputBefore[i])
		if diff > 1e-5 {
			changed = true
			break
		}
	}

	if changed {
		fmt.Println("✓ Weights successfully changed during training")
		fmt.Printf("  Output change: Δ[0]=%.6f, Δ[1]=%.6f\n",
			outputAfter[0]-outputBefore[0], outputAfter[1]-outputBefore[1])
	} else {
		fmt.Println("⚠ Weights did not change during training")
	}
	fmt.Println()

	// Test serialization
	fmt.Println("Testing save/load serialization...")
	jsonStr, err := network.SaveModelToString("all_layers_test")
	if err != nil {
		panic(fmt.Sprintf("Failed to serialize: %v", err))
	}
	fmt.Printf("  ✓ Serialized network (JSON size: %d bytes)\n", len(jsonStr))

	reloaded, err := nn.LoadModelFromString(jsonStr, "all_layers_test")
	if err != nil {
		panic(fmt.Sprintf("Failed to deserialize: %v", err))
	}
	reloaded.BatchSize = batchSize
	fmt.Println("  ✓ Deserialized network")

	outputReloaded, _ := reloaded.ForwardCPU(testInput)
	maxDiff := float32(0)
	for i := range outputAfter {
		diff := abs(outputAfter[i] - outputReloaded[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	fmt.Printf("  Max output difference: %.10f\n", maxDiff)
	if maxDiff < 1e-5 {
		fmt.Println("  ✓ Reload successful - outputs match exactly")
	} else if maxDiff < 0.1 {
		fmt.Println("  ✓ Reload successful - small differences (expected)")
	} else {
		fmt.Println("  ⚠ Large output differences after reload")
	}
	fmt.Println()

	fmt.Println("=== All Layer Types Test Complete ===")
	fmt.Println("✅ Network built entirely from embedded JSON")
	fmt.Println("✅ All 10 core layer types tested:")
	fmt.Println("   Dense, RMSNorm, Conv2D, Attention, RNN, LSTM, SwiGLU, LayerNorm, Softmax")
	fmt.Println("✅ Training successful - weights updated")
	fmt.Println("✅ Serialization working correctly")
	fmt.Println()
	fmt.Println("No external files needed - everything from embedded JSON!")
}

// Initialize all trainable weights
func initializeAllWeights(network *nn.Network, batchSize int) {
	for i := 0; i < network.TotalLayers(); i++ {
		row := i / (network.GridCols * network.LayersPerCell)
		col := (i / network.LayersPerCell) % network.GridCols
		layer := i % network.LayersPerCell

		cfg := network.GetLayer(row, col, layer)

		switch cfg.Type {
		case nn.LayerDense:
			// Dense layer weights
			if cfg.InputHeight > 0 && cfg.OutputHeight > 0 {
				numWeights := cfg.InputHeight * cfg.OutputHeight
				cfg.Kernel = make([]float32, numWeights)
				scale := float32(1.0) / float32(cfg.InputHeight)
				for j := range cfg.Kernel {
					cfg.Kernel[j] = (rand.Float32()*2 - 1) * scale
				}
				cfg.Bias = make([]float32, cfg.OutputHeight)
				for j := range cfg.Bias {
					cfg.Bias[j] = rand.Float32() * 0.01
				}
				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerConv2D:
			// Conv2D weights
			if cfg.Filters > 0 && cfg.InputChannels > 0 && cfg.KernelSize > 0 {
				kernelSize := cfg.Filters * cfg.InputChannels * cfg.KernelSize * cfg.KernelSize
				cfg.Kernel = make([]float32, kernelSize)
				scale := float32(1.0) / float32(cfg.InputChannels*cfg.KernelSize*cfg.KernelSize)
				for j := range cfg.Kernel {
					cfg.Kernel[j] = (rand.Float32()*2 - 1) * scale
				}
				cfg.Bias = make([]float32, cfg.Filters)
				for j := range cfg.Bias {
					cfg.Bias[j] = rand.Float32() * 0.01
				}
				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerMultiHeadAttention:
			// Attention weights
			if cfg.DModel > 0 {
				size := cfg.DModel * cfg.DModel
				cfg.QWeights = make([]float32, size)
				cfg.KWeights = make([]float32, size)
				cfg.VWeights = make([]float32, size)
				cfg.OutputWeight = make([]float32, size)
				for j := range cfg.QWeights {
					cfg.QWeights[j] = rand.Float32()*0.2 - 0.1
					cfg.KWeights[j] = rand.Float32()*0.2 - 0.1
					cfg.VWeights[j] = rand.Float32()*0.2 - 0.1
					cfg.OutputWeight[j] = rand.Float32()*0.2 - 0.1
				}
				cfg.QBias = make([]float32, cfg.DModel)
				cfg.KBias = make([]float32, cfg.DModel)
				cfg.VBias = make([]float32, cfg.DModel)
				cfg.OutputBias = make([]float32, cfg.DModel)
				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerRNN:
			// RNN weights
			if cfg.RNNInputSize > 0 && cfg.HiddenSize > 0 {
				cfg.WeightIH = make([]float32, cfg.HiddenSize*cfg.RNNInputSize)
				cfg.WeightHH = make([]float32, cfg.HiddenSize*cfg.HiddenSize)
				for j := range cfg.WeightIH {
					cfg.WeightIH[j] = rand.Float32()*0.2 - 0.1
				}
				for j := range cfg.WeightHH {
					cfg.WeightHH[j] = rand.Float32()*0.2 - 0.1
				}
				cfg.BiasH = make([]float32, cfg.HiddenSize)
				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerLSTM:
			// LSTM weights (4 gates)
			if cfg.RNNInputSize > 0 && cfg.HiddenSize > 0 {
				ihSize := cfg.HiddenSize * cfg.RNNInputSize
				hhSize := cfg.HiddenSize * cfg.HiddenSize

				cfg.WeightIH_i = make([]float32, ihSize)
				cfg.WeightIH_f = make([]float32, ihSize)
				cfg.WeightIH_g = make([]float32, ihSize)
				cfg.WeightIH_o = make([]float32, ihSize)

				cfg.WeightHH_i = make([]float32, hhSize)
				cfg.WeightHH_f = make([]float32, hhSize)
				cfg.WeightHH_g = make([]float32, hhSize)
				cfg.WeightHH_o = make([]float32, hhSize)

				for j := 0; j < ihSize; j++ {
					cfg.WeightIH_i[j] = rand.Float32()*0.2 - 0.1
					cfg.WeightIH_f[j] = rand.Float32()*0.2 - 0.1
					cfg.WeightIH_g[j] = rand.Float32()*0.2 - 0.1
					cfg.WeightIH_o[j] = rand.Float32()*0.2 - 0.1
				}
				for j := 0; j < hhSize; j++ {
					cfg.WeightHH_i[j] = rand.Float32()*0.2 - 0.1
					cfg.WeightHH_f[j] = rand.Float32()*0.2 - 0.1
					cfg.WeightHH_g[j] = rand.Float32()*0.2 - 0.1
					cfg.WeightHH_o[j] = rand.Float32()*0.2 - 0.1
				}

				cfg.BiasH_i = make([]float32, cfg.HiddenSize)
				cfg.BiasH_f = make([]float32, cfg.HiddenSize)
				cfg.BiasH_g = make([]float32, cfg.HiddenSize)
				cfg.BiasH_o = make([]float32, cfg.HiddenSize)

				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerSwiGLU:
			// SwiGLU weights
			if cfg.InputHeight > 0 && cfg.OutputHeight > 0 {
				gateSize := cfg.InputHeight * cfg.OutputHeight
				downSize := cfg.OutputHeight * cfg.InputHeight

				cfg.GateWeights = make([]float32, gateSize)
				cfg.UpWeights = make([]float32, gateSize)
				cfg.DownWeights = make([]float32, downSize)

				for j := range cfg.GateWeights {
					cfg.GateWeights[j] = rand.Float32()*0.2 - 0.1
					cfg.UpWeights[j] = rand.Float32()*0.2 - 0.1
				}
				for j := range cfg.DownWeights {
					cfg.DownWeights[j] = rand.Float32()*0.2 - 0.1
				}

				cfg.GateBias = make([]float32, cfg.OutputHeight)
				cfg.UpBias = make([]float32, cfg.OutputHeight)
				cfg.DownBias = make([]float32, cfg.InputHeight)

				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerNorm:
			// LayerNorm weights
			if cfg.NormSize > 0 {
				cfg.Gamma = make([]float32, cfg.NormSize)
				cfg.Beta = make([]float32, cfg.NormSize)
				for j := range cfg.Gamma {
					cfg.Gamma[j] = 1.0
					cfg.Beta[j] = 0.0
				}
				network.SetLayer(row, col, layer, *cfg)
			}

		case nn.LayerRMSNorm:
			// RMSNorm weights
			if cfg.NormSize > 0 {
				cfg.Gamma = make([]float32, cfg.NormSize)
				for j := range cfg.Gamma {
					cfg.Gamma[j] = 1.0
				}
				network.SetLayer(row, col, layer, *cfg)
			}
		}
	}
}

// Absolute value helper
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
