package main

import (
	"fmt"
	"log"

	"github.com/openfluke/loom/nn"
)

// Simple example showing how to use JSON configuration files to build networks
func main() {
	fmt.Println("=== JSON Network Builder - Simple Usage Example ===\n")

	// Example 1: Build from JSON file
	fmt.Println("1. Loading network from JSON file...")
	network1, err := nn.BuildNetworkFromFile("example_network_config.json")
	if err != nil {
		log.Fatalf("Failed to load network from file: %v", err)
	}
	fmt.Printf("   ✓ Loaded network: %d rows × %d cols × %d layers/cell = %d total layers\n",
		network1.GridRows, network1.GridCols, network1.LayersPerCell, network1.TotalLayers())

	// Example 2: Build from JSON string
	fmt.Println("\n2. Building network from JSON string...")
	jsonConfig := `{
		"id": "inline_network",
		"batch_size": 16,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 100,
				"output_height": 50
			},
			{
				"type": "softmax",
				"softmax_variant": "standard",
				"softmax_rows": 1,
				"softmax_cols": 10
			}
		]
	}`

	network2, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		log.Fatalf("Failed to build network from JSON: %v", err)
	}
	fmt.Printf("   ✓ Built network: %d total layers\n", network2.TotalLayers())

	// Example 3: Inspect layer configurations
	fmt.Println("\n3. Inspecting layer configurations...")
	for i := 0; i < network1.TotalLayers(); i++ {
		row := i / (network1.GridCols * network1.LayersPerCell)
		col := (i / network1.LayersPerCell) % network1.GridCols
		layer := i % network1.LayersPerCell

		cfg := network1.GetLayer(row, col, layer)

		var layerType string
		switch cfg.Type {
		case nn.LayerDense:
			layerType = "Dense"
		case nn.LayerConv2D:
			layerType = "Conv2D"
		case nn.LayerMultiHeadAttention:
			layerType = "Multi-Head Attention"
		case nn.LayerRNN:
			layerType = "RNN"
		case nn.LayerLSTM:
			layerType = "LSTM"
		case nn.LayerSoftmax:
			layerType = "Softmax"
		case nn.LayerNorm:
			layerType = "LayerNorm"
		case nn.LayerRMSNorm:
			layerType = "RMSNorm"
		case nn.LayerSwiGLU:
			layerType = "SwiGLU"
		default:
			layerType = "Unknown"
		}

		var activation string
		switch cfg.Activation {
		case nn.ActivationScaledReLU:
			activation = "ReLU"
		case nn.ActivationSigmoid:
			activation = "Sigmoid"
		case nn.ActivationTanh:
			activation = "Tanh"
		case nn.ActivationSoftplus:
			activation = "Softplus"
		case nn.ActivationLeakyReLU:
			activation = "LeakyReLU"
		default:
			activation = "Unknown"
		}

		fmt.Printf("   Layer [%d,%d,%d]: %s (activation: %s)\n", row, col, layer, layerType, activation)
	}

	// Example 4: Save network back to JSON
	fmt.Println("\n4. Saving network to JSON...")
	err = network1.SaveModel("saved_network.json", "my_network")
	if err != nil {
		log.Fatalf("Failed to save network: %v", err)
	}
	fmt.Println("   ✓ Saved to saved_network.json")

	// Example 5: Load it back
	fmt.Println("\n5. Loading saved network...")
	network3, err := nn.LoadModel("saved_network.json", "my_network")
	if err != nil {
		log.Fatalf("Failed to load saved network: %v", err)
	}
	fmt.Printf("   ✓ Loaded network: %d total layers\n", network3.TotalLayers())

	fmt.Println("\n=== Usage Example Complete ===")
	fmt.Println("\nTry these example config files:")
	fmt.Println("  - example_network_config.json     (Mixed layer types)")
	fmt.Println("  - advanced_network_config.json    (Conv2D, Attention, LSTM, RNN)")
	fmt.Println("  - softmax_variants_config.json    (All 10 softmax variants)")
}
