package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== JSON Network Builder Test ===")
	fmt.Println("Testing all layer types with JSON-based network creation\n")

	// Test 1: Dense Network
	fmt.Println("1. Testing Dense Network")
	testDenseNetwork()

	// Test 2: Conv2D Network
	fmt.Println("\n2. Testing Conv2D Network")
	testConv2DNetwork()

	// Test 3: Multi-Head Attention Network
	fmt.Println("\n3. Testing Multi-Head Attention Network")
	testMHANetwork()

	// Test 4: RNN Network
	fmt.Println("\n4. Testing RNN Network")
	testRNNNetwork()

	// Test 5: LSTM Network
	fmt.Println("\n5. Testing LSTM Network")
	testLSTMNetwork()

	// Test 6: Softmax Variants Network
	fmt.Println("\n6. Testing Softmax Variants Network")
	testSoftmaxNetwork()

	// Test 7: Normalization Layers Network
	fmt.Println("\n7. Testing Normalization Layers Network")
	testNormalizationNetwork()

	// Test 8: SwiGLU Network
	fmt.Println("\n8. Testing SwiGLU Network")
	testSwiGLUNetwork()

	// Test 9: Mixed Layer Types Network (all in one)
	fmt.Println("\n9. Testing Mixed Layer Types Network")
	testMixedNetwork()

	// Test 10: Serialization Round-trip
	fmt.Println("\n10. Testing Serialization Round-trip")
	testSerializationRoundtrip()

	fmt.Println("\n=== All Tests Completed Successfully ===")
}

// Test 1: Dense Network
func testDenseNetwork() {
	config := `{
		"id": "dense_network",
		"batch_size": 32,
		"grid_rows": 2,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 128,
				"output_height": 64
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 64,
				"output_height": 32
			},
			{
				"type": "dense",
				"activation": "tanh",
				"input_height": 32,
				"output_height": 16
			},
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 16,
				"output_height": 10
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build dense network: %v", err)
	}

	fmt.Printf("   ✓ Created dense network: %d rows × %d cols × %d layers/cell = %d total layers\n",
		network.GridRows, network.GridCols, network.LayersPerCell, network.TotalLayers())

	// Verify layer configurations
	for i := 0; i < network.TotalLayers(); i++ {
		row := i / (network.GridCols * network.LayersPerCell)
		col := (i / network.LayersPerCell) % network.GridCols
		layer := i % network.LayersPerCell
		cfg := network.GetLayer(row, col, layer)
		if cfg.Type != nn.LayerDense {
			log.Fatalf("Expected dense layer at position %d, got type %d", i, cfg.Type)
		}
	}
	fmt.Println("   ✓ All layers configured correctly as dense")
}

// Test 2: Conv2D Network
func testConv2DNetwork() {
	config := `{
		"id": "conv2d_network",
		"batch_size": 16,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "conv2d",
				"activation": "relu",
				"input_channels": 3,
				"filters": 32,
				"kernel_size": 3,
				"stride": 1,
				"padding": 1,
				"input_height": 28,
				"input_width": 28,
				"output_height": 28,
				"output_width": 28
			},
			{
				"type": "conv2d",
				"activation": "relu",
				"input_channels": 32,
				"filters": 64,
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"input_height": 28,
				"input_width": 28,
				"output_height": 14,
				"output_width": 14
			},
			{
				"type": "conv2d",
				"activation": "relu",
				"input_channels": 64,
				"filters": 128,
				"kernel_size": 3,
				"stride": 2,
				"padding": 1,
				"input_height": 14,
				"input_width": 14,
				"output_height": 7,
				"output_width": 7
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build conv2d network: %v", err)
	}

	fmt.Printf("   ✓ Created Conv2D network: %d total layers\n", network.TotalLayers())

	// Verify first layer
	cfg := network.GetLayer(0, 0, 0)
	if cfg.Type != nn.LayerConv2D {
		log.Fatalf("Expected Conv2D layer, got type %d", cfg.Type)
	}
	if cfg.Filters != 32 || cfg.KernelSize != 3 {
		log.Fatalf("Conv2D layer params incorrect: filters=%d, kernel=%d", cfg.Filters, cfg.KernelSize)
	}
	fmt.Println("   ✓ Conv2D layers configured correctly")
}

// Test 3: Multi-Head Attention Network
func testMHANetwork() {
	config := `{
		"id": "mha_network",
		"batch_size": 8,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "multi_head_attention",
				"activation": "relu",
				"d_model": 512,
				"num_heads": 8,
				"seq_length": 64
			},
			{
				"type": "multi_head_attention",
				"activation": "relu",
				"d_model": 512,
				"num_heads": 8,
				"seq_length": 64
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build MHA network: %v", err)
	}

	fmt.Printf("   ✓ Created MHA network: %d total layers\n", network.TotalLayers())

	cfg := network.GetLayer(0, 0, 0)
	if cfg.Type != nn.LayerMultiHeadAttention {
		log.Fatalf("Expected MHA layer, got type %d", cfg.Type)
	}
	if cfg.DModel != 512 || cfg.NumHeads != 8 {
		log.Fatalf("MHA params incorrect: d_model=%d, num_heads=%d", cfg.DModel, cfg.NumHeads)
	}
	fmt.Println("   ✓ MHA layers configured correctly")
}

// Test 4: RNN Network
func testRNNNetwork() {
	config := `{
		"id": "rnn_network",
		"batch_size": 32,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "rnn",
				"activation": "tanh",
				"input_size": 100,
				"hidden_size": 256,
				"seq_length": 50
			},
			{
				"type": "rnn",
				"activation": "tanh",
				"input_size": 256,
				"hidden_size": 128,
				"seq_length": 50
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build RNN network: %v", err)
	}

	fmt.Printf("   ✓ Created RNN network: %d total layers\n", network.TotalLayers())

	cfg := network.GetLayer(0, 0, 0)
	if cfg.Type != nn.LayerRNN {
		log.Fatalf("Expected RNN layer, got type %d", cfg.Type)
	}
	if cfg.RNNInputSize != 100 || cfg.HiddenSize != 256 {
		log.Fatalf("RNN params incorrect: input_size=%d, hidden_size=%d", cfg.RNNInputSize, cfg.HiddenSize)
	}
	fmt.Println("   ✓ RNN layers configured correctly")
}

// Test 5: LSTM Network
func testLSTMNetwork() {
	config := `{
		"id": "lstm_network",
		"batch_size": 32,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "lstm",
				"activation": "tanh",
				"input_size": 100,
				"hidden_size": 256,
				"seq_length": 50
			},
			{
				"type": "lstm",
				"activation": "tanh",
				"input_size": 256,
				"hidden_size": 128,
				"seq_length": 50
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build LSTM network: %v", err)
	}

	fmt.Printf("   ✓ Created LSTM network: %d total layers\n", network.TotalLayers())

	cfg := network.GetLayer(0, 0, 0)
	if cfg.Type != nn.LayerLSTM {
		log.Fatalf("Expected LSTM layer, got type %d", cfg.Type)
	}
	if cfg.RNNInputSize != 100 || cfg.HiddenSize != 256 {
		log.Fatalf("LSTM params incorrect: input_size=%d, hidden_size=%d", cfg.RNNInputSize, cfg.HiddenSize)
	}
	fmt.Println("   ✓ LSTM layers configured correctly")
}

// Test 6: Softmax Variants Network
func testSoftmaxNetwork() {
	config := `{
		"id": "softmax_network",
		"batch_size": 32,
		"grid_rows": 2,
		"grid_cols": 5,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "softmax",
				"softmax_variant": "standard",
				"softmax_rows": 1,
				"softmax_cols": 10
			},
			{
				"type": "softmax",
				"softmax_variant": "grid",
				"softmax_rows": 4,
				"softmax_cols": 8
			},
			{
				"type": "softmax",
				"softmax_variant": "hierarchical",
				"hierarchy_levels": [5, 10, 20]
			},
			{
				"type": "softmax",
				"softmax_variant": "temperature",
				"softmax_rows": 1,
				"softmax_cols": 10,
				"temperature": 0.7
			},
			{
				"type": "softmax",
				"softmax_variant": "gumbel",
				"softmax_rows": 1,
				"softmax_cols": 10,
				"temperature": 1.0,
				"gumbel_noise": true
			},
			{
				"type": "softmax",
				"softmax_variant": "masked",
				"softmax_rows": 1,
				"softmax_cols": 10,
				"mask": [true, true, true, false, false, true, true, true, true, true]
			},
			{
				"type": "softmax",
				"softmax_variant": "sparse",
				"softmax_rows": 1,
				"softmax_cols": 10
			},
			{
				"type": "softmax",
				"softmax_variant": "adaptive",
				"adaptive_clusters": [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
			},
			{
				"type": "softmax",
				"softmax_variant": "mixture",
				"softmax_rows": 1,
				"softmax_cols": 10,
				"mixture_weights": [0.3, 0.5, 0.2]
			},
			{
				"type": "softmax",
				"softmax_variant": "entmax",
				"softmax_rows": 1,
				"softmax_cols": 10,
				"entmax_alpha": 1.5
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build softmax network: %v", err)
	}

	fmt.Printf("   ✓ Created Softmax network: %d total layers\n", network.TotalLayers())

	// Verify different softmax variants
	variants := []struct {
		row, col, layer int
		expectedVariant nn.SoftmaxType
	}{
		{0, 0, 0, nn.SoftmaxStandard},
		{0, 1, 0, nn.SoftmaxGrid},
		{0, 2, 0, nn.SoftmaxHierarchical},
		{0, 3, 0, nn.SoftmaxTemperature},
		{0, 4, 0, nn.SoftmaxGumbel},
		{1, 0, 0, nn.SoftmaxMasked},
		{1, 1, 0, nn.SoftmaxSparse},
		{1, 2, 0, nn.SoftmaxAdaptive},
		{1, 3, 0, nn.SoftmaxMixture},
		{1, 4, 0, nn.SoftmaxEntmax},
	}

	for _, v := range variants {
		cfg := network.GetLayer(v.row, v.col, v.layer)
		if cfg.Type != nn.LayerSoftmax {
			log.Fatalf("Expected Softmax layer at (%d,%d,%d), got type %d", v.row, v.col, v.layer, cfg.Type)
		}
		if cfg.SoftmaxVariant != v.expectedVariant {
			log.Fatalf("Expected variant %d at (%d,%d,%d), got %d", v.expectedVariant, v.row, v.col, v.layer, cfg.SoftmaxVariant)
		}
	}
	fmt.Println("   ✓ All softmax variants configured correctly")
}

// Test 7: Normalization Layers Network
func testNormalizationNetwork() {
	config := `{
		"id": "norm_network",
		"batch_size": 32,
		"grid_rows": 1,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "layer_norm",
				"norm_size": 512,
				"epsilon": 1e-5
			},
			{
				"type": "rms_norm",
				"norm_size": 512,
				"epsilon": 1e-6
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build normalization network: %v", err)
	}

	fmt.Printf("   ✓ Created Normalization network: %d total layers\n", network.TotalLayers())

	// Verify LayerNorm
	cfg := network.GetLayer(0, 0, 0)
	if cfg.Type != nn.LayerNorm {
		log.Fatalf("Expected LayerNorm, got type %d", cfg.Type)
	}
	if cfg.NormSize != 512 {
		log.Fatalf("LayerNorm size incorrect: %d", cfg.NormSize)
	}

	// Verify RMSNorm
	cfg = network.GetLayer(0, 1, 0)
	if cfg.Type != nn.LayerRMSNorm {
		log.Fatalf("Expected RMSNorm, got type %d", cfg.Type)
	}
	if cfg.NormSize != 512 {
		log.Fatalf("RMSNorm size incorrect: %d", cfg.NormSize)
	}

	fmt.Println("   ✓ Normalization layers configured correctly")
}

// Test 8: SwiGLU Network
func testSwiGLUNetwork() {
	config := `{
		"id": "swiglu_network",
		"batch_size": 32,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "swiglu",
				"input_height": 512,
				"output_height": 2048
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build SwiGLU network: %v", err)
	}

	fmt.Printf("   ✓ Created SwiGLU network: %d total layers\n", network.TotalLayers())

	cfg := network.GetLayer(0, 0, 0)
	if cfg.Type != nn.LayerSwiGLU {
		log.Fatalf("Expected SwiGLU layer, got type %d", cfg.Type)
	}
	if cfg.InputHeight != 512 || cfg.OutputHeight != 2048 {
		log.Fatalf("SwiGLU params incorrect: input=%d, output=%d", cfg.InputHeight, cfg.OutputHeight)
	}
	fmt.Println("   ✓ SwiGLU layer configured correctly")
}

// Test 9: Mixed Layer Types Network
func testMixedNetwork() {
	config := `{
		"id": "mixed_network",
		"batch_size": 16,
		"grid_rows": 2,
		"grid_cols": 4,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 784,
				"output_height": 512
			},
			{
				"type": "layer_norm",
				"norm_size": 512
			},
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 512,
				"output_height": 256
			},
			{
				"type": "rms_norm",
				"norm_size": 256
			},
			{
				"type": "swiglu",
				"input_height": 256,
				"output_height": 1024
			},
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 256,
				"output_height": 128
			},
			{
				"type": "dense",
				"activation": "tanh",
				"input_height": 128,
				"output_height": 64
			},
			{
				"type": "softmax",
				"softmax_variant": "standard",
				"softmax_rows": 1,
				"softmax_cols": 10
			}
		]
	}`

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		log.Fatalf("Failed to build mixed network: %v", err)
	}

	fmt.Printf("   ✓ Created Mixed network: %d rows × %d cols × %d layers/cell = %d total layers\n",
		network.GridRows, network.GridCols, network.LayersPerCell, network.TotalLayers())

	// Verify layer types are diverse
	expectedTypes := []nn.LayerType{
		nn.LayerDense,
		nn.LayerNorm,
		nn.LayerDense,
		nn.LayerRMSNorm,
		nn.LayerSwiGLU,
		nn.LayerDense,
		nn.LayerDense,
		nn.LayerSoftmax,
	}

	for i, expectedType := range expectedTypes {
		row := i / (network.GridCols * network.LayersPerCell)
		col := (i / network.LayersPerCell) % network.GridCols
		layer := i % network.LayersPerCell
		cfg := network.GetLayer(row, col, layer)
		if cfg.Type != expectedType {
			log.Fatalf("Expected type %d at position %d, got %d", expectedType, i, cfg.Type)
		}
	}

	fmt.Println("   ✓ All mixed layer types configured correctly")
}

// Test 10: Serialization Round-trip
func testSerializationRoundtrip() {
	// Create a network from JSON
	originalConfig := `{
		"id": "roundtrip_test",
		"batch_size": 32,
		"grid_rows": 1,
		"grid_cols": 3,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"activation": "relu",
				"input_height": 128,
				"output_height": 64
			},
			{
				"type": "layer_norm",
				"norm_size": 64,
				"epsilon": 1e-5
			},
			{
				"type": "softmax",
				"softmax_variant": "temperature",
				"softmax_rows": 1,
				"softmax_cols": 10,
				"temperature": 0.8
			}
		]
	}`

	network1, err := nn.BuildNetworkFromJSON(originalConfig)
	if err != nil {
		log.Fatalf("Failed to build network for round-trip: %v", err)
	}

	fmt.Println("   ✓ Created network from JSON")

	// Serialize to JSON string
	jsonStr, err := network1.SaveModelToString("roundtrip_test")
	if err != nil {
		log.Fatalf("Failed to serialize network: %v", err)
	}

	fmt.Println("   ✓ Serialized network to JSON")

	// Pretty print for inspection
	var prettyJSON map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &prettyJSON); err != nil {
		log.Fatalf("Failed to parse JSON: %v", err)
	}

	// Deserialize back
	network2, err := nn.LoadModelFromString(jsonStr, "roundtrip_test")
	if err != nil {
		log.Fatalf("Failed to deserialize network: %v", err)
	}

	fmt.Println("   ✓ Deserialized network from JSON")

	// Verify structure matches
	if network2.GridRows != network1.GridRows ||
		network2.GridCols != network1.GridCols ||
		network2.LayersPerCell != network1.LayersPerCell ||
		network2.BatchSize != network1.BatchSize {
		log.Fatalf("Network structure mismatch after round-trip")
	}

	// Verify layer configurations
	for i := 0; i < network1.TotalLayers(); i++ {
		row := i / (network1.GridCols * network1.LayersPerCell)
		col := (i / network1.LayersPerCell) % network1.GridCols
		layer := i % network1.LayersPerCell

		cfg1 := network1.GetLayer(row, col, layer)
		cfg2 := network2.GetLayer(row, col, layer)

		if cfg1.Type != cfg2.Type {
			log.Fatalf("Layer type mismatch at position %d: %d vs %d", i, cfg1.Type, cfg2.Type)
		}

		if cfg1.Activation != cfg2.Activation {
			log.Fatalf("Activation mismatch at position %d: %d vs %d", i, cfg1.Activation, cfg2.Activation)
		}
	}

	fmt.Println("   ✓ Round-trip serialization successful")
	fmt.Println("   ✓ All layer configurations preserved")

	// Save to file for testing
	err = network1.SaveModel("test_roundtrip.json", "roundtrip_test")
	if err != nil {
		log.Fatalf("Failed to save to file: %v", err)
	}
	fmt.Println("   ✓ Saved network to test_roundtrip.json")

	// Load from file
	network3, err := nn.LoadModel("test_roundtrip.json", "roundtrip_test")
	if err != nil {
		log.Fatalf("Failed to load from file: %v", err)
	}
	fmt.Println("   ✓ Loaded network from test_roundtrip.json")

	if network3.TotalLayers() != network1.TotalLayers() {
		log.Fatalf("Layer count mismatch after file round-trip")
	}
	fmt.Println("   ✓ File round-trip successful")
}
