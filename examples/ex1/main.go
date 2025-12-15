package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Neural Tween Experiment - True Bidirectional Approach
// NOT backpropagation! Instead:
// 1. Forward: capture what each layer actually produces
// 2. Backward estimate: from expected output, estimate what each layer SHOULD produce
// 3. Link budget: measure information flow quality at each layer
// 4. Tween: directly morph weights to close the gap (no gradients!)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Neural Tween - Bidirectional Weight Morphing Experiment    ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║ NOT backpropagation! Uses:                                   ║")
	fmt.Println("║ • Forward analysis + Backward target estimation              ║")
	fmt.Println("║ • Link budgeting (like WiFi signal loss)                     ║")
	fmt.Println("║ • Direct weight tweening (like Flash morphing)               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Test 1: Simple binary classification
	test1()

	// Test 2: Larger network
	test2()
}

func test1() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 1: Simple Network (8→32→16→2)")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	network := createSimpleNetwork()
	inputs, expected := generateData(500)

	// Initial evaluation
	initial, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("Before: Score=%.1f/100\n", initial.Score)

	// Initialize tween state
	ts := nn.NewTweenState(network)
	ts.CalculateLinkBudgetsFromSample(network, inputs[0])
	avgB, minB, maxB := ts.GetBudgetSummary()
	fmt.Printf("Initial Link Budgets: avg=%.2f, min=%.2f, max=%.2f\n\n", avgB, minB, maxB)

	// Run neural tweening (NOT backprop!)
	epochs := 50
	tweenRate := float32(0.5)

	fmt.Println("Training with bidirectional tweening...")
	ts.Train(network, inputs, expected, epochs, tweenRate,
		func(epoch int, avgLoss float32, metrics *nn.DeviationMetrics) {
			avgB, _, _ := ts.GetBudgetSummary()
			avgGap, _ := ts.GetGapSummary()
			fmt.Printf("Epoch %3d: Score=%5.1f | Loss=%.3f | Budget=%.2f | Gap=%.4f\n",
				epoch, metrics.Score, avgLoss, avgB, avgGap)
		})

	// Final evaluation
	final, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("\nAfter: Score=%.1f/100 (improvement: +%.1f%%)\n", final.Score, final.Score-initial.Score)

	if final.Score >= 90 {
		fmt.Println("✅ TEST 1: Neural tweening works!\n")
	} else {
		fmt.Println("⚠️  TEST 1: Needs more tuning\n")
	}
}

func test2() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 2: Deeper Network (8→64→128→64→2) - LAYERWISE TRAINING")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	network := createDeepNetwork()
	inputs, expected := generateData(1000)

	// Initial evaluation
	initial, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("Before: Score=%.1f/100\n", initial.Score)

	// Initialize tween state
	ts := nn.NewTweenState(network)
	ts.CalculateLinkBudgetsFromSample(network, inputs[0])
	fmt.Printf("Network has %d layers\n\n", ts.TotalLayers)

	// Use LAYERWISE training for deep networks
	// Train each layer from output to input
	fmt.Println("Training layer-by-layer (output → input)...")
	epochsPerLayer := 15
	tweenRate := float32(0.5)

	ts.TrainLayerwise(network, inputs, expected, epochsPerLayer, tweenRate,
		func(layer int, epochs int, score float64) {
			fmt.Printf("  Layer %d trained (%d epochs): Score=%.1f\n", layer, epochs, score)
		})

	// Final evaluation
	final, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("\nAfter: Score=%.1f/100 (improvement: +%.1f%%)\n", final.Score, final.Score-initial.Score)

	if final.Score >= 90 {
		fmt.Println("✅ TEST 2: Deep network layerwise tweening works!")
	} else if final.Score > initial.Score+10 {
		fmt.Println("⚠️  TEST 2: Improved but needs more epochs")
	} else {
		fmt.Println("❌ TEST 2: Needs algorithm tuning")
	}
}

func createSimpleNetwork() *nn.Network {
	jsonConfig := `{
		"batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 3,
		"layers": [
			{"type": "dense", "activation": "tanh", "input_height": 8, "output_height": 32},
			{"type": "dense", "activation": "tanh", "input_height": 32, "output_height": 16},
			{"type": "dense", "activation": "sigmoid", "input_height": 16, "output_height": 2}
		]
	}`
	network, _ := nn.BuildNetworkFromJSON(jsonConfig)
	network.InitializeWeights()
	return network
}

func createDeepNetwork() *nn.Network {
	jsonConfig := `{
		"batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 4,
		"layers": [
			{"type": "dense", "activation": "tanh", "input_height": 8, "output_height": 64},
			{"type": "dense", "activation": "tanh", "input_height": 64, "output_height": 128},
			{"type": "dense", "activation": "tanh", "input_height": 128, "output_height": 64},
			{"type": "dense", "activation": "sigmoid", "input_height": 64, "output_height": 2}
		]
	}`
	network, _ := nn.BuildNetworkFromJSON(jsonConfig)
	network.InitializeWeights()
	return network
}

func generateData(n int) ([][]float32, []float64) {
	inputs := make([][]float32, n)
	expected := make([]float64, n)

	for i := 0; i < n; i++ {
		input := make([]float32, 8)
		sum := float32(0)
		for j := 0; j < 8; j++ {
			input[j] = rand.Float32()
			sum += input[j]
		}
		inputs[i] = input
		if sum < 4.0 {
			expected[i] = 0
		} else {
			expected[i] = 1
		}
	}
	return inputs, expected
}
