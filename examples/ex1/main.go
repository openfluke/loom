package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Neural Tween Experiment - True Bidirectional Approach
// Compares speed vs standard backpropagation

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Neural Tween - Speed Comparison vs Backpropagation         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Test 1: Neural Tweening
	test1_tween()

	// Test 2: Standard Backpropagation (for comparison)
	test2_backprop()

	// Test 3: Deep network with tweening
	test3_deep_tween()
}

func test1_tween() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 1: Neural Tweening (8→32→16→2)")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	network := createNetwork()
	inputs, expected := generateData(500)

	initial, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("Before: Score=%.1f/100\n", initial.Score)

	ts := nn.NewTweenState(network)

	start := time.Now()
	epochs := 50
	var finalScore float64

	ts.Train(network, inputs, expected, epochs, 0.5,
		func(epoch int, avgLoss float32, metrics *nn.DeviationMetrics) {
			finalScore = metrics.Score
		})

	elapsed := time.Since(start)

	fmt.Printf("After: Score=%.1f/100\n", finalScore)
	fmt.Printf("⏱️  Time: %v | Epochs: %d | ms/epoch: %.1f\n\n",
		elapsed, epochs, float64(elapsed.Milliseconds())/float64(epochs))
}

func test2_backprop() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 2: Standard Backpropagation (8→32→16→2)")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	network := createNetwork()
	inputs, expected := generateData(500)

	initial, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("Before: Score=%.1f/100\n", initial.Score)

	// Use standard training with BackwardCPU
	start := time.Now()
	epochs := 50
	learningRate := float32(0.1)

	for epoch := 0; epoch < epochs; epoch++ {
		for i := range inputs {
			output, _ := network.ForwardCPU(inputs[i])

			// Compute error gradient
			errorGrad := make([]float32, len(output))
			for j := range output {
				target := float32(0)
				if j == int(expected[i]) {
					target = 1.0
				}
				errorGrad[j] = output[j] - target
			}

			// Backward pass
			network.BackwardCPU(errorGrad)

			// Update weights manually
			updateWeights(network, learningRate)
		}
	}

	elapsed := time.Since(start)

	final, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("After: Score=%.1f/100\n", final.Score)
	fmt.Printf("⏱️  Time: %v | Epochs: %d | ms/epoch: %.1f\n\n",
		elapsed, epochs, float64(elapsed.Milliseconds())/float64(epochs))
}

func test3_deep_tween() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 3: Deep Network Tweening (8→64→128→64→2)")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	network := createDeepNetwork()
	inputs, expected := generateData(1000)

	initial, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("Before: Score=%.1f/100\n", initial.Score)

	ts := nn.NewTweenState(network)

	start := time.Now()
	epochs := 100
	var finalScore float64

	ts.Train(network, inputs, expected, epochs, 0.3,
		func(epoch int, avgLoss float32, metrics *nn.DeviationMetrics) {
			finalScore = metrics.Score
			if epoch%20 == 0 {
				fmt.Printf("  Epoch %d: Score=%.1f\n", epoch, metrics.Score)
			}
		})

	elapsed := time.Since(start)

	fmt.Printf("After: Score=%.1f/100 (best found during training)\n", finalScore)
	fmt.Printf("⏱️  Time: %v | Epochs: %d | ms/epoch: %.1f\n",
		elapsed, epochs, float64(elapsed.Milliseconds())/float64(epochs))
}

func updateWeights(n *nn.Network, lr float32) {
	kernelGrads := n.KernelGradients()
	biasGrads := n.BiasGradients()

	for i := 0; i < n.TotalLayers(); i++ {
		row := i / (n.GridCols * n.LayersPerCell)
		col := (i / n.LayersPerCell) % n.GridCols
		layer := i % n.LayersPerCell

		cfg := n.GetLayer(row, col, layer)
		if cfg == nil || len(cfg.Kernel) == 0 {
			continue
		}

		// Update kernel
		if i < len(kernelGrads) && len(kernelGrads[i]) == len(cfg.Kernel) {
			for j := range cfg.Kernel {
				cfg.Kernel[j] -= lr * kernelGrads[i][j]
			}
		}

		// Update bias
		if i < len(biasGrads) && len(biasGrads[i]) == len(cfg.Bias) {
			for j := range cfg.Bias {
				cfg.Bias[j] -= lr * biasGrads[i][j]
			}
		}

		n.SetLayer(row, col, layer, *cfg)
	}
}

func createNetwork() *nn.Network {
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
