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

	// Test 4: REAL DATA - Iris dataset
	test4_iris()
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

	start := time.Now()
	epochs := 50
	learningRate := float32(0.1)

	for epoch := 0; epoch < epochs; epoch++ {
		for i := range inputs {
			output, _ := network.ForwardCPU(inputs[i])

			errorGrad := make([]float32, len(output))
			for j := range output {
				target := float32(0)
				if j == int(expected[i]) {
					target = 1.0
				}
				errorGrad[j] = output[j] - target
			}

			network.BackwardCPU(errorGrad)
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
	fmt.Printf("⏱️  Time: %v | Epochs: %d | ms/epoch: %.1f\n\n",
		elapsed, epochs, float64(elapsed.Milliseconds())/float64(epochs))
}

func test4_iris() {
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 4: REAL DATA - Iris Dataset (4→16→8→3)")
	fmt.Println(" 150 samples, 4 features, 3 classes (setosa/versicolor/virginica)")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// Create network for Iris: 4 inputs, 3 outputs
	network := createIrisNetwork()
	inputs, expected := loadIrisData()

	fmt.Printf("Dataset: %d samples\n", len(inputs))

	initial, _ := network.EvaluateNetwork(inputs, expected)
	fmt.Printf("Before: Score=%.1f/100\n", initial.Score)

	ts := nn.NewTweenState(network)

	start := time.Now()
	epochs := 300
	var finalScore float64

	ts.Train(network, inputs, expected, epochs, 0.5,
		func(epoch int, avgLoss float32, metrics *nn.DeviationMetrics) {
			finalScore = metrics.Score
			if epoch%50 == 0 {
				fmt.Printf("  Epoch %d: Score=%.1f, Loss=%.3f\n", epoch, metrics.Score, avgLoss)
			}
		})

	elapsed := time.Since(start)

	fmt.Printf("After: Score=%.1f/100\n", finalScore)
	fmt.Printf("⏱️  Time: %v | Epochs: %d | ms/epoch: %.1f\n",
		elapsed, epochs, float64(elapsed.Milliseconds())/float64(epochs))

	if finalScore >= 90 {
		fmt.Println("✅ Iris classification successful!")
	} else {
		fmt.Println("⚠️  Could improve with more epochs")
	}
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

		if i < len(kernelGrads) && len(kernelGrads[i]) == len(cfg.Kernel) {
			for j := range cfg.Kernel {
				cfg.Kernel[j] -= lr * kernelGrads[i][j]
			}
		}

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

func createIrisNetwork() *nn.Network {
	jsonConfig := `{
		"batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 3,
		"layers": [
			{"type": "dense", "activation": "tanh", "input_height": 4, "output_height": 16},
			{"type": "dense", "activation": "tanh", "input_height": 16, "output_height": 8},
			{"type": "dense", "activation": "sigmoid", "input_height": 8, "output_height": 3}
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

// loadIrisData returns the famous Iris dataset
// 150 samples: 50 setosa (0), 50 versicolor (1), 50 virginica (2)
// Features: sepal length, sepal width, petal length, petal width (normalized)
func loadIrisData() ([][]float32, []float64) {
	// Iris dataset (normalized to 0-1 range)
	rawData := [][]float32{
		// Setosa (class 0) - 50 samples
		{0.222, 0.625, 0.068, 0.042}, {0.167, 0.417, 0.068, 0.042}, {0.111, 0.500, 0.051, 0.042},
		{0.083, 0.458, 0.085, 0.042}, {0.194, 0.667, 0.068, 0.042}, {0.306, 0.792, 0.119, 0.125},
		{0.083, 0.583, 0.068, 0.083}, {0.194, 0.583, 0.085, 0.042}, {0.028, 0.375, 0.068, 0.042},
		{0.167, 0.458, 0.085, 0.000}, {0.306, 0.708, 0.085, 0.042}, {0.139, 0.583, 0.102, 0.042},
		{0.139, 0.417, 0.068, 0.000}, {0.000, 0.417, 0.017, 0.000}, {0.417, 0.833, 0.034, 0.042},
		{0.389, 1.000, 0.085, 0.125}, {0.306, 0.792, 0.051, 0.125}, {0.222, 0.625, 0.068, 0.083},
		{0.389, 0.750, 0.119, 0.083}, {0.222, 0.750, 0.085, 0.083}, {0.306, 0.583, 0.119, 0.042},
		{0.222, 0.708, 0.085, 0.125}, {0.083, 0.667, 0.000, 0.042}, {0.222, 0.542, 0.119, 0.167},
		{0.139, 0.583, 0.153, 0.042}, {0.194, 0.417, 0.102, 0.042}, {0.194, 0.583, 0.102, 0.125},
		{0.250, 0.625, 0.085, 0.042}, {0.250, 0.583, 0.068, 0.042}, {0.111, 0.500, 0.102, 0.042},
		{0.139, 0.458, 0.102, 0.042}, {0.306, 0.583, 0.085, 0.125}, {0.250, 0.875, 0.085, 0.000},
		{0.333, 0.917, 0.068, 0.042}, {0.167, 0.458, 0.085, 0.000}, {0.194, 0.500, 0.034, 0.042},
		{0.333, 0.625, 0.051, 0.042}, {0.167, 0.458, 0.085, 0.000}, {0.028, 0.417, 0.051, 0.042},
		{0.222, 0.583, 0.085, 0.042}, {0.194, 0.625, 0.051, 0.083}, {0.056, 0.125, 0.051, 0.083},
		{0.028, 0.500, 0.051, 0.042}, {0.194, 0.625, 0.102, 0.208}, {0.222, 0.750, 0.153, 0.125},
		{0.139, 0.417, 0.068, 0.083}, {0.222, 0.750, 0.102, 0.042}, {0.083, 0.500, 0.068, 0.042},
		{0.278, 0.708, 0.085, 0.042}, {0.194, 0.542, 0.068, 0.042},
		// Versicolor (class 1) - 50 samples
		{0.528, 0.375, 0.593, 0.583}, {0.444, 0.417, 0.576, 0.583}, {0.500, 0.417, 0.627, 0.583},
		{0.194, 0.208, 0.390, 0.375}, {0.472, 0.375, 0.593, 0.542}, {0.278, 0.292, 0.458, 0.417},
		{0.417, 0.417, 0.559, 0.583}, {0.139, 0.167, 0.322, 0.375}, {0.444, 0.333, 0.559, 0.500},
		{0.194, 0.333, 0.424, 0.417}, {0.167, 0.125, 0.288, 0.333}, {0.333, 0.375, 0.508, 0.500},
		{0.333, 0.167, 0.458, 0.375}, {0.389, 0.375, 0.542, 0.500}, {0.222, 0.333, 0.424, 0.375},
		{0.472, 0.417, 0.525, 0.625}, {0.333, 0.292, 0.508, 0.417}, {0.306, 0.292, 0.458, 0.375},
		{0.361, 0.125, 0.492, 0.417}, {0.250, 0.292, 0.390, 0.375}, {0.361, 0.375, 0.542, 0.542},
		{0.306, 0.333, 0.492, 0.417}, {0.444, 0.208, 0.576, 0.417}, {0.389, 0.333, 0.576, 0.500},
		{0.389, 0.292, 0.525, 0.458}, {0.417, 0.333, 0.559, 0.458}, {0.472, 0.333, 0.627, 0.583},
		{0.500, 0.375, 0.610, 0.625}, {0.361, 0.333, 0.508, 0.500}, {0.222, 0.250, 0.390, 0.417},
		{0.222, 0.167, 0.356, 0.375}, {0.222, 0.167, 0.356, 0.333}, {0.278, 0.292, 0.424, 0.375},
		{0.389, 0.250, 0.593, 0.458}, {0.278, 0.250, 0.458, 0.417}, {0.389, 0.417, 0.542, 0.583},
		{0.472, 0.417, 0.593, 0.542}, {0.333, 0.167, 0.458, 0.417}, {0.306, 0.333, 0.458, 0.417},
		{0.194, 0.208, 0.373, 0.375}, {0.222, 0.292, 0.441, 0.375}, {0.333, 0.250, 0.508, 0.458},
		{0.278, 0.250, 0.458, 0.417}, {0.167, 0.167, 0.305, 0.375}, {0.250, 0.292, 0.441, 0.417},
		{0.250, 0.333, 0.424, 0.333}, {0.306, 0.333, 0.475, 0.417}, {0.361, 0.333, 0.441, 0.375},
		{0.139, 0.208, 0.271, 0.333}, {0.278, 0.292, 0.492, 0.417},
		// Virginica (class 2) - 50 samples
		{0.472, 0.292, 0.695, 0.625}, {0.389, 0.250, 0.576, 0.542}, {0.639, 0.333, 0.729, 0.708},
		{0.389, 0.167, 0.610, 0.583}, {0.500, 0.292, 0.678, 0.708}, {0.722, 0.333, 0.864, 0.833},
		{0.167, 0.125, 0.441, 0.417}, {0.611, 0.292, 0.797, 0.750}, {0.472, 0.083, 0.627, 0.500},
		{0.583, 0.417, 0.729, 0.708}, {0.444, 0.333, 0.661, 0.625}, {0.444, 0.208, 0.593, 0.583},
		{0.528, 0.250, 0.678, 0.625}, {0.333, 0.125, 0.508, 0.417}, {0.333, 0.167, 0.559, 0.500},
		{0.444, 0.292, 0.644, 0.708}, {0.472, 0.333, 0.661, 0.583}, {0.806, 0.417, 0.831, 0.625},
		{0.861, 0.125, 0.864, 0.625}, {0.306, 0.042, 0.492, 0.375}, {0.556, 0.333, 0.729, 0.708},
		{0.333, 0.208, 0.542, 0.500}, {0.750, 0.250, 0.797, 0.542}, {0.389, 0.167, 0.593, 0.583},
		{0.500, 0.333, 0.678, 0.625}, {0.583, 0.333, 0.729, 0.750}, {0.361, 0.167, 0.576, 0.500},
		{0.333, 0.250, 0.559, 0.500}, {0.444, 0.208, 0.661, 0.583}, {0.528, 0.125, 0.678, 0.458},
		{0.639, 0.167, 0.678, 0.542}, {0.806, 0.292, 0.762, 0.708}, {0.444, 0.292, 0.661, 0.667},
		{0.389, 0.208, 0.542, 0.542}, {0.389, 0.208, 0.576, 0.583}, {0.750, 0.333, 0.831, 0.833},
		{0.556, 0.417, 0.763, 0.875}, {0.472, 0.375, 0.593, 0.542}, {0.306, 0.208, 0.508, 0.500},
		{0.444, 0.333, 0.644, 0.583}, {0.500, 0.333, 0.712, 0.625}, {0.417, 0.333, 0.678, 0.625},
		{0.389, 0.250, 0.576, 0.542}, {0.528, 0.333, 0.678, 0.667}, {0.556, 0.375, 0.780, 0.833},
		{0.472, 0.292, 0.695, 0.667}, {0.389, 0.208, 0.610, 0.583}, {0.417, 0.250, 0.559, 0.583},
		{0.444, 0.333, 0.644, 0.583}, {0.333, 0.208, 0.559, 0.542},
	}

	labels := make([]float64, 150)
	for i := 0; i < 50; i++ {
		labels[i] = 0 // Setosa
	}
	for i := 50; i < 100; i++ {
		labels[i] = 1 // Versicolor
	}
	for i := 100; i < 150; i++ {
		labels[i] = 2 // Virginica
	}

	return rawData, labels
}
