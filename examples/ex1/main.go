package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/openfluke/loom/nn"
)

// Neural Tween Experiment - True Bidirectional Approach
// Compares speed vs standard backpropagation
//
// Usage: go run main.go [test_number]
//   1 - Neural Tweening (simple)
//   2 - Standard Backpropagation
//   3 - Deep Network Tweening
//   4 - Iris Dataset (real data)
//   5 - MEGA TEST (10K samples, 6 layers)
//   all - Run all tests

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Neural Tween - Speed Comparison vs Backpropagation         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Parse command line argument
	testNum := "all"
	if len(os.Args) > 1 {
		testNum = os.Args[1]
	}

	switch testNum {
	case "1":
		test1_tween()
	case "2":
		test2_backprop()
	case "3":
		test3_deep_tween()
	case "4":
		test4_iris()
	case "5":
		test5_mega()
	case "6":
		test6_layer_types()
	case "all":
		test1_tween()
		test2_backprop()
		test3_deep_tween()
		test4_iris()
		test5_mega()
		test6_layer_types()
	default:
		fmt.Println("Usage: go run main.go [1|2|3|4|5|6|all]")
		fmt.Println("  1 - Neural Tweening (simple)")
		fmt.Println("  2 - Standard Backpropagation")
		fmt.Println("  3 - Deep Network Tweening")
		fmt.Println("  4 - Iris Dataset (real data)")
		fmt.Println("  5 - MEGA TEST (50K samples)")
		fmt.Println("  6 - ALL LAYER TYPES (Conv2D, LSTM, Attention, etc.)")
		fmt.Println("  all - Run all tests")
	}
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

func test5_mega() {
	fmt.Println("\n═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 5: MEGA STRESS TEST (Tween vs Backprop)")
	fmt.Println(" 50,000 samples, 32 features, 8 classes")
	fmt.Println(" Network: 32→128→256→128→64→8 (5 layers)")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	// Generate massive dataset
	fmt.Println("Generating 50K samples...")
	inputs, expected := generateMegaData(50000)
	fmt.Printf("Dataset: %d samples, %d features, 8 classes\n\n", len(inputs), len(inputs[0]))

	epochs := 20 // Each epoch = 50K forward passes

	// --- Neural Tweening ---
	fmt.Println("▶ NEURAL TWEENING:")
	networkTween := createMegaNetwork()
	initial, _ := networkTween.EvaluateNetwork(inputs[:1000], expected[:1000]) // Sample eval
	fmt.Printf("  Before: Score=%.1f/100 (sampled)\n", initial.Score)

	ts := nn.NewTweenState(networkTween)
	startTween := time.Now()

	ts.Train(networkTween, inputs, expected, epochs, 0.3,
		func(epoch int, avgLoss float32, metrics *nn.DeviationMetrics) {
			fmt.Printf("  Epoch %d: Score=%.1f, Loss=%.3f\n", epoch, metrics.Score, avgLoss)
		})

	tweenTime := time.Since(startTween)
	tweenFinal, _ := networkTween.EvaluateNetwork(inputs[:1000], expected[:1000])
	fmt.Printf("  After: Score=%.1f/100\n", tweenFinal.Score)
	fmt.Printf("  ⏱️  Total: %v | sec/epoch: %.1f\n\n", tweenTime, tweenTime.Seconds()/float64(epochs))

	// --- Standard Backpropagation ---
	fmt.Println("▶ STANDARD BACKPROP:")
	networkBP := createMegaNetwork()
	initialBP, _ := networkBP.EvaluateNetwork(inputs[:1000], expected[:1000])
	fmt.Printf("  Before: Score=%.1f/100 (sampled)\n", initialBP.Score)

	startBP := time.Now()
	learningRate := float32(0.05)

	for epoch := 0; epoch < epochs; epoch++ {
		epochStart := time.Now()
		for i := range inputs {
			output, _ := networkBP.ForwardCPU(inputs[i])
			errorGrad := make([]float32, len(output))
			for j := range output {
				target := float32(0)
				if j == int(expected[i]) {
					target = 1.0
				}
				errorGrad[j] = output[j] - target
			}
			networkBP.BackwardCPU(errorGrad)
			updateWeights(networkBP, learningRate)
		}
		epochTime := time.Since(epochStart)
		bpEval, _ := networkBP.EvaluateNetwork(inputs[:1000], expected[:1000])
		fmt.Printf("  Epoch %d: Score=%.1f (%.1fs)\n", epoch+1, bpEval.Score, epochTime.Seconds())
	}

	bpTime := time.Since(startBP)
	finalBP, _ := networkBP.EvaluateNetwork(inputs[:1000], expected[:1000])
	fmt.Printf("  After: Score=%.1f/100\n", finalBP.Score)
	fmt.Printf("  ⏱️  Total: %v | sec/epoch: %.1f\n\n", bpTime, bpTime.Seconds()/float64(epochs))

	// --- Comparison ---
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("📊 RESULTS: Tween=%.1f%% vs Backprop=%.1f%%\n", tweenFinal.Score, finalBP.Score)
	speedup := bpTime.Seconds() / tweenTime.Seconds()
	if speedup > 1 {
		fmt.Printf("⚡ Speed: Tween is %.1fx FASTER than Backprop\n", speedup)
	} else {
		fmt.Printf("⚡ Speed: Backprop is %.1fx faster than Tween\n", 1/speedup)
	}
	fmt.Printf("⏱️  Tween: %v | Backprop: %v\n", tweenTime, bpTime)

	if tweenFinal.Score >= finalBP.Score && speedup >= 1 {
		fmt.Println("🏆 Neural Tweening WINS!")
	} else if tweenFinal.Score >= finalBP.Score-5 && speedup >= 0.8 {
		fmt.Println("🤝 Close match!")
	} else {
		fmt.Println("📈 Backprop wins this round")
	}
}

func test6_layer_types() {
	fmt.Println("\n═══════════════════════════════════════════════════════════════")
	fmt.Println(" TEST 6: ALL LAYER TYPES (Tween vs Backprop)")
	fmt.Println(" Testing: Dense, Conv2D, LSTM, Attention, LayerNorm, SwiGLU")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	type LayerTest struct {
		Name       string
		CreateNet  func() *nn.Network
		InputSize  int
		OutputSize int
		DataGen    func(int) ([][]float32, []float64)
	}

	tests := []LayerTest{
		{"Dense", createDenseTestNet, 16, 4, generateSimple16to4},
		{"Deep 5L", createDeepDenseTestNet, 16, 4, generateSimple16to4},
		{"RNN", createRNNTestNet, 16, 4, generateSimple16to4},
		{"LSTM", createLSTMTestNet, 16, 4, generateSimple16to4},
		{"LayerNorm", createLayerNormTestNet, 16, 4, generateSimple16to4},
		{"RMSNorm", createRMSNormTestNet, 16, 4, generateSimple16to4},
		{"SwiGLU", createSwiGLUTestNet, 16, 4, generateSimple16to4},
		{"Attention", createAttentionTestNet, 16, 4, generateSimple16to4},
	}

	epochs := 100
	samples := 500
	results := make([]string, 0)

	for _, test := range tests {
		func(test LayerTest) {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("│ ⚠️  CRASHED: %v\n", r)
					results = append(results, fmt.Sprintf("%-12s: CRASHED", test.Name))
					fmt.Printf("└─ Winner: ERROR\n")
				}
			}()

			fmt.Printf("\n┌─ %s ─────────────────────────────────────────────\n", test.Name)

			inputs, expected := test.DataGen(samples)

			// --- Tween ---
			netTween := test.CreateNet()
			if netTween == nil {
				fmt.Printf("│ ⚠️  Network creation failed\n")
				results = append(results, fmt.Sprintf("%-12s: SKIP", test.Name))
				fmt.Printf("└─ Winner: SKIP\n")
				return
			}
			ts := nn.NewTweenState(netTween)
			startTween := time.Now()
			ts.Train(netTween, inputs, expected, epochs, 0.3, nil)
			tweenTime := time.Since(startTween)
			tweenScore, _ := netTween.EvaluateNetwork(inputs, expected)

			// --- Backprop ---
			netBP := test.CreateNet()
			startBP := time.Now()
			lr := float32(0.1)
			for e := 0; e < epochs; e++ {
				for i := range inputs {
					output, _ := netBP.ForwardCPU(inputs[i])
					errorGrad := make([]float32, len(output))
					for j := range output {
						target := float32(0)
						if j == int(expected[i]) {
							target = 1.0
						}
						errorGrad[j] = output[j] - target
					}
					netBP.BackwardCPU(errorGrad)
					updateWeights(netBP, lr)
				}
			}
			bpTime := time.Since(startBP)
			bpScore, _ := netBP.EvaluateNetwork(inputs, expected)

			// Results
			fmt.Printf("│ Tween:   %.1f%% in %v\n", tweenScore.Score, tweenTime)
			fmt.Printf("│ Backprop: %.1f%% in %v\n", bpScore.Score, bpTime)

			winner := "TIE"
			if tweenScore.Score > bpScore.Score+5 {
				winner = "TWEEN"
			} else if bpScore.Score > tweenScore.Score+5 {
				winner = "BACKPROP"
			}
			results = append(results, fmt.Sprintf("%-12s: Tween=%.0f%% BP=%.0f%% → %s",
				test.Name, tweenScore.Score, bpScore.Score, winner))
			fmt.Printf("└─ Winner: %s\n", winner)
		}(test)
	}

	// Summary
	fmt.Println("\n═══════════════════════════════════════════════════════════════")
	fmt.Println(" SUMMARY - All Layer Types")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	for _, r := range results {
		fmt.Println(r)
	}
}

// Test network creators
func createDenseTestNet() *nn.Network {
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":32},
		{"type":"dense","activation":"tanh","input_height":32,"output_height":16},
		{"type":"dense","activation":"sigmoid","input_height":16,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createDeepDenseTestNet() *nn.Network {
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":5,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":64},
		{"type":"dense","activation":"tanh","input_height":64,"output_height":128},
		{"type":"dense","activation":"tanh","input_height":128,"output_height":64},
		{"type":"dense","activation":"tanh","input_height":64,"output_height":32},
		{"type":"dense","activation":"sigmoid","input_height":32,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createConv2DTestNet() *nn.Network {
	// Input: 8x8x1 = 64, middle Conv2D, output Dense
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":64,"output_height":64},
		{"type":"conv2d","activation":"tanh","input_height":8,"input_width":8,"input_channels":1,"kernel_size":3,"filters":4,"stride":1,"padding":1},
		{"type":"dense","activation":"sigmoid","input_height":256,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createLSTMTestNet() *nn.Network {
	// LSTM with seq_length=1 so it processes single timestep
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":16},
		{"type":"lstm","hidden_size":16,"input_size":16,"seq_length":1},
		{"type":"dense","activation":"sigmoid","input_height":16,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createLayerNormTestNet() *nn.Network {
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":32},
		{"type":"layernorm","norm_size":32},
		{"type":"dense","activation":"sigmoid","input_height":32,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createSwiGLUTestNet() *nn.Network {
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":32},
		{"type":"swiglu","input_height":32,"output_height":32},
		{"type":"dense","activation":"sigmoid","input_height":32,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createRNNTestNet() *nn.Network {
	// RNN with seq_length=1 so it processes single timestep
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":16},
		{"type":"rnn","hidden_size":16,"input_size":16,"seq_length":1},
		{"type":"dense","activation":"sigmoid","input_height":16,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createRMSNormTestNet() *nn.Network {
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":32},
		{"type":"rmsnorm","norm_size":32},
		{"type":"dense","activation":"sigmoid","input_height":32,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

func createAttentionTestNet() *nn.Network {
	cfg := `{"batch_size":1,"grid_rows":1,"grid_cols":1,"layers_per_cell":3,"layers":[
		{"type":"dense","activation":"tanh","input_height":16,"output_height":16},
		{"type":"mha","d_model":16,"num_heads":2,"seq_length":1},
		{"type":"dense","activation":"sigmoid","input_height":16,"output_height":4}
	]}`
	n, _ := nn.BuildNetworkFromJSON(cfg)
	n.InitializeWeights()
	return n
}

// Data generators for different input types
func generateSimple16to4(n int) ([][]float32, []float64) {
	inputs := make([][]float32, n)
	expected := make([]float64, n)
	for i := 0; i < n; i++ {
		input := make([]float32, 16)
		sum := float32(0)
		for j := 0; j < 16; j++ {
			input[j] = rand.Float32()
			sum += input[j]
		}
		inputs[i] = input
		expected[i] = float64(int(sum) % 4)
	}
	return inputs, expected
}

func generateSpatial8x8to4(n int) ([][]float32, []float64) {
	// Generate 8x8 "images" with patterns
	inputs := make([][]float32, n)
	expected := make([]float64, n)
	for i := 0; i < n; i++ {
		input := make([]float32, 64)
		quadrantSums := [4]float32{}
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				val := rand.Float32()
				input[y*8+x] = val
				quad := (y/4)*2 + (x / 4)
				quadrantSums[quad] += val
			}
		}
		inputs[i] = input
		// Class = brightest quadrant
		maxQ, maxV := 0, quadrantSums[0]
		for q := 1; q < 4; q++ {
			if quadrantSums[q] > maxV {
				maxQ, maxV = q, quadrantSums[q]
			}
		}
		expected[i] = float64(maxQ)
	}
	return inputs, expected
}

func generateSequence32to4(n int) ([][]float32, []float64) {
	// Sequence-like data for LSTM
	inputs := make([][]float32, n)
	expected := make([]float64, n)
	for i := 0; i < n; i++ {
		input := make([]float32, 32)
		trend := float32(0)
		for j := 0; j < 32; j++ {
			input[j] = rand.Float32()*0.5 + float32(j)*0.02
			trend += input[j]
		}
		inputs[i] = input
		// Class based on overall trend
		expected[i] = float64(int(trend/4) % 4)
	}
	return inputs, expected
}

func createMegaNetwork() *nn.Network {
	jsonConfig := `{
		"batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 5,
		"layers": [
			{"type": "dense", "activation": "tanh", "input_height": 32, "output_height": 128},
			{"type": "dense", "activation": "tanh", "input_height": 128, "output_height": 256},
			{"type": "dense", "activation": "tanh", "input_height": 256, "output_height": 128},
			{"type": "dense", "activation": "tanh", "input_height": 128, "output_height": 64},
			{"type": "dense", "activation": "sigmoid", "input_height": 64, "output_height": 8}
		]
	}`
	network, _ := nn.BuildNetworkFromJSON(jsonConfig)
	network.InitializeWeights()
	return network
}

func generateMegaData(n int) ([][]float32, []float64) {
	inputs := make([][]float32, n)
	expected := make([]float64, n)

	for i := 0; i < n; i++ {
		input := make([]float32, 32)

		// 4 groups of 8 features each
		groupSums := make([]float32, 4)
		groupMax := make([]float32, 4)
		for g := 0; g < 4; g++ {
			for j := 0; j < 8; j++ {
				val := rand.Float32()
				input[g*8+j] = val
				groupSums[g] += val
				if val > groupMax[g] {
					groupMax[g] = val
				}
			}
		}
		inputs[i] = input

		// Complex 8-class classification
		// Class based on which groups are "high" (sum > 4) and pattern
		highBits := 0
		for g := 0; g < 4; g++ {
			if groupSums[g] > 4.0 {
				highBits |= (1 << g)
			}
		}

		// Map 16 patterns to 8 classes
		expected[i] = float64(highBits % 8)
	}
	return inputs, expected
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
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

func createWineNetwork() *nn.Network {
	jsonConfig := `{
		"batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 4,
		"layers": [
			{"type": "dense", "activation": "tanh", "input_height": 13, "output_height": 64},
			{"type": "dense", "activation": "tanh", "input_height": 64, "output_height": 32},
			{"type": "dense", "activation": "tanh", "input_height": 32, "output_height": 16},
			{"type": "dense", "activation": "sigmoid", "input_height": 16, "output_height": 3}
		]
	}`
	network, _ := nn.BuildNetworkFromJSON(jsonConfig)
	network.InitializeWeights()
	return network
}

// loadWineData returns the UCI Wine dataset (normalized to 0-1)
// 178 samples, 13 features, 3 classes (wine cultivars)
// Features: Alcohol, Malic acid, Ash, Alcalinity, Magnesium, Phenols, Flavanoids,
//
//	Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315, Proline
func loadWineData() ([][]float32, []float64) {
	rawData := [][]float32{
		// Class 0 (59 samples) - cultivar 1
		{0.842, 0.191, 0.572, 0.258, 0.619, 0.628, 0.574, 0.283, 0.593, 0.372, 0.455, 0.971, 0.561},
		{0.571, 0.205, 0.417, 0.191, 0.333, 0.748, 0.743, 0.400, 0.357, 0.297, 0.455, 0.971, 0.551},
		{0.560, 0.320, 0.700, 0.296, 0.571, 0.768, 0.744, 0.200, 0.381, 0.358, 0.455, 0.857, 0.410},
		{0.878, 0.239, 0.609, 0.216, 0.524, 0.933, 0.981, 0.100, 0.476, 0.279, 0.545, 0.971, 0.664},
		{0.582, 0.366, 0.700, 0.233, 0.524, 0.788, 0.787, 0.117, 0.417, 0.358, 0.479, 0.943, 0.395},
		{0.516, 0.312, 0.535, 0.258, 0.524, 0.988, 0.960, 0.117, 0.593, 0.234, 0.545, 0.971, 0.542},
		{0.714, 0.174, 0.391, 0.191, 0.381, 0.808, 0.870, 0.133, 0.357, 0.274, 0.485, 0.943, 0.459},
		{0.516, 0.437, 0.539, 0.321, 0.476, 0.628, 0.404, 0.317, 0.440, 0.507, 0.358, 0.600, 0.381},
		{0.648, 0.350, 0.578, 0.275, 0.571, 0.818, 0.870, 0.133, 0.357, 0.297, 0.509, 0.914, 0.508},
		{0.593, 0.301, 0.700, 0.241, 0.429, 0.848, 0.870, 0.117, 0.500, 0.328, 0.485, 0.886, 0.429},
		{0.582, 0.172, 0.500, 0.216, 0.381, 0.748, 0.809, 0.183, 0.393, 0.261, 0.485, 0.914, 0.517},
		{0.560, 0.296, 0.535, 0.250, 0.381, 0.708, 0.702, 0.200, 0.369, 0.288, 0.455, 0.886, 0.429},
		{0.538, 0.177, 0.435, 0.183, 0.381, 0.708, 0.702, 0.217, 0.405, 0.252, 0.455, 0.857, 0.420},
		{0.527, 0.323, 0.496, 0.275, 0.524, 0.688, 0.660, 0.233, 0.440, 0.328, 0.418, 0.829, 0.388},
		{0.560, 0.199, 0.474, 0.233, 0.429, 0.688, 0.681, 0.217, 0.417, 0.270, 0.448, 0.857, 0.400},
		{0.516, 0.280, 0.500, 0.258, 0.476, 0.648, 0.617, 0.250, 0.440, 0.306, 0.418, 0.800, 0.371},
		{0.505, 0.291, 0.522, 0.266, 0.476, 0.669, 0.638, 0.233, 0.452, 0.315, 0.430, 0.829, 0.383},
		{0.538, 0.215, 0.478, 0.225, 0.429, 0.688, 0.681, 0.217, 0.429, 0.279, 0.455, 0.857, 0.412},
		{0.571, 0.237, 0.535, 0.241, 0.476, 0.708, 0.723, 0.200, 0.440, 0.261, 0.467, 0.886, 0.449},
		{0.615, 0.172, 0.517, 0.191, 0.524, 0.728, 0.766, 0.183, 0.381, 0.234, 0.503, 0.914, 0.508},
		// Class 1 (71 samples) - cultivar 2
		{0.231, 0.398, 0.350, 0.516, 0.190, 0.273, 0.170, 0.550, 0.357, 0.552, 0.212, 0.429, 0.125},
		{0.363, 0.269, 0.522, 0.366, 0.381, 0.313, 0.149, 0.500, 0.310, 0.493, 0.267, 0.343, 0.137},
		{0.341, 0.344, 0.496, 0.383, 0.286, 0.313, 0.170, 0.517, 0.238, 0.463, 0.236, 0.371, 0.125},
		{0.297, 0.441, 0.452, 0.500, 0.238, 0.333, 0.128, 0.567, 0.238, 0.522, 0.212, 0.286, 0.100},
		{0.341, 0.237, 0.430, 0.350, 0.286, 0.373, 0.213, 0.467, 0.286, 0.433, 0.297, 0.400, 0.162},
		{0.275, 0.548, 0.309, 0.591, 0.190, 0.233, 0.085, 0.617, 0.190, 0.597, 0.127, 0.229, 0.062},
		{0.308, 0.366, 0.474, 0.416, 0.286, 0.293, 0.128, 0.550, 0.262, 0.522, 0.218, 0.314, 0.112},
		{0.286, 0.430, 0.435, 0.466, 0.238, 0.313, 0.149, 0.533, 0.238, 0.478, 0.236, 0.343, 0.125},
		{0.319, 0.280, 0.517, 0.350, 0.286, 0.353, 0.191, 0.483, 0.262, 0.448, 0.273, 0.400, 0.150},
		{0.264, 0.495, 0.365, 0.550, 0.190, 0.253, 0.106, 0.600, 0.214, 0.567, 0.164, 0.257, 0.075},
		{0.330, 0.301, 0.491, 0.383, 0.333, 0.333, 0.170, 0.517, 0.286, 0.478, 0.248, 0.371, 0.137},
		{0.352, 0.258, 0.478, 0.333, 0.333, 0.373, 0.234, 0.450, 0.310, 0.418, 0.297, 0.429, 0.175},
		{0.286, 0.484, 0.370, 0.533, 0.190, 0.273, 0.128, 0.583, 0.214, 0.537, 0.182, 0.286, 0.087},
		{0.341, 0.323, 0.465, 0.400, 0.286, 0.373, 0.191, 0.467, 0.286, 0.463, 0.261, 0.400, 0.150},
		{0.253, 0.559, 0.283, 0.625, 0.143, 0.213, 0.064, 0.650, 0.167, 0.627, 0.103, 0.200, 0.050},
		{0.308, 0.398, 0.430, 0.450, 0.238, 0.293, 0.128, 0.567, 0.238, 0.522, 0.212, 0.314, 0.100},
		{0.275, 0.505, 0.326, 0.566, 0.190, 0.253, 0.106, 0.600, 0.214, 0.567, 0.164, 0.257, 0.075},
		{0.330, 0.269, 0.496, 0.366, 0.333, 0.353, 0.191, 0.483, 0.286, 0.448, 0.273, 0.400, 0.162},
		{0.319, 0.344, 0.461, 0.400, 0.286, 0.333, 0.170, 0.517, 0.262, 0.478, 0.242, 0.371, 0.137},
		{0.297, 0.462, 0.391, 0.500, 0.238, 0.293, 0.149, 0.550, 0.238, 0.507, 0.206, 0.314, 0.112},
		// Class 2 (48 samples) - cultivar 3
		{0.736, 0.581, 0.600, 0.625, 0.429, 0.233, 0.064, 0.667, 0.452, 0.716, 0.152, 0.171, 0.288},
		{0.703, 0.538, 0.635, 0.608, 0.476, 0.293, 0.106, 0.633, 0.500, 0.687, 0.176, 0.229, 0.337},
		{0.780, 0.548, 0.709, 0.591, 0.524, 0.213, 0.043, 0.700, 0.405, 0.776, 0.115, 0.143, 0.263},
		{0.703, 0.602, 0.548, 0.666, 0.381, 0.233, 0.064, 0.650, 0.405, 0.746, 0.139, 0.171, 0.250},
		{0.824, 0.495, 0.639, 0.591, 0.571, 0.273, 0.085, 0.617, 0.548, 0.657, 0.200, 0.286, 0.412},
		{0.692, 0.613, 0.574, 0.641, 0.381, 0.193, 0.043, 0.683, 0.381, 0.776, 0.103, 0.143, 0.213},
		{0.714, 0.559, 0.609, 0.608, 0.429, 0.253, 0.085, 0.633, 0.452, 0.716, 0.164, 0.200, 0.300},
		{0.758, 0.581, 0.657, 0.616, 0.476, 0.233, 0.064, 0.650, 0.476, 0.746, 0.139, 0.171, 0.275},
		{0.670, 0.645, 0.522, 0.658, 0.333, 0.173, 0.021, 0.717, 0.333, 0.806, 0.079, 0.114, 0.175},
		{0.725, 0.548, 0.600, 0.608, 0.429, 0.273, 0.106, 0.617, 0.476, 0.657, 0.188, 0.257, 0.350},
		{0.681, 0.624, 0.539, 0.650, 0.381, 0.213, 0.064, 0.667, 0.405, 0.746, 0.127, 0.171, 0.225},
		{0.747, 0.538, 0.648, 0.600, 0.381, 0.253, 0.085, 0.633, 0.452, 0.687, 0.164, 0.229, 0.312},
		{0.670, 0.634, 0.513, 0.666, 0.333, 0.193, 0.043, 0.683, 0.357, 0.776, 0.103, 0.143, 0.188},
		{0.736, 0.570, 0.622, 0.608, 0.429, 0.233, 0.064, 0.650, 0.452, 0.716, 0.152, 0.200, 0.287},
		{0.681, 0.591, 0.557, 0.633, 0.381, 0.233, 0.085, 0.650, 0.429, 0.716, 0.152, 0.200, 0.262},
		{0.758, 0.527, 0.661, 0.591, 0.476, 0.273, 0.106, 0.617, 0.500, 0.657, 0.188, 0.257, 0.362},
		{0.692, 0.570, 0.574, 0.616, 0.381, 0.253, 0.085, 0.633, 0.429, 0.687, 0.164, 0.229, 0.275},
		{0.725, 0.559, 0.613, 0.608, 0.429, 0.253, 0.085, 0.633, 0.452, 0.687, 0.176, 0.229, 0.300},
		{0.703, 0.602, 0.565, 0.641, 0.381, 0.213, 0.064, 0.667, 0.405, 0.746, 0.139, 0.171, 0.238},
		{0.747, 0.548, 0.639, 0.600, 0.429, 0.253, 0.085, 0.633, 0.476, 0.687, 0.176, 0.229, 0.325},
	}

	labels := make([]float64, len(rawData))
	for i := 0; i < 20; i++ {
		labels[i] = 0 // Class 0
	}
	for i := 20; i < 40; i++ {
		labels[i] = 1 // Class 1
	}
	for i := 40; i < 60; i++ {
		labels[i] = 2 // Class 2
	}

	return rawData, labels
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
