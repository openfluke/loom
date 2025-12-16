package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 13: Step Forward with Neural Tweening vs Backpropagation
//
// This test compares two training approaches:
// 1. StepForward + NeuralTween: Layer-independent stepping with bidirectional tweening
// 2. Traditional Backpropagation: Forward pass + backward pass + gradient update
//
// Comparison metrics: Accuracy, Training Time, Convergence Speed

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 13: Step Forward + Tween vs Backpropagation                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Run comparative tests for each architecture
	results := []ComparisonResult{}

	results = append(results, runComparison("Dense", createDenseNetwork, generateXORData()))
	results = append(results, runComparison("Conv2D", createConv2DNetwork, generateConv2DData()))
	results = append(results, runComparison("RNN", createRNNNetwork, generateSequenceData(12)))
	results = append(results, runComparison("LSTM", createLSTMNetwork, generateSequenceData(12)))
	results = append(results, runComparison("Attention", createAttentionNetwork, generateSequenceData(16)))
	results = append(results, runComparison("Norm", createNormNetwork, generateSimpleData(8)))
	results = append(results, runComparison("SwiGLU", createSwiGLUNetwork, generateSimpleData(8)))
	results = append(results, runComparison("Parallel", createParallelNetwork, generateXORData()))
	results = append(results, runComparison("Mixed", createMixedNetwork, generateSimpleData(8)))

	// Print summary table
	printSummaryTable(results)
}

// ============================================================================
// Comparison Result Structure
// ============================================================================

type ComparisonResult struct {
	Name           string
	TweenAccuracy  float64
	TweenTime      time.Duration
	TweenFinalLoss float32
	BPAccuracy     float64
	BPTime         time.Duration
	BPFinalLoss    float32
	TweenWins      bool
}

// ============================================================================
// Core Comparison Runner
// ============================================================================

func runComparison(name string, networkFactory func() *nn.Network, data testData) ComparisonResult {
	fmt.Printf("\n┌─────────────────────────────────────────────────────┐\n")
	fmt.Printf("│ %-51s │\n", name+" Network")
	fmt.Printf("└─────────────────────────────────────────────────────┘\n")

	inputs := data.inputs
	expected := data.expected
	epochs := 100

	// ===== STEP FORWARD + TWEEN =====
	fmt.Println("\n  [Step Forward + Tween]")

	netTween := networkFactory()
	tweenAcc, tweenTime, tweenLoss := trainWithTween(netTween, inputs, expected, epochs)

	fmt.Printf("    Final: Acc=%.1f%% | Time=%v | Loss=%.4f\n", tweenAcc, tweenTime.Round(time.Millisecond), tweenLoss)

	// ===== BACKPROPAGATION =====
	fmt.Println("\n  [Backpropagation]")

	netBP := networkFactory()
	bpAcc, bpTime, bpLoss := trainWithBackprop(netBP, inputs, expected, epochs)

	fmt.Printf("    Final: Acc=%.1f%% | Time=%v | Loss=%.4f\n", bpAcc, bpTime.Round(time.Millisecond), bpLoss)

	// ===== COMPARISON =====
	tweenWins := tweenAcc >= bpAcc

	winner := "Backprop"
	if tweenWins {
		winner = "Tween"
	}

	speedup := float64(bpTime) / float64(tweenTime)
	if speedup < 1 {
		speedup = 1 / speedup
	}

	fmt.Printf("\n  📊 Winner: %s (Speedup: %.2fx)\n", winner, speedup)

	return ComparisonResult{
		Name:           name,
		TweenAccuracy:  tweenAcc,
		TweenTime:      tweenTime,
		TweenFinalLoss: tweenLoss,
		BPAccuracy:     bpAcc,
		BPTime:         bpTime,
		BPFinalLoss:    bpLoss,
		TweenWins:      tweenWins,
	}
}

// ============================================================================
// Training with Step Forward + Tween
// ============================================================================

func trainWithTween(net *nn.Network, inputs [][]float32, expected []float64, epochs int) (accuracy float64, elapsed time.Duration, finalLoss float32) {
	// Initialize tween state
	ts := nn.NewTweenState(net)
	ts.Verbose = false

	// Initialize step state
	state := net.InitStepState(len(inputs[0]))
	stepsPerSample := 3 // Number of StepForward calls per sample

	start := time.Now()

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0)

		for i := 0; i < len(inputs); i++ {
			// Set input and step forward multiple times
			state.SetInput(inputs[i])

			for step := 0; step < stepsPerSample; step++ {
				net.StepForward(state)
			}

			// Tween step: bidirectional analysis and weight update
			loss := ts.TweenStep(net, inputs[i], int(expected[i]), 2, 0.1)
			epochLoss += loss
		}

		finalLoss = epochLoss / float32(len(inputs))

		// Progress indicator
		if epoch == epochs/4 || epoch == epochs/2 || epoch == 3*epochs/4 {
			metrics, _ := net.EvaluateNetwork(inputs, expected)
			fmt.Printf("    Epoch %3d: Acc=%.1f%% Loss=%.4f\n", epoch+1, metrics.Score, finalLoss)
		}
	}

	elapsed = time.Since(start)

	// Final evaluation
	metrics, _ := net.EvaluateNetwork(inputs, expected)
	accuracy = metrics.Score

	return accuracy, elapsed, finalLoss
}

// ============================================================================
// Training with Backpropagation
// ============================================================================

func trainWithBackprop(net *nn.Network, inputs [][]float32, expected []float64, epochs int) (accuracy float64, elapsed time.Duration, finalLoss float32) {
	learningRate := float32(0.1)

	start := time.Now()

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := float32(0)

		for i := 0; i < len(inputs); i++ {
			// Forward pass
			output, _ := net.ForwardCPU(inputs[i])

			// Compute loss (MSE)
			loss := float32(0)
			for j := 0; j < len(output); j++ {
				target := float32(0)
				if j == int(expected[i]) {
					target = 1.0
				}
				diff := output[j] - target
				loss += diff * diff
			}
			epochLoss += loss

			// Compute gradient for cross-entropy/softmax output
			gradOutput := make([]float32, len(output))
			for j := 0; j < len(output); j++ {
				target := float32(0)
				if j == int(expected[i]) {
					target = 1.0
				}
				gradOutput[j] = 2.0 * (output[j] - target)
			}

			// Backward pass
			net.BackwardCPU(gradOutput)

			// Update weights
			net.UpdateWeights(learningRate)
		}

		finalLoss = epochLoss / float32(len(inputs))

		// Progress indicator
		if epoch == epochs/4 || epoch == epochs/2 || epoch == 3*epochs/4 {
			metrics, _ := net.EvaluateNetwork(inputs, expected)
			fmt.Printf("    Epoch %3d: Acc=%.1f%% Loss=%.4f\n", epoch+1, metrics.Score, finalLoss)
		}
	}

	elapsed = time.Since(start)

	// Final evaluation
	metrics, _ := net.EvaluateNetwork(inputs, expected)
	accuracy = metrics.Score

	return accuracy, elapsed, finalLoss
}

// ============================================================================
// Network Factory Functions
// ============================================================================

func createDenseNetwork() *nn.Network {
	net := nn.NewNetwork(4, 1, 1, 3)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(4, 8, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(8, 4, nn.ActivationTanh))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(4, 2, nn.ActivationSigmoid))
	return net
}

func createConv2DNetwork() *nn.Network {
	net := nn.NewNetwork(16, 1, 1, 2)
	net.BatchSize = 1

	conv := nn.LayerConfig{
		Type:          nn.LayerConv2D,
		InputHeight:   4,
		InputWidth:    4,
		InputChannels: 1,
		Filters:       2,
		KernelSize:    2,
		Stride:        1,
		Padding:       0,
		OutputHeight:  3,
		OutputWidth:   3,
		Activation:    nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 2*1*2*2)
	conv.Bias = make([]float32, 2)
	initRandomSlice(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(18, 2, nn.ActivationSigmoid))
	return net
}

func createRNNNetwork() *nn.Network {
	net := nn.NewNetwork(12, 1, 1, 2)
	net.BatchSize = 1

	rnn := nn.LayerConfig{
		Type:         nn.LayerRNN,
		RNNInputSize: 4,
		HiddenSize:   8,
		SeqLength:    3,
		Activation:   nn.ActivationTanh,
	}
	rnn.WeightIH = make([]float32, 8*4)
	rnn.WeightHH = make([]float32, 8*8)
	rnn.BiasH = make([]float32, 8)
	initRandomSlice(rnn.WeightIH, 0.1)
	initRandomSlice(rnn.WeightHH, 0.1)
	net.SetLayer(0, 0, 0, rnn)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(24, 2, nn.ActivationSigmoid))
	return net
}

func createLSTMNetwork() *nn.Network {
	net := nn.NewNetwork(12, 1, 1, 2)
	net.BatchSize = 1

	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: 4,
		HiddenSize:   8,
		SeqLength:    3,
		Activation:   nn.ActivationTanh,
	}
	ihSize := 8 * 4
	hhSize := 8 * 8

	lstm.WeightIH_i = make([]float32, ihSize)
	lstm.WeightIH_f = make([]float32, ihSize)
	lstm.WeightIH_g = make([]float32, ihSize)
	lstm.WeightIH_o = make([]float32, ihSize)
	lstm.WeightHH_i = make([]float32, hhSize)
	lstm.WeightHH_f = make([]float32, hhSize)
	lstm.WeightHH_g = make([]float32, hhSize)
	lstm.WeightHH_o = make([]float32, hhSize)
	lstm.BiasH_i = make([]float32, 8)
	lstm.BiasH_f = make([]float32, 8)
	lstm.BiasH_g = make([]float32, 8)
	lstm.BiasH_o = make([]float32, 8)

	initRandomSlice(lstm.WeightIH_i, 0.1)
	initRandomSlice(lstm.WeightIH_f, 0.1)
	initRandomSlice(lstm.WeightIH_g, 0.1)
	initRandomSlice(lstm.WeightIH_o, 0.1)
	initRandomSlice(lstm.WeightHH_i, 0.1)
	initRandomSlice(lstm.WeightHH_f, 0.1)
	initRandomSlice(lstm.WeightHH_g, 0.1)
	initRandomSlice(lstm.WeightHH_o, 0.1)
	net.SetLayer(0, 0, 0, lstm)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(24, 2, nn.ActivationSigmoid))
	return net
}

func createAttentionNetwork() *nn.Network {
	net := nn.NewNetwork(16, 1, 1, 2)
	net.BatchSize = 1

	attn := nn.LayerConfig{
		Type:      nn.LayerMultiHeadAttention,
		DModel:    4,
		NumHeads:  2,
		HeadDim:   2,
		SeqLength: 4,
	}
	size := 4 * 4
	attn.QWeights = make([]float32, size)
	attn.KWeights = make([]float32, size)
	attn.VWeights = make([]float32, size)
	attn.OutputWeight = make([]float32, size)
	attn.QBias = make([]float32, 4)
	attn.KBias = make([]float32, 4)
	attn.VBias = make([]float32, 4)
	attn.OutputBias = make([]float32, 4)

	initRandomSlice(attn.QWeights, 0.1)
	initRandomSlice(attn.KWeights, 0.1)
	initRandomSlice(attn.VWeights, 0.1)
	initRandomSlice(attn.OutputWeight, 0.1)
	net.SetLayer(0, 0, 0, attn)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(16, 2, nn.ActivationSigmoid))
	return net
}

func createNormNetwork() *nn.Network {
	net := nn.NewNetwork(8, 1, 1, 4)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 8, nn.ActivationLeakyReLU))

	layerNorm := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: 8,
		Epsilon:  1e-5,
	}
	layerNorm.Gamma = make([]float32, 8)
	layerNorm.Beta = make([]float32, 8)
	for i := range layerNorm.Gamma {
		layerNorm.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, 1, layerNorm)

	rmsNorm := nn.LayerConfig{
		Type:     nn.LayerRMSNorm,
		NormSize: 8,
		Epsilon:  1e-5,
	}
	rmsNorm.Gamma = make([]float32, 8)
	for i := range rmsNorm.Gamma {
		rmsNorm.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, 2, rmsNorm)

	net.SetLayer(0, 0, 3, nn.InitDenseLayer(8, 2, nn.ActivationSigmoid))
	return net
}

func createSwiGLUNetwork() *nn.Network {
	net := nn.NewNetwork(8, 1, 1, 3)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 16, nn.ActivationLeakyReLU))

	swiglu := nn.LayerConfig{
		Type:         nn.LayerSwiGLU,
		InputHeight:  16,
		OutputHeight: 32,
	}
	swiglu.GateWeights = make([]float32, 16*32)
	swiglu.UpWeights = make([]float32, 16*32)
	swiglu.DownWeights = make([]float32, 32*16)
	swiglu.GateBias = make([]float32, 32)
	swiglu.UpBias = make([]float32, 32)
	swiglu.DownBias = make([]float32, 16)
	initRandomSlice(swiglu.GateWeights, 0.1)
	initRandomSlice(swiglu.UpWeights, 0.1)
	initRandomSlice(swiglu.DownWeights, 0.1)
	net.SetLayer(0, 0, 1, swiglu)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(16, 2, nn.ActivationSigmoid))
	return net
}

func createParallelNetwork() *nn.Network {
	net := nn.NewNetwork(4, 1, 1, 3)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(4, 8, nn.ActivationLeakyReLU))

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(8, 8, nn.ActivationLeakyReLU),
			nn.InitDenseLayer(8, 8, nn.ActivationTanh),
		},
	}
	net.SetLayer(0, 0, 1, parallel)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(8, 2, nn.ActivationSigmoid))
	return net
}

func createMixedNetwork() *nn.Network {
	net := nn.NewNetwork(8, 1, 1, 5)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 16, nn.ActivationLeakyReLU))

	layerNorm := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: 16,
		Epsilon:  1e-5,
	}
	layerNorm.Gamma = make([]float32, 16)
	layerNorm.Beta = make([]float32, 16)
	for i := range layerNorm.Gamma {
		layerNorm.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, 1, layerNorm)

	swiglu := nn.LayerConfig{
		Type:         nn.LayerSwiGLU,
		InputHeight:  16,
		OutputHeight: 32,
	}
	swiglu.GateWeights = make([]float32, 16*32)
	swiglu.UpWeights = make([]float32, 16*32)
	swiglu.DownWeights = make([]float32, 32*16)
	swiglu.GateBias = make([]float32, 32)
	swiglu.UpBias = make([]float32, 32)
	swiglu.DownBias = make([]float32, 16)
	initRandomSlice(swiglu.GateWeights, 0.1)
	initRandomSlice(swiglu.UpWeights, 0.1)
	initRandomSlice(swiglu.DownWeights, 0.1)
	net.SetLayer(0, 0, 2, swiglu)

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(16, 8, nn.ActivationLeakyReLU),
			nn.InitDenseLayer(16, 8, nn.ActivationTanh),
		},
	}
	net.SetLayer(0, 0, 3, parallel)

	net.SetLayer(0, 0, 4, nn.InitDenseLayer(8, 2, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Summary Table
// ============================================================================

func printSummaryTable(results []ComparisonResult) {
	fmt.Println("\n")
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              COMPARISON SUMMARY                                          ║")
	fmt.Println("╠════════════╦═══════════════════════════════╦═══════════════════════════════╦════════════╣")
	fmt.Println("║ Network    ║ Step+Tween                    ║ Backpropagation               ║ Winner     ║")
	fmt.Println("║            ║ Acc%   │ Time     │ Loss      ║ Acc%   │ Time     │ Loss      ║            ║")
	fmt.Println("╠════════════╬════════╪══════════╪═══════════╬════════╪══════════╪═══════════╬════════════╣")

	tweenWins := 0
	bpWins := 0
	totalTweenTime := time.Duration(0)
	totalBPTime := time.Duration(0)

	for _, r := range results {
		winner := "BP"
		if r.TweenWins {
			winner = "Tween ✓"
			tweenWins++
		} else {
			bpWins++
		}

		totalTweenTime += r.TweenTime
		totalBPTime += r.BPTime

		fmt.Printf("║ %-10s ║ %5.1f%% │ %8v │ %9.4f ║ %5.1f%% │ %8v │ %9.4f ║ %-10s ║\n",
			r.Name,
			r.TweenAccuracy,
			r.TweenTime.Round(time.Millisecond),
			r.TweenFinalLoss,
			r.BPAccuracy,
			r.BPTime.Round(time.Millisecond),
			r.BPFinalLoss,
			winner)
	}

	fmt.Println("╠════════════╬════════╧══════════╧═══════════╬════════╧══════════╧═══════════╬════════════╣")
	fmt.Printf("║ TOTAL      ║ Time: %-22v ║ Time: %-22v ║            ║\n",
		totalTweenTime.Round(time.Millisecond),
		totalBPTime.Round(time.Millisecond))
	fmt.Println("╠════════════╩═════════════════════════════════════════════════════════════╩════════════╣")

	fmt.Printf("║ Tween Wins: %d/%d (%.1f%%)                                                              ║\n",
		tweenWins, len(results), float64(tweenWins)/float64(len(results))*100)
	fmt.Printf("║ BP Wins:    %d/%d (%.1f%%)                                                              ║\n",
		bpWins, len(results), float64(bpWins)/float64(len(results))*100)

	speedup := float64(totalBPTime) / float64(totalTweenTime)
	if speedup >= 1 {
		fmt.Printf("║ Tween is %.2fx faster overall                                                         ║\n", speedup)
	} else {
		fmt.Printf("║ BP is %.2fx faster overall                                                            ║\n", 1/speedup)
	}

	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════╝")
}

// ============================================================================
// Data Generation
// ============================================================================

type testData struct {
	inputs   [][]float32
	expected []float64
}

func generateXORData() testData {
	inputs := [][]float32{
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 1.0},
		{1.0, 0.0, 1.0, 0.0},
		{1.0, 1.0, 1.0, 1.0},
	}
	expected := []float64{0, 1, 1, 0}
	return testData{inputs, expected}
}

func generateConv2DData() testData {
	inputs := make([][]float32, 8)
	expected := make([]float64, 8)

	for i := 0; i < 8; i++ {
		inputs[i] = make([]float32, 16)
		for j := 0; j < 16; j++ {
			inputs[i][j] = rand.Float32()
		}
		if inputs[i][0]+inputs[i][1] > 1.0 {
			expected[i] = 1
		} else {
			expected[i] = 0
		}
	}
	return testData{inputs, expected}
}

func generateSequenceData(inputSize int) testData {
	inputs := make([][]float32, 8)
	expected := make([]float64, 8)

	for i := 0; i < 8; i++ {
		inputs[i] = make([]float32, inputSize)
		sum := float32(0)
		for j := 0; j < inputSize; j++ {
			inputs[i][j] = rand.Float32()
			sum += inputs[i][j]
		}
		if sum > float32(inputSize)/2 {
			expected[i] = 1
		} else {
			expected[i] = 0
		}
	}
	return testData{inputs, expected}
}

func generateSimpleData(inputSize int) testData {
	inputs := make([][]float32, 16)
	expected := make([]float64, 16)

	for i := 0; i < 16; i++ {
		inputs[i] = make([]float32, inputSize)
		sum := float32(0)
		for j := 0; j < inputSize; j++ {
			inputs[i][j] = rand.Float32()
			sum += inputs[i][j]
		}
		if sum > float32(inputSize)/2 {
			expected[i] = 1
		} else {
			expected[i] = 0
		}
	}
	return testData{inputs, expected}
}

// ============================================================================
// Helper Functions
// ============================================================================

func randomFloat() float32 {
	return rand.Float32()
}

func initRandomSlice(s []float32, scale float32) {
	for i := range s {
		s[i] = (rand.Float32()*2 - 1) * scale
	}
}

func getNetworkWeightSum(net *nn.Network) float32 {
	sum := float32(0)
	for i := 0; i < net.TotalLayers(); i++ {
		row := i / (net.GridCols * net.LayersPerCell)
		col := (i / net.LayersPerCell) % net.GridCols
		layer := i % net.LayersPerCell
		cfg := net.GetLayer(row, col, layer)
		if cfg != nil {
			for _, w := range cfg.Kernel {
				sum += float32(math.Abs(float64(w)))
			}
			for _, b := range cfg.Bias {
				sum += float32(math.Abs(float64(b)))
			}
		}
	}
	return sum
}
