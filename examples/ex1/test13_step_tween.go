package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 13: Real-time Step Training Comparison
//
// Compares two CONTINUOUS training approaches:
// 1. Step + Backprop: StepForward → StepBackward → ApplyGradients (traditional stepping)
// 2. Step + Tween: StepForward → TweenStep (gradient-free, bidirectional)
//
// Both run for ~2 seconds to demonstrate real-time learning while running.

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 13: Real-time Stepping Training Comparison                         ║")
	fmt.Println("║  Step + Backprop vs Step + Tween — Learning While Running                ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	runDuration := 2 * time.Second

	// Run tests for each architecture
	results := []ComparisonResult{}

	results = append(results, runRealtimeComparison("Dense", createDenseNetwork, generateTrainingData(4, 3), runDuration))
	results = append(results, runRealtimeComparison("Conv2D", createConv2DNetwork, generateTrainingData(16, 3), runDuration))
	results = append(results, runRealtimeComparison("RNN", createRNNNetwork, generateTrainingData(12, 3), runDuration))
	results = append(results, runRealtimeComparison("LSTM", createLSTMNetwork, generateTrainingData(12, 3), runDuration))
	results = append(results, runRealtimeComparison("Attention", createAttentionNetwork, generateTrainingData(16, 3), runDuration))
	results = append(results, runRealtimeComparison("Norm", createNormNetwork, generateTrainingData(8, 3), runDuration))
	results = append(results, runRealtimeComparison("SwiGLU", createSwiGLUNetwork, generateTrainingData(8, 3), runDuration))
	results = append(results, runRealtimeComparison("Parallel", createParallelNetwork, generateTrainingData(4, 3), runDuration))
	results = append(results, runRealtimeComparison("Mixed", createMixedNetwork, generateTrainingData(8, 3), runDuration))

	// Print summary
	printSummaryTable(results)
}

// ============================================================================
// Result Structure
// ============================================================================

type ComparisonResult struct {
	Name string

	// Step + Backprop
	BPSteps    int
	BPAccuracy float64
	BPLoss     float32
	BPStepsPS  float64 // Steps per second

	// Step + Tween
	TweenSteps    int
	TweenAccuracy float64
	TweenLoss     float32
	TweenStepsPS  float64 // Steps per second

	Winner string
}

// ============================================================================
// Real-time Comparison Runner
// ============================================================================

func runRealtimeComparison(name string, netFactory func() *nn.Network, data TrainingData, duration time.Duration) ComparisonResult {
	fmt.Printf("\n┌─────────────────────────────────────────────────────────────────────┐\n")
	fmt.Printf("│ %-67s │\n", name+" Network — Running for "+duration.String())
	fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

	// Run both in parallel to ensure fair comparison
	var wg sync.WaitGroup
	wg.Add(2)

	var bpResult, tweenResult struct {
		steps    int
		accuracy float64
		loss     float32
	}

	// Step + Backprop
	go func() {
		defer wg.Done()
		net := netFactory()
		bpResult.steps, bpResult.accuracy, bpResult.loss = runStepBackprop(net, data, duration, name+" BP")
	}()

	// Step + Tween
	go func() {
		defer wg.Done()
		net := netFactory()
		tweenResult.steps, tweenResult.accuracy, tweenResult.loss = runStepTween(net, data, duration, name+" Tween")
	}()

	wg.Wait()

	// Determine winner based on accuracy (then steps if tied)
	winner := "Backprop"
	if tweenResult.accuracy > bpResult.accuracy {
		winner = "Tween ✓"
	} else if tweenResult.accuracy == bpResult.accuracy && tweenResult.steps > bpResult.steps {
		winner = "Tween ✓"
	}

	return ComparisonResult{
		Name:          name,
		BPSteps:       bpResult.steps,
		BPAccuracy:    bpResult.accuracy,
		BPLoss:        bpResult.loss,
		BPStepsPS:     float64(bpResult.steps) / duration.Seconds(),
		TweenSteps:    tweenResult.steps,
		TweenAccuracy: tweenResult.accuracy,
		TweenLoss:     tweenResult.loss,
		TweenStepsPS:  float64(tweenResult.steps) / duration.Seconds(),
		Winner:        winner,
	}
}

// ============================================================================
// Step + Backprop Training (Continuous Real-time)
// ============================================================================

func runStepBackprop(net *nn.Network, data TrainingData, duration time.Duration, label string) (steps int, accuracy float64, finalLoss float32) {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	// Target queue for delayed targets (accounts for network depth)
	targetDelay := net.TotalLayers()
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.02)
	decayRate := float32(0.9999)
	minLR := float32(0.001)

	start := time.Now()
	sampleIdx := 0
	logInterval := 50000 // Log every N steps

	fmt.Printf("  [Step+Backprop] Starting continuous training...\n")

	for time.Since(start) < duration {
		// Rotate samples
		if steps%20 == 0 {
			sampleIdx = rand.Intn(len(data.Samples))
		}
		sample := data.Samples[sampleIdx]

		// 1. Set Input
		state.SetInput(sample.Input)

		// 2. Step Forward (produces output while training!)
		net.StepForward(state)

		// 3. Queue target (accounts for propagation delay)
		targetQueue.Push(sample.Target)

		// 4. When queue is full, we can compare output to delayed target
		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

			// Calculate loss and gradient
			loss := float32(0)
			gradOutput := make([]float32, len(output))
			for i := 0; i < len(output); i++ {
				p := clamp(output[i], 1e-7, 1-1e-7)
				if delayedTarget[i] > 0.5 {
					loss -= float32(math.Log(float64(p)))
				}
				gradOutput[i] = output[i] - delayedTarget[i]
			}
			finalLoss = loss

			// 5. Step Backward
			net.StepBackward(state, gradOutput)

			// 6. Apply Gradients
			net.ApplyGradients(learningRate)

			// Decay LR
			learningRate *= decayRate
			if learningRate < minLR {
				learningRate = minLR
			}
		}

		steps++

		// Real-time logging
		if steps%logInterval == 0 {
			elapsed := time.Since(start)
			stepsPS := float64(steps) / elapsed.Seconds()
			fmt.Printf("    Step %6d | %.0f steps/sec | Loss: %.4f\n", steps, stepsPS, finalLoss)
		}
	}

	// Final evaluation
	accuracy = evaluateSteppingNetwork(net, data, state)
	fmt.Printf("  [Step+Backprop] Done: %d steps | Acc: %.1f%% | Loss: %.4f\n", steps, accuracy, finalLoss)

	return steps, accuracy, finalLoss
}

// ============================================================================
// Step + Tween Training (Continuous Real-time)
// ============================================================================

func runStepTween(net *nn.Network, data TrainingData, duration time.Duration, label string) (steps int, accuracy float64, finalLoss float32) {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	// Initialize Tween state
	ts := nn.NewTweenState(net)
	ts.Verbose = false

	start := time.Now()
	sampleIdx := 0
	logInterval := 50000 // Log every N steps
	tweenEvery := 10     // Tween every N steps (balance speed vs learning)

	fmt.Printf("  [Step+Tween] Starting continuous training...\n")

	for time.Since(start) < duration {
		// Rotate samples
		if steps%20 == 0 {
			sampleIdx = rand.Intn(len(data.Samples))
		}
		sample := data.Samples[sampleIdx]

		// 1. Set Input
		state.SetInput(sample.Input)

		// 2. Step Forward (produces output while training!)
		net.StepForward(state)

		// 3. Tween step (every N steps to avoid overhead)
		if steps%tweenEvery == 0 {
			targetClass := argmax(sample.Target)
			loss := ts.TweenStep(net, sample.Input, targetClass, len(sample.Target), 0.1)
			finalLoss = loss
		}

		steps++

		// Real-time logging
		if steps%logInterval == 0 {
			elapsed := time.Since(start)
			stepsPS := float64(steps) / elapsed.Seconds()
			avgBudget, _, _ := ts.GetBudgetSummary()
			fmt.Printf("    Step %6d | %.0f steps/sec | Loss: %.4f | Budget: %.3f\n", steps, stepsPS, finalLoss, avgBudget)
		}
	}

	// Final evaluation
	accuracy = evaluateSteppingNetwork(net, data, state)
	fmt.Printf("  [Step+Tween] Done: %d steps | Acc: %.1f%% | Loss: %.4f\n", steps, accuracy, finalLoss)

	return steps, accuracy, finalLoss
}

// ============================================================================
// Evaluation (uses stepping to settle network)
// ============================================================================

func evaluateSteppingNetwork(net *nn.Network, data TrainingData, state *nn.StepState) float64 {
	correct := 0
	settleSteps := 10 // Let network settle

	for _, sample := range data.Samples {
		state.SetInput(sample.Input)
		for i := 0; i < settleSteps; i++ {
			net.StepForward(state)
		}
		output := state.GetOutput()

		predicted := argmax(output)
		expected := argmax(sample.Target)

		if predicted == expected {
			correct++
		}
	}

	return float64(correct) / float64(len(data.Samples)) * 100.0
}

// ============================================================================
// Target Queue (for delayed target matching in stepping)
// ============================================================================

type TargetQueue struct {
	targets [][]float32
	maxSize int
}

func NewTargetQueue(size int) *TargetQueue {
	return &TargetQueue{
		targets: make([][]float32, 0, size),
		maxSize: size,
	}
}

func (q *TargetQueue) Push(target []float32) {
	q.targets = append(q.targets, target)
}

func (q *TargetQueue) Pop() []float32 {
	if len(q.targets) == 0 {
		return nil
	}
	t := q.targets[0]
	q.targets = q.targets[1:]
	return t
}

func (q *TargetQueue) IsFull() bool {
	return len(q.targets) >= q.maxSize
}

// ============================================================================
// Network Factories
// ============================================================================

func createDenseNetwork() *nn.Network {
	net := nn.NewNetwork(4, 1, 1, 3)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(4, 16, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(16, 8, nn.ActivationTanh))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(8, 3, nn.ActivationSigmoid))
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
		Filters:       4,
		KernelSize:    2,
		Stride:        1,
		Padding:       0,
		OutputHeight:  3,
		OutputWidth:   3,
		Activation:    nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 4*1*2*2)
	conv.Bias = make([]float32, 4)
	initRandomSlice(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(36, 3, nn.ActivationSigmoid))
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

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(24, 3, nn.ActivationSigmoid))
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

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(24, 3, nn.ActivationSigmoid))
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

	net.SetLayer(0, 0, 1, nn.InitDenseLayer(16, 3, nn.ActivationSigmoid))
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

	net.SetLayer(0, 0, 3, nn.InitDenseLayer(8, 3, nn.ActivationSigmoid))
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

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(16, 3, nn.ActivationSigmoid))
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

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(8, 3, nn.ActivationSigmoid))
	return net
}

func createMixedNetwork() *nn.Network {
	net := nn.NewNetwork(8, 1, 1, 4)
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

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(16, 8, nn.ActivationLeakyReLU),
			nn.InitDenseLayer(16, 8, nn.ActivationTanh),
		},
	}
	net.SetLayer(0, 0, 2, parallel)

	net.SetLayer(0, 0, 3, nn.InitDenseLayer(8, 3, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Training Data
// ============================================================================

type Sample struct {
	Input  []float32
	Target []float32
	Label  string
}

type TrainingData struct {
	Samples []Sample
}

func generateTrainingData(inputSize, numClasses int) TrainingData {
	samples := []Sample{}

	// Generate samples for each class
	for class := 0; class < numClasses; class++ {
		for i := 0; i < 4; i++ { // 4 samples per class
			input := make([]float32, inputSize)
			for j := 0; j < inputSize; j++ {
				// Base pattern per class + noise
				base := float32(class) / float32(numClasses)
				input[j] = base + rand.Float32()*0.3 - 0.15
				input[j] = clamp(input[j], 0, 1)
			}

			target := make([]float32, numClasses)
			target[class] = 1.0

			samples = append(samples, Sample{
				Input:  input,
				Target: target,
				Label:  fmt.Sprintf("Class%d", class),
			})
		}
	}

	return TrainingData{Samples: samples}
}

// ============================================================================
// Summary Table
// ============================================================================

func printSummaryTable(results []ComparisonResult) {
	fmt.Println("\n")
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                          REAL-TIME STEPPING COMPARISON SUMMARY                                        ║")
	fmt.Println("╠═══════════╦═══════════════════════════════════════╦═══════════════════════════════════════╦═══════════╣")
	fmt.Println("║ Network   ║ Step + Backprop                       ║ Step + Tween                          ║ Winner    ║")
	fmt.Println("║           ║ Steps   │ Steps/s │ Acc%   │ Loss    ║ Steps   │ Steps/s │ Acc%   │ Loss    ║           ║")
	fmt.Println("╠═══════════╬═════════╪═════════╪════════╪═════════╬═════════╪═════════╪════════╪═════════╬═══════════╣")

	tweenWins := 0
	bpWins := 0
	totalBPSteps := 0
	totalTweenSteps := 0

	for _, r := range results {
		totalBPSteps += r.BPSteps
		totalTweenSteps += r.TweenSteps

		if r.Winner == "Tween ✓" {
			tweenWins++
		} else {
			bpWins++
		}

		fmt.Printf("║ %-9s ║ %7d │ %7.0f │ %5.1f%% │ %7.4f ║ %7d │ %7.0f │ %5.1f%% │ %7.4f ║ %-9s ║\n",
			r.Name,
			r.BPSteps, r.BPStepsPS, r.BPAccuracy, r.BPLoss,
			r.TweenSteps, r.TweenStepsPS, r.TweenAccuracy, r.TweenLoss,
			r.Winner)
	}

	fmt.Println("╠═══════════╬═════════╧═════════╧════════╧═════════╬═════════╧═════════╧════════╧═════════╬═══════════╣")
	fmt.Printf("║ TOTAL     ║ Steps: %-28d ║ Steps: %-28d ║           ║\n", totalBPSteps, totalTweenSteps)
	fmt.Println("╠═══════════╩═══════════════════════════════════════╩═══════════════════════════════════════╩═══════════╣")

	fmt.Printf("║ Tween Wins: %d/%d (%.1f%%)  |  Backprop Wins: %d/%d (%.1f%%)                                            ║\n",
		tweenWins, len(results), float64(tweenWins)/float64(len(results))*100,
		bpWins, len(results), float64(bpWins)/float64(len(results))*100)

	speedRatio := float64(totalTweenSteps) / float64(totalBPSteps)
	if speedRatio >= 1 {
		fmt.Printf("║ Tween processed %.2fx more steps (gradient-free overhead is lower)                                   ║\n", speedRatio)
	} else {
		fmt.Printf("║ Backprop processed %.2fx more steps                                                                   ║\n", 1/speedRatio)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Key Insight: Both methods run CONTINUOUSLY — the network produces outputs while learning!")
}

// ============================================================================
// Helper Functions
// ============================================================================

func initRandomSlice(s []float32, scale float32) {
	for i := range s {
		s[i] = (rand.Float32()*2 - 1) * scale
	}
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func argmax(s []float32) int {
	maxIdx := 0
	maxVal := s[0]
	for i := 1; i < len(s); i++ {
		if s[i] > maxVal {
			maxVal = s[i]
			maxIdx = i
		}
	}
	return maxIdx
}
