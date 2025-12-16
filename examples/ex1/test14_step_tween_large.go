package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 13: Real-time Step Training Comparison
//
// LARGE SCALE VERSION - Bigger networks, longer training
// Compares Step + Backprop vs Step + Tween with comprehensive metrics:
// 1. Time: Total training duration (10 seconds per network)
// 2. Memory: Peak RAM allocation during training
// 3. Accuracy: Final classification accuracy
// 4. Speed: Time to reach target accuracy threshold

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 14: LARGE SCALE Step Training Comparison                           ║")
	fmt.Println("║  Bigger Networks | 10s Training | Comprehensive Metrics                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	runDuration := 10 * time.Second // 10 seconds per network (5x longer)
	targetAcc := 90.0               // Higher target for bigger networks

	// Run tests for each architecture (with larger input sizes)
	results := []ComparisonResult{}

	results = append(results, runComparison("Dense", createDenseNetwork, generateTrainingData(8, 3), runDuration, targetAcc))
	results = append(results, runComparison("Conv2D", createConv2DNetwork, generateTrainingData(64, 3), runDuration, targetAcc))
	results = append(results, runComparison("RNN", createRNNNetwork, generateTrainingData(32, 3), runDuration, targetAcc))
	results = append(results, runComparison("LSTM", createLSTMNetwork, generateTrainingData(32, 3), runDuration, targetAcc))
	results = append(results, runComparison("Attention", createAttentionNetwork, generateTrainingData(64, 3), runDuration, targetAcc))
	results = append(results, runComparison("Norm", createNormNetwork, generateTrainingData(16, 3), runDuration, targetAcc))
	results = append(results, runComparison("SwiGLU", createSwiGLUNetwork, generateTrainingData(16, 3), runDuration, targetAcc))
	results = append(results, runComparison("Parallel", createParallelNetwork, generateTrainingData(8, 3), runDuration, targetAcc))
	results = append(results, runComparison("Mixed", createMixedNetwork, generateTrainingData(16, 3), runDuration, targetAcc))

	// Print summary
	printSummaryTable(results, targetAcc)
}

// ============================================================================
// Result Structure with Comprehensive Metrics
// ============================================================================

type Metrics struct {
	Steps        int
	Accuracy     float64
	Loss         float32
	TimeTotal    time.Duration
	TimeToTarget time.Duration         // Time to reach target accuracy (0 if never reached)
	MemoryPeakMB float64               // Peak memory allocation in MB
	StepsPS      float64               // Steps per second
	Milestones   map[int]time.Duration // Time to reach each accuracy milestone (10%, 20%, ...)
}

type ComparisonResult struct {
	Name   string
	BP     Metrics
	Tween  Metrics
	Winner string
}

// ============================================================================
// Real-time Comparison Runner
// ============================================================================

func runComparison(name string, netFactory func() *nn.Network, data TrainingData, duration time.Duration, targetAcc float64) ComparisonResult {
	fmt.Printf("\n┌─────────────────────────────────────────────────────────────────────┐\n")
	fmt.Printf("│ %-67s │\n", name+" Network — Running for "+duration.String())
	fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

	// Run both in parallel
	var wg sync.WaitGroup
	wg.Add(2)

	var bpMetrics, tweenMetrics Metrics

	// Step + Backprop
	go func() {
		defer wg.Done()
		net := netFactory()
		bpMetrics = runStepBackprop(net, data, duration, targetAcc, name+" BP")
	}()

	// Step + Tween
	go func() {
		defer wg.Done()
		net := netFactory()
		tweenMetrics = runStepTween(net, data, duration, targetAcc, name+" Tween")
	}()

	wg.Wait()

	// Determine winner based on accuracy, then loss, then time-to-target
	winner := determineWinner(bpMetrics, tweenMetrics)

	return ComparisonResult{
		Name:   name,
		BP:     bpMetrics,
		Tween:  tweenMetrics,
		Winner: winner,
	}
}

func determineWinner(bp, tween Metrics) string {
	// 1. Higher accuracy wins
	if tween.Accuracy > bp.Accuracy+1 {
		return "Tween ✓"
	}
	if bp.Accuracy > tween.Accuracy+1 {
		return "BP ✓"
	}

	// 2. Accuracy tied - compare loss (lower is better)
	tweenLossValid := !math.IsNaN(float64(tween.Loss)) && !math.IsInf(float64(tween.Loss), 0)
	bpLossValid := !math.IsNaN(float64(bp.Loss)) && !math.IsInf(float64(bp.Loss), 0)

	if tweenLossValid && bpLossValid {
		if tween.Loss < bp.Loss*0.8 {
			return "Tween ✓"
		}
		if bp.Loss < tween.Loss*0.8 {
			return "BP ✓"
		}
	} else if tweenLossValid && !bpLossValid {
		return "Tween ✓"
	} else if bpLossValid && !tweenLossValid {
		return "BP ✓"
	}

	// 3. Compare time to reach target (faster is better)
	if tween.TimeToTarget > 0 && bp.TimeToTarget > 0 {
		if tween.TimeToTarget < bp.TimeToTarget*8/10 {
			return "Tween ✓"
		}
		if bp.TimeToTarget < tween.TimeToTarget*8/10 {
			return "BP ✓"
		}
	} else if tween.TimeToTarget > 0 && bp.TimeToTarget == 0 {
		return "Tween ✓"
	} else if bp.TimeToTarget > 0 && tween.TimeToTarget == 0 {
		return "BP ✓"
	}

	return "Tie"
}

// ============================================================================
// Memory Tracking Helpers
// ============================================================================

func getMemoryMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / 1024 / 1024
}

func trackPeakMemory(done chan bool, peakMB *float64) {
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			current := getMemoryMB()
			if current > *peakMB {
				*peakMB = current
			}
		}
	}
}

// ============================================================================
// Step + Backprop Training
// ============================================================================

func runStepBackprop(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64, label string) Metrics {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	// Target queue for delayed targets
	targetDelay := net.TotalLayers()
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.02)
	decayRate := float32(0.9999)
	minLR := float32(0.001)

	// Memory tracking
	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking (10%, 20%, ... 100%)
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	steps := 0
	sampleIdx := 0
	var finalLoss float32
	evalInterval := 5000 // More frequent evaluation for milestone tracking

	fmt.Printf("  [Step+Backprop] Starting...\n")

	for time.Since(start) < duration {
		if steps%20 == 0 {
			sampleIdx = rand.Intn(len(data.Samples))
		}
		sample := data.Samples[sampleIdx]

		state.SetInput(sample.Input)
		net.StepForward(state)
		targetQueue.Push(sample.Target)

		if targetQueue.IsFull() {
			delayedTarget := targetQueue.Pop()
			output := state.GetOutput()

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

			net.StepBackward(state, gradOutput)
			net.ApplyGradients(learningRate)

			learningRate *= decayRate
			if learningRate < minLR {
				learningRate = minLR
			}
		}

		steps++

		// Check accuracy and milestones periodically
		if steps%evalInterval == 0 {
			acc := evaluateSteppingNetwork(net, data, state)
			elapsed := time.Since(start)

			// Record milestones
			for threshold := 10; threshold <= 100; threshold += 10 {
				if milestones[threshold] == 0 && acc >= float64(threshold) {
					milestones[threshold] = elapsed
				}
			}

			if timeToTarget == 0 && acc >= targetAcc {
				timeToTarget = elapsed
			}
		}
	}

	done <- true
	elapsed := time.Since(start)
	accuracy := evaluateSteppingNetwork(net, data, state)

	fmt.Printf("  [Step+Backprop] Done: %dk steps | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		steps/1000, accuracy, finalLoss, peakMB)

	return Metrics{
		Steps:        steps,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    elapsed,
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		StepsPS:      float64(steps) / elapsed.Seconds(),
		Milestones:   milestones,
	}
}

// ============================================================================
// Step + Tween Training
// ============================================================================

func runStepTween(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64, label string) Metrics {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	ts := nn.NewTweenState(net)
	ts.Verbose = false

	// Memory tracking
	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking (10%, 20%, ... 100%)
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	steps := 0
	sampleIdx := 0
	var finalLoss float32
	tweenEvery := 5 // Tween more frequently for better learning
	evalInterval := 5000

	fmt.Printf("  [Step+Tween] Starting...\n")

	for time.Since(start) < duration {
		if steps%20 == 0 {
			sampleIdx = rand.Intn(len(data.Samples))
		}
		sample := data.Samples[sampleIdx]

		state.SetInput(sample.Input)
		net.StepForward(state)

		if steps%tweenEvery == 0 {
			targetClass := argmax(sample.Target)
			loss := ts.TweenStep(net, sample.Input, targetClass, len(sample.Target), 0.15)
			if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
				finalLoss = loss
			}
		}

		steps++

		// Check accuracy and milestones periodically
		if steps%evalInterval == 0 {
			acc := evaluateSteppingNetwork(net, data, state)
			elapsed := time.Since(start)

			// Record milestones
			for threshold := 10; threshold <= 100; threshold += 10 {
				if milestones[threshold] == 0 && acc >= float64(threshold) {
					milestones[threshold] = elapsed
				}
			}

			if timeToTarget == 0 && acc >= targetAcc {
				timeToTarget = elapsed
			}
		}
	}

	done <- true
	elapsed := time.Since(start)
	accuracy := evaluateSteppingNetwork(net, data, state)

	fmt.Printf("  [Step+Tween] Done: %dk steps | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		steps/1000, accuracy, finalLoss, peakMB)

	return Metrics{
		Steps:        steps,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    elapsed,
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		StepsPS:      float64(steps) / elapsed.Seconds(),
		Milestones:   milestones,
	}
}

// ============================================================================
// Evaluation
// ============================================================================

func evaluateSteppingNetwork(net *nn.Network, data TrainingData, state *nn.StepState) float64 {
	correct := 0
	settleSteps := 10

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
// Target Queue
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
	// LARGE: 4 layers, more neurons
	net := nn.NewNetwork(8, 1, 1, 4)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(64, 32, nn.ActivationTanh))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 16, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(16, 3, nn.ActivationSigmoid))
	return net
}

func createConv2DNetwork() *nn.Network {
	// LARGE: Bigger conv with more filters
	net := nn.NewNetwork(64, 1, 1, 3) // 8x8 input
	net.BatchSize = 1

	conv := nn.LayerConfig{
		Type:          nn.LayerConv2D,
		InputHeight:   8,
		InputWidth:    8,
		InputChannels: 1,
		Filters:       8,
		KernelSize:    3,
		Stride:        1,
		Padding:       0,
		OutputHeight:  6,
		OutputWidth:   6,
		Activation:    nn.ActivationLeakyReLU,
	}
	conv.Kernel = make([]float32, 8*1*3*3)
	conv.Bias = make([]float32, 8)
	initRandomSlice(conv.Kernel, 0.2)
	net.SetLayer(0, 0, 0, conv)

	// 8*6*6 = 288 -> 32 -> 3
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(288, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

func createRNNNetwork() *nn.Network {
	// LARGE: Longer sequence, bigger hidden
	net := nn.NewNetwork(32, 1, 1, 2) // 8 seqlen * 4 input
	net.BatchSize = 1

	rnn := nn.LayerConfig{
		Type:         nn.LayerRNN,
		RNNInputSize: 4,
		HiddenSize:   32,
		SeqLength:    8,
		Activation:   nn.ActivationTanh,
	}
	rnn.WeightIH = make([]float32, 32*4)
	rnn.WeightHH = make([]float32, 32*32)
	rnn.BiasH = make([]float32, 32)
	initRandomSlice(rnn.WeightIH, 0.1)
	initRandomSlice(rnn.WeightHH, 0.1)
	net.SetLayer(0, 0, 0, rnn)

	// 8*32=256 -> 3
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(256, 3, nn.ActivationSigmoid))
	return net
}

func createLSTMNetwork() *nn.Network {
	// LARGE: Longer sequence, bigger hidden
	net := nn.NewNetwork(32, 1, 1, 2) // 8 seqlen * 4 input
	net.BatchSize = 1

	lstm := nn.LayerConfig{
		Type:         nn.LayerLSTM,
		RNNInputSize: 4,
		HiddenSize:   32,
		SeqLength:    8,
		Activation:   nn.ActivationTanh,
	}
	ihSize := 32 * 4
	hhSize := 32 * 32

	lstm.WeightIH_i = make([]float32, ihSize)
	lstm.WeightIH_f = make([]float32, ihSize)
	lstm.WeightIH_g = make([]float32, ihSize)
	lstm.WeightIH_o = make([]float32, ihSize)
	lstm.WeightHH_i = make([]float32, hhSize)
	lstm.WeightHH_f = make([]float32, hhSize)
	lstm.WeightHH_g = make([]float32, hhSize)
	lstm.WeightHH_o = make([]float32, hhSize)
	lstm.BiasH_i = make([]float32, 32)
	lstm.BiasH_f = make([]float32, 32)
	lstm.BiasH_g = make([]float32, 32)
	lstm.BiasH_o = make([]float32, 32)

	initRandomSlice(lstm.WeightIH_i, 0.1)
	initRandomSlice(lstm.WeightIH_f, 0.1)
	initRandomSlice(lstm.WeightIH_g, 0.1)
	initRandomSlice(lstm.WeightIH_o, 0.1)
	initRandomSlice(lstm.WeightHH_i, 0.1)
	initRandomSlice(lstm.WeightHH_f, 0.1)
	initRandomSlice(lstm.WeightHH_g, 0.1)
	initRandomSlice(lstm.WeightHH_o, 0.1)
	net.SetLayer(0, 0, 0, lstm)

	// 8*32=256 -> 3
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(256, 3, nn.ActivationSigmoid))
	return net
}

func createAttentionNetwork() *nn.Network {
	// LARGE: Bigger attention with more heads
	net := nn.NewNetwork(64, 1, 1, 3) // 8 seq * 8 dmodel = 64
	net.BatchSize = 1

	// First: Dense layer to stabilize input
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))

	// Attention layer with larger dimensions
	attn := nn.LayerConfig{
		Type:      nn.LayerMultiHeadAttention,
		DModel:    8,
		NumHeads:  4,
		HeadDim:   2,
		SeqLength: 8,
	}
	size := 8 * 8
	attn.QWeights = make([]float32, size)
	attn.KWeights = make([]float32, size)
	attn.VWeights = make([]float32, size)
	attn.OutputWeight = make([]float32, size)
	attn.QBias = make([]float32, 8)
	attn.KBias = make([]float32, 8)
	attn.VBias = make([]float32, 8)
	attn.OutputBias = make([]float32, 8)

	initRandomSlice(attn.QWeights, 0.05)
	initRandomSlice(attn.KWeights, 0.05)
	initRandomSlice(attn.VWeights, 0.05)
	initRandomSlice(attn.OutputWeight, 0.05)
	net.SetLayer(0, 0, 1, attn)

	// Output layer
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 3, nn.ActivationSigmoid))
	return net
}

func createNormNetwork() *nn.Network {
	// LARGE: More layers, bigger dimensions
	net := nn.NewNetwork(16, 1, 1, 5)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(16, 32, nn.ActivationLeakyReLU))

	layerNorm := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: 32,
		Epsilon:  1e-5,
	}
	layerNorm.Gamma = make([]float32, 32)
	layerNorm.Beta = make([]float32, 32)
	for i := range layerNorm.Gamma {
		layerNorm.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, 1, layerNorm)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 16, nn.ActivationLeakyReLU))

	rmsNorm := nn.LayerConfig{
		Type:     nn.LayerRMSNorm,
		NormSize: 16,
		Epsilon:  1e-5,
	}
	rmsNorm.Gamma = make([]float32, 16)
	for i := range rmsNorm.Gamma {
		rmsNorm.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, 3, rmsNorm)

	net.SetLayer(0, 0, 4, nn.InitDenseLayer(16, 3, nn.ActivationSigmoid))
	return net
}

func createSwiGLUNetwork() *nn.Network {
	// LARGE: Bigger intermediate size
	net := nn.NewNetwork(16, 1, 1, 3)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(16, 32, nn.ActivationLeakyReLU))

	swiglu := nn.LayerConfig{
		Type:         nn.LayerSwiGLU,
		InputHeight:  32,
		OutputHeight: 64,
	}
	swiglu.GateWeights = make([]float32, 32*64)
	swiglu.UpWeights = make([]float32, 32*64)
	swiglu.DownWeights = make([]float32, 64*32)
	swiglu.GateBias = make([]float32, 64)
	swiglu.UpBias = make([]float32, 64)
	swiglu.DownBias = make([]float32, 32)
	initRandomSlice(swiglu.GateWeights, 0.05)
	initRandomSlice(swiglu.UpWeights, 0.05)
	initRandomSlice(swiglu.DownWeights, 0.05)
	net.SetLayer(0, 0, 1, swiglu)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

func createParallelNetwork() *nn.Network {
	// LARGE: More branches, bigger dimensions
	net := nn.NewNetwork(8, 1, 1, 3)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 32, nn.ActivationLeakyReLU))

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU),
			nn.InitDenseLayer(32, 32, nn.ActivationTanh),
			nn.InitDenseLayer(32, 32, nn.ActivationSigmoid),
		},
	}
	net.SetLayer(0, 0, 1, parallel)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

func createMixedNetwork() *nn.Network {
	// LARGE: More layers, bigger dimensions
	net := nn.NewNetwork(16, 1, 1, 5)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(16, 64, nn.ActivationLeakyReLU))

	layerNorm := nn.LayerConfig{
		Type:     nn.LayerNorm,
		NormSize: 64,
		Epsilon:  1e-5,
	}
	layerNorm.Gamma = make([]float32, 64)
	layerNorm.Beta = make([]float32, 64)
	for i := range layerNorm.Gamma {
		layerNorm.Gamma[i] = 1.0
	}
	net.SetLayer(0, 0, 1, layerNorm)

	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))

	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(32, 16, nn.ActivationLeakyReLU),
			nn.InitDenseLayer(32, 16, nn.ActivationTanh),
			nn.InitDenseLayer(32, 16, nn.ActivationSigmoid),
		},
	}
	net.SetLayer(0, 0, 3, parallel)

	net.SetLayer(0, 0, 4, nn.InitDenseLayer(16, 3, nn.ActivationSigmoid))
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

	for class := 0; class < numClasses; class++ {
		for i := 0; i < 4; i++ {
			input := make([]float32, inputSize)
			for j := 0; j < inputSize; j++ {
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

func printSummaryTable(results []ComparisonResult, targetAcc float64) {
	fmt.Println("\n")
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                               COMPREHENSIVE COMPARISON SUMMARY                                                    ║")
	fmt.Printf("║                               Target Accuracy: %.0f%%                                                               ║\n", targetAcc)
	fmt.Println("╠═══════════╦═════════════════════════════════════════════╦═════════════════════════════════════════════╦═══════════╣")
	fmt.Println("║ Network   ║ Step + Backprop                             ║ Step + Tween                                ║ Winner    ║")
	fmt.Println("║           ║ Acc%   │ Loss   │ Mem MB │ TimeToTgt        ║ Acc%   │ Loss   │ Mem MB │ TimeToTgt        ║           ║")
	fmt.Println("╠═══════════╬════════╪════════╪════════╪══════════════════╬════════╪════════╪════════╪══════════════════╬═══════════╣")

	bpWins := 0
	tweenWins := 0
	ties := 0

	for _, r := range results {
		if r.Winner == "Tween ✓" {
			tweenWins++
		} else if r.Winner == "BP ✓" {
			bpWins++
		} else {
			ties++
		}

		bpTTT := "N/A"
		if r.BP.TimeToTarget > 0 {
			bpTTT = r.BP.TimeToTarget.Round(time.Millisecond).String()
		}
		tweenTTT := "N/A"
		if r.Tween.TimeToTarget > 0 {
			tweenTTT = r.Tween.TimeToTarget.Round(time.Millisecond).String()
		}

		bpLoss := fmt.Sprintf("%.4f", r.BP.Loss)
		if math.IsNaN(float64(r.BP.Loss)) || math.IsInf(float64(r.BP.Loss), 0) {
			bpLoss = "NaN"
		}
		tweenLoss := fmt.Sprintf("%.4f", r.Tween.Loss)
		if math.IsNaN(float64(r.Tween.Loss)) || math.IsInf(float64(r.Tween.Loss), 0) {
			tweenLoss = "NaN"
		}

		fmt.Printf("║ %-9s ║ %5.1f%% │ %6s │ %6.1f │ %16s ║ %5.1f%% │ %6s │ %6.1f │ %16s ║ %-9s ║\n",
			r.Name,
			r.BP.Accuracy, bpLoss, r.BP.MemoryPeakMB, bpTTT,
			r.Tween.Accuracy, tweenLoss, r.Tween.MemoryPeakMB, tweenTTT,
			r.Winner)
	}

	fmt.Println("╠═══════════╩═════════════════════════════════════════════╩═════════════════════════════════════════════╩═══════════╣")
	fmt.Printf("║ Results: Backprop Wins: %d | Tween Wins: %d | Ties: %d                                                             ║\n",
		bpWins, tweenWins, ties)
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Print convergence curves
	fmt.Println("\n")
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                               CONVERGENCE SPEED (Time to reach each accuracy %)                                  ║")
	fmt.Println("╠═══════════╦═══════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Network   ║ 10%    │ 20%    │ 30%    │ 40%    │ 50%    │ 60%    │ 70%    │ 80%    │ 90%    │ 100%   ║ Method    ║")
	fmt.Println("╠═══════════╬════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╬═══════════╣")

	for _, r := range results {
		// Print BP row
		fmt.Printf("║ %-9s ║ ", r.Name)
		for pct := 10; pct <= 100; pct += 10 {
			t := r.BP.Milestones[pct]
			if t > 0 {
				fmt.Printf("%6s │ ", formatDuration(t))
			} else {
				fmt.Printf("   N/A │ ")
			}
		}
		fmt.Println("Backprop   ║")

		// Print Tween row
		fmt.Printf("║           ║ ")
		for pct := 10; pct <= 100; pct += 10 {
			t := r.Tween.Milestones[pct]
			if t > 0 {
				fmt.Printf("%6s │ ", formatDuration(t))
			} else {
				fmt.Printf("   N/A │ ")
			}
		}
		fmt.Println("Tween      ║")
		fmt.Println("╠───────────╬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────╬───────────╣")
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Legend: Shorter times = faster convergence. '—' means milestone not reached.")
}

// formatDuration formats a duration compactly for the table
func formatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%dμs", d.Microseconds())
	}
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
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
