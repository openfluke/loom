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

// Test 16: Batch Training Comparison
//
// Compares 4 training modes:
// 1. Step + Backprop (baseline stepping with backpropagation)
// 2. Step + Tween (stepping with neural tweening)
// 3. Batch Tween (non-stepping, batch training with tweening)
// 4. Step + Batch Tween (stepping with batch-accumulated tweening)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 16: BATCH TRAINING Comparison                                      ║")
	fmt.Println("║  6 Modes | Variable Depth | Step vs Batch Analysis                       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	runDuration := 10 * time.Second
	targetAcc := 90.0
	batchSize := 32

	// Layer depths to test
	layerDepths := []int{3, 5, 9, 15, 20}

	results := []nn.ComparisonResult{}

	for _, numLayers := range layerDepths {
		fmt.Printf("\n========== TESTING WITH %d LAYERS ==========\n", numLayers)

		// Networks compatible with Step modes (Dense, Conv2D)
		results = append(results, runComparison(fmt.Sprintf("Dense-%dL", numLayers), func() *nn.Network { return createDenseNetwork(numLayers) }, generateTrainingData(8, 3), runDuration, targetAcc, batchSize, true))
		results = append(results, runComparison(fmt.Sprintf("Conv2D-%dL", numLayers), func() *nn.Network { return createConv2DNetwork(numLayers) }, generateTrainingData(64, 3), runDuration, targetAcc, batchSize, true))

		// Networks that crash with Step modes - run only Batch Tween
		results = append(results, runComparison(fmt.Sprintf("RNN-%dL", numLayers), func() *nn.Network { return createRNNNetwork(numLayers) }, generateTrainingData(32, 3), runDuration, targetAcc, batchSize, false))
		results = append(results, runComparison(fmt.Sprintf("LSTM-%dL", numLayers), func() *nn.Network { return createLSTMNetwork(numLayers) }, generateTrainingData(32, 3), runDuration, targetAcc, batchSize, false))
		results = append(results, runComparison(fmt.Sprintf("Attn-%dL", numLayers), func() *nn.Network { return createAttentionNetwork(numLayers) }, generateTrainingData(64, 3), runDuration, targetAcc, batchSize, false))
		results = append(results, runComparison(fmt.Sprintf("Norm-%dL", numLayers), func() *nn.Network { return createNormNetwork(numLayers) }, generateTrainingData(16, 3), runDuration, targetAcc, batchSize, false))
		results = append(results, runComparison(fmt.Sprintf("SwiGLU-%dL", numLayers), func() *nn.Network { return createSwiGLUNetwork(numLayers) }, generateTrainingData(16, 3), runDuration, targetAcc, batchSize, false))
	}

	printSummaryTable(results, targetAcc)
}

// createNormNetwork creates a network with LayerNorm - variable depth
func createNormNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(16, 1, 1, numLayers)
	net.BatchSize = 1

	// First layer: input -> hidden
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(16, 32, nn.ActivationLeakyReLU))

	// Middle layers: alternating Dense and LayerNorm
	for i := 1; i < numLayers-1; i++ {
		if i%2 == 1 {
			// LayerNorm
			norm := nn.LayerConfig{
				Type:         nn.LayerNorm,
				InputHeight:  32,
				OutputHeight: 32,
			}
			norm.Gamma = make([]float32, 32)
			norm.Beta = make([]float32, 32)
			for j := range norm.Gamma {
				norm.Gamma[j] = 1.0
			}
			net.SetLayer(0, 0, i, norm)
		} else {
			// Dense
			net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
		}
	}

	// Output layer
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

// createSwiGLUNetwork creates a network with SwiGLU - variable depth
func createSwiGLUNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(16, 1, 1, numLayers)
	net.BatchSize = 1

	// First layer: input -> hidden
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(16, 32, nn.ActivationLeakyReLU))

	// Middle layers: alternating Dense and SwiGLU
	for i := 1; i < numLayers-1; i++ {
		if i%2 == 1 {
			// SwiGLU
			swiglu := nn.LayerConfig{
				Type:         nn.LayerSwiGLU,
				InputHeight:  32,
				OutputHeight: 32,
			}
			swiglu.GateWeights = make([]float32, 32*32)
			swiglu.UpWeights = make([]float32, 32*32)
			swiglu.DownWeights = make([]float32, 32*32)
			swiglu.GateBias = make([]float32, 32)
			swiglu.UpBias = make([]float32, 32)
			swiglu.DownBias = make([]float32, 32)
			initRandomSlice(swiglu.GateWeights, 0.1)
			initRandomSlice(swiglu.UpWeights, 0.1)
			initRandomSlice(swiglu.DownWeights, 0.1)
			net.SetLayer(0, 0, i, swiglu)
		} else {
			// Dense
			net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
		}
	}

	// Output layer
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Comparison Runner
// ============================================================================

func runComparison(name string, netFactory func() *nn.Network, data TrainingData, duration time.Duration, targetAcc float64, batchSize int, supportsStep bool) nn.ComparisonResult {
	fmt.Printf("\n┌─────────────────────────────────────────────────────────────────────┐\n")
	fmt.Printf("│ %-67s │\n", name+" Network — Running for "+duration.String())
	fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

	var wg sync.WaitGroup
	var normalBP, normalTween, stepBP, stepTween, stepTweenChain, batchTween, stepBatchTween nn.TrainingMetrics

	// Count modes to run:
	// - NormalBP + NormalTween + BatchTween always run (they use network.Train/TweenStep which work on all layers)
	// - StepBP + StepTween + StepTweenChain + StepBatchTween only if supportsStep (Dense/Conv2D)
	modesToRun := 3 // Normal BP, Normal Tween, Batch Tween always run
	if supportsStep {
		modesToRun += 4 // Step BP, Step Tween, Step Tween Chain, Step Batch Tween
	}
	wg.Add(modesToRun)

	// 1. Normal Backprop (uses network.Train - works on all layers)
	go func() {
		defer wg.Done()
		net := netFactory()
		normalBP = runNormalBackprop(net, data, duration, targetAcc)
	}()

	// 2. Normal Tween (always runs - uses TweenStep which works on all layers)
	go func() {
		defer wg.Done()
		net := netFactory()
		normalTween = runNormalTween(net, data, duration, targetAcc)
	}()

	// 3. Step + Backprop (only if layer supports stepping)
	if supportsStep {
		go func() {
			defer wg.Done()
			net := netFactory()
			stepBP = runStepBackprop(net, data, duration, targetAcc)
		}()
	}

	// 4. Step + Tween Legacy (only if layer supports stepping)
	if supportsStep {
		go func() {
			defer wg.Done()
			net := netFactory()
			stepTween = runStepTween(net, data, duration, targetAcc)
		}()
	}

	// 5. Step + Tween Chain Rule (only if layer supports stepping)
	if supportsStep {
		go func() {
			defer wg.Done()
			net := netFactory()
			stepTweenChain = runStepTweenChain(net, data, duration, targetAcc)
		}()
	}

	// 6. Batch Tween (non-stepping) - always runs
	go func() {
		defer wg.Done()
		net := netFactory()
		batchTween = runBatchTween(net, data, duration, targetAcc, batchSize)
	}()

	// 7. Step + Batch Tween (only if layer supports stepping)
	if supportsStep {
		go func() {
			defer wg.Done()
			net := netFactory()
			stepBatchTween = runStepBatchTween(net, data, duration, targetAcc, batchSize)
		}()
	}

	wg.Wait()

	return nn.ComparisonResult{
		Name:           name,
		NormalBP:       normalBP,
		NormalTween:    normalTween,
		StepBP:         stepBP,
		StepTween:      stepTween,
		StepTweenChain: stepTweenChain,
		BatchTween:     batchTween,
		StepBatchTween: stepBatchTween,
	}
}

// ============================================================================
// Training Mode 1: Normal Backprop (no stepping)
// ============================================================================

func runNormalBackprop(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64) nn.TrainingMetrics {
	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	// Convert to TrainingBatch format
	batches := make([]nn.TrainingBatch, len(data.Samples))
	for i, s := range data.Samples {
		batches[i] = nn.TrainingBatch{
			Input:  s.Input,
			Target: s.Target,
		}
	}

	start := time.Now()
	var timeToTarget time.Duration
	totalEpochs := 0
	var finalLoss float32

	fmt.Printf("  [Normal BP] Starting...\n")

	// Train in small epoch chunks to track milestones
	epochsPerChunk := 5
	for time.Since(start) < duration {
		config := &nn.TrainingConfig{
			Epochs:          epochsPerChunk,
			LearningRate:    0.02,
			UseGPU:          false,
			PrintEveryBatch: 0,
			GradientClip:    1.0,
			LossType:        "mse",
			Verbose:         false,
		}

		result, err := net.Train(batches, config)
		if err != nil {
			break
		}

		totalEpochs += epochsPerChunk
		finalLoss = float32(result.FinalLoss)

		// Check milestones
		acc := evaluateNetwork(net, data)
		elapsed := time.Since(start)
		for threshold := 10; threshold <= 100; threshold += 10 {
			if milestones[threshold] == 0 && acc >= float64(threshold) {
				milestones[threshold] = elapsed
			}
		}
		if timeToTarget == 0 && acc >= targetAcc {
			timeToTarget = elapsed
		}
	}

	done <- true
	accuracy := evaluateNetwork(net, data)

	fmt.Printf("  [Normal BP] Done: %d epochs | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		totalEpochs, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        totalEpochs * len(data.Samples),
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Training Mode 2: Normal Tween (no stepping)
// ============================================================================

func runNormalTween(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64) nn.TrainingMetrics {
	ts := nn.NewTweenState(net, nil)
	ts.Verbose = false
	outputSize := len(data.Samples[0].Target)

	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	epochs := 0
	var finalLoss float32

	fmt.Printf("  [Normal Tween] Starting...\n")

	for time.Since(start) < duration {
		epochLoss := float32(0)
		for _, sample := range data.Samples {
			targetClass := argmax(sample.Target)
			loss := ts.TweenStep(net, sample.Input, targetClass, outputSize, 0.15)
			if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
				epochLoss += loss
			}
		}

		avgLoss := epochLoss / float32(len(data.Samples))
		if !math.IsNaN(float64(avgLoss)) && !math.IsInf(float64(avgLoss), 0) {
			finalLoss = avgLoss
		}

		epochs++

		if epochs%10 == 0 {
			acc := evaluateNetwork(net, data)
			elapsed := time.Since(start)
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
	accuracy := evaluateNetwork(net, data)

	fmt.Printf("  [Normal Tween] Done: %d epochs | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		epochs, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        epochs * len(data.Samples),
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Training Mode 3: Step + Backprop
// ============================================================================

func runStepBackprop(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64) nn.TrainingMetrics {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	targetDelay := net.TotalLayers()
	targetQueue := NewTargetQueue(targetDelay)

	learningRate := float32(0.02)
	decayRate := float32(0.9999)
	minLR := float32(0.001)

	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	steps := 0
	sampleIdx := 0
	var finalLoss float32

	fmt.Printf("  [Step+BP] Starting...\n")

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

			// Calculate gradients
			gradOutput := make([]float32, len(output))
			for i := 0; i < len(output) && i < len(delayedTarget); i++ {
				gradOutput[i] = output[i] - delayedTarget[i]
			}

			net.StepBackward(state, gradOutput)
			net.ApplyGradients(learningRate)

			loss := calculateLoss(output, delayedTarget)
			if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
				finalLoss = loss
			}
		}

		learningRate = max32(learningRate*decayRate, minLR)
		steps++

		if steps%5000 == 0 {
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
	accuracy := evaluateSteppingNetwork(net, data, state)

	fmt.Printf("  [Step+BP] Done: %dk steps | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		steps/1000, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        steps,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Training Mode 2: Step + Tween
// ============================================================================

func runStepTween(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64) nn.TrainingMetrics {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	ts := nn.NewTweenState(net, nil)
	ts.Verbose = false

	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	steps := 0
	sampleIdx := 0
	var finalLoss float32
	tweenEvery := 5

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

		if steps%5000 == 0 {
			acc := evaluateSteppingNetwork(net, data, state)
			elapsed := time.Since(start)
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
	accuracy := evaluateSteppingNetwork(net, data, state)

	fmt.Printf("  [Step+Tween] Done: %dk steps | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		steps/1000, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        steps,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Training Mode 5: Step + Tween Chain Rule
// ============================================================================

func runStepTweenChain(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64) nn.TrainingMetrics {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	ts := nn.NewTweenState(net, nil)
	ts.Verbose = false
	ts.Config.UseChainRule = true // Use chain rule gradient propagation

	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	steps := 0
	sampleIdx := 0
	var finalLoss float32
	tweenEvery := 5

	fmt.Printf("  [TChain] Starting...\n")

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

		if steps%5000 == 0 {
			acc := evaluateSteppingNetwork(net, data, state)
			elapsed := time.Since(start)
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
	accuracy := evaluateSteppingNetwork(net, data, state)

	fmt.Printf("  [TChain] Done: %dk steps | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		steps/1000, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        steps,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Training Mode 6: Batch Tween (non-stepping)
// ============================================================================

func runBatchTween(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64, batchSize int) nn.TrainingMetrics {
	ts := nn.NewTweenState(net, nil)
	ts.Verbose = false
	ts.Config.BatchSize = batchSize

	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	// Prepare batch inputs/targets
	numSamples := len(data.Samples)
	inputs := make([][]float32, numSamples)
	targets := make([]int, numSamples)
	for i, s := range data.Samples {
		inputs[i] = s.Input
		targets[i] = argmax(s.Target)
	}
	outputSize := len(data.Samples[0].Target)

	start := time.Now()
	var timeToTarget time.Duration
	batches := 0
	var finalLoss float32

	fmt.Printf("  [Batch Tween] Starting...\n")

	for time.Since(start) < duration {
		// Shuffle and create batch
		perm := rand.Perm(numSamples)
		batchInputs := make([][]float32, batchSize)
		batchTargets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			idx := perm[i%numSamples]
			batchInputs[i] = inputs[idx]
			batchTargets[i] = targets[idx]
		}

		// Train batch
		loss := ts.TweenBatch(net, batchInputs, batchTargets, outputSize, 0.15)
		if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
			finalLoss = loss
		}

		batches++

		if batches%100 == 0 {
			acc := evaluateNetwork(net, data)
			elapsed := time.Since(start)
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
	accuracy := evaluateNetwork(net, data)

	fmt.Printf("  [Batch Tween] Done: %d batches | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		batches, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        batches * batchSize,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Training Mode 4: Step + Batch Tween
// ============================================================================

func runStepBatchTween(net *nn.Network, data TrainingData, duration time.Duration, targetAcc float64, batchSize int) nn.TrainingMetrics {
	inputSize := len(data.Samples[0].Input)
	state := net.InitStepState(inputSize)

	ts := nn.NewTweenState(net, nil)
	ts.Verbose = false
	ts.Config.BatchSize = batchSize

	outputSize := len(data.Samples[0].Target)

	peakMB := getMemoryMB()
	done := make(chan bool)
	go trackPeakMemory(done, &peakMB)

	// Milestone tracking
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}

	start := time.Now()
	var timeToTarget time.Duration
	steps := 0
	sampleIdx := 0
	var finalLoss float32

	fmt.Printf("  [Step+Batch Tween] Starting...\n")

	for time.Since(start) < duration {
		if steps%20 == 0 {
			sampleIdx = rand.Intn(len(data.Samples))
		}
		sample := data.Samples[sampleIdx]

		state.SetInput(sample.Input)
		net.StepForward(state)

		// Accumulate gaps
		targetClass := argmax(sample.Target)
		loss := ts.TweenStepAccumulate(net, sample.Input, targetClass, outputSize)
		if !math.IsNaN(float64(loss)) && !math.IsInf(float64(loss), 0) {
			finalLoss = loss
		}

		steps++

		// Apply batch when full
		if ts.BatchCount >= batchSize {
			ts.TweenBatchApply(net, 0.15)
		}

		if steps%5000 == 0 {
			acc := evaluateSteppingNetwork(net, data, state)
			elapsed := time.Since(start)
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
	accuracy := evaluateSteppingNetwork(net, data, state)

	fmt.Printf("  [Step+Batch Tween] Done: %dk steps | Acc: %.1f%% | Loss: %.4f | Mem: %.1fMB\n",
		steps/1000, accuracy, finalLoss, peakMB)

	return nn.TrainingMetrics{
		Steps:        steps,
		Accuracy:     accuracy,
		Loss:         finalLoss,
		TimeTotal:    time.Since(start),
		TimeToTarget: timeToTarget,
		MemoryPeakMB: peakMB,
		Milestones:   milestones,
	}
}

// ============================================================================
// Evaluation Functions
// ============================================================================

func evaluateSteppingNetwork(net *nn.Network, data TrainingData, state *nn.StepState) float64 {
	correct := 0
	stepsNeeded := net.TotalLayers() + 2

	for _, sample := range data.Samples {
		state.SetInput(sample.Input)
		for s := 0; s < stepsNeeded; s++ {
			net.StepForward(state)
		}
		predicted := argmax(state.GetOutput())
		expected := argmax(sample.Target)
		if predicted == expected {
			correct++
		}
	}

	return float64(correct) / float64(len(data.Samples)) * 100.0
}

func evaluateNetwork(net *nn.Network, data TrainingData) float64 {
	correct := 0
	for _, sample := range data.Samples {
		output, _ := net.ForwardCPU(sample.Input)
		predicted := argmax(output)
		expected := argmax(sample.Target)
		if predicted == expected {
			correct++
		}
	}
	return float64(correct) / float64(len(data.Samples)) * 100.0
}

func calculateLoss(output, target []float32) float32 {
	loss := float32(0)
	for i := range output {
		if i < len(target) {
			diff := output[i] - target[i]
			loss += diff * diff
		}
	}
	return loss
}

// ============================================================================
// Memory Tracking
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

func createDenseNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(8, 1, 1, numLayers)
	net.BatchSize = 1

	// First layer: input -> hidden
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 64, nn.ActivationLeakyReLU))

	// Middle layers with varying sizes
	hiddenSizes := []int{64, 48, 32, 24, 16}
	for i := 1; i < numLayers-1; i++ {
		inSize := hiddenSizes[(i-1)%len(hiddenSizes)]
		outSize := hiddenSizes[i%len(hiddenSizes)]
		activation := nn.ActivationLeakyReLU
		if i%2 == 0 {
			activation = nn.ActivationTanh
		}
		net.SetLayer(0, 0, i, nn.InitDenseLayer(inSize, outSize, activation))
	}

	// Output layer
	lastHidden := hiddenSizes[(numLayers-2)%len(hiddenSizes)]
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(lastHidden, 3, nn.ActivationSigmoid))
	return net
}

func createConv2DNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(64, 1, 1, numLayers)
	net.BatchSize = 1

	// First layer: Conv2D
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

	// Middle layers: Dense
	for i := 1; i < numLayers-1; i++ {
		if i == 1 {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(288, 64, nn.ActivationLeakyReLU))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
		}
	}

	// Output layer
	if numLayers > 2 {
		net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(64, 3, nn.ActivationSigmoid))
	} else {
		net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(288, 3, nn.ActivationSigmoid))
	}
	return net
}

func createRNNNetwork(numLayers int) *nn.Network {
	// Input: 32, RNN expects seqLength * featureSize = 32 (4 steps x 8 features)
	net := nn.NewNetwork(32, 1, 1, numLayers)
	net.BatchSize = 1

	// First layer: Dense
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))

	// Middle layers: alternating RNN and Dense
	for i := 1; i < numLayers-1; i++ {
		if i%2 == 1 {
			// RNN layer
			rnn := nn.InitRNNLayer(8, 8, 1, 4) // inputSize=8, hiddenSize=8, seqLength=4 -> output 32
			net.SetLayer(0, 0, i, rnn)
		} else {
			// Dense layer
			net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
		}
	}

	// Output layer
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

func createLSTMNetwork(numLayers int) *nn.Network {
	// Input: 32, LSTM expects seqLength * featureSize = 32 (4 steps x 8 features)
	net := nn.NewNetwork(32, 1, 1, numLayers)
	net.BatchSize = 1

	// First layer: Dense
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))

	// Middle layers: alternating LSTM and Dense
	for i := 1; i < numLayers-1; i++ {
		if i%2 == 1 {
			// LSTM layer
			lstm := nn.InitLSTMLayer(8, 8, 1, 4) // inputSize=8, hiddenSize=8, seqLength=4 -> output 32
			net.SetLayer(0, 0, i, lstm)
		} else {
			// Dense layer
			net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
		}
	}

	// Output layer
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(32, 3, nn.ActivationSigmoid))
	return net
}

func createAttentionNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(64, 1, 1, numLayers)
	net.BatchSize = 1
	dModel := 64
	numHeads := 4
	headDim := dModel / numHeads

	// Middle layers: alternating Attention and Dense
	for i := 0; i < numLayers-1; i++ {
		if i%2 == 0 {
			// Multi-head attention
			mha := nn.LayerConfig{
				Type:     nn.LayerMultiHeadAttention,
				DModel:   dModel,
				NumHeads: numHeads,
			}
			mha.QWeights = make([]float32, dModel*dModel)
			mha.KWeights = make([]float32, dModel*dModel)
			mha.VWeights = make([]float32, dModel*dModel)
			mha.OutputWeight = make([]float32, dModel*dModel)
			mha.QBias = make([]float32, dModel)
			mha.KBias = make([]float32, dModel)
			mha.VBias = make([]float32, dModel)
			mha.OutputBias = make([]float32, dModel)
			initRandomSlice(mha.QWeights, 0.1/float32(math.Sqrt(float64(headDim))))
			initRandomSlice(mha.KWeights, 0.1/float32(math.Sqrt(float64(headDim))))
			initRandomSlice(mha.VWeights, 0.1/float32(math.Sqrt(float64(headDim))))
			initRandomSlice(mha.OutputWeight, 0.1/float32(math.Sqrt(float64(dModel))))
			net.SetLayer(0, 0, i, mha)
		} else {
			// Dense layer
			net.SetLayer(0, 0, i, nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU))
		}
	}

	// Output layer
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(dModel, 3, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Data Generation
// ============================================================================

type Sample struct {
	Input  []float32
	Target []float32
}

type TrainingData struct {
	Samples []Sample
}

func generateTrainingData(inputSize, numClasses int) TrainingData {
	data := TrainingData{Samples: make([]Sample, 500)} // More samples
	for i := range data.Samples {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32()*2 - 1
		}

		// HARDER: XOR-like pattern based on quadrants + products
		// Class depends on sign of (x0 * x1) and (x2 * x3) - nonlinear!
		var class int
		if inputSize >= 4 {
			// Quadrant-based: combines multiple dimensions with products
			prod1 := input[0] * input[1]
			prod2 := input[2] * input[3]
			quadrant := 0
			if prod1 > 0 {
				quadrant += 1
			}
			if prod2 > 0 {
				quadrant += 2
			}
			// Add more complexity with sum of remaining features
			extraSum := float32(0)
			for j := 4; j < inputSize; j++ {
				extraSum += input[j]
			}
			if extraSum > 0 {
				quadrant = (quadrant + 1) % 4
			}
			class = quadrant % numClasses
		} else {
			// Fallback for small inputs: simple product rule
			prod := float32(1)
			for _, v := range input {
				prod *= v
			}
			if prod > 0 {
				class = 0
			} else {
				class = 1
			}
			class = class % numClasses
		}

		// Add noise: 10% chance of random class
		if rand.Float32() < 0.1 {
			class = rand.Intn(numClasses)
		}

		target := make([]float32, numClasses)
		target[class] = 1.0

		data.Samples[i] = Sample{Input: input, Target: target}
	}
	return data
}

// ============================================================================
// Summary Table
// ============================================================================

func printSummaryTable(results []nn.ComparisonResult, targetAcc float64) {
	fmt.Println("\n")
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                       TRAINING METHOD COMPARISON (7 Modes)                                                                                                        ║")
	fmt.Printf("║                                                       Target Accuracy: %.0f%%                                                                                                                          ║\n", targetAcc)
	fmt.Println("╠═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════════════════╗")
	fmt.Println("║ Network   ║ NormalBP  ║ NormTween ║ Step+BP   ║ StepTween ║ TChain ║ BatchTwn  ║ StepBatch ║ Best Method       ║")
	fmt.Println("╠═══════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════════════════╣")

	for _, r := range results {
		best := r.DetermineBest()
		fmt.Printf("║ %-9s ║ %6.1f%%   ║ %6.1f%%   ║ %6.1f%%   ║ %6.1f%%   ║ %6.1f%%   ║ %6.1f%%   ║ %6.1f%%   ║ %-17s ║\n",
			r.Name,
			r.NormalBP.Accuracy,
			r.NormalTween.Accuracy,
			r.StepBP.Accuracy,
			r.StepTween.Accuracy,
			r.StepTweenChain.Accuracy,
			r.BatchTween.Accuracy,
			r.StepBatchTween.Accuracy,
			best)
	}

	fmt.Println("╚═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════════════════╝")

	// Print convergence milestone table
	fmt.Println("\n")
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                      CONVERGENCE MILESTONES (Time to reach accuracy %)                                                   ║")
	fmt.Println("╠═══════════╦═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════════╣")
	fmt.Println("║ Network   ║  10%   │  20%   │  30%   │  40%   │  50%   │  60%   │  70%   │  80%   │  90%   │ 100%   ║ Method    ║")
	fmt.Println("╠═══════════╬════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╬═══════════╣")

	for _, r := range results {
		// Print each method that was tested
		methods := []struct {
			name    string
			metrics nn.TrainingMetrics
		}{
			{"NormalBP", r.NormalBP},
			{"NormTween", r.NormalTween},
			{"Step+BP", r.StepBP},
			{"StepTween", r.StepTween},
			{"TChain", r.StepTweenChain},
			{"BatchTween", r.BatchTween},
			{"StepBatch", r.StepBatchTween},
		}

		for _, m := range methods {
			if m.metrics.Accuracy == 0 && m.metrics.Steps == 0 {
				continue // Skip methods that weren't run
			}
			fmt.Printf("║ %-9s ║ ", r.Name)
			for pct := 10; pct <= 100; pct += 10 {
				if m.metrics.Milestones != nil {
					t := m.metrics.Milestones[pct]
					if t > 0 {
						fmt.Printf("%6s │ ", formatDuration(t))
					} else {
						fmt.Printf("   N/A │ ")
					}
				} else {
					fmt.Printf("   N/A │ ")
				}
			}
			fmt.Printf("%-11s║\n", m.name)
		}
		fmt.Println("╠───────────╬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────╬───────────╣")
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println("\nLegend: Shorter times = faster convergence. 'N/A' means milestone not reached.")
}

func formatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%dμs", d.Microseconds())
	}
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func formatLoss(loss float32) string {
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		return "NaN"
	}
	return fmt.Sprintf("%.4f", loss)
}

// ============================================================================
// Helper Functions
// ============================================================================

func initRandomSlice(s []float32, scale float32) {
	for i := range s {
		s[i] = (rand.Float32()*2 - 1) * scale
	}
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
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
