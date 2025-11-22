package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	nn "github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Stepping Neural Network v2: Real-Time Micro-Learning (FIXED) ===")
	fmt.Println("Continuous propagation with EFFECTIVE micro-learning")
	fmt.Println()

	// Create a simple network for binary classification
	// 2x2 grid with heterogeneous agents
	networkJSON := `{
		"batch_size": 1,
		"grid_rows": 2,
		"grid_cols": 2,
		"layers_per_cell": 1,
		"layers": [
			{
				"type": "dense",
				"input_height": 4,
				"output_height": 8,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_height": 8,
				"output_height": 8,
				"activation": "gelu"
			},
			{
				"type": "dense",
				"input_height": 8,
				"output_height": 4,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_height": 4,
				"output_height": 2,
				"activation": "sigmoid"
			}
		]
	}`

	net, err := nn.BuildNetworkFromJSON(networkJSON)
	if err != nil {
		log.Fatalf("Failed to build network: %v", err)
	}
	net.InitializeWeights()

	fmt.Println("Network Architecture (2x2 Grid):")
	fmt.Println("  ┌─────────────────┬─────────────────┐")
	fmt.Println("  │ [0,0] Extractor │ [0,1] Transform │")
	fmt.Println("  │   4→8 (ReLU)    │   8→8 (GeLU)    │")
	fmt.Println("  ├─────────────────┼─────────────────┤")
	fmt.Println("  │ [1,0] Reducer   │ [1,1] Decider   │")
	fmt.Println("  │   8→4 (Tanh)    │   4→2 (Sigmoid) │")
	fmt.Println("  └─────────────────┴─────────────────┘")
	fmt.Println()

	// Initialize stepping state
	inputSize := 4
	state := net.InitStepState(inputSize)

	fmt.Printf("Initialized stepping state with %d layers\n", len(state.GetLayerData())-1)
	fmt.Println()

	// Create training data: XOR-like problem
	// Pattern: if sum(first_half) > sum(second_half) → [1,0], else → [0,1]
	trainingData := []struct {
		input  []float32
		target []float32
		label  string
	}{
		{
			input:  []float32{0.8, 0.9, 0.1, 0.2},
			target: []float32{1.0, 0.0},
			label:  "High-Low",
		},
		{
			input:  []float32{0.2, 0.1, 0.9, 0.8},
			target: []float32{0.0, 1.0},
			label:  "Low-High",
		},
		{
			input:  []float32{0.7, 0.8, 0.2, 0.3},
			target: []float32{1.0, 0.0},
			label:  "High-Low",
		},
		{
			input:  []float32{0.3, 0.2, 0.7, 0.8},
			target: []float32{0.0, 1.0},
			label:  "Low-High",
		},
	}

	fmt.Println("Training Task: Binary Classification")
	fmt.Println("Rule: Compare sum of first half vs second half")
	for i, data := range trainingData {
		fmt.Printf("  Sample %d (%s): %v → %v\n", i, data.label, data.input, data.target)
	}
	fmt.Println()

	// Stepping parameters
	stepsPerSecond := 50
	stepInterval := time.Duration(1000000/stepsPerSecond) * time.Microsecond
	totalSeconds := 20 // Longer to see learning
	totalSteps := stepsPerSecond * totalSeconds

	fmt.Printf("=== Starting Continuous Stepping with IMPROVED Micro-Learning ===\n")
	fmt.Printf("Step Rate: %d steps/second (%v per step)\n", stepsPerSecond, stepInterval)
	fmt.Printf("Duration: %d seconds (%d total steps)\n", totalSeconds, totalSteps)
	fmt.Println()

	// IMPROVED Micro-training configuration
	initialLR := float32(0.15) // Higher initial learning rate
	microConfig := &nn.TrainingConfig{
		Epochs:          3, // 3 epochs per step for better convergence
		LearningRate:    initialLR,
		UseGPU:          false,
		PrintEveryBatch: 0,
		GradientClip:    1.0,
		LossType:        "mse",
		Verbose:         false,
	}

	fmt.Println("IMPROVED Learning Strategy:")
	fmt.Printf("  • Higher learning rate: %.3f (vs 0.05)\n", initialLR)
	fmt.Println("  • 3 epochs per step (vs 1)")
	fmt.Println("  • Slower sample rotation: 25 steps (vs 5)")
	fmt.Println("  • Experience replay: 8 sample buffer")
	fmt.Println("  • Learning rate decay: 0.995 per step")
	fmt.Println()

	// Experience replay buffer
	replayBuffer := make([]nn.TrainingBatch, 0, 8)

	// Sample index for slower rotation
	currentSample := 0

	// Metrics tracking
	lossHistory := make([]float32, 0)
	stepTimes := make([]time.Duration, 0)
	trainTimes := make([]time.Duration, 0)

	// Start stepping loop
	fmt.Println("Network is now ALIVE - stepping and learning continuously...")
	fmt.Println("Each step: Forward → Micro-train (3 epochs) → Decay LR → Weight update")
	fmt.Println()

	ticker := time.NewTicker(stepInterval)
	defer ticker.Stop()

	startTime := time.Now()
	stepCount := 0

	// Display header
	fmt.Printf("%-8s %-12s %-10s %-15s %-25s %-15s %-10s\n",
		"Step", "Sample", "LR", "Train Time", "Output", "Target", "Loss")
	fmt.Println("──────────────────────────────────────────────────────────────────────────────────────────────────────")

	for stepCount < totalSteps {
		select {
		case <-ticker.C:
			// Get current training sample
			sample := trainingData[currentSample]

			// 1. Set input from current training sample
			state.SetInput(sample.input)

			// 2. Execute ONE step for ALL layers (forward propagation)
			stepTime := net.StepForward(state)
			stepTimes = append(stepTimes, stepTime)

			// 3. Get current output
			output := state.GetOutput()

			// 4. Calculate loss (MSE)
			loss := float32(0.0)
			for i := 0; i < len(output); i++ {
				diff := output[i] - sample.target[i]
				loss += diff * diff
			}
			loss /= float32(len(output))
			lossHistory = append(lossHistory, loss)

			// 5. Add to experience replay buffer
			replayBuffer = append(replayBuffer, nn.TrainingBatch{
				Input:  sample.input,
				Target: sample.target,
			})
			if len(replayBuffer) > 8 {
				replayBuffer = replayBuffer[1:] // Keep only last 8
			}

			// 6. MICRO-TRAINING with experience replay
			trainStart := time.Now()

			// Train on current sample + random sample from buffer
			trainBatch := []nn.TrainingBatch{
				{Input: sample.input, Target: sample.target},
			}

			// Add a random replay sample (if buffer has enough)
			if len(replayBuffer) > 2 {
				replayIdx := rand.Intn(len(replayBuffer))
				trainBatch = append(trainBatch, replayBuffer[replayIdx])
			}

			_, err := net.Train(trainBatch, microConfig)
			trainTime := time.Since(trainStart)
			trainTimes = append(trainTimes, trainTime)

			if err != nil {
				fmt.Printf("Warning: micro-training error at step %d: %v\n", stepCount, err)
			}

			// 7. Decay learning rate
			microConfig.LearningRate *= 0.995
			if microConfig.LearningRate < 0.001 {
				microConfig.LearningRate = 0.001 // Floor
			}

			// Print update every 20 steps
			if stepCount%20 == 0 || stepCount < 5 {
				fmt.Printf("%-8d %-12s %-10.4f %-15v [%.3f, %.3f] [%.3f, %.3f] %.6f\n",
					stepCount,
					sample.label,
					microConfig.LearningRate,
					trainTime,
					output[0], output[1],
					sample.target[0], sample.target[1],
					loss)
			}

			stepCount++

			// SLOWER rotation: change sample every 25 steps
			if stepCount%25 == 0 {
				currentSample = (currentSample + 1) % len(trainingData)
			}
		}
	}

	totalTime := time.Since(startTime)
	ticker.Stop()

	fmt.Println()
	fmt.Println("=== Stepping Complete ===")
	fmt.Printf("Total steps: %d\n", stepCount)
	fmt.Printf("Total time: %v\n", totalTime)
	fmt.Printf("Actual step rate: %.1f steps/second\n", float64(stepCount)/totalTime.Seconds())
	fmt.Println()

	// Calculate average step and train times
	avgStepTime := time.Duration(0)
	for _, t := range stepTimes {
		avgStepTime += t
	}
	avgStepTime /= time.Duration(len(stepTimes))

	avgTrainTime := time.Duration(0)
	for _, t := range trainTimes {
		avgTrainTime += t
	}
	avgTrainTime /= time.Duration(len(trainTimes))

	fmt.Printf("Average step time (forward): %v\n", avgStepTime)
	fmt.Printf("Average train time (3 epochs): %v\n", avgTrainTime)
	fmt.Printf("Total time per cycle: %v\n", avgStepTime+avgTrainTime)
	fmt.Printf("Final learning rate: %.6f\n", microConfig.LearningRate)
	fmt.Println()

	// Show loss progression
	fmt.Println("Loss Progression:")
	fmt.Printf("  Initial loss: %.6f\n", lossHistory[0])
	fmt.Printf("  Final loss:   %.6f\n", lossHistory[len(lossHistory)-1])

	// Calculate average loss in first 100 steps vs last 100 steps
	firstLoss := float32(0)
	count := 100
	if count > len(lossHistory) {
		count = len(lossHistory)
	}
	for i := 0; i < count; i++ {
		firstLoss += lossHistory[i]
	}
	firstLoss /= float32(count)

	lastLoss := float32(0)
	start := len(lossHistory) - 100
	if start < 0 {
		start = 0
	}
	for i := start; i < len(lossHistory); i++ {
		lastLoss += lossHistory[i]
	}
	lastLoss /= float32(len(lossHistory) - start)

	fmt.Printf("  Avg first %d: %.6f\n", count, firstLoss)
	fmt.Printf("  Avg last 100:  %.6f\n", lastLoss)

	if firstLoss > lastLoss {
		improvement := (firstLoss - lastLoss) / firstLoss * 100
		fmt.Printf("  ✓ Improvement: %.2f%%\n", improvement)
	} else {
		fmt.Printf("  ⚠ Loss increased by: %.2f%%\n", (lastLoss-firstLoss)/firstLoss*100)
	}
	fmt.Println()

	// Test final predictions on all samples
	fmt.Println("=== Final Network State (After Continuous Learning) ===")
	fmt.Println("Testing all samples:")
	fmt.Println()

	correctCount := 0
	for i, sample := range trainingData {
		state.SetInput(sample.input)

		// Run a few steps to let it propagate
		for s := 0; s < 5; s++ {
			net.StepForward(state)
		}

		output := state.GetOutput()

		// Determine predicted class
		predClass := 0
		if output[1] > output[0] {
			predClass = 1
		}

		expClass := 0
		if sample.target[1] > sample.target[0] {
			expClass = 1
		}

		correct := "✓"
		if predClass == expClass {
			correctCount++
		} else {
			correct = "✗"
		}

		confidence := output[predClass]

		fmt.Printf("Sample %d (%s):\n", i, sample.label)
		fmt.Printf("  Input:      %v\n", sample.input)
		fmt.Printf("  Output:     [%.3f, %.3f]\n", output[0], output[1])
		fmt.Printf("  Target:     [%.3f, %.3f]\n", sample.target[0], sample.target[1])
		fmt.Printf("  Predicted:  Class %d (expected %d) %s (confidence: %.1f%%)\n",
			predClass, expClass, correct, confidence*100)
		fmt.Println()
	}

	accuracy := float32(correctCount) / float32(len(trainingData)) * 100
	fmt.Printf("Final Accuracy: %d/%d (%.1f%%)\n", correctCount, len(trainingData), accuracy)

	if accuracy >= 75.0 {
		fmt.Println("✓ LEARNING SUCCESSFUL! Network learned the pattern!")
	} else if accuracy >= 50.0 {
		fmt.Println("⚠ Partial learning - network is on the right track")
	} else {
		fmt.Println("✗ Learning failed - network performing worse than random")
	}
	fmt.Println()

	// Plot loss over time (text-based)
	fmt.Println("=== Loss History (text plot) ===")
	plotLossHistory(lossHistory, 20)
	fmt.Println()

	// Demonstrate continuous stepping behavior
	fmt.Println("=== Demonstrating Continuous Behavior ===")
	fmt.Println("Setting input and watching network 'think' for 2 seconds...")
	fmt.Println()

	state.SetInput([]float32{0.9, 0.8, 0.2, 0.1})
	fmt.Println("Input: [0.9, 0.8, 0.2, 0.1] (should output ~[1.0, 0.0])")
	fmt.Println()

	// Watch it step
	watchSteps := 50
	watchInterval := 40 * time.Millisecond

	fmt.Printf("%-6s %-25s %-15s\n", "Step", "Output", "Prediction")
	fmt.Println("──────────────────────────────────────────────────────")

	for step := 0; step < watchSteps; step++ {
		net.StepForward(state)
		output := state.GetOutput()

		if step%5 == 0 {
			predClass := 0
			if output[1] > output[0] {
				predClass = 1
			}
			confidence := output[predClass] * 100

			fmt.Printf("%-6d [%.3f, %.3f] Class %d (%.1f%%)\n",
				step, output[0], output[1], predClass, confidence)
		}

		time.Sleep(watchInterval)
	}

	fmt.Println()
	finalOut := state.GetOutput()
	finalClass := 0
	if finalOut[1] > finalOut[0] {
		finalClass = 1
	}
	fmt.Printf("Final: [%.3f, %.3f] → Class %d ",
		finalOut[0], finalOut[1], finalClass)
	if finalClass == 0 {
		fmt.Println("✓ (Correct!)")
	} else {
		fmt.Println("✗ (Should be Class 0)")
	}
	fmt.Println()

	fmt.Println("=== Key Improvements Over v1 ===")
	fmt.Println()
	fmt.Println("1. HIGHER LEARNING RATE:")
	fmt.Printf("   • Started at %.3f (vs 0.05)\n", initialLR)
	fmt.Println("   • More impactful weight updates per step")
	fmt.Println()
	fmt.Println("2. MORE EPOCHS PER STEP:")
	fmt.Println("   • 3 epochs (vs 1) allows better convergence")
	fmt.Println("   • Each sample gets more training time")
	fmt.Println()
	fmt.Println("3. SLOWER SAMPLE ROTATION:")
	fmt.Println("   • 25 steps per sample (vs 5)")
	fmt.Println("   • Prevents catastrophic interference")
	fmt.Println()
	fmt.Println("4. EXPERIENCE REPLAY:")
	fmt.Println("   • Buffer of 8 recent samples")
	fmt.Println("   • Trains on current + 1 random replay")
	fmt.Println("   • Reduces forgetting")
	fmt.Println()
	fmt.Println("5. LEARNING RATE DECAY:")
	fmt.Println("   • 0.995 decay per step")
	fmt.Println("   • Stabilizes as learning progresses")
	fmt.Println()
	fmt.Println("Result: Network should now ACTUALLY LEARN! 🎓✨")
}

// Helper functions

func minDuration(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}
	min := durations[0]
	for _, d := range durations {
		if d < min {
			min = d
		}
	}
	return min
}

func maxDuration(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}
	max := durations[0]
	for _, d := range durations {
		if d > max {
			max = d
		}
	}
	return max
}

// plotLossHistory creates a simple text-based plot of loss over time
func plotLossHistory(history []float32, height int) {
	if len(history) == 0 {
		return
	}

	// Find min and max loss
	minLoss := history[0]
	maxLoss := history[0]
	for _, loss := range history {
		if loss < minLoss {
			minLoss = loss
		}
		if loss > maxLoss {
			maxLoss = loss
		}
	}

	// Add some padding
	lossRange := maxLoss - minLoss
	if lossRange < 0.001 {
		lossRange = 0.001
	}
	minLoss -= lossRange * 0.1
	maxLoss += lossRange * 0.1
	lossRange = maxLoss - minLoss

	// Sample points for plotting (max 80 columns)
	width := 80
	if len(history) < width {
		width = len(history)
	}
	stride := len(history) / width
	if stride < 1 {
		stride = 1
	}

	fmt.Printf("Loss: %.6f (min) to %.6f (max)\n", minLoss, maxLoss)
	fmt.Println()

	// Draw plot from top to bottom
	for row := 0; row < height; row++ {
		threshold := maxLoss - (float32(row)/float32(height-1))*lossRange

		// Y-axis label
		fmt.Printf("%8.5f │", threshold)

		// Plot points
		for col := 0; col < width; col++ {
			idx := col * stride
			if idx >= len(history) {
				idx = len(history) - 1
			}

			loss := history[idx]

			// Check if this cell should be filled
			prevLoss := loss
			if idx > 0 {
				prevLoss = history[idx-1]
			}

			// Simple filled plot
			if (loss <= threshold && prevLoss >= threshold) ||
				(loss >= threshold && prevLoss <= threshold) {
				fmt.Print("●")
			} else if loss >= threshold {
				fmt.Print("·")
			} else {
				fmt.Print(" ")
			}
		}
		fmt.Println()
	}

	// X-axis
	fmt.Print("         └")
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Println()

	fmt.Printf("          0%s%d steps\n",
		spaces(width-15), len(history))
}

func spaces(n int) string {
	if n <= 0 {
		return ""
	}
	s := ""
	for i := 0; i < n; i++ {
		s += " "
	}
	return s
}
