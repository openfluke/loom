package main

import (
	"fmt"
	"log"
	"time"

	nn "github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=== LOOM Stepping Neural Network: Real-Time Learning ===")
	fmt.Println("Continuous propagation where ALL layers step simultaneously")
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
				"input_size": 4,
				"output_size": 8,
				"activation": "relu",
				"comment": "Agent [0,0]: Feature extractor"
			},
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 8,
				"activation": "gelu",
				"comment": "Agent [0,1]: Feature transformer"
			},
			{
				"type": "dense",
				"input_size": 8,
				"output_size": 4,
				"activation": "tanh",
				"comment": "Agent [1,0]: Feature reducer"
			},
			{
				"type": "dense",
				"input_size": 4,
				"output_size": 2,
				"activation": "sigmoid",
				"comment": "Agent [1,1]: Decision maker"
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
	stepsPerSecond := 100
	stepInterval := time.Duration(1000000/stepsPerSecond) * time.Microsecond
	totalSeconds := 5
	totalSteps := stepsPerSecond * totalSeconds

	fmt.Printf("=== Starting Continuous Stepping ===\n")
	fmt.Printf("Step Rate: %d steps/second (%v per step)\n", stepsPerSecond, stepInterval)
	fmt.Printf("Duration: %d seconds (%d total steps)\n", totalSeconds, totalSteps)
	fmt.Println()

	// Sample index for round-robin training
	currentSample := 0

	// Metrics tracking
	lossHistory := make([]float32, 0)
	stepTimes := make([]time.Duration, 0)

	// Start stepping loop
	fmt.Println("Network is now ALIVE - stepping continuously...")
	fmt.Println("Press Ctrl+C to stop (or wait for completion)")
	fmt.Println()

	ticker := time.NewTicker(stepInterval)
	defer ticker.Stop()

	startTime := time.Now()
	stepCount := 0

	// Display header
	fmt.Printf("%-8s %-12s %-15s %-25s %-15s %-10s\n",
		"Step", "Sample", "Step Time", "Output", "Target", "Loss")
	fmt.Println("─────────────────────────────────────────────────────────────────────────────────────")

	for stepCount < totalSteps {
		select {
		case <-ticker.C:
			// Set input from current training sample
			sample := trainingData[currentSample]
			state.SetInput(sample.input)

			// Execute ONE step for ALL layers
			stepTime := net.StepForward(state)
			stepTimes = append(stepTimes, stepTime)

			// Get current output
			output := state.GetOutput()

			// Calculate loss (MSE)
			loss := float32(0.0)
			for i := 0; i < len(output); i++ {
				diff := output[i] - sample.target[i]
				loss += diff * diff
			}
			loss /= float32(len(output))
			lossHistory = append(lossHistory, loss)

			// Print update every 10 steps
			if stepCount%10 == 0 || stepCount < 5 {
				fmt.Printf("%-8d %-12s %-15v [%.3f, %.3f] [%.3f, %.3f] %.6f\n",
					stepCount,
					sample.label,
					stepTime,
					output[0], output[1],
					sample.target[0], sample.target[1],
					loss)
			}

			stepCount++

			// Rotate to next sample every 25 steps (show it all samples)
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

	// Calculate average step time
	avgStepTime := time.Duration(0)
	for _, t := range stepTimes {
		avgStepTime += t
	}
	avgStepTime /= time.Duration(len(stepTimes))

	fmt.Printf("Average step time: %v\n", avgStepTime)
	fmt.Printf("Min step time: %v\n", minDuration(stepTimes))
	fmt.Printf("Max step time: %v\n", maxDuration(stepTimes))
	fmt.Println()

	// Show loss progression
	fmt.Println("Loss Progression:")
	fmt.Printf("  Initial loss: %.6f\n", lossHistory[0])
	fmt.Printf("  Final loss:   %.6f\n", lossHistory[len(lossHistory)-1])

	// Calculate average loss in first 100 steps vs last 100 steps
	firstLoss := float32(0)
	for i := 0; i < 100 && i < len(lossHistory); i++ {
		firstLoss += lossHistory[i]
	}
	firstLoss /= 100

	lastLoss := float32(0)
	start := len(lossHistory) - 100
	if start < 0 {
		start = 0
	}
	for i := start; i < len(lossHistory); i++ {
		lastLoss += lossHistory[i]
	}
	lastLoss /= float32(len(lossHistory) - start)

	fmt.Printf("  Avg first 100: %.6f\n", firstLoss)
	fmt.Printf("  Avg last 100:  %.6f\n", lastLoss)

	if firstLoss > lastLoss {
		improvement := (firstLoss - lastLoss) / firstLoss * 100
		fmt.Printf("  Improvement: %.2f%%\n", improvement)
	} else {
		fmt.Println("  No improvement (may need learning mechanism)")
	}
	fmt.Println()

	// Test final predictions on all samples
	fmt.Println("=== Final Network State ===")
	fmt.Println("Testing all samples:")
	fmt.Println()

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
		if predClass != expClass {
			correct = "✗"
		}

		fmt.Printf("Sample %d (%s):\n", i, sample.label)
		fmt.Printf("  Input:    %v\n", sample.input)
		fmt.Printf("  Output:   [%.3f, %.3f]\n", output[0], output[1])
		fmt.Printf("  Target:   [%.3f, %.3f]\n", sample.target[0], sample.target[1])
		fmt.Printf("  Predicted: Class %d (expected %d) %s\n", predClass, expClass, correct)
		fmt.Println()
	}

	// Demonstrate continuous stepping behavior
	fmt.Println("=== Demonstrating Continuous Behavior ===")
	fmt.Println("Setting input and watching network 'think' for 2 seconds...")
	fmt.Println()

	state.SetInput([]float32{0.9, 0.8, 0.2, 0.1})
	fmt.Println("Input: [0.9, 0.8, 0.2, 0.1] (should output ~[1.0, 0.0])")
	fmt.Println()

	// Watch it step for 2 seconds at higher rate
	watchSteps := 200
	watchInterval := 10 * time.Millisecond

	fmt.Printf("%-6s %-20s %-15s\n", "Step", "Output", "All Layers Active")
	fmt.Println("─────────────────────────────────────────────────")

	for step := 0; step < watchSteps; step++ {
		net.StepForward(state)
		output := state.GetOutput()

		if step%20 == 0 {
			// Show activity in all layers
			allActive := true
			layerActivity := "["
			for l := 0; l < 4; l++ {
				layerOut := state.GetLayerOutput(l + 1)
				if layerOut != nil && len(layerOut) > 0 {
					layerActivity += "✓"
				} else {
					layerActivity += "✗"
					allActive = false
				}
			}
			layerActivity += "]"

			fmt.Printf("%-6d [%.3f, %.3f] %s\n",
				step, output[0], output[1], layerActivity)
		}

		time.Sleep(watchInterval)
	}

	fmt.Println()
	fmt.Printf("Final output after %d continuous steps: [%.3f, %.3f]\n",
		watchSteps, state.GetOutput()[0], state.GetOutput()[1])
	fmt.Println()

	fmt.Println("=== Key Insights ===")
	fmt.Println()
	fmt.Println("1. CONTINUOUS PROPAGATION:")
	fmt.Println("   • Network never stops - always stepping")
	fmt.Println("   • All layers process simultaneously")
	fmt.Println("   • Information flows continuously through grid")
	fmt.Println()
	fmt.Println("2. REAL-TIME BEHAVIOR:")
	fmt.Println("   • Timer-driven execution (not event-driven)")
	fmt.Printf("   • Achieved ~%.1f steps/second\n", float64(stepCount)/totalTime.Seconds())
	fmt.Println("   • Predictable, rhythmic computation")
	fmt.Println()
	fmt.Println("3. SPATIAL COMPUTATION:")
	fmt.Println("   • Each grid cell [row,col] is an independent agent")
	fmt.Println("   • All agents step in parallel")
	fmt.Println("   • Creates wave-like propagation pattern")
	fmt.Println()
	fmt.Println("4. LIVING NETWORK:")
	fmt.Println("   • Network has 'heartbeat' - continuous activity")
	fmt.Println("   • Can process streams of data in real-time")
	fmt.Println("   • More biological - neurons fire continuously")
	fmt.Println()
	fmt.Println("Next steps to add learning:")
	fmt.Println("→ Implement step-wise backprop (gradient per step)")
	fmt.Println("→ Add temporal credit assignment")
	fmt.Println("→ Hebbian-style local learning rules")
	fmt.Println("→ Online gradient descent with step-based updates")
	fmt.Println()
	fmt.Println("Potential applications:")
	fmt.Println("→ Real-time control systems (robotics)")
	fmt.Println("→ Streaming data processing")
	fmt.Println("→ Asynchronous neural computation")
	fmt.Println("→ Spiking neural network simulation")
	fmt.Println("→ Brain-like continuous processing")
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

// Bonus: Demonstrate circular/recurrent behavior
func demonstrateCircularPropagation() {
	fmt.Println("\n=== BONUS: Circular Propagation Demo ===")
	fmt.Println("Create a network where outputs can feed back to inputs")
	fmt.Println("(This requires extending the stepping mechanism)")
	fmt.Println()

	// This would show how you could:
	// 1. Connect layer N output back to layer 0 input
	// 2. Create true recurrent loops in the grid
	// 3. Make information cycle through the network indefinitely

	fmt.Println("Pseudocode for circular connection:")
	fmt.Println("  for each step:")
	fmt.Println("    output = StepForward(state)")
	fmt.Println("    // Feed output back as input")
	fmt.Println("    state.SetInput(mix(external_input, output))")
	fmt.Println("    // Network now processes its own output!")
	fmt.Println()
	fmt.Println("This creates a true 'thinking' loop - network processes,")
	fmt.Println("outputs, receives its output, processes again, forever.")
}
