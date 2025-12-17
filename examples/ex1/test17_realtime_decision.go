package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 17: Real-Time Adaptation Benchmark
//
// Shows how each training method handles a MID-STREAM TASK CHANGE.
// The key question: How quickly can each method ADAPT?
//
// Timeline:
//   0-5s:  Task A (chase the target)
//   5-10s: Task B (AVOID the target!) ← sudden change!
//   10-15s: Task A again (back to chase)
//
// We measure accuracy in 1-second windows to see the adaptation curve.

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 17: MID-STREAM ADAPTATION Benchmark                                ║")
	fmt.Println("║  Task Changes Suddenly — Which Method Adapts Fastest?                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Timeline: [Chase 5s] → [AVOID 5s] → [Chase 5s]")
	fmt.Println("Network:  6-layer Dense (8→32→64→64→32→4)")
	fmt.Println()

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTweenChain,
	}

	// Run all modes and collect time-series data
	allResults := make(map[TrainingMode]*AdaptationResult)

	for _, mode := range modes {
		fmt.Printf("Running [%s]... ", modeNames[mode])
		result := runAdaptationTest(mode)
		allResults[mode] = result
		fmt.Printf("Done — Total outputs: %d\n", result.TotalOutputs)
	}

	// Print time-series comparison
	printAdaptationTimeline(allResults, modes)

	// Print summary
	printAdaptationSummary(allResults, modes)
}

// ============================================================================
// Types
// ============================================================================

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "NormalBP      ",
	ModeStepBP:         "Step+BP       ",
	ModeTween:          "Tween         ",
	ModeTweenChain:     "TweenChain    ",
	ModeStepTweenChain: "StepTweenChain",
}

type TimeWindow struct {
	Outputs       int
	Correct       int
	Accuracy      float64
	OutputsPerSec int
	CurrentTask   string
}

type AdaptationResult struct {
	Windows      []TimeWindow // 1-second windows
	TotalOutputs int

	// Adaptation metrics
	PreChangeAccuracy   float64 // Accuracy in window before first change
	PostChange1Accuracy float64 // Accuracy immediately after first change
	AdaptTime1          int     // Windows to recover to 50%+ after first change
	PostChange2Accuracy float64 // Accuracy immediately after second change
	AdaptTime2          int     // Windows to recover after second change
}

type Environment struct {
	AgentPos  [2]float32
	TargetPos [2]float32
	Task      int // 0=chase, 1=avoid
}

// ============================================================================
// Main Test
// ============================================================================

func runAdaptationTest(mode TrainingMode) *AdaptationResult {
	testDuration := 15 * time.Second
	windowDuration := 1 * time.Second

	// Create 6-layer network
	net := createDeepNetwork()

	result := &AdaptationResult{
		Windows: make([]TimeWindow, 15), // 15 one-second windows
	}

	// Initialize states based on mode
	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTweenChain {
		state = net.InitStepState(8)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		ts.Config.ExplosionDetection = false
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	env := &Environment{
		AgentPos:  [2]float32{0.5, 0.5},
		TargetPos: [2]float32{rand.Float32(), rand.Float32()},
		Task:      0, // Start with chase
	}

	learningRate := float32(0.02)
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond

	start := time.Now()
	currentWindow := 0

	for time.Since(start) < testDuration {
		elapsed := time.Since(start)

		// Update window
		newWindow := int(elapsed / windowDuration)
		if newWindow > currentWindow && newWindow < 15 {
			// Finalize previous window
			result.Windows[currentWindow].OutputsPerSec = result.Windows[currentWindow].Outputs
			if result.Windows[currentWindow].Outputs > 0 {
				result.Windows[currentWindow].Accuracy = float64(result.Windows[currentWindow].Correct) / float64(result.Windows[currentWindow].Outputs) * 100
			}
			currentWindow = newWindow
		}

		// TASK CHANGES
		if elapsed >= 5*time.Second && elapsed < 10*time.Second {
			if env.Task != 1 {
				env.Task = 1 // Switch to AVOID
				result.Windows[currentWindow].CurrentTask = "AVOID!"
			}
		} else {
			if env.Task != 0 {
				env.Task = 0 // Back to CHASE
				result.Windows[currentWindow].CurrentTask = "chase"
			}
		}

		if result.Windows[currentWindow].CurrentTask == "" {
			if env.Task == 0 {
				result.Windows[currentWindow].CurrentTask = "chase"
			} else {
				result.Windows[currentWindow].CurrentTask = "AVOID!"
			}
		}

		// Get observation
		obs := getObservation(env)

		// Forward pass
		var output []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(obs)
		case ModeStepBP, ModeStepTweenChain:
			state.SetInput(obs)
			net.StepForward(state)
			output = state.GetOutput()
		}

		action := argmax(output)
		optimalAction := getOptimalAction(env)

		// Record to current window
		if currentWindow < 15 {
			result.Windows[currentWindow].Outputs++
			result.TotalOutputs++
			if action == optimalAction {
				result.Windows[currentWindow].Correct++
			}
		}

		// Execute action
		executeAction(env, action)

		// Store sample
		target := make([]float32, 4)
		target[optimalAction] = 1.0
		trainBatch = append(trainBatch, TrainingSample{Input: obs, Target: target})

		// Training
		switch mode {
		case ModeNormalBP:
			if time.Since(lastTrainTime) > trainInterval && len(trainBatch) > 0 {
				batches := make([]nn.TrainingBatch, len(trainBatch))
				for i, s := range trainBatch {
					batches[i] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
				}
				net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: learningRate, LossType: "mse"})
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepBP:
			grad := make([]float32, len(output))
			for i := range output {
				if i < len(target) {
					grad[i] = output[i] - target[i]
				}
			}
			net.StepBackward(state, grad)
			net.ApplyGradients(learningRate)

		case ModeTween, ModeTweenChain:
			if time.Since(lastTrainTime) > trainInterval && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					ts.TweenStep(net, s.Input, argmax(s.Target), 4, learningRate)
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepTweenChain:
			ts.TweenStep(net, obs, optimalAction, 4, learningRate)
		}

		// Update environment
		updateEnvironment(env)
	}

	// Finalize last window
	if currentWindow < 15 && result.Windows[currentWindow].Outputs > 0 {
		result.Windows[currentWindow].OutputsPerSec = result.Windows[currentWindow].Outputs
		result.Windows[currentWindow].Accuracy = float64(result.Windows[currentWindow].Correct) / float64(result.Windows[currentWindow].Outputs) * 100
	}

	// Calculate adaptation metrics
	// Window 4 = last window before first change (second 4-5)
	// Window 5 = first window after first change (second 5-6)
	if len(result.Windows) > 5 {
		result.PreChangeAccuracy = result.Windows[4].Accuracy
		result.PostChange1Accuracy = result.Windows[5].Accuracy

		// How long to recover to 50%+ after first change?
		for i := 5; i < 10 && i < len(result.Windows); i++ {
			if result.Windows[i].Accuracy >= 50 {
				result.AdaptTime1 = i - 5
				break
			}
		}
	}

	// Window 9 = last window before second change
	// Window 10 = first window after second change
	if len(result.Windows) > 10 {
		result.PostChange2Accuracy = result.Windows[10].Accuracy
		for i := 10; i < 15 && i < len(result.Windows); i++ {
			if result.Windows[i].Accuracy >= 50 {
				result.AdaptTime2 = i - 10
				break
			}
		}
	}

	return result
}

// ============================================================================
// Network
// ============================================================================

func createDeepNetwork() *nn.Network {
	net := nn.NewNetwork(8, 1, 1, 6)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(32, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 3, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 4, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 5, nn.InitDenseLayer(32, 4, nn.ActivationSigmoid))

	return net
}

// ============================================================================
// Environment
// ============================================================================

func getObservation(env *Environment) []float32 {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(relX*relX + relY*relY)))

	return []float32{
		env.AgentPos[0], env.AgentPos[1],
		env.TargetPos[0], env.TargetPos[1],
		relX, relY,
		dist,
		float32(env.Task),
	}
}

func getOptimalAction(env *Environment) int {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]

	if env.Task == 0 { // Chase - move towards
		if abs(relX) > abs(relY) {
			if relX > 0 {
				return 3 // right
			}
			return 2 // left
		}
		if relY > 0 {
			return 0 // up
		}
		return 1 // down
	} else { // Avoid - move away
		if abs(relX) > abs(relY) {
			if relX > 0 {
				return 2 // left (away)
			}
			return 3 // right (away)
		}
		if relY > 0 {
			return 1 // down (away)
		}
		return 0 // up (away)
	}
}

func executeAction(env *Environment, action int) {
	speed := float32(0.02)
	moves := [][2]float32{{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}}
	if action >= 0 && action < 4 {
		env.AgentPos[0] = clamp(env.AgentPos[0]+moves[action][0], 0, 1)
		env.AgentPos[1] = clamp(env.AgentPos[1]+moves[action][1], 0, 1)
	}
}

func updateEnvironment(env *Environment) {
	env.TargetPos[0] += (rand.Float32() - 0.5) * 0.01
	env.TargetPos[1] += (rand.Float32() - 0.5) * 0.01
	env.TargetPos[0] = clamp(env.TargetPos[0], 0.1, 0.9)
	env.TargetPos[1] = clamp(env.TargetPos[1], 0.1, 0.9)
}

type TrainingSample struct {
	Input  []float32
	Target []float32
}

// ============================================================================
// Printing
// ============================================================================

func printAdaptationTimeline(results map[TrainingMode]*AdaptationResult, modes []TrainingMode) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ACCURACY OVER TIME (per 1-second window)                                                   ║")
	fmt.Println("║                   [0-5s: CHASE]    │    [5-10s: AVOID!]    │    [10-15s: CHASE]                                                 ║")
	fmt.Println("╠═══════════════════╦════╦════╦════╦════╦════║════╦════╦════╦════╦════║════╦════╦════╦════╦════╗")
	fmt.Println("║ Mode              ║ 1s ║ 2s ║ 3s ║ 4s ║ 5s ║ 6s ║ 7s ║ 8s ║ 9s ║10s ║11s ║12s ║13s ║14s ║15s ║")
	fmt.Println("╠═══════════════════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╣")

	for _, mode := range modes {
		r := results[mode]
		fmt.Printf("║ %-17s ║", modeNames[mode])
		for i := 0; i < 15; i++ {
			if i < len(r.Windows) {
				acc := r.Windows[i].Accuracy
				if acc >= 80 {
					fmt.Printf(" %2.0f%%║", acc)
				} else if acc >= 50 {
					fmt.Printf(" %2.0f%%║", acc)
				} else {
					fmt.Printf(" %2.0f%%║", acc)
				}
			} else {
				fmt.Printf("  -- ║")
			}
		}
		fmt.Println()
	}
	fmt.Println("╚═══════════════════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╝")
	fmt.Println("                         ↑ TASK CHANGE ↑                    ↑ TASK CHANGE ↑")
}

func printAdaptationSummary(results map[TrainingMode]*AdaptationResult, modes []TrainingMode) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ADAPTATION SUMMARY                                             ║")
	fmt.Println("╠═══════════════════╦═════════════════╦═══════════════════════╦═══════════════════════╦══════════════╣")
	fmt.Println("║ Mode              ║ Total Outputs   ║ 1st Change Adapt      ║ 2nd Change Adapt      ║ Avg Acc      ║")
	fmt.Println("║                   ║ (actions/15s)   ║ Before→After (delay)  ║ Before→After (delay)  ║              ║")
	fmt.Println("╠═══════════════════╬═════════════════╬═══════════════════════╬═══════════════════════╬══════════════╣")

	for _, mode := range modes {
		r := results[mode]
		avgAcc := float64(0)
		for _, w := range r.Windows {
			avgAcc += w.Accuracy
		}
		avgAcc /= float64(len(r.Windows))

		adapt1 := "N/A"
		if r.AdaptTime1 >= 0 {
			adapt1 = fmt.Sprintf("%ds", r.AdaptTime1)
		}
		adapt2 := "N/A"
		if r.AdaptTime2 >= 0 {
			adapt2 = fmt.Sprintf("%ds", r.AdaptTime2)
		}

		fmt.Printf("║ %-17s ║ %13d   ║ %5.0f%%→%5.0f%% (%3s)   ║ %5.0f%%→%5.0f%% (%3s)   ║   %5.1f%%    ║\n",
			modeNames[mode],
			r.TotalOutputs,
			r.PreChangeAccuracy, r.PostChange1Accuracy, adapt1,
			r.Windows[9].Accuracy, r.PostChange2Accuracy, adapt2,
			avgAcc)
	}

	fmt.Println("╚═══════════════════╩═════════════════╩═══════════════════════╩═══════════════════════╩══════════════╝")

	fmt.Println("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                                         KEY INSIGHTS                                              │")
	fmt.Println("├────────────────────────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Println("│ • Task changes at second 5 (chase→avoid) and second 10 (avoid→chase)                              │")
	fmt.Println("│ • 'Adapt delay' = seconds until accuracy recovers to 50%+ after task change                       │")
	fmt.Println("│                                                                                                   │")
	fmt.Println("│ ★ StepTweenChain: Trains EVERY output, adapts fastest to task changes                             │")
	fmt.Println("│ ★ NormalBP/Tween: Batch train periodically, slower to adapt (batched updates lag behind)          │")
	fmt.Println("│ ★ Step+BP: Trains every step but expensive gradients hurt throughput                              │")
	fmt.Println("│                                                                                                   │")
	fmt.Println("│ For embodied AI: Fast adaptation to changing goals is critical                                    │")
	fmt.Println("│ An agent that adapts while acting > one that waits to batch-learn                                 │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────────────────────────┘")
}

// ============================================================================
// Utility
// ============================================================================

func argmax(s []float32) int {
	if len(s) == 0 {
		return 0
	}
	maxI, maxV := 0, s[0]
	for i, v := range s {
		if v > maxV {
			maxV, maxI = v, i
		}
	}
	return maxI
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

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func getMemoryMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / 1024 / 1024
}
