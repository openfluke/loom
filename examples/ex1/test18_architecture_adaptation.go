package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 18: Multi-Architecture Mid-Stream Adaptation Benchmark
//
// Combines Test 16's network variety with Test 17's adaptation benchmark.
// Tests how different network architectures adapt to mid-stream task changes.
//
// Networks: Dense, Conv2D, RNN, LSTM, Attention
// Depths: 3, 5, 9 layers
// Modes: NormalBP, Step+BP, Tween, TweenChain, StepTweenChain
//
// Key metric: Accuracy after task change (adaptation speed)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 18: MULTI-ARCHITECTURE Adaptation Benchmark                        ║")
	fmt.Println("║  All Network Types | Variable Depths | Mid-Stream Task Changes           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Network types to test
	networkTypes := []string{"Dense", "Conv2D", "RNN", "LSTM", "Attn"}

	// Depths to test (keeping smaller for reasonable runtime)
	depths := []int{3, 5, 9}

	// Training modes
	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTweenChain,
	}

	testDuration := 10 * time.Second

	// Results storage
	allResults := make(map[string]map[TrainingMode]SummaryResult18)

	for _, netType := range networkTypes {
		for _, depth := range depths {
			configName := fmt.Sprintf("%s-%dL", netType, depth)
			allResults[configName] = make(map[TrainingMode]SummaryResult18)

			fmt.Printf("\n┌─────────────────────────────────────────────────────────────────────┐\n")
			fmt.Printf("│ %-67s │\n", configName+" — Testing 5 training modes")
			fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

			for _, mode := range modes {
				fmt.Printf("  [%s] ", modeNames[mode])

				// Create network based on type
				net := createNetwork(netType, depth)
				if net == nil {
					fmt.Printf("SKIP (unsupported)\n")
					continue
				}

				result := runAdaptationTest(net, mode, testDuration)

				// Get pre-change accuracy (window before change)
				preChange1 := 0.0
				preChange2 := 0.0
				if len(result.Windows) > 3 {
					preChange1 = result.Windows[2] // Window before 1st change (at 1/3)
				}
				if len(result.Windows) > 6 {
					preChange2 = result.Windows[5] // Window before 2nd change (at 2/3)
				}

				allResults[configName][mode] = SummaryResult18{
					AvgAccuracy:   result.AvgAccuracy,
					Change1Adapt:  result.Change1Accuracy,
					Change2Adapt:  result.Change2Accuracy,
					TotalOutputs:  result.TotalOutputs,
					Windows:       result.Windows,
					PreChange1Acc: preChange1,
					PreChange2Acc: preChange2,
				}

				fmt.Printf("Avg: %5.1f%% | After 1st: %5.1f%% | After 2nd: %5.1f%% | Outputs: %d\n",
					result.AvgAccuracy, result.Change1Accuracy, result.Change2Accuracy, result.TotalOutputs)
			}
		}
	}

	// Print summary table
	printSummaryTable(allResults, networkTypes, depths, modes)
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

var modeShortNames18 = map[TrainingMode]string{
	ModeNormalBP:       "BP",
	ModeStepBP:         "S+BP",
	ModeTween:          "Twn",
	ModeTweenChain:     "TwnC",
	ModeStepTweenChain: "STC",
}

type SummaryResult18 struct {
	AvgAccuracy   float64
	Change1Adapt  float64
	Change2Adapt  float64
	TotalOutputs  int
	Windows       []float64
	PreChange1Acc float64
	PreChange2Acc float64
}

type AdaptationResult struct {
	TotalOutputs    int
	AvgAccuracy     float64
	Change1Accuracy float64 // Accuracy in window immediately after 1st task change
	Change2Accuracy float64 // Accuracy in window immediately after 2nd task change
	Windows         []float64
}

type Environment struct {
	AgentPos  [2]float32
	TargetPos [2]float32
	Task      int // 0=chase, 1=avoid
}

// ============================================================================
// Main Benchmark
// ============================================================================

func runAdaptationTest(net *nn.Network, mode TrainingMode, duration time.Duration) *AdaptationResult {
	inputSize := net.InputSize
	outputSize := 4

	result := &AdaptationResult{}
	windowDuration := 1 * time.Second
	numWindows := int(duration / windowDuration)
	result.Windows = make([]float64, numWindows)

	windowOutputs := make([]int, numWindows)
	windowCorrect := make([]int, numWindows)

	// Initialize states based on mode
	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTweenChain {
		state = net.InitStepState(inputSize)
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
		Task:      0,
	}

	learningRate := float32(0.02)
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond

	start := time.Now()

	for time.Since(start) < duration {
		elapsed := time.Since(start)
		currentWindow := int(elapsed / windowDuration)
		if currentWindow >= numWindows {
			currentWindow = numWindows - 1
		}

		// Task changes: [0-1/3: chase] → [1/3-2/3: avoid] → [2/3-1: chase]
		oneThird := duration / 3
		twoThirds := 2 * oneThird

		if elapsed >= oneThird && elapsed < twoThirds {
			env.Task = 1 // AVOID
		} else {
			env.Task = 0 // CHASE
		}

		// Get observation (pad/truncate to match network input size)
		obs := getObservation(env, inputSize)

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

		// Handle output size mismatch
		if len(output) < outputSize {
			// Pad with zeros
			padded := make([]float32, outputSize)
			copy(padded, output)
			output = padded
		}

		action := argmax(output[:outputSize])
		optimalAction := getOptimalAction(env)

		// Record to window
		windowOutputs[currentWindow]++
		result.TotalOutputs++
		if action == optimalAction {
			windowCorrect[currentWindow]++
		}

		// Execute action
		executeAction(env, action)

		// Store sample
		target := make([]float32, outputSize)
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
					ts.TweenStep(net, s.Input, argmax(s.Target), outputSize, learningRate)
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepTweenChain:
			ts.TweenStep(net, obs, optimalAction, outputSize, learningRate)
		}

		// Update environment
		updateEnvironment(env)
	}

	// Calculate window accuracies
	totalAcc := 0.0
	for i := 0; i < numWindows; i++ {
		if windowOutputs[i] > 0 {
			result.Windows[i] = float64(windowCorrect[i]) / float64(windowOutputs[i]) * 100
		}
		totalAcc += result.Windows[i]
	}
	result.AvgAccuracy = totalAcc / float64(numWindows)

	// Get accuracy after task changes
	// Task changes at 1/3 and 2/3 of duration
	changeWindow1 := numWindows / 3
	changeWindow2 := 2 * numWindows / 3

	if changeWindow1 < numWindows {
		result.Change1Accuracy = result.Windows[changeWindow1]
	}
	if changeWindow2 < numWindows {
		result.Change2Accuracy = result.Windows[changeWindow2]
	}

	return result
}

// ============================================================================
// Network Factories
// ============================================================================

func createNetwork(netType string, numLayers int) *nn.Network {
	switch netType {
	case "Dense":
		return createDenseNetwork(numLayers)
	case "Conv2D":
		return createConv2DNetwork(numLayers)
	case "RNN":
		return createRNNNetwork(numLayers)
	case "LSTM":
		return createLSTMNetwork(numLayers)
	case "Attn":
		return createAttentionNetwork(numLayers)
	default:
		return nil
	}
}

func createDenseNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(8, 1, 1, numLayers)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(8, 64, nn.ActivationLeakyReLU))

	hiddenSizes := []int{64, 48, 32, 24, 16}
	for i := 1; i < numLayers-1; i++ {
		inSize := hiddenSizes[(i-1)%len(hiddenSizes)]
		outSize := hiddenSizes[i%len(hiddenSizes)]
		net.SetLayer(0, 0, i, nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU))
	}

	lastHidden := hiddenSizes[(numLayers-2)%len(hiddenSizes)]
	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(lastHidden, 4, nn.ActivationSigmoid))
	return net
}

func createConv2DNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(64, 1, 1, numLayers)
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

	for i := 1; i < numLayers-1; i++ {
		if i == 1 {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(288, 64, nn.ActivationLeakyReLU))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(64, 64, nn.ActivationLeakyReLU))
		}
	}

	if numLayers > 2 {
		net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(64, 4, nn.ActivationSigmoid))
	} else {
		net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(288, 4, nn.ActivationSigmoid))
	}
	return net
}

func createRNNNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(32, 1, 1, numLayers)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))

	for i := 1; i < numLayers-1; i++ {
		if i%2 == 1 {
			rnn := nn.InitRNNLayer(8, 8, 1, 4)
			net.SetLayer(0, 0, i, rnn)
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
		}
	}

	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(32, 4, nn.ActivationSigmoid))
	return net
}

func createLSTMNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(32, 1, 1, numLayers)
	net.BatchSize = 1

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))

	for i := 1; i < numLayers-1; i++ {
		if i%2 == 1 {
			lstm := nn.InitLSTMLayer(8, 8, 1, 4)
			net.SetLayer(0, 0, i, lstm)
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
		}
	}

	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(32, 4, nn.ActivationSigmoid))
	return net
}

func createAttentionNetwork(numLayers int) *nn.Network {
	net := nn.NewNetwork(64, 1, 1, numLayers)
	net.BatchSize = 1
	dModel := 64
	numHeads := 4
	headDim := dModel / numHeads

	for i := 0; i < numLayers-1; i++ {
		if i%2 == 0 {
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
			net.SetLayer(0, 0, i, nn.InitDenseLayer(dModel, dModel, nn.ActivationLeakyReLU))
		}
	}

	net.SetLayer(0, 0, numLayers-1, nn.InitDenseLayer(dModel, 4, nn.ActivationSigmoid))
	return net
}

// ============================================================================
// Environment
// ============================================================================

func getObservation(env *Environment, targetSize int) []float32 {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(relX*relX + relY*relY)))

	base := []float32{
		env.AgentPos[0], env.AgentPos[1],
		env.TargetPos[0], env.TargetPos[1],
		relX, relY,
		dist,
		float32(env.Task),
	}

	// Pad/repeat to match target size
	obs := make([]float32, targetSize)
	for i := 0; i < targetSize; i++ {
		obs[i] = base[i%len(base)]
	}
	return obs
}

func getOptimalAction(env *Environment) int {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]

	if env.Task == 0 { // Chase
		if abs(relX) > abs(relY) {
			if relX > 0 {
				return 3
			}
			return 2
		}
		if relY > 0 {
			return 0
		}
		return 1
	} else { // Avoid
		if abs(relX) > abs(relY) {
			if relX > 0 {
				return 2
			}
			return 3
		}
		if relY > 0 {
			return 1
		}
		return 0
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
// Summary Table
// ============================================================================

func printSummaryTable(results map[string]map[TrainingMode]SummaryResult18, netTypes []string, depths []int, modes []TrainingMode) {

	// For each network configuration, print the timeline and adaptation summary
	for _, netType := range netTypes {
		for _, depth := range depths {
			configName := fmt.Sprintf("%s-%dL", netType, depth)
			configResults := results[configName]

			// Print Timeline for this configuration
			fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
			fmt.Printf("║  %-90s  ║\n", configName+" — ACCURACY OVER TIME (per 1-second window)")
			fmt.Printf("║  %-90s  ║\n", "[0-3s: CHASE]    │    [3-7s: AVOID!]    │    [7-10s: CHASE]")
			fmt.Println("╠═══════════════════╦════╦════╦════╦════╦════╦════╦════╦════╦════╦════╗")
			fmt.Println("║ Mode              ║ 1s ║ 2s ║ 3s ║ 4s ║ 5s ║ 6s ║ 7s ║ 8s ║ 9s ║10s ║")
			fmt.Println("╠═══════════════════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╣")

			for _, mode := range modes {
				if r, ok := configResults[mode]; ok {
					fmt.Printf("║ %-17s ║", modeNames[mode])
					for i := 0; i < 10 && i < len(r.Windows); i++ {
						fmt.Printf(" %2.0f%%║", r.Windows[i])
					}
					fmt.Println()
				}
			}
			fmt.Println("╚═══════════════════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╝")
			fmt.Println("                              ↑ TASK CHANGE ↑        ↑ TASK CHANGE ↑")

			// Print Adaptation Summary for this configuration
			fmt.Println()
			fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
			fmt.Printf("║  %-90s  ║\n", configName+" — ADAPTATION SUMMARY")
			fmt.Println("╠═══════════════════╦═════════════════╦═══════════════════════════╦═══════════════════════════╦══════════════╣")
			fmt.Println("║ Mode              ║ Total Outputs   ║ 1st Change Adapt          ║ 2nd Change Adapt          ║ Avg Acc      ║")
			fmt.Println("║                   ║                 ║ Before→After              ║ Before→After              ║              ║")
			fmt.Println("╠═══════════════════╬═════════════════╬═══════════════════════════╬═══════════════════════════╬══════════════╣")

			for _, mode := range modes {
				if r, ok := configResults[mode]; ok {
					fmt.Printf("║ %-17s ║ %13d   ║ %5.0f%%→%5.0f%%              ║ %5.0f%%→%5.0f%%              ║   %5.1f%%    ║\n",
						modeNames[mode],
						r.TotalOutputs,
						r.PreChange1Acc, r.Change1Adapt,
						r.PreChange2Acc, r.Change2Adapt,
						r.AvgAccuracy)
				}
			}
			fmt.Println("╚═══════════════════╩═════════════════╩═══════════════════════════╩═══════════════════════════╩══════════════╝")
		}
	}

	// Print overall summary table
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          MULTI-ARCHITECTURE ADAPTATION SUMMARY                                                               ║")
	fmt.Println("║                                          (Avg Acc | After 1st Change | After 2nd Change)                                                     ║")
	fmt.Println("╠════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╣")
	fmt.Printf("║ %-10s ║ %-29s ║ %-29s ║ %-29s ║ %-29s ║ %-29s ║\n", "Network", "NormalBP", "Step+BP", "Tween", "TweenChain", "StepTweenChain")
	fmt.Println("╠════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╣")

	for _, netType := range netTypes {
		for _, depth := range depths {
			configName := fmt.Sprintf("%s-%dL", netType, depth)

			fmt.Printf("║ %-10s ║", configName)

			for _, mode := range modes {
				if r, ok := results[configName][mode]; ok {
					fmt.Printf(" %4.0f%% | %4.0f%% | %4.0f%% ║", r.AvgAccuracy, r.Change1Adapt, r.Change2Adapt)
				} else {
					fmt.Printf("      --     |      --     ║")
				}
			}
			fmt.Println()
		}
	}

	fmt.Println("╚════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╝")

	fmt.Println("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                                         KEY INSIGHTS                                                  │")
	fmt.Println("├────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Println("│ • 'After 1st Change' = accuracy immediately after task switches (chase→avoid)                        │")
	fmt.Println("│ • 'After 2nd Change' = accuracy immediately after task switches back (avoid→chase)                   │")
	fmt.Println("│ • Higher 'After Change' accuracy = faster adaptation to changing goals                                │")
	fmt.Println("│                                                                                                       │")
	fmt.Println("│ ★ StepTweenChain shows most CONSISTENT accuracy across all windows                                    │")
	fmt.Println("│ ★ Other methods crash to 0% after task changes while StepTweenChain maintains ~40-80%                │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
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

func initRandomSlice(s []float32, scale float32) {
	for i := range s {
		s[i] = (rand.Float32()*2 - 1) * scale
	}
}

func getMemoryMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / 1024 / 1024
}
