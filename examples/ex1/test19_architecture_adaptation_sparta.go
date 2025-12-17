package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Test 19: SPARTA - Statistical Parallel Adaptation Run Test Architecture
//
// Runs each network configuration 100 times in parallel to gather statistically
// significant data about adaptation performance. Results are saved to JSON files.
//
// Structure:
//   results/
//     Dense-3L/
//       NormalBP/
//         run_001.json, run_002.json, ... run_100.json
//         summary.json
//       StepTweenChain/
//         run_001.json, run_002.json, ... run_100.json
//         summary.json
//     Dense-5L/
//       ...
//     overall_summary.json

const (
	NumRuns       = 100              // Number of runs per configuration
	TestDuration  = 10 * time.Second // Duration per run
	MaxConcurrent = 16               // Max concurrent goroutines
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Test 19: SPARTA — Statistical Parallel Adaptation Run Test              ║")
	fmt.Println("║  100 runs per config | Parallel execution | JSON output                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create results directory
	resultsDir := "results_sparta"
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		fmt.Printf("Failed to create results directory: %v\n", err)
		return
	}

	networkTypes := []string{"Dense", "Conv2D", "RNN", "LSTM", "Attn"}
	depths := []int{3, 5, 9}
	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTweenChain,
	}

	overallSummary := &OverallSummary{
		Timestamp:    time.Now().Format(time.RFC3339),
		NumRuns:      NumRuns,
		TestDuration: TestDuration.String(),
		Configs:      make(map[string]ConfigSummary),
	}

	startTime := time.Now()

	for _, netType := range networkTypes {
		for _, depth := range depths {
			configName := fmt.Sprintf("%s-%dL", netType, depth)
			configDir := filepath.Join(resultsDir, configName)

			fmt.Printf("\n┌─────────────────────────────────────────────────────────────────────┐\n")
			fmt.Printf("│ %-67s │\n", configName+" — Running "+fmt.Sprintf("%d", NumRuns)+" trials per mode")
			fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

			configSummary := ConfigSummary{
				NetworkType: netType,
				Depth:       depth,
				Modes:       make(map[string]ModeSummary),
			}

			for _, mode := range modes {
				modeName := modeNames[mode]
				modeDir := filepath.Join(configDir, modeName)
				if err := os.MkdirAll(modeDir, 0755); err != nil {
					fmt.Printf("Failed to create mode directory: %v\n", err)
					continue
				}

				// Check if summary already exists - skip if so
				summaryFile := filepath.Join(modeDir, "summary.json")
				if existingSummary, err := loadModeSummaryJSON(summaryFile); err == nil {
					fmt.Printf("  [%s] SKIP (summary.json exists)\n", modeName)
					configSummary.Modes[modeName] = existingSummary
					continue
				}

				fmt.Printf("  [%s] Running %d parallel trials... ", modeName, NumRuns)
				startMode := time.Now()

				// Run trials in parallel
				results := runParallelTrials(netType, depth, mode, NumRuns)

				// Save individual results
				for i, result := range results {
					filename := filepath.Join(modeDir, fmt.Sprintf("run_%03d.json", i+1))
					saveResultJSON(filename, result)
				}

				// Calculate statistics
				summary := calculateModeSummary(results, modeName)
				configSummary.Modes[modeName] = summary

				// Save mode summary
				saveSummaryJSON(summaryFile, summary)

				elapsed := time.Since(startMode)
				fmt.Printf("Done in %v\n", elapsed.Round(time.Millisecond))
				fmt.Printf("           Avg Accuracy: %.1f%% (±%.1f%%) | Best: %.1f%% | Worst: %.1f%%\n",
					summary.AvgAccuracy.Mean, summary.AvgAccuracy.StdDev,
					summary.AvgAccuracy.Max, summary.AvgAccuracy.Min)
			}

			overallSummary.Configs[configName] = configSummary

			// Save config summary
			configSummaryFile := filepath.Join(configDir, "summary.json")
			saveConfigSummaryJSON(configSummaryFile, configSummary)
		}
	}

	// Save overall summary
	overallFile := filepath.Join(resultsDir, "overall_summary.json")
	saveOverallSummaryJSON(overallFile, overallSummary)

	totalElapsed := time.Since(startTime)
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  SPARTA Complete — Total time: %-40s      ║\n", totalElapsed.Round(time.Second))
	fmt.Printf("║  Results saved to: %-52s      ║\n", resultsDir+"/")
	fmt.Printf("╚══════════════════════════════════════════════════════════════════════════╝\n")

	// Print final comparison
	printFinalComparison(overallSummary)
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
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTweenChain: "StepTweenChain",
}

type Environment struct {
	AgentPos  [2]float32
	TargetPos [2]float32
	Task      int
}

type TrainingSample struct {
	Input  []float32
	Target []float32
}

// Statistics for a single metric
type MetricStats struct {
	Mean   float64   `json:"mean"`
	StdDev float64   `json:"std_dev"`
	Min    float64   `json:"min"`
	Max    float64   `json:"max"`
	Median float64   `json:"median"`
	Values []float64 `json:"values,omitempty"` // Optional: all values for detailed analysis
}

// Summary for a single training mode
type ModeSummary struct {
	ModeName         string        `json:"mode_name"`
	NumRuns          int           `json:"num_runs"`
	AvgAccuracy      MetricStats   `json:"avg_accuracy"`
	Change1Accuracy  MetricStats   `json:"change1_accuracy"`
	Change2Accuracy  MetricStats   `json:"change2_accuracy"`
	RecoveryTime1    MetricStats   `json:"recovery_time1"`
	RecoveryTime2    MetricStats   `json:"recovery_time2"`
	TotalOutputs     MetricStats   `json:"total_outputs"`
	WindowAccuracies []MetricStats `json:"window_accuracies"` // Per-window stats across all runs
}

// Summary for a network configuration
type ConfigSummary struct {
	NetworkType string                 `json:"network_type"`
	Depth       int                    `json:"depth"`
	Modes       map[string]ModeSummary `json:"modes"`
	BestMode    string                 `json:"best_mode"`
	MostStable  string                 `json:"most_stable"`
}

// Overall summary
type OverallSummary struct {
	Timestamp    string                   `json:"timestamp"`
	NumRuns      int                      `json:"num_runs"`
	TestDuration string                   `json:"test_duration"`
	Configs      map[string]ConfigSummary `json:"configs"`
}

// ============================================================================
// Parallel Execution
// ============================================================================

func runParallelTrials(netType string, depth int, mode TrainingMode, numRuns int) []*nn.AdaptationResult {
	results := make([]*nn.AdaptationResult, numRuns)
	var wg sync.WaitGroup
	sem := make(chan struct{}, MaxConcurrent)

	for i := 0; i < numRuns; i++ {
		wg.Add(1)
		go func(runIdx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			// Each run gets its own random seed
			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(runIdx)))

			// Create fresh network for this run
			net := createNetwork(netType, depth)
			if net == nil {
				return
			}

			// Run the adaptation test
			result := runAdaptationTest(net, mode, TestDuration, localRand)
			result.ModelName = fmt.Sprintf("%s-%dL", netType, depth)
			result.ModeName = modeNames[mode]
			results[runIdx] = result
		}(i)
	}

	wg.Wait()

	// Filter out nil results
	validResults := make([]*nn.AdaptationResult, 0, numRuns)
	for _, r := range results {
		if r != nil {
			validResults = append(validResults, r)
		}
	}
	return validResults
}

func runAdaptationTest(net *nn.Network, mode TrainingMode, duration time.Duration, rng *rand.Rand) *nn.AdaptationResult {
	inputSize := net.InputSize
	outputSize := 4
	windowDuration := 1 * time.Second

	// Create the AdaptationTracker from the framework
	tracker := nn.NewAdaptationTracker(windowDuration, duration)
	tracker.SetModelInfo("", modeNames[mode])

	// Schedule task changes: [0-1/3: chase] → [1/3-2/3: avoid] → [2/3-1: chase]
	oneThird := duration / 3
	twoThirds := 2 * oneThird

	tracker.ScheduleTaskChange(oneThird, 1, "AVOID")
	tracker.ScheduleTaskChange(twoThirds, 0, "CHASE")

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
		TargetPos: [2]float32{rng.Float32(), rng.Float32()},
		Task:      0,
	}

	learningRate := float32(0.02)
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond

	tracker.Start("CHASE", 0)
	start := time.Now()

	for time.Since(start) < duration {
		currentTaskID := tracker.GetCurrentTask()
		env.Task = currentTaskID

		obs := getObservation(env, inputSize)

		var output []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(obs)
		case ModeStepBP, ModeStepTweenChain:
			state.SetInput(obs)
			net.StepForward(state)
			output = state.GetOutput()
		}

		if len(output) < outputSize {
			padded := make([]float32, outputSize)
			copy(padded, output)
			output = padded
		}

		action := argmax(output[:outputSize])
		optimalAction := getOptimalAction(env)
		isCorrect := action == optimalAction
		tracker.RecordOutput(isCorrect)

		executeAction(env, action)

		target := make([]float32, outputSize)
		target[optimalAction] = 1.0
		trainBatch = append(trainBatch, TrainingSample{Input: obs, Target: target})

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

		updateEnvironment(env, rng)
	}

	return tracker.Finalize()
}

// ============================================================================
// Statistics Calculation
// ============================================================================

func calculateModeSummary(results []*nn.AdaptationResult, modeName string) ModeSummary {
	n := len(results)
	if n == 0 {
		return ModeSummary{ModeName: modeName, NumRuns: 0}
	}

	// Collect metrics
	avgAccs := make([]float64, n)
	change1Accs := make([]float64, n)
	change2Accs := make([]float64, n)
	recovery1s := make([]float64, n)
	recovery2s := make([]float64, n)
	outputs := make([]float64, n)

	numWindows := 0
	if len(results) > 0 && len(results[0].Windows) > 0 {
		numWindows = len(results[0].Windows)
	}
	windowAccs := make([][]float64, numWindows)
	for i := range windowAccs {
		windowAccs[i] = make([]float64, n)
	}

	for i, r := range results {
		avgAccs[i] = r.AvgAccuracy
		outputs[i] = float64(r.TotalOutputs)

		if len(r.TaskChanges) > 0 {
			change1Accs[i] = r.TaskChanges[0].PostAccuracy
			recovery1s[i] = float64(r.TaskChanges[0].RecoveryWindows)
		}
		if len(r.TaskChanges) > 1 {
			change2Accs[i] = r.TaskChanges[1].PostAccuracy
			recovery2s[i] = float64(r.TaskChanges[1].RecoveryWindows)
		}

		for w := 0; w < numWindows && w < len(r.Windows); w++ {
			windowAccs[w][i] = r.Windows[w].Accuracy
		}
	}

	// Calculate per-window statistics
	windowStats := make([]MetricStats, numWindows)
	for w := 0; w < numWindows; w++ {
		windowStats[w] = calculateStats(windowAccs[w])
	}

	return ModeSummary{
		ModeName:         modeName,
		NumRuns:          n,
		AvgAccuracy:      calculateStats(avgAccs),
		Change1Accuracy:  calculateStats(change1Accs),
		Change2Accuracy:  calculateStats(change2Accs),
		RecoveryTime1:    calculateStats(recovery1s),
		RecoveryTime2:    calculateStats(recovery2s),
		TotalOutputs:     calculateStats(outputs),
		WindowAccuracies: windowStats,
	}
}

func calculateStats(values []float64) MetricStats {
	n := len(values)
	if n == 0 {
		return MetricStats{}
	}

	// Mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)

	// Variance and StdDev
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(n)
	stdDev := math.Sqrt(variance)

	// Min, Max
	minVal, maxVal := values[0], values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Median (simple sort)
	sorted := make([]float64, n)
	copy(sorted, values)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	median := sorted[n/2]
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	}

	return MetricStats{
		Mean:   mean,
		StdDev: stdDev,
		Min:    minVal,
		Max:    maxVal,
		Median: median,
		Values: values, // Store all values for detailed analysis
	}
}

// ============================================================================
// JSON Saving
// ============================================================================

func saveResultJSON(filename string, result *nn.AdaptationResult) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(filename, data, 0644)
}

func saveSummaryJSON(filename string, summary ModeSummary) {
	// Don't include all individual values in the summary to save space
	summaryCopy := summary
	summaryCopy.AvgAccuracy.Values = nil
	summaryCopy.Change1Accuracy.Values = nil
	summaryCopy.Change2Accuracy.Values = nil
	summaryCopy.RecoveryTime1.Values = nil
	summaryCopy.RecoveryTime2.Values = nil
	summaryCopy.TotalOutputs.Values = nil
	for i := range summaryCopy.WindowAccuracies {
		summaryCopy.WindowAccuracies[i].Values = nil
	}

	data, err := json.MarshalIndent(summaryCopy, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(filename, data, 0644)
}

func loadModeSummaryJSON(filename string) (ModeSummary, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return ModeSummary{}, err
	}
	var summary ModeSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		return ModeSummary{}, err
	}
	return summary, nil
}

func saveConfigSummaryJSON(filename string, summary ConfigSummary) {
	// Find best mode by average accuracy
	bestAcc := 0.0
	leastVariance := math.MaxFloat64
	for name, mode := range summary.Modes {
		if mode.AvgAccuracy.Mean > bestAcc {
			bestAcc = mode.AvgAccuracy.Mean
			summary.BestMode = name
		}
		if mode.AvgAccuracy.StdDev < leastVariance {
			leastVariance = mode.AvgAccuracy.StdDev
			summary.MostStable = name
		}
	}

	// Clear values arrays to save space
	for modeName, mode := range summary.Modes {
		mode.AvgAccuracy.Values = nil
		mode.Change1Accuracy.Values = nil
		mode.Change2Accuracy.Values = nil
		mode.RecoveryTime1.Values = nil
		mode.RecoveryTime2.Values = nil
		mode.TotalOutputs.Values = nil
		for i := range mode.WindowAccuracies {
			mode.WindowAccuracies[i].Values = nil
		}
		summary.Modes[modeName] = mode
	}

	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(filename, data, 0644)
}

func saveOverallSummaryJSON(filename string, summary *OverallSummary) {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(filename, data, 0644)
}

// ============================================================================
// Final Comparison
// ============================================================================

func printFinalComparison(summary *OverallSummary) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                         SPARTA STATISTICAL SUMMARY                                                             ║")
	fmt.Println("║                                         (Mean ± StdDev across 100 runs)                                                        ║")
	fmt.Println("╠════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════╣")
	fmt.Println("║ Config     ║ NormalBP                      ║ StepBP                        ║ StepTweenChain                ║ Best Mode         ║")
	fmt.Println("╠════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════╣")

	configs := []string{
		"Dense-3L", "Dense-5L", "Dense-9L",
		"Conv2D-3L", "Conv2D-5L", "Conv2D-9L",
		"RNN-3L", "RNN-5L", "RNN-9L",
		"LSTM-3L", "LSTM-5L", "LSTM-9L",
		"Attn-3L", "Attn-5L", "Attn-9L",
	}

	stcWins := 0
	totalConfigs := 0

	for _, configName := range configs {
		config, ok := summary.Configs[configName]
		if !ok {
			continue
		}
		totalConfigs++

		normalBP := config.Modes["NormalBP"]
		stepBP := config.Modes["StepBP"]
		stc := config.Modes["StepTweenChain"]

		if config.BestMode == "StepTweenChain" {
			stcWins++
		}

		fmt.Printf("║ %-10s ║ %5.1f%% (±%4.1f%%)               ║ %5.1f%% (±%4.1f%%)               ║ %5.1f%% (±%4.1f%%)               ║ %-17s ║\n",
			configName,
			normalBP.AvgAccuracy.Mean, normalBP.AvgAccuracy.StdDev,
			stepBP.AvgAccuracy.Mean, stepBP.AvgAccuracy.StdDev,
			stc.AvgAccuracy.Mean, stc.AvgAccuracy.StdDev,
			config.BestMode)
	}

	fmt.Println("╚════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════╝")

	fmt.Println("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                                         KEY FINDINGS                                                  │")
	fmt.Println("├────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ ★ StepTweenChain won %d/%d configurations (%.1f%%)                                                      │\n",
		stcWins, totalConfigs, float64(stcWins)/float64(totalConfigs)*100)
	fmt.Println("│ ★ Statistical significance: 100 runs per config with mean ± standard deviation                        │")
	fmt.Println("│ ★ Lower StdDev = more consistent/stable performance                                                   │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
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
	}
	return nil
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

func initRandomSlice(slice []float32, scale float32) {
	for i := range slice {
		slice[i] = (rand.Float32()*2 - 1) * scale
	}
}

// ============================================================================
// Environment
// ============================================================================

func getObservation(env *Environment, targetSize int) []float32 {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(relX*relX + relY*relY)))

	baseObs := []float32{
		env.AgentPos[0], env.AgentPos[1],
		env.TargetPos[0], env.TargetPos[1],
		relX, relY, dist, float32(env.Task),
	}

	if len(baseObs) >= targetSize {
		return baseObs[:targetSize]
	}

	result := make([]float32, targetSize)
	copy(result, baseObs)
	for i := len(baseObs); i < targetSize; i++ {
		result[i] = baseObs[i%len(baseObs)]
	}
	return result
}

func getOptimalAction(env *Environment) int {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]

	if env.Task == 0 {
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
	} else {
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

func updateEnvironment(env *Environment, rng *rand.Rand) {
	env.TargetPos[0] += (rng.Float32() - 0.5) * 0.01
	env.TargetPos[1] += (rng.Float32() - 0.5) * 0.01
	env.TargetPos[0] = clamp(env.TargetPos[0], 0.1, 0.9)
	env.TargetPos[1] = clamp(env.TargetPos[1], 0.1, 0.9)
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
