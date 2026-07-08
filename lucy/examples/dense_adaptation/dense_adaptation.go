package denseadaptation

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	denseAdaptationDuration          = 15 * time.Second
	denseAdaptationWindowDuration    = time.Second
	denseAdaptationWindowCount       = 15
	denseAdaptationLearningRate      = float32(0.02)
	denseAdaptationTrainInterval     = 50 * time.Millisecond
	denseAdaptationMaxBatch          = 20
	denseAdaptationRecoveryThreshold = 50.0
	denseAdaptationDecisionDeadline  = 10 * time.Millisecond
	denseAdaptationSeed              = uint64(0xD17EADA9)
)

// RunDenseForwardComparison runs the dense adaptation benchmark (compat alias).
func RunDenseForwardComparison(reader *bufio.Reader) {
	RunDenseAdaptationBenchmark(reader)
}

// RunDenseBench runs the dense adaptation benchmark (compat alias).
func RunDenseBench(reader *bufio.Reader) {
	RunDenseAdaptationBenchmark(reader)
}

// RunDenseAdaptationBenchmark compares how dense poly training paths adapt to a mid-stream task flip.
func RunDenseAdaptationBenchmark(_ *bufio.Reader) {
	path, err := denseAdaptationStartOutputCapture()
	if err != nil {
		fmt.Printf("⚠️ dense adaptation output capture disabled: %v\n", err)
		runDenseAdaptationBenchmark()
		return
	}
	defer denseAdaptationStopOutputCapture(path)
	runDenseAdaptationBenchmark()
}

func runDenseAdaptationBenchmark() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Dense Mid-Stream Adaptation Benchmark                                  ║")
	fmt.Println("║  Task Changes Suddenly — Which poly Training Path Adapts Fastest?       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Timeline: [Chase 5s] → [AVOID 5s] → [Chase 5s]")
	fmt.Println("Network:  6-layer Dense (8→32→64→64→64→32→4)")
	fmt.Println("Engine:   loom/poly VolumetricNetwork")
	if poly.Plan9SimdEnabled() {
		fmt.Println("SIMD:     linked (AVX2/NEON) — each training path runs scalar + SIMD forward variants")
	} else {
		fmt.Println("SIMD:     not linked for this build/arch — SIMD rows use scalar forward")
	}
	fmt.Println()

	master, err := createDenseAdaptationNetwork()
	if err != nil {
		fmt.Printf("❌ network: %v\n", err)
		return
	}
	wire, err := poly.SerializeNetwork(master)
	if err != nil {
		fmt.Printf("❌ serialize master: %v\n", err)
		return
	}

	modes := []denseAdaptationMode{
		denseAdaptationModeNormalBP,
		denseAdaptationModeNormalBPSimd,
		denseAdaptationModeStepBP,
		denseAdaptationModeStepBPSimd,
		denseAdaptationModeTween,
		denseAdaptationModeTweenSimd,
		denseAdaptationModeTweenChain,
		denseAdaptationModeTweenChainSimd,
		denseAdaptationModeStepTween,
		denseAdaptationModeStepTweenSimd,
		denseAdaptationModeStepTweenChain,
		denseAdaptationModeStepTweenChainSimd,
	}

	allResults := make(map[denseAdaptationMode]*denseAdaptationResult)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, mode := range modes {
		wg.Add(1)
		go func(m denseAdaptationMode) {
			defer wg.Done()
			fmt.Printf("Starting [%s]...\n", denseAdaptationModeNames[m])
			result := runDenseAdaptationTest(m, wire, denseAdaptationSeed)
			mu.Lock()
			allResults[m] = result
			mu.Unlock()
			if result.Err != "" {
				fmt.Printf("Finished [%s] — ERROR: %s\n", denseAdaptationModeNames[m], result.Err)
				return
			}
			fmt.Printf("Finished [%s] — Total outputs: %d\n", denseAdaptationModeNames[m], result.TotalOutputs)
		}(mode)
	}

	wg.Wait()
	fmt.Println("\nAll tests complete.")

	printDenseAdaptationTimeline(allResults, modes)
	printDenseAdaptationSummary(allResults, modes)
	printDensePhaseRecoveryMetrics(allResults, modes)
	printDenseDecisionActivityMetrics(allResults, modes)
	printDenseOperationalMetrics(allResults, modes)
}

type denseAdaptationMode int

const (
	denseAdaptationModeNormalBP denseAdaptationMode = iota
	denseAdaptationModeNormalBPSimd
	denseAdaptationModeStepBP
	denseAdaptationModeStepBPSimd
	denseAdaptationModeTween
	denseAdaptationModeTweenSimd
	denseAdaptationModeTweenChain
	denseAdaptationModeTweenChainSimd
	denseAdaptationModeStepTween
	denseAdaptationModeStepTweenSimd
	denseAdaptationModeStepTweenChain
	denseAdaptationModeStepTweenChainSimd
)

var denseAdaptationModeNames = map[denseAdaptationMode]string{
	denseAdaptationModeNormalBP:           "NormalBP      ",
	denseAdaptationModeNormalBPSimd:       "NormalBP      ",
	denseAdaptationModeStepBP:             "Step+BP       ",
	denseAdaptationModeStepBPSimd:         "Step+BP       ",
	denseAdaptationModeTween:              "Tween         ",
	denseAdaptationModeTweenSimd:          "Tween         ",
	denseAdaptationModeTweenChain:         "TweenChain    ",
	denseAdaptationModeTweenChainSimd:     "TweenChain    ",
	denseAdaptationModeStepTween:          "StepTween     ",
	denseAdaptationModeStepTweenSimd:      "StepTween     ",
	denseAdaptationModeStepTweenChain:     "StepTweenChain",
	denseAdaptationModeStepTweenChainSimd: "StepTweenChain",
}

type denseAdaptationWindow struct {
	Outputs       int
	Correct       int
	Accuracy      float64
	OutputsPerSec int
	CurrentTask   string
}

type denseAdaptationResult struct {
	Windows      []denseAdaptationWindow
	TotalOutputs int
	Err          string

	PreChangeAccuracy   float64
	PostChange1Accuracy float64
	AdaptTime1          int
	DipDepth1           float64
	PostChange2Accuracy float64
	AdaptTime2          int
	DipDepth2           float64

	TotalDuration       time.Duration
	BlockedTime         time.Duration
	PeakLatency         time.Duration
	TotalLatency        time.Duration
	LatencySamples      int
	DeadlineHits        int
	DeadlineMisses      int
	AwaitTime           time.Duration
	AvgAccuracy         float64
	Stability           float64
	CompetenceStability float64
	Throughput          float64
	Availability        float64
	Score               float64

	Chase1Avg         float64
	AvoidAvg          float64
	Chase2Avg         float64
	RecoveryPeak1     float64
	RecoveryPeak2     float64
	CompetenceFloor   float64
	Retention         float64
	ZeroDowntimeScore float64
	OnlineEfficiency  float64

	DeadlineHitRate       float64
	DecisionActivityScore float64
	ReactiveScore         float64
}

type denseAdaptationEnvironment struct {
	AgentPos  [2]float32
	TargetPos [2]float32
	Task      int // 0=chase, 1=avoid
}

type denseAdaptationSample struct {
	Input  *poly.Tensor[float32]
	Target *poly.Tensor[float32]
}

func runDenseAdaptationTest(mode denseAdaptationMode, wire []byte, seed uint64) (result *denseAdaptationResult) {
	result = &denseAdaptationResult{
		Windows:    make([]denseAdaptationWindow, denseAdaptationWindowCount),
		AdaptTime1: -1,
		AdaptTime2: -1,
	}
	defer func() {
		if panicVal := recover(); panicVal != nil {
			result.Err = fmt.Sprintf("panic: %v", panicVal)
		}
	}()

	net, err := poly.DeserializeNetwork(wire)
	if err != nil {
		result.Err = err.Error()
		return result
	}
	configureDenseAdaptationRuntime(net)
	net.SetSimdForwardRecursive(denseAdaptationModeUseSimd(mode))

	var tweenState *poly.TweenState[float32]
	if denseAdaptationModeUseTween(mode) {
		cfg := poly.DefaultTweenConfig()
		cfg.UseChainRule = denseAdaptationModeUseTweenChain(mode)
		cfg.LearningRate = denseAdaptationLearningRate
		tweenState = poly.NewTweenState[float32](net, cfg)
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xA5A5A5A5A5A5A5A5))
	env := &denseAdaptationEnvironment{
		AgentPos:  [2]float32{0.5, 0.5},
		TargetPos: [2]float32{rng.Float32(), rng.Float32()},
		Task:      0,
	}

	trainBatch := make([]denseAdaptationSample, 0, denseAdaptationMaxBatch)
	lastTrainTime := time.Now()
	start := time.Now()
	currentWindow := 0

	for time.Since(start) < denseAdaptationDuration {
		actionStart := time.Now()
		elapsed := time.Since(start)
		newWindow := int(elapsed / denseAdaptationWindowDuration)
		if newWindow >= denseAdaptationWindowCount {
			newWindow = denseAdaptationWindowCount - 1
		}
		for newWindow > currentWindow && currentWindow < denseAdaptationWindowCount-1 {
			finalizeDenseAdaptationWindow(&result.Windows[currentWindow])
			currentWindow++
		}

		env.Task = denseAdaptationTaskAt(elapsed)
		result.Windows[currentWindow].CurrentTask = denseAdaptationTaskName(env.Task)

		obs := denseAdaptationObservation(env)
		targetAction := denseAdaptationOptimalAction(env)
		target := denseAdaptationTarget(targetAction)

		var output *poly.Tensor[float32]
		var histIn, histPre []*poly.Tensor[float32]
		switch mode {
		case denseAdaptationModeNormalBP, denseAdaptationModeNormalBPSimd:
			output, _, _ = poly.ForwardPolymorphic(net, obs)
		case denseAdaptationModeStepBP, denseAdaptationModeStepBPSimd:
			output, histIn, histPre = denseAdaptationForwardWithHistory(net, obs)
		case denseAdaptationModeTween, denseAdaptationModeTweenSimd,
			denseAdaptationModeTweenChain, denseAdaptationModeTweenChainSimd,
			denseAdaptationModeStepTween, denseAdaptationModeStepTweenSimd,
			denseAdaptationModeStepTweenChain, denseAdaptationModeStepTweenChainSimd:
			output = poly.TweenForward(net, tweenState, obs)
		}

		action := denseAdaptationArgmaxTensor(output)
		result.Windows[currentWindow].Outputs++
		result.TotalOutputs++
		if action == targetAction {
			result.Windows[currentWindow].Correct++
		}

		denseAdaptationExecuteAction(env, action)
		trainBatch = appendDenseAdaptationSample(trainBatch, obs, target)

		switch mode {
		case denseAdaptationModeNormalBP, denseAdaptationModeNormalBPSimd:
			if time.Since(lastTrainTime) > denseAdaptationTrainInterval && len(trainBatch) > 0 {
				blockedStart := time.Now()
				denseAdaptationTrainBatchBP(net, trainBatch)
				result.BlockedTime += time.Since(blockedStart)
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case denseAdaptationModeStepBP, denseAdaptationModeStepBPSimd:
			if output != nil && denseAdaptationStepHistoryReady(net, histIn, histPre) {
				grad := poly.ComputeLossGradient(output, target, "mse")
				_, layerGradients, _ := poly.BackwardPolymorphic(net, grad, histIn, histPre)
				denseAdaptationApplyGradients(net, layerGradients, denseAdaptationLearningRate)
			}
		case denseAdaptationModeTween, denseAdaptationModeTweenSimd,
			denseAdaptationModeTweenChain, denseAdaptationModeTweenChainSimd:
			if time.Since(lastTrainTime) > denseAdaptationTrainInterval && len(trainBatch) > 0 {
				blockedStart := time.Now()
				for _, sample := range trainBatch {
					poly.TweenForward(net, tweenState, sample.Input)
					poly.TweenBackward(net, tweenState, sample.Target)
					tweenState.CalculateLinkBudgets()
					poly.ApplyTweenGaps(net, tweenState, denseAdaptationLearningRate)
				}
				result.BlockedTime += time.Since(blockedStart)
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case denseAdaptationModeStepTween, denseAdaptationModeStepTweenSimd,
			denseAdaptationModeStepTweenChain, denseAdaptationModeStepTweenChainSimd:
			poly.TweenBackward(net, tweenState, target)
			tweenState.CalculateLinkBudgets()
			poly.ApplyTweenGaps(net, tweenState, denseAdaptationLearningRate)
		}

		denseAdaptationUpdateEnvironment(env, rng)
		latency := time.Since(actionStart)
		result.TotalLatency += latency
		result.LatencySamples++
		if latency <= denseAdaptationDecisionDeadline {
			result.DeadlineHits++
		} else {
			result.DeadlineMisses++
			result.AwaitTime += latency - denseAdaptationDecisionDeadline
		}
		if latency > result.PeakLatency {
			result.PeakLatency = latency
		}
	}

	result.TotalDuration = time.Since(start)
	if currentWindow < denseAdaptationWindowCount {
		finalizeDenseAdaptationWindow(&result.Windows[currentWindow])
	}
	calculateDenseAdaptationMetrics(result)
	return result
}

func createDenseAdaptationNetwork() (*poly.VolumetricNetwork, error) {
	spec := denseAdaptationNetSpecJSON()
	net, err := poly.BuildNetworkFromJSON(spec)
	if err != nil {
		return nil, err
	}
	configureDenseAdaptationRuntime(net)
	return net, nil
}

func configureDenseAdaptationRuntime(net *poly.VolumetricNetwork) {
	net.UseTiling = false
	net.EnableMultiCoreTiling = false
	for i := range net.Layers {
		net.Layers[i].UseTiling = false
		net.Layers[i].EnableMultiCoreTiling = false
		if net.Layers[i].WeightStore != nil {
			net.Layers[i].WeightStore.Scale = 1.0
			net.Layers[i].WeightStore.InvalidateVersions()
		}
	}
	net.SyncToCPU()
}

func denseAdaptationModeUseSimd(mode denseAdaptationMode) bool {
	switch mode {
	case denseAdaptationModeNormalBPSimd,
		denseAdaptationModeStepBPSimd,
		denseAdaptationModeTweenSimd,
		denseAdaptationModeTweenChainSimd,
		denseAdaptationModeStepTweenSimd,
		denseAdaptationModeStepTweenChainSimd:
		return true
	default:
		return false
	}
}

func denseAdaptationModeUseTween(mode denseAdaptationMode) bool {
	switch mode {
	case denseAdaptationModeTween, denseAdaptationModeTweenSimd,
		denseAdaptationModeTweenChain, denseAdaptationModeTweenChainSimd,
		denseAdaptationModeStepTween, denseAdaptationModeStepTweenSimd,
		denseAdaptationModeStepTweenChain, denseAdaptationModeStepTweenChainSimd:
		return true
	default:
		return false
	}
}

func denseAdaptationModeUseTweenChain(mode denseAdaptationMode) bool {
	switch mode {
	case denseAdaptationModeTweenChain, denseAdaptationModeTweenChainSimd,
		denseAdaptationModeStepTweenChain, denseAdaptationModeStepTweenChainSimd:
		return true
	default:
		return false
	}
}

func denseAdaptationSimdLabel(mode denseAdaptationMode) string {
	if denseAdaptationModeUseSimd(mode) {
		return "on"
	}
	return "off"
}

func denseAdaptationNetSpecJSON() []byte {
	dims := []int{8, 32, 64, 64, 64, 32, 4}
	var b strings.Builder
	b.WriteString(`{"id":"lucy-dense-adaptation","depth":1,"rows":1,"cols":1,"layers_per_cell":`)
	b.WriteString(fmt.Sprintf("%d", len(dims)-1))
	b.WriteString(`,"layers":[`)
	for i := 0; i < len(dims)-1; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		activation := "LEAKYRELU"
		if i == len(dims)-2 {
			activation = "SIGMOID"
		}
		b.WriteString(fmt.Sprintf(
			`{"z":0,"y":0,"x":0,"l":%d,"type":"DENSE","activation":"%s","dtype":"FLOAT32","input_height":%d,"output_height":%d}`,
			i, activation, dims[i], dims[i+1],
		))
	}
	b.WriteString(`]}`)
	return []byte(b.String())
}

func denseAdaptationForwardWithHistory(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) (*poly.Tensor[float32], []*poly.Tensor[float32], []*poly.Tensor[float32]) {
	histIn := make([]*poly.Tensor[float32], len(net.Layers))
	histPre := make([]*poly.Tensor[float32], len(net.Layers))
	current := input
	for idx := range net.Layers {
		layer := &net.Layers[idx]
		if layer.IsDisabled {
			continue
		}
		histIn[idx] = current
		pre, post := poly.DispatchLayer(layer, current, nil)
		histPre[idx] = pre
		current = post
	}
	return current, histIn, histPre
}

func denseAdaptationStepHistoryReady(net *poly.VolumetricNetwork, histIn, histPre []*poly.Tensor[float32]) bool {
	if len(histIn) != len(net.Layers) || len(histPre) != len(net.Layers) {
		return false
	}
	for i := range net.Layers {
		if net.Layers[i].IsDisabled {
			continue
		}
		if histIn[i] == nil || histPre[i] == nil {
			return false
		}
	}
	return true
}

func denseAdaptationTrainBatchBP(net *poly.VolumetricNetwork, samples []denseAdaptationSample) {
	batches := make([]poly.TrainingBatch[float32], len(samples))
	for i, sample := range samples {
		batches[i] = poly.TrainingBatch[float32]{Input: sample.Input, Target: sample.Target}
	}
	_, _ = poly.Train(net, batches, &poly.TrainingConfig{
		Epochs:       1,
		LearningRate: denseAdaptationLearningRate,
		LossType:     "mse",
		Verbose:      false,
	})
}

func denseAdaptationApplyGradients(net *poly.VolumetricNetwork, layerGradients [][2]*poly.Tensor[float32], lr float32) {
	for idx := range net.Layers {
		if idx < len(layerGradients) && layerGradients[idx][1] != nil {
			poly.ApplyRecursiveGradients(&net.Layers[idx], layerGradients[idx][1], lr, 0)
		}
	}
}

func appendDenseAdaptationSample(samples []denseAdaptationSample, input, target *poly.Tensor[float32]) []denseAdaptationSample {
	sample := denseAdaptationSample{Input: input, Target: target}
	if len(samples) < denseAdaptationMaxBatch {
		return append(samples, sample)
	}
	copy(samples, samples[1:])
	samples[len(samples)-1] = sample
	return samples
}

func denseAdaptationObservation(env *denseAdaptationEnvironment) *poly.Tensor[float32] {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]
	dist := float32(math.Sqrt(float64(relX*relX + relY*relY)))
	t := poly.NewTensor[float32](1, 8)
	copy(t.Data, []float32{
		env.AgentPos[0], env.AgentPos[1],
		env.TargetPos[0], env.TargetPos[1],
		relX, relY,
		dist,
		float32(env.Task),
	})
	return t
}

func denseAdaptationTarget(action int) *poly.Tensor[float32] {
	t := poly.NewTensor[float32](1, 4)
	if action >= 0 && action < len(t.Data) {
		t.Data[action] = 1
	}
	return t
}

func denseAdaptationTaskAt(elapsed time.Duration) int {
	if elapsed >= 5*time.Second && elapsed < 10*time.Second {
		return 1
	}
	return 0
}

func denseAdaptationTaskName(task int) string {
	if task == 1 {
		return "AVOID!"
	}
	return "chase"
}

func denseAdaptationOptimalAction(env *denseAdaptationEnvironment) int {
	relX := env.TargetPos[0] - env.AgentPos[0]
	relY := env.TargetPos[1] - env.AgentPos[1]

	if env.Task == 0 {
		if denseAdaptationAbs(relX) > denseAdaptationAbs(relY) {
			if relX > 0 {
				return 3 // right
			}
			return 2 // left
		}
		if relY > 0 {
			return 0 // up
		}
		return 1 // down
	}

	if denseAdaptationAbs(relX) > denseAdaptationAbs(relY) {
		if relX > 0 {
			return 2 // left, away
		}
		return 3 // right, away
	}
	if relY > 0 {
		return 1 // down, away
	}
	return 0 // up, away
}

func denseAdaptationExecuteAction(env *denseAdaptationEnvironment, action int) {
	speed := float32(0.02)
	moves := [][2]float32{{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}}
	if action >= 0 && action < len(moves) {
		env.AgentPos[0] = denseAdaptationClamp(env.AgentPos[0]+moves[action][0], 0, 1)
		env.AgentPos[1] = denseAdaptationClamp(env.AgentPos[1]+moves[action][1], 0, 1)
	}
}

func denseAdaptationUpdateEnvironment(env *denseAdaptationEnvironment, rng *rand.Rand) {
	env.TargetPos[0] += (rng.Float32() - 0.5) * 0.01
	env.TargetPos[1] += (rng.Float32() - 0.5) * 0.01
	env.TargetPos[0] = denseAdaptationClamp(env.TargetPos[0], 0.1, 0.9)
	env.TargetPos[1] = denseAdaptationClamp(env.TargetPos[1], 0.1, 0.9)
}

func finalizeDenseAdaptationWindow(w *denseAdaptationWindow) {
	w.OutputsPerSec = w.Outputs
	if w.Outputs > 0 {
		w.Accuracy = float64(w.Correct) / float64(w.Outputs) * 100
	}
}

func calculateDenseAdaptationMetrics(result *denseAdaptationResult) {
	if len(result.Windows) > 5 {
		result.PreChangeAccuracy = result.Windows[4].Accuracy
		result.PostChange1Accuracy = result.Windows[5].Accuracy
		result.DipDepth1 = denseAdaptationDipDepth(result.PreChangeAccuracy, result.PostChange1Accuracy)
		for i := 5; i < 10 && i < len(result.Windows); i++ {
			if result.Windows[i].Accuracy >= denseAdaptationRecoveryThreshold {
				result.AdaptTime1 = i - 5
				break
			}
		}
	}
	if len(result.Windows) > 10 {
		result.PostChange2Accuracy = result.Windows[10].Accuracy
		result.DipDepth2 = denseAdaptationDipDepth(result.Windows[9].Accuracy, result.PostChange2Accuracy)
		for i := 10; i < denseAdaptationWindowCount && i < len(result.Windows); i++ {
			if result.Windows[i].Accuracy >= denseAdaptationRecoveryThreshold {
				result.AdaptTime2 = i - 10
				break
			}
		}
	}
	result.AvgAccuracy = denseAdaptationAverageAccuracy(result.Windows)
	avgDelta := denseAdaptationAverageAccuracyDelta(result.Windows)
	result.Stability = denseAdaptationClampFloat(100-avgDelta, 0, 100)
	result.CompetenceStability = denseAdaptationClampFloat(result.AvgAccuracy-avgDelta, 0, 100)
	result.Chase1Avg = denseAdaptationAverageWindowRange(result.Windows, 0, 5)
	result.AvoidAvg = denseAdaptationAverageWindowRange(result.Windows, 5, 10)
	result.Chase2Avg = denseAdaptationAverageWindowRange(result.Windows, 10, 15)
	result.RecoveryPeak1 = denseAdaptationPeakWindowRange(result.Windows, 5, 10)
	result.RecoveryPeak2 = denseAdaptationPeakWindowRange(result.Windows, 10, 15)
	result.CompetenceFloor = denseAdaptationFloorWindowRange(result.Windows, 0, denseAdaptationWindowCount)
	result.Retention = denseAdaptationRetention(result.Chase1Avg, result.Chase2Avg)
	if result.TotalDuration > 0 {
		result.Throughput = float64(result.TotalOutputs) / result.TotalDuration.Seconds()
		available := result.TotalDuration - result.BlockedTime
		if available < 0 {
			available = 0
		}
		result.Availability = float64(available) / float64(result.TotalDuration) * 100
	}
	result.Score = result.Throughput * result.Availability * result.AvgAccuracy / 10000
	result.ZeroDowntimeScore = result.AvgAccuracy * result.Availability / 100
	if result.LatencySamples > 0 {
		result.DeadlineHitRate = float64(result.DeadlineHits) / float64(result.LatencySamples) * 100
	}
	result.DecisionActivityScore = result.Availability * result.DeadlineHitRate / 100
	result.ReactiveScore = result.ZeroDowntimeScore * result.DeadlineHitRate / 100
	if result.LatencySamples > 0 && result.TotalLatency > 0 {
		avgLatencyMS := (float64(result.TotalLatency) / float64(result.LatencySamples)) / float64(time.Millisecond)
		if avgLatencyMS > 0 {
			result.OnlineEfficiency = result.AvgAccuracy / avgLatencyMS
		}
	}
}

func denseAdaptationAverageAccuracy(windows []denseAdaptationWindow) float64 {
	if len(windows) == 0 {
		return 0
	}
	sum := 0.0
	for _, w := range windows {
		sum += w.Accuracy
	}
	return sum / float64(len(windows))
}

func denseAdaptationAverageAccuracyDelta(windows []denseAdaptationWindow) float64 {
	if len(windows) < 2 {
		return 0
	}
	totalDelta := 0.0
	transitions := 0
	for i := 1; i < len(windows); i++ {
		totalDelta += math.Abs(windows[i].Accuracy - windows[i-1].Accuracy)
		transitions++
	}
	if transitions == 0 {
		return 0
	}
	return totalDelta / float64(transitions)
}

func denseAdaptationAverageWindowRange(windows []denseAdaptationWindow, start, end int) float64 {
	start, end = denseAdaptationBoundWindowRange(len(windows), start, end)
	if start >= end {
		return 0
	}
	sum := 0.0
	for i := start; i < end; i++ {
		sum += windows[i].Accuracy
	}
	return sum / float64(end-start)
}

func denseAdaptationPeakWindowRange(windows []denseAdaptationWindow, start, end int) float64 {
	start, end = denseAdaptationBoundWindowRange(len(windows), start, end)
	if start >= end {
		return 0
	}
	peak := windows[start].Accuracy
	for i := start + 1; i < end; i++ {
		if windows[i].Accuracy > peak {
			peak = windows[i].Accuracy
		}
	}
	return peak
}

func denseAdaptationFloorWindowRange(windows []denseAdaptationWindow, start, end int) float64 {
	start, end = denseAdaptationBoundWindowRange(len(windows), start, end)
	if start >= end {
		return 0
	}
	floor := windows[start].Accuracy
	for i := start + 1; i < end; i++ {
		if windows[i].Accuracy < floor {
			floor = windows[i].Accuracy
		}
	}
	return floor
}

func denseAdaptationBoundWindowRange(n, start, end int) (int, int) {
	if start < 0 {
		start = 0
	}
	if end > n {
		end = n
	}
	if end < start {
		end = start
	}
	return start, end
}

func denseAdaptationRetention(chase1, chase2 float64) float64 {
	if chase1 < 1 {
		if chase2 < 1 {
			return 0
		}
		return 100
	}
	return denseAdaptationClampFloat(chase2/chase1*100, 0, 999)
}

func denseAdaptationDipDepth(before, after float64) float64 {
	if before <= after {
		return 0
	}
	return before - after
}

func printDenseAdaptationTimeline(results map[denseAdaptationMode]*denseAdaptationResult, modes []denseAdaptationMode) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ACCURACY OVER TIME (per 1-second window)                                                   ║")
	fmt.Println("║                   [0-5s: CHASE]    │    [5-10s: AVOID!]    │    [10-15s: CHASE]                                                 ║")
	fmt.Println("╠═══════════════════╦════╦════╦════╦════╦════║════╦════╦════╦════╦════║════╦════╦════╦════╦════╗")
	fmt.Println("║ Mode              ║SIMD║ 1s ║ 2s ║ 3s ║ 4s ║ 5s ║ 6s ║ 7s ║ 8s ║ 9s ║10s ║11s ║12s ║13s ║14s ║15s ║")
	fmt.Println("╠═══════════════════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╬════╣")

	for _, mode := range modes {
		r := results[mode]
		fmt.Printf("║ %-17s ║ %-3s", denseAdaptationModeNames[mode], denseAdaptationSimdLabel(mode))
		if r == nil || r.Err != "" {
			for i := 0; i < denseAdaptationWindowCount; i++ {
				fmt.Printf("║ ERR")
			}
			fmt.Println("║")
			continue
		}
		for i := 0; i < denseAdaptationWindowCount; i++ {
			acc := 0.0
			if i < len(r.Windows) {
				acc = r.Windows[i].Accuracy
			}
			fmt.Printf("║ %2.0f%%", acc)
		}
		fmt.Println("║")
	}
	fmt.Println("╚═══════════════════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╩════╝")
	fmt.Println("                         ↑ TASK CHANGE ↑                    ↑ TASK CHANGE ↑")
}

func printDenseAdaptationSummary(results map[denseAdaptationMode]*denseAdaptationResult, modes []denseAdaptationMode) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                           ADAPTATION SUMMARY                                                  ║")
	fmt.Println("╠═══════════════════╦════╦═══════════════╦════════════════════╦════════════╦════════════════════╦════════════╦═══════╣")
	fmt.Println("║ Mode              ║SIMD║ Total Outputs ║ 1st Change         ║ 1st Dip    ║ 2nd Change         ║ 2nd Dip    ║ Avg   ║")
	fmt.Println("║                   ║    ║ (actions/15s) ║ Before→After(delay)║            ║ Before→After(delay)║            ║ Acc   ║")
	fmt.Println("╠═══════════════════╬════╬═══════════════╬════════════════════╬════════════╬════════════════════╬════════════╬═══════╣")

	for _, mode := range modes {
		r := results[mode]
		if r == nil || r.Err != "" {
			errMsg := "unknown"
			if r != nil {
				errMsg = r.Err
			}
			fmt.Printf("║ %-17s ║ %-3s ║ %-13s ║ %-18s ║ %-10s ║ %-18s ║ %-10s ║ %-5s ║\n",
				denseAdaptationModeNames[mode],
				denseAdaptationSimdLabel(mode),
				"ERROR",
				denseAdaptationTruncate(errMsg, 21),
				"",
				"",
				"",
				"",
			)
			continue
		}

		fmt.Printf("║ %-17s ║ %-3s ║ %13d ║ %5.0f%%→%5.0f%% (%3s) ║ %8.1f%% ║ %5.0f%%→%5.0f%% (%3s) ║ %8.1f%% ║ %5.1f ║\n",
			denseAdaptationModeNames[mode],
			denseAdaptationSimdLabel(mode),
			r.TotalOutputs,
			r.PreChangeAccuracy, r.PostChange1Accuracy, denseAdaptationDelayLabel(r.AdaptTime1),
			r.DipDepth1,
			r.Windows[9].Accuracy, r.PostChange2Accuracy, denseAdaptationDelayLabel(r.AdaptTime2),
			r.DipDepth2,
			r.AvgAccuracy)
	}

	fmt.Println("╚═══════════════════╩════╩═══════════════╩════════════════════╩════════════╩════════════════════╩════════════╩═══════╝")
}

func printDenseOperationalMetrics(results map[denseAdaptationMode]*denseAdaptationResult, modes []denseAdaptationMode) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                           OPERATIONAL METRICS                                                              ║")
	fmt.Println("╠═══════════════════╦════╦═══════════╦═══════════╦═════════════╦════════════╦════════════╦════════════╦═════════════╦═══════════╣")
	fmt.Println("║ Mode              ║SIMD║ Smoothness║ Thruput/s ║ Availability║ Blocked ms ║ Peak Lat   ║ Avg Lat    ║ Score       ║ Runtime   ║")
	fmt.Println("╠═══════════════════╬════╬═══════════╬═══════════╬═════════════╬════════════╬════════════╬════════════╬═════════════╬═══════════╣")
	for _, mode := range modes {
		r := results[mode]
		if r == nil || r.Err != "" {
			errMsg := "unknown"
			if r != nil {
				errMsg = r.Err
			}
			fmt.Printf("║ %-17s ║ %-3s ║ %-9s ║ %-9s ║ %-11s ║ %-10s ║ %-10s ║ %-10s ║ %-11s ║ %-9s ║\n",
				denseAdaptationModeNames[mode],
				denseAdaptationSimdLabel(mode),
				"ERROR",
				"",
				denseAdaptationTruncate(errMsg, 11),
				"",
				"",
				"",
				"",
				"",
			)
			continue
		}
		avgLatency := time.Duration(0)
		if r.LatencySamples > 0 {
			avgLatency = r.TotalLatency / time.Duration(r.LatencySamples)
		}
		fmt.Printf("║ %-17s ║ %-3s ║ %8.1f%% ║ %9.0f ║ %10.1f%% ║ %10.0f ║ %-10s ║ %-10s ║ %11.0f ║ %-9s ║\n",
			denseAdaptationModeNames[mode],
			denseAdaptationSimdLabel(mode),
			r.Stability,
			r.Throughput,
			r.Availability,
			float64(r.BlockedTime.Microseconds())/1000,
			denseAdaptationLatencyLabel(r.PeakLatency),
			denseAdaptationLatencyLabel(avgLatency),
			r.Score,
			r.TotalDuration.Round(time.Millisecond).String(),
		)
	}
	fmt.Println("╚═══════════════════╩════╩═══════════╩═══════════╩═════════════╩════════════╩════════════╩════════════╩═════════════╩═══════════╝")
	fmt.Println("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                                         KEY INSIGHTS                                              │")
	fmt.Println("├────────────────────────────────────────────────────────────────────────────────────────────────────┤")
	fmt.Println("│ • Task changes at second 5 (chase→avoid) and second 10 (avoid→chase)                              │")
	fmt.Println("│ • 'Adapt delay' = seconds until accuracy recovers to 50%+ after task change                       │")
	fmt.Println("│ • 'Dip' = immediate accuracy drop at a task change; 0 means the first post-change window improved │")
	fmt.Println("│ • Score = throughput × availability% × average accuracy% / 10,000                                 │")
	fmt.Println("│ • ZDT = average accuracy × availability / 100; Eff/ms = average accuracy per avg latency ms       │")
	fmt.Println("│ • Activity = no-pause availability × deadline-hit rate; Reactive = ZDT × deadline-hit rate        │")
	fmt.Println("│ • Competence Smooth = avg accuracy minus volatility; Smoothness only measures low window jitter   │")
	fmt.Println("│ • NormalBP uses poly.Train on small periodic batches; Step+BP updates after each action output     │")
	fmt.Println("│ • Tween modes use target propagation; StepTween variants apply it after every output               │")
	fmt.Println("│ • SIMD column = Plan 9 AVX2/NEON dense forward; training/backward paths are unchanged (scalar)       │")
	fmt.Println("│                                                                                                   │")
	fmt.Println("│ For embodied AI: fast adaptation to changing goals matters more than offline convergence alone     │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────────────────────────┘")
}

func printDensePhaseRecoveryMetrics(results map[denseAdaptationMode]*denseAdaptationResult, modes []denseAdaptationMode) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                        PHASE / RECOVERY METRICS                                                            ║")
	fmt.Println("╠═══════════════════╦════╦════════╦════════╦════════╦════════╦════════╦════════╦═════════╦════════╦════════╦════════════════════╣")
	fmt.Println("║ Mode              ║SIMD║ Chase1 ║ Avoid  ║ Chase2 ║ Peak1  ║ Peak2  ║ Floor  ║ Return% ║ ZDT    ║ Eff/ms ║ Competence Smooth  ║")
	fmt.Println("╠═══════════════════╬════╬════════╬════════╬════════╬════════╬════════╬════════╬═════════╬════════╬════════╬════════════════════╣")
	for _, mode := range modes {
		r := results[mode]
		if r == nil || r.Err != "" {
			errMsg := "unknown"
			if r != nil {
				errMsg = r.Err
			}
			fmt.Printf("║ %-17s ║ %-3s ║ %-6s ║ %-6s ║ %-6s ║ %-6s ║ %-6s ║ %-6s ║ %-7s ║ %-6s ║ %-6s ║ %-18s ║\n",
				denseAdaptationModeNames[mode],
				denseAdaptationSimdLabel(mode),
				"ERROR",
				"",
				"",
				"",
				"",
				"",
				"",
				"",
				"",
				denseAdaptationTruncate(errMsg, 18),
			)
			continue
		}
		fmt.Printf("║ %-17s ║ %-3s ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %5.1f%% ║ %6.1f%% ║ %6.1f ║ %6.1f ║ %7.1f / %7.1f ║\n",
			denseAdaptationModeNames[mode],
			denseAdaptationSimdLabel(mode),
			r.Chase1Avg,
			r.AvoidAvg,
			r.Chase2Avg,
			r.RecoveryPeak1,
			r.RecoveryPeak2,
			r.CompetenceFloor,
			r.Retention,
			r.ZeroDowntimeScore,
			r.OnlineEfficiency,
			r.CompetenceStability,
			r.Stability,
		)
	}
	fmt.Println("╚═══════════════════╩════╩════════╩════════╩════════╩════════╩════════╩════════╩═════════╩════════╩════════╩════════════════════╝")
}

func printDenseDecisionActivityMetrics(results map[denseAdaptationMode]*denseAdaptationResult, modes []denseAdaptationMode) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                         DECISION ACTIVITY METRICS (deadline: 10ms)                                      ║")
	fmt.Println("╠═══════════════════╦════╦══════════╦══════════╦═══════════╦═══════════╦═══════════╦══════════╦═══════════╦════════════╣")
	fmt.Println("║ Mode              ║SIMD║ Decisions║ Hit Rate ║ Misses    ║ Await ms  ║ Worst Gap ║ Activity ║ Reactive  ║ No-Pause   ║")
	fmt.Println("╠═══════════════════╬════╬══════════╬══════════╬═══════════╬═══════════╬═══════════╬══════════╬═══════════╬════════════╣")
	for _, mode := range modes {
		r := results[mode]
		if r == nil || r.Err != "" {
			errMsg := "unknown"
			if r != nil {
				errMsg = r.Err
			}
			fmt.Printf("║ %-17s ║ %-3s ║ %-8s ║ %-8s ║ %-9s ║ %-9s ║ %-9s ║ %-8s ║ %-9s ║ %-10s ║\n",
				denseAdaptationModeNames[mode],
				denseAdaptationSimdLabel(mode),
				"ERROR",
				"",
				denseAdaptationTruncate(errMsg, 9),
				"",
				"",
				"",
				"",
				"",
			)
			continue
		}
		fmt.Printf("║ %-17s ║ %-3s ║ %8d ║ %7.1f%% ║ %9d ║ %9.1f ║ %-9s ║ %7.1f%% ║ %9.1f ║ %9.1f%% ║\n",
			denseAdaptationModeNames[mode],
			denseAdaptationSimdLabel(mode),
			r.LatencySamples,
			r.DeadlineHitRate,
			r.DeadlineMisses,
			float64(r.AwaitTime.Microseconds())/1000,
			denseAdaptationLatencyLabel(r.PeakLatency),
			r.DecisionActivityScore,
			r.ReactiveScore,
			r.Availability,
		)
	}
	fmt.Println("╚═══════════════════╩════╩══════════╩══════════╩═══════════╩═══════════╩═══════════╩══════════╩═══════════╩════════════╝")
}

func denseAdaptationDelayLabel(seconds int) string {
	if seconds < 0 {
		return "N/A"
	}
	return fmt.Sprintf("%ds", seconds)
}

func denseAdaptationLatencyLabel(d time.Duration) string {
	if d <= 0 {
		return "0s"
	}
	if d < time.Millisecond {
		return d.Round(time.Microsecond).String()
	}
	return d.Round(time.Millisecond).String()
}

func denseAdaptationArgmaxTensor(t *poly.Tensor[float32]) int {
	if t == nil {
		return 0
	}
	return denseAdaptationArgmax(t.Data)
}

func denseAdaptationArgmax(s []float32) int {
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

func denseAdaptationClamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func denseAdaptationClampFloat(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func denseAdaptationAbs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func denseAdaptationTruncate(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	if max <= 1 {
		return "…"
	}
	return string(r[:max-1]) + "…"
}

type denseAdaptationCaptureState struct {
	origStdout *os.File
	pipeReader *os.File
	pipeWriter *os.File
	logFile    *os.File
	done       chan struct{}
}

var denseCaptureState *denseAdaptationCaptureState

func denseAdaptationStartOutputCapture() (string, error) {
	if denseCaptureState != nil {
		return "", fmt.Errorf("capture already active")
	}
	if err := os.MkdirAll("lucy_testing_output", 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("dense_adaptation_%s.txt", time.Now().Format("20060102_150405"))
	path := filepath.Join("lucy_testing_output", name)
	logFile, err := os.Create(path)
	if err != nil {
		return "", err
	}
	orig := os.Stdout
	pipeReader, pipeWriter, err := os.Pipe()
	if err != nil {
		_ = logFile.Close()
		return "", err
	}

	state := &denseAdaptationCaptureState{
		origStdout: orig,
		pipeReader: pipeReader,
		pipeWriter: pipeWriter,
		logFile:    logFile,
		done:       make(chan struct{}),
	}
	denseCaptureState = state
	os.Stdout = pipeWriter

	go func() {
		_, _ = io.Copy(io.MultiWriter(orig, logFile), pipeReader)
		close(state.done)
	}()

	return path, nil
}

func denseAdaptationStopOutputCapture(path string) {
	state := denseCaptureState
	if state == nil {
		return
	}
	_ = state.pipeWriter.Close()
	os.Stdout = state.origStdout
	<-state.done
	_ = state.pipeReader.Close()
	_ = state.logFile.Close()
	denseCaptureState = nil
	fmt.Printf("💾 Dense adaptation output saved to %s\n", path)
}
