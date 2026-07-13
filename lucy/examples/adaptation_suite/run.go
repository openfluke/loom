package adaptationsuite

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
)

// Window holds per-window adaptation stats.
type Window struct {
	Outputs       int
	ScoreSum      float64
	Accuracy      float64
	OutputsPerSec int
	Phase         int
	PhaseName     string
}

// Result is the outcome of one adaptation path run.
type Result struct {
	Path     Path
	DType    string
	Layer    string
	Windows  []Window
	Steps    int
	Err      string
	Duration time.Duration

	PreChangeAccuracy   float64
	PostChange1Accuracy float64
	AdaptTime1          int
	DipDepth1           float64
	PostChange2Accuracy float64
	AdaptTime2          int
	DipDepth2           float64

	BlockedTime    time.Duration
	PeakLatency    time.Duration
	TotalLatency   time.Duration
	LatencySamples int
	DeadlineHits   int
	DeadlineMisses int
	AwaitTime      time.Duration
	AvgAccuracy    float64
	Stability      float64
	Throughput     float64
	Availability   float64
	Score          float64

	Phase1Avg float64
	Phase2Avg float64
	Phase3Avg float64
}

// Run executes one mid-stream adaptation benchmark.
func Run(wire []byte, path Path, scenario Scenario, cfg Config) (res *Result) {
	res = &Result{
		Path:  path,
		Steps: cfg.Steps,
	}
	defer func() {
		if v := recover(); v != nil {
			if res == nil {
				res = &Result{Path: path, Steps: cfg.Steps}
			}
			res.Err = fmt.Sprintf("panic: %v", v)
		}
	}()

	net, err := poly.DeserializeNetwork(wire)
	if err != nil {
		res.Err = err.Error()
		return res
	}
	configureRuntime(net, path)

	var tweenState *poly.TweenState[float32]
	if path.Mode.UsesTween() {
		tc := poly.DefaultTweenConfig()
		tc.UseChainRule = path.Mode.UsesTweenChain()
		tc.LearningRate = cfg.LearningRate
		tweenState = poly.NewTweenState[float32](net, tc)
	}

	input := scenario.MakeInput()
	if input == nil {
		res.Err = "nil input"
		return res
	}

	outShape := resolveOutputShape(net, input)
	if outShape == nil {
		res.Err = "output shape"
		return res
	}

	res.Windows = make([]Window, cfg.Windows)
	stepsPerWindow := cfg.Steps / cfg.Windows
	if stepsPerWindow < 1 {
		stepsPerWindow = 1
	}

	trainBatch := make([]sample, 0, cfg.MaxBatch)
	lastTrain := time.Now()
	start := time.Now()
	currentWindow := 0

	for step := 0; step < cfg.Steps; step++ {
		actionStart := time.Now()
		newWindow := step / stepsPerWindow
		if newWindow >= cfg.Windows {
			newWindow = cfg.Windows - 1
		}
		for newWindow > currentWindow && currentWindow < cfg.Windows-1 {
			finalizeWindow(&res.Windows[currentWindow], res.Duration)
			currentWindow++
		}

		phase := phaseAt(step, cfg.Steps)
		res.Windows[currentWindow].Phase = phase
		res.Windows[currentWindow].PhaseName = phaseName(phase)

		target := scenario.PhaseTarget(phase, outShape)
		if target == nil {
			res.Err = "nil target"
			return res
		}

		var output *poly.Tensor[float32]
		var histIn, histPre []*poly.Tensor[float32]
		switch path.Mode {
		case UpdateNormalBP:
			output, _, _ = poly.ForwardPolymorphic(net, input)
		case UpdateStepBP:
			output, histIn, histPre = forwardWithHistory(net, input)
		default:
			output = poly.TweenForward(net, tweenState, input)
		}

		if output == nil {
			res.Err = "nil output"
			return res
		}

		score := stepScore(output, target)
		res.Windows[currentWindow].Outputs++
		res.Windows[currentWindow].ScoreSum += score

		trainBatch = appendSample(trainBatch, input, target, cfg.MaxBatch)

		switch path.Mode {
		case UpdateNormalBP:
			if time.Since(lastTrain) > cfg.TrainInterval && len(trainBatch) > 0 {
				blocked := time.Now()
				trainBatchBP(net, trainBatch, cfg.LearningRate)
				res.BlockedTime += time.Since(blocked)
				trainBatch = trainBatch[:0]
				lastTrain = time.Now()
			}
		case UpdateStepBP:
			if stepHistoryReady(net, histIn, histPre) {
				grad := poly.ComputeLossGradient(output, target, "mse")
				_, layerGradients, _ := poly.BackwardPolymorphic(net, grad, histIn, histPre)
				applyGradients(net, layerGradients, cfg.LearningRate)
			}
		case UpdateTween, UpdateTweenChain:
			if time.Since(lastTrain) > cfg.TrainInterval && len(trainBatch) > 0 {
				blocked := time.Now()
				for _, s := range trainBatch {
					poly.TweenForward(net, tweenState, s.Input)
					poly.TweenBackward(net, tweenState, s.Target)
					tweenState.CalculateLinkBudgets()
					poly.ApplyTweenGaps(net, tweenState, cfg.LearningRate)
				}
				res.BlockedTime += time.Since(blocked)
				trainBatch = trainBatch[:0]
				lastTrain = time.Now()
			}
		case UpdateStepTween, UpdateStepTweenChain:
			poly.TweenBackward(net, tweenState, target)
			tweenState.CalculateLinkBudgets()
			poly.ApplyTweenGaps(net, tweenState, cfg.LearningRate)
		}

		latency := time.Since(actionStart)
		res.TotalLatency += latency
		res.LatencySamples++
		if latency <= cfg.Deadline {
			res.DeadlineHits++
		} else {
			res.DeadlineMisses++
			res.AwaitTime += latency - cfg.Deadline
		}
		if latency > res.PeakLatency {
			res.PeakLatency = latency
		}
	}

	res.Duration = time.Since(start)
	if currentWindow < cfg.Windows {
		finalizeWindow(&res.Windows[currentWindow], res.Duration)
	}
	calculateMetrics(res, cfg)
	return res
}

func configureRuntime(net *poly.VolumetricNetwork, path Path) {
	net.UseTiling = false
	net.EnableMultiCoreTiling = false
	net.UseExactDType = path.Paradigm == ParadigmNative
	net.SetSimdForwardRecursive(path.SIMD)
	for i := range net.Layers {
		net.Layers[i].UseTiling = path.Paradigm == ParadigmQAT
		net.Layers[i].EnableMultiCoreTiling = false
		if net.Layers[i].WeightStore != nil {
			net.Layers[i].WeightStore.Scale = 1.0
		}
	}
	net.SyncToCPU()
}

func phaseAt(step, total int) int {
	third := total / 3
	if step < third {
		return 0
	}
	if step < 2*third {
		return 1
	}
	return 2
}

func phaseName(p int) string {
	switch p {
	case 0:
		return "A"
	case 1:
		return "B"
	default:
		return "A'"
	}
}

type sample struct {
	Input  *poly.Tensor[float32]
	Target *poly.Tensor[float32]
}

func appendSample(samples []sample, input, target *poly.Tensor[float32], max int) []sample {
	s := sample{Input: input, Target: target}
	if len(samples) < max {
		return append(samples, s)
	}
	copy(samples, samples[1:])
	samples[len(samples)-1] = s
	return samples
}

func stepScore(output, target *poly.Tensor[float32]) float64 {
	if output == nil || target == nil || len(output.Data) != len(target.Data) {
		return 0
	}
	loss := poly.CalculateLoss(output, target, "mse")
	if loss >= 1 {
		return 0
	}
	return (1 - loss) * 100
}

func forwardWithHistory(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) (*poly.Tensor[float32], []*poly.Tensor[float32], []*poly.Tensor[float32]) {
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

func stepHistoryReady(net *poly.VolumetricNetwork, histIn, histPre []*poly.Tensor[float32]) bool {
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

func trainBatchBP(net *poly.VolumetricNetwork, samples []sample, lr float32) {
	batches := make([]poly.TrainingBatch[float32], len(samples))
	for i, s := range samples {
		batches[i] = poly.TrainingBatch[float32]{Input: s.Input, Target: s.Target}
	}
	_, _ = poly.Train(net, batches, &poly.TrainingConfig{
		Epochs:       1,
		LearningRate: lr,
		LossType:     "mse",
		Verbose:      false,
	})
}

func applyGradients(net *poly.VolumetricNetwork, layerGradients [][2]*poly.Tensor[float32], lr float32) {
	for idx := range net.Layers {
		if idx < len(layerGradients) && layerGradients[idx][1] != nil {
			poly.ApplyRecursiveGradients(&net.Layers[idx], layerGradients[idx][1], lr, 0)
		}
	}
}

func finalizeWindow(w *Window, totalDur time.Duration) {
	if w.Outputs > 0 {
		w.Accuracy = w.ScoreSum / float64(w.Outputs)
	}
	if totalDur > 0 {
		w.OutputsPerSec = int(float64(w.Outputs) / totalDur.Seconds())
	}
}

func calculateMetrics(res *Result, cfg Config) {
	n := len(res.Windows)
	if n == 0 {
		return
	}
	phaseSize := cfg.Windows / 3
	if phaseSize < 1 {
		phaseSize = 1
	}
	p1End := phaseSize
	p2End := 2 * phaseSize
	if p1End < n {
		res.PreChangeAccuracy = avgWindowRange(res.Windows, 0, p1End)
	}
	if p2End <= n && p1End < n {
		res.PostChange1Accuracy = avgWindowRange(res.Windows, p1End, p2End)
		res.DipDepth1 = dipDepth(res.PreChangeAccuracy, res.PostChange1Accuracy)
		for i := p1End; i < p2End && i < n; i++ {
			if res.Windows[i].Accuracy >= cfg.RecoveryThresh {
				res.AdaptTime1 = i - p1End
				break
			}
		}
	}
	if p2End < n {
		res.PostChange2Accuracy = avgWindowRange(res.Windows, p2End, n)
		if p2End > 0 {
			res.DipDepth2 = dipDepth(res.Windows[p2End-1].Accuracy, res.PostChange2Accuracy)
		}
		for i := p2End; i < n; i++ {
			if res.Windows[i].Accuracy >= cfg.RecoveryThresh {
				res.AdaptTime2 = i - p2End
				break
			}
		}
	}
	res.Phase1Avg = avgWindowRange(res.Windows, 0, p1End)
	res.Phase2Avg = avgWindowRange(res.Windows, p1End, p2End)
	res.Phase3Avg = avgWindowRange(res.Windows, p2End, n)
	res.AvgAccuracy = avgWindowRange(res.Windows, 0, n)
	res.Stability = clamp(100-avgWindowDelta(res.Windows), 0, 100)
	if res.Duration > 0 {
		res.Throughput = float64(res.Steps) / res.Duration.Seconds()
		avail := res.Duration - res.BlockedTime
		if avail < 0 {
			avail = 0
		}
		res.Availability = float64(avail) / float64(res.Duration) * 100
	}
	res.Score = res.Throughput * res.Availability * res.AvgAccuracy / 10000
	if math.IsNaN(res.Score) || math.IsInf(res.Score, 0) {
		res.Score = 0
	}
}

func avgWindowRange(w []Window, start, end int) float64 {
	start, end = boundRange(len(w), start, end)
	if start >= end {
		return 0
	}
	sum := 0.0
	for i := start; i < end; i++ {
		sum += w[i].Accuracy
	}
	return sum / float64(end-start)
}

func avgWindowDelta(w []Window) float64 {
	if len(w) < 2 {
		return 0
	}
	total := 0.0
	for i := 1; i < len(w); i++ {
		total += math.Abs(w[i].Accuracy - w[i-1].Accuracy)
	}
	return total / float64(len(w)-1)
}

func dipDepth(before, after float64) float64 {
	if before <= after {
		return 0
	}
	return before - after
}

func boundRange(n, start, end int) (int, int) {
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

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// RegressionScenario builds phase-flipped sin/cos targets for any layer stack.
func RegressionScenario(makeInput func() *poly.Tensor[float32]) Scenario {
	return Scenario{
		MakeInput: makeInput,
		PhaseTarget: func(phase int, outShape []int) *poly.Tensor[float32] {
			tgt := poly.NewTensor[float32](outShape...)
			for i := range tgt.Data {
				switch phase {
				case 0:
					tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
				case 1:
					tgt.Data[i] = 0.5 + 0.3*float32(math.Cos(float64(i)*0.31+0.5))
				default:
					tgt.Data[i] = 0.5 + 0.3*float32(math.Sin(float64(i)*0.17))
				}
			}
			return tgt
		},
	}
}

func resolveOutputShape(net *poly.VolumetricNetwork, input *poly.Tensor[float32]) []int {
	out, _, _ := poly.ForwardPolymorphic(net, input)
	if out == nil || len(out.Shape) == 0 {
		return nil
	}
	shape := make([]int, len(out.Shape))
	copy(shape, out.Shape)
	for i := range net.Layers {
		net.Layers[i].ResetState()
	}
	return shape
}
