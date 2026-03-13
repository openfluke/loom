package poly

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// ============================================================================
// Training Metrics (Polymorphic)
// ============================================================================

// TrainingMetrics captures performance metrics for a training run.
type TrainingMetrics struct {
	Steps        int                   `json:"steps"`
	Accuracy     float64               `json:"accuracy"`
	Loss         float64               `json:"loss"`
	TimeTotal    time.Duration         `json:"time_total"`
	TimeToTarget time.Duration         `json:"time_to_target"`
	MemoryPeakMB float64               `json:"memory_peak_mb"`
	Milestones   map[int]time.Duration `json:"milestones"`
}

// NewTrainingMetrics creates an initialized TrainingMetrics.
func NewTrainingMetrics() TrainingMetrics {
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}
	return TrainingMetrics{
		Milestones: milestones,
	}
}

// ComparisonResult holds results from comparing multiple training methods.
type ComparisonResult struct {
	Name      string                     `json:"name"`
	NumLayers int                        `json:"num_layers"`
	Methods   map[string]TrainingMetrics `json:"methods"`
}

// NewComparisonResult initializes aComparisonResult.
func NewComparisonResult(name string, numLayers int) *ComparisonResult {
	return &ComparisonResult{
		Name:      name,
		NumLayers: numLayers,
		Methods:   make(map[string]TrainingMetrics),
	}
}

// DetermineBest returns the name of the best performing training method.
func (cr *ComparisonResult) DetermineBest() string {
	bestName := ""
	bestAcc := -1.0
	bestLoss := math.MaxFloat64

	for name, m := range cr.Methods {
		if m.Accuracy > bestAcc+1 {
			bestAcc = m.Accuracy
			bestLoss = m.Loss
			bestName = name
		} else if math.Abs(m.Accuracy-bestAcc) <= 1 {
			if m.Loss < bestLoss {
				bestLoss = m.Loss
				bestName = name
			}
		}
	}
	return bestName + " \u2713"
}

// MultiNetworkEvaluation benchmarks multiple models on the same data.
func MultiNetworkEvaluation[T Numeric](models map[string]*VolumetricNetwork, inputs []*Tensor[T], expected []float64) (map[string]*DeviationMetrics, error) {
	results := make(map[string]*DeviationMetrics)
	for name, n := range models {
		m, err := EvaluateNetworkPolymorphic(n, inputs, expected)
		if err != nil {
			return nil, fmt.Errorf("model %s: %v", name, err)
		}
		results[name] = m
	}
	return results, nil
}

func PrintMultiNetworkSummary(results map[string]*DeviationMetrics) {
	fmt.Println("\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
	fmt.Println("\u2551           MULTI-MODEL PERFORMANCE COMPARISON            \u2551")
	fmt.Println("\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")
	fmt.Println("\u2551 Model Name             | Accuracy | Score | Avg Dev    \u2551")
	fmt.Println("\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")

	for name, m := range results {
		fmt.Printf("\u2551 %-22s | %7.2f%% | %5.1f | %7.2f%%   \u2551\n", name, m.Accuracy, m.Score, m.AverageDeviation)
	}
	fmt.Println("\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569")
}

// ============================================================================
// Deviation Metrics (Polymorphic)
// ============================================================================

// DeviationBucket represents a specific deviation percentage range.
type DeviationBucket struct {
	RangeMin float64 `json:"range_min"`
	RangeMax float64 `json:"range_max"`
	Count    int     `json:"count"`
	Samples  []int   `json:"samples"`
}

// PredictionResult represents model performance on a single prediction.
type PredictionResult struct {
	SampleIndex    int     `json:"sample_index"`
	ExpectedOutput float64 `json:"expected"`
	ActualOutput   float64 `json:"actual"`
	Deviation      float64 `json:"deviation"` // % error
	Bucket         string  `json:"bucket"`
}

// DeviationMetrics stores the model performance breakdown.
type DeviationMetrics struct {
	Buckets          map[string]*DeviationBucket `json:"buckets"`
	Score            float64                     `json:"score"` // 0-100 quality score
	TotalSamples     int                         `json:"total_samples"`
	Failures         int                         `json:"failures"` // 100%+ deviations
	Results          []PredictionResult          `json:"results"`
	AverageDeviation float64                     `json:"avg_deviation"`
	CorrectCount     int                         `json:"correct_count"`
	Accuracy         float64                     `json:"accuracy"`
}

// NewDeviationMetrics initializes empty metrics.
func NewDeviationMetrics() *DeviationMetrics {
	return &DeviationMetrics{
		Buckets: map[string]*DeviationBucket{
			"0-10%":   {0, 10, 0, []int{}},
			"10-20%":  {10, 20, 0, []int{}},
			"20-30%":  {20, 30, 0, []int{}},
			"30-40%":  {30, 40, 0, []int{}},
			"40-50%":  {40, 50, 0, []int{}},
			"50-100%": {50, 100, 0, []int{}},
			"100%+":   {100, math.Inf(1), 0, []int{}},
		},
		Results: []PredictionResult{},
	}
}

// EvaluatePrediction categorizes expected vs actual results.
func EvaluatePrediction(sampleIndex int, expected, actual float64) PredictionResult {
	var deviation float64
	if math.Abs(expected) < 1e-10 {
		deviation = math.Abs(actual-expected) * 100
	} else {
		deviation = math.Abs((actual - expected) / expected * 100)
	}

	if math.IsNaN(deviation) || math.IsInf(deviation, 0) {
		deviation = 100
	}

	var bucketName string
	switch {
	case deviation <= 10: bucketName = "0-10%"
	case deviation <= 20: bucketName = "10-20%"
	case deviation <= 30: bucketName = "20-30%"
	case deviation <= 40: bucketName = "30-40%"
	case deviation <= 50: bucketName = "40-50%"
	case deviation <= 100: bucketName = "50-100%"
	default: bucketName = "100%+"
	}

	return PredictionResult{
		SampleIndex:    sampleIndex,
		ExpectedOutput: expected,
		ActualOutput:   actual,
		Deviation:      deviation,
		Bucket:         bucketName,
	}
}

// UpdateMetrics adds one prediction to the metrics.
func (dm *DeviationMetrics) UpdateMetrics(result PredictionResult) {
	bucket := dm.Buckets[result.Bucket]
	bucket.Count++
	bucket.Samples = append(bucket.Samples, result.SampleIndex)

	dm.TotalSamples++
	if result.Bucket == "100%+" {
		dm.Failures++
	}

	dm.Results = append(dm.Results, result)
	dm.Score += math.Max(0, 100-result.Deviation)

	if result.ActualOutput == result.ExpectedOutput {
		dm.CorrectCount++
	}
}

// ComputeFinalMetrics completes the scoring.
func (dm *DeviationMetrics) ComputeFinalMetrics() {
	if dm.TotalSamples == 0 { return }
	dm.Score = math.Max(0, dm.Score/float64(dm.TotalSamples))
	
	totalDev := 0.0
	for _, r := range dm.Results {
		totalDev += r.Deviation
	}
	dm.AverageDeviation = totalDev / float64(dm.TotalSamples)
	dm.Accuracy = float64(dm.CorrectCount) / float64(dm.TotalSamples) * 100
}

// EvaluateNetworkPolymorphic evaluates a VolumetricNetwork across multiple inputs.
func EvaluateNetworkPolymorphic[T Numeric](n *VolumetricNetwork, inputs []*Tensor[T], expected []float64) (*DeviationMetrics, error) {
	if len(inputs) != len(expected) {
		return nil, fmt.Errorf("length mismatch: inputs (%d) vs expected (%d)", len(inputs), len(expected))
	}

	metrics := NewDeviationMetrics()
	for i, input := range inputs {
		output, _, _ := ForwardPolymorphic(n, input)

		var actual float64
		if len(output.Data) == 1 {
			actual = float64(output.Data[0])
		} else {
			// Argmax for classification
			maxIdx := 0
			maxVal := output.Data[0]
			for j := 1; j < len(output.Data); j++ {
				if output.Data[j] > maxVal {
					maxVal = output.Data[j]
					maxIdx = j
				}
			}
			actual = float64(maxIdx)
		}

		res := EvaluatePrediction(i, expected[i], actual)
		metrics.UpdateMetrics(res)
	}

	metrics.ComputeFinalMetrics()
	return metrics, nil
}

// ============================================================================
// Adaptation Tracker (Polymorphic)
// ============================================================================

type TimeWindow struct {
	WindowIndex int           `json:"window_index"`
	Duration    time.Duration `json:"duration"`
	Outputs     int           `json:"outputs"`
	Correct     int           `json:"correct"`
	Accuracy    float64       `json:"accuracy"`
	OutputsPerSec int         `json:"outputs_per_sec"`
	CurrentTask string        `json:"current_task"`
	TaskID      int           `json:"task_id"`
}

type TaskChange struct {
	AtTime           time.Duration `json:"at_time"`
	FromTask         string        `json:"from_task"`
	ToTask           string        `json:"to_task"`
	PreChangeWindow  int           `json:"pre_change_window"`
	PostChangeWindow int           `json:"post_change_window"`
	PreAccuracy      float64       `json:"pre_accuracy"`
	PostAccuracy     float64       `json:"post_accuracy"`
	RecoveryTime     time.Duration `json:"recovery_time"`
}

type AdaptationResult struct {
	ModelName    string        `json:"model_name"`
	ModeName     string        `json:"mode_name"`
	TotalOutputs int           `json:"total_outputs"`
	AvgAccuracy  float64       `json:"avg_accuracy"`
	Windows      []TimeWindow  `json:"windows"`
	TaskChanges  []TaskChange  `json:"task_changes"`
	Duration     time.Duration `json:"duration"`
}

type AdaptationTracker struct {
	mu             sync.RWMutex
	windowDuration time.Duration
	numWindows     int
	windows        []TimeWindow
	taskChanges    []TaskChange
	currentWindow  int
	startTime      time.Time
	totalOutputs   int
	modelName      string
	modeName       string

	currentTask   string
	currentTaskID int
}

func NewAdaptationTracker(winDur, totalDur time.Duration) *AdaptationTracker {
	numWin := int(totalDur / winDur)
	if numWin < 1 { numWin = 1 }
	
	windows := make([]TimeWindow, numWin)
	for i := range windows {
		windows[i] = TimeWindow{WindowIndex: i, Duration: winDur}
	}

	return &AdaptationTracker{
		windowDuration: winDur,
		numWindows:     numWin,
		windows:        windows,
	}
}

func (at *AdaptationTracker) Start(initialTask string, initialTaskID int) {
	at.mu.Lock()
	defer at.mu.Unlock()
	at.startTime = time.Now()
	at.currentTask = initialTask
	at.currentTaskID = initialTaskID
	at.currentWindow = 0
}

func (at *AdaptationTracker) RecordOutput(correct bool) {
	at.mu.Lock()
	defer at.mu.Unlock()

	elapsed := time.Since(at.startTime)
	winIdx := int(elapsed / at.windowDuration)

	if winIdx < at.numWindows {
		at.currentWindow = winIdx
		w := &at.windows[winIdx]
		w.Outputs++
		at.totalOutputs++
		if correct { w.Correct++ }
		w.CurrentTask = at.currentTask
		w.TaskID = at.currentTaskID
	}
}

func (at *AdaptationTracker) Finalize() *AdaptationResult {
	at.mu.Lock()
	defer at.mu.Unlock()

	totalAcc := 0.0
	validWin := 0
	for i := range at.windows {
		w := &at.windows[i]
		if w.Outputs > 0 {
			w.Accuracy = float64(w.Correct) / float64(w.Outputs) * 100
			w.OutputsPerSec = int(float64(w.Outputs) / w.Duration.Seconds())
			totalAcc += w.Accuracy
			validWin++
		}
	}

	avgAcc := 0.0
	if validWin > 0 { avgAcc = totalAcc / float64(validWin) }

	return &AdaptationResult{
		ModelName:    at.modelName,
		ModeName:     at.modeName,
		TotalOutputs: at.totalOutputs,
		AvgAccuracy:  avgAcc,
		Windows:      at.windows,
		TaskChanges:  at.taskChanges,
		Duration:     time.Since(at.startTime),
	}
}

// ============================================================================
// Printing Utilities
// ============================================================================

func (dm *DeviationMetrics) PrintSummary() {
	fmt.Printf("\n\u2551 MODEL EVALUATION SUMMARY \u2551\n")
	fmt.Printf("Total Samples: %d\n", dm.TotalSamples)
	fmt.Printf("Quality Score: %.2f/100\n", dm.Score)
	fmt.Printf("Avg Deviation: %.2f%%\n", dm.AverageDeviation)
	fmt.Printf("Accuracy:      %.1f%%\n", dm.Accuracy)
	
	fmt.Println("\nDistribution:")
	order := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, name := range order {
		b := dm.Buckets[name]
		percent := float64(b.Count) / float64(dm.TotalSamples) * 100
		bar := ""
		for i := 0; i < int(percent/2); i++ { bar += "\u2588" }
		fmt.Printf("  %7s: %4d samples (%5.1f%%) %s\n", name, b.Count, percent, bar)
	}
}
