package nn

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"time"
)

// ============================================================================
// Training Comparison Metrics
// ============================================================================

// TrainingMetrics captures performance metrics for a training run
type TrainingMetrics struct {
	Steps        int                   `json:"steps"`          // Total training steps/iterations
	Accuracy     float64               `json:"accuracy"`       // Final accuracy percentage
	Loss         float32               `json:"loss"`           // Final loss value
	TimeTotal    time.Duration         `json:"time_total"`     // Total training time
	TimeToTarget time.Duration         `json:"time_to_target"` // Time to reach target accuracy
	MemoryPeakMB float64               `json:"memory_peak_mb"` // Peak memory usage in MB
	Milestones   map[int]time.Duration `json:"milestones"`     // Time to reach 10%, 20%, ... 100% accuracy
}

// NewTrainingMetrics creates an initialized TrainingMetrics with milestone tracking
func NewTrainingMetrics() TrainingMetrics {
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}
	return TrainingMetrics{
		Milestones: milestones,
	}
}

// ComparisonResult holds results from comparing multiple training methods
type ComparisonResult struct {
	Name           string          `json:"name"`             // Network/test name
	NumLayers      int             `json:"num_layers"`       // Number of layers tested
	NormalBP       TrainingMetrics `json:"normal_bp"`        // Normal backprop (no stepping)
	NormalTween    TrainingMetrics `json:"normal_tween"`     // Normal tween training
	StepBP         TrainingMetrics `json:"step_bp"`          // Stepping + backprop
	StepTween      TrainingMetrics `json:"step_tween"`       // Stepping + tween (legacy mode)
	StepTweenChain TrainingMetrics `json:"step_tween_chain"` // Stepping + tween (chain rule mode)
	BatchTween     TrainingMetrics `json:"batch_tween"`      // Batch tween (non-stepping)
	StepBatchTween TrainingMetrics `json:"step_batch_tween"` // Stepping + batch tween
}

// DetermineBest returns the name of the best performing training method
func (cr *ComparisonResult) DetermineBest() string {
	methods := []struct {
		name string
		acc  float64
		loss float32
	}{
		{"NormalBP", cr.NormalBP.Accuracy, cr.NormalBP.Loss},
		{"NormTween", cr.NormalTween.Accuracy, cr.NormalTween.Loss},
		{"Step+BP", cr.StepBP.Accuracy, cr.StepBP.Loss},
		{"StepTween", cr.StepTween.Accuracy, cr.StepTween.Loss},
		{"TChain", cr.StepTweenChain.Accuracy, cr.StepTweenChain.Loss},
		{"BatchTween", cr.BatchTween.Accuracy, cr.BatchTween.Loss},
		{"StepBatch", cr.StepBatchTween.Accuracy, cr.StepBatchTween.Loss},
	}

	bestIdx := 0
	for i := 1; i < len(methods); i++ {
		if methods[i].acc > methods[bestIdx].acc+1 {
			bestIdx = i
		} else if math.Abs(methods[i].acc-methods[bestIdx].acc) <= 1 {
			// Tie on accuracy, prefer lower loss
			if !math.IsNaN(float64(methods[i].loss)) && methods[i].loss < methods[bestIdx].loss {
				bestIdx = i
			}
		}
	}
	return methods[bestIdx].name + " ✓"
}

// UpdateMilestone records the time when an accuracy milestone is reached
func (tm *TrainingMetrics) UpdateMilestone(accuracy float64, elapsed time.Duration) {
	if tm.Milestones == nil {
		tm.Milestones = make(map[int]time.Duration)
	}
	for threshold := 10; threshold <= 100; threshold += 10 {
		if tm.Milestones[threshold] == 0 && accuracy >= float64(threshold) {
			tm.Milestones[threshold] = elapsed
		}
	}
}

// DeviationMetrics - Accuracy Deviation Heatmap Distribution
// Provides detailed performance breakdown showing how far predictions deviate from expected values
/*
Color Representation (for visualization):
  Red → High confidence, model is highly accurate (0-10% deviation)
  Orange/Yellow → Medium error range (10%-50% deviation)
  Blue → Predictions are significantly off (50%-100% deviation)
  Black → Beyond 100% deviation (model is extremely wrong)
*/

// DeviationBucket represents a specific deviation percentage range
type DeviationBucket struct {
	RangeMin float64 `json:"range_min"`
	RangeMax float64 `json:"range_max"`
	Count    int     `json:"count"`
	Samples  []int   `json:"samples"` // Track which sample indices fall in this bucket
}

// MarshalJSON implements custom JSON marshaling to handle infinity values
func (db DeviationBucket) MarshalJSON() ([]byte, error) {
	type Alias DeviationBucket
	return json.Marshal(&struct {
		RangeMin float64 `json:"range_min"`
		RangeMax float64 `json:"range_max"`
		Count    int     `json:"count"`
		Samples  []int   `json:"samples"`
	}{
		RangeMin: sanitizeFloat(db.RangeMin),
		RangeMax: sanitizeFloat(db.RangeMax),
		Count:    db.Count,
		Samples:  db.Samples,
	})
}

// PredictionResult represents the performance of the model on one prediction
type PredictionResult struct {
	SampleIndex    int     `json:"sample_index"`
	ExpectedOutput float64 `json:"expected"`
	ActualOutput   float64 `json:"actual"`
	Deviation      float64 `json:"deviation"` // Percentage deviation
	Bucket         string  `json:"bucket"`
}

// DeviationMetrics stores the full model performance breakdown
type DeviationMetrics struct {
	Buckets          map[string]*DeviationBucket `json:"buckets"`
	Score            float64                     `json:"score"` // Average quality score (0-100)
	TotalSamples     int                         `json:"total_samples"`
	Failures         int                         `json:"failures"`      // Count of 100%+ deviations
	Results          []PredictionResult          `json:"results"`       // All individual results
	AverageDeviation float64                     `json:"avg_deviation"` // Mean deviation across all samples
}

// NewDeviationMetrics initializes an empty metrics struct
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
		Score:   0,
		Results: []PredictionResult{},
	}
}

// EvaluatePrediction categorizes an expected vs actual output into a deviation bucket
func EvaluatePrediction(sampleIndex int, expected, actual float64) PredictionResult {
	var deviation float64
	if math.Abs(expected) < 1e-10 { // Handle near-zero expected values
		deviation = math.Abs(actual-expected) * 100 // Scale to percentage
	} else {
		deviation = math.Abs((actual - expected) / expected * 100) // % error
	}

	// Prevent NaN/Inf issues
	if math.IsNaN(deviation) || math.IsInf(deviation, 0) {
		deviation = 100 // Default worst case
	}

	var bucketName string
	switch {
	case deviation <= 10:
		bucketName = "0-10%"
	case deviation <= 20:
		bucketName = "10-20%"
	case deviation <= 30:
		bucketName = "20-30%"
	case deviation <= 40:
		bucketName = "30-40%"
	case deviation <= 50:
		bucketName = "40-50%"
	case deviation <= 100:
		bucketName = "50-100%"
	default:
		bucketName = "100%+"
	}

	return PredictionResult{
		SampleIndex:    sampleIndex,
		ExpectedOutput: expected,
		ActualOutput:   actual,
		Deviation:      deviation,
		Bucket:         bucketName,
	}
}

// UpdateMetrics updates the metrics with a single prediction result
func (dm *DeviationMetrics) UpdateMetrics(result PredictionResult) {
	bucket := dm.Buckets[result.Bucket]
	bucket.Count++
	bucket.Samples = append(bucket.Samples, result.SampleIndex)
	dm.Buckets[result.Bucket] = bucket

	dm.TotalSamples++
	if result.Bucket == "100%+" {
		dm.Failures++
	}

	dm.Results = append(dm.Results, result)

	// Compute score: lower deviations contribute more positively
	dm.Score += math.Max(0, 100-result.Deviation)
}

// ComputeFinalMetrics calculates final scores and averages
func (dm *DeviationMetrics) ComputeFinalMetrics() {
	if dm.TotalSamples == 0 {
		dm.Score = 0
		dm.AverageDeviation = 0
		return
	}

	// Average quality score (0-100, higher is better)
	dm.Score = math.Max(0, dm.Score/float64(dm.TotalSamples))

	// Average deviation percentage
	totalDeviation := 0.0
	for _, result := range dm.Results {
		totalDeviation += result.Deviation
	}
	dm.AverageDeviation = totalDeviation / float64(dm.TotalSamples)
}

// EvaluateModel evaluates model performance on a batch of predictions
// Returns detailed deviation metrics
func EvaluateModel(expectedOutputs, actualOutputs []float64) (*DeviationMetrics, error) {
	if len(expectedOutputs) != len(actualOutputs) {
		return nil, fmt.Errorf("mismatched expected (%d) vs actual (%d) output sizes", len(expectedOutputs), len(actualOutputs))
	}

	metrics := NewDeviationMetrics()

	for i := range expectedOutputs {
		result := EvaluatePrediction(i, expectedOutputs[i], actualOutputs[i])
		metrics.UpdateMetrics(result)
	}

	metrics.ComputeFinalMetrics()
	return metrics, nil
}

// EvaluateNetwork evaluates a network's predictions against expected outputs
// Runs forward passes and computes deviation metrics
func (n *Network) EvaluateNetwork(inputs [][]float32, expectedOutputs []float64) (*DeviationMetrics, error) {
	if len(inputs) != len(expectedOutputs) {
		return nil, fmt.Errorf("mismatched inputs (%d) vs expected outputs (%d)", len(inputs), len(expectedOutputs))
	}

	// Save and restore batch size for single-sample evaluation
	originalBatchSize := n.BatchSize
	n.BatchSize = 1
	defer func() { n.BatchSize = originalBatchSize }()

	metrics := NewDeviationMetrics()
	actualOutputs := make([]float64, len(inputs))

	// Run forward passes
	for i, input := range inputs {
		output, _ := n.ForwardCPU(input)

		// Extract prediction (argmax or first value depending on task)
		if len(output) == 1 {
			actualOutputs[i] = float64(output[0])
		} else {
			// Classification: argmax
			maxIdx := 0
			maxVal := output[0]
			for j := 1; j < len(output); j++ {
				if output[j] > maxVal {
					maxVal = output[j]
					maxIdx = j
				}
			}
			actualOutputs[i] = float64(maxIdx)
		}

		result := EvaluatePrediction(i, expectedOutputs[i], actualOutputs[i])
		metrics.UpdateMetrics(result)
	}

	metrics.ComputeFinalMetrics()
	return metrics, nil
}

// EvaluateFromCheckpointFiles loads checkpoint files and evaluates model performance
// Returns metrics and timing information
func (n *Network) EvaluateFromCheckpointFiles(checkpointFiles []string, expectedOutputs []float64) (*DeviationMetrics, time.Duration, time.Duration, error) {
	if len(checkpointFiles) != len(expectedOutputs) {
		return nil, 0, 0, fmt.Errorf("mismatched checkpoint files (%d) vs expected outputs (%d)", len(checkpointFiles), len(expectedOutputs))
	}

	metrics := NewDeviationMetrics()
	actualOutputs := make([]float64, len(checkpointFiles))

	var totalLoadTime, totalForwardTime time.Duration

	// Process each checkpoint file
	for i, cpFile := range checkpointFiles {
		startLoad := time.Now()
		data, err := os.ReadFile(cpFile)
		if err != nil {
			log.Printf("Failed to read checkpoint file %s: %v", cpFile, err)
			continue
		}
		var cpState []float32
		if err := json.Unmarshal(data, &cpState); err != nil {
			log.Printf("Failed to unmarshal checkpoint file %s: %v", cpFile, err)
			continue
		}
		totalLoadTime += time.Since(startLoad)

		startForward := time.Now()
		output, _ := n.ForwardCPU(cpState)
		totalForwardTime += time.Since(startForward)

		// Extract prediction
		if len(output) == 1 {
			actualOutputs[i] = float64(output[0])
		} else {
			maxIdx := 0
			maxVal := output[0]
			for j := 1; j < len(output); j++ {
				if output[j] > maxVal {
					maxVal = output[j]
					maxIdx = j
				}
			}
			actualOutputs[i] = float64(maxIdx)
		}

		result := EvaluatePrediction(i, expectedOutputs[i], actualOutputs[i])
		metrics.UpdateMetrics(result)
	}

	metrics.ComputeFinalMetrics()
	return metrics, totalLoadTime, totalForwardTime, nil
}

// PrintSummary prints a human-readable summary of the deviation metrics
func (dm *DeviationMetrics) PrintSummary() {
	fmt.Printf("\n=== Model Evaluation Summary ===\n")
	fmt.Printf("Total Samples: %d\n", dm.TotalSamples)
	fmt.Printf("Quality Score: %.2f/100\n", dm.Score)
	fmt.Printf("Average Deviation: %.2f%%\n", dm.AverageDeviation)
	fmt.Printf("Failures (>100%% deviation): %d (%.1f%%)\n", dm.Failures, float64(dm.Failures)/float64(dm.TotalSamples)*100)

	fmt.Printf("\nDeviation Distribution:\n")
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucketName := range bucketOrder {
		bucket := dm.Buckets[bucketName]
		percentage := float64(bucket.Count) / float64(dm.TotalSamples) * 100
		bar := ""
		for i := 0; i < int(percentage/2); i++ {
			bar += "█"
		}
		fmt.Printf("  %8s: %4d samples (%.1f%%) %s\n", bucketName, bucket.Count, percentage, bar)
	}
	fmt.Println()
}

// GetSamplesInBucket returns the sample indices that fall within a specific bucket
func (dm *DeviationMetrics) GetSamplesInBucket(bucketName string) []int {
	if bucket, exists := dm.Buckets[bucketName]; exists {
		return bucket.Samples
	}
	return []int{}
}

// GetWorstSamples returns the N samples with the highest deviation
func (dm *DeviationMetrics) GetWorstSamples(n int) []PredictionResult {
	if n > len(dm.Results) {
		n = len(dm.Results)
	}

	// Sort results by deviation (descending)
	sorted := make([]PredictionResult, len(dm.Results))
	copy(sorted, dm.Results)

	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].Deviation > sorted[i].Deviation {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted[:n]
}

// SaveMetrics saves the metrics to a JSON file
func (dm *DeviationMetrics) SaveMetrics(filepath string) error {
	// Create a copy with sanitized buckets (convert pointers to values and sanitize range values)
	sanitizedBuckets := make(map[string]DeviationBucket)
	for k, v := range dm.Buckets {
		bucket := *v
		bucket.RangeMin = sanitizeFloat(bucket.RangeMin)
		bucket.RangeMax = sanitizeFloat(bucket.RangeMax)
		sanitizedBuckets[k] = bucket
	}

	// Sanitize results (replace Inf with large numbers)
	sanitizedResults := make([]PredictionResult, len(dm.Results))
	for i, r := range dm.Results {
		sanitizedResults[i] = r
		sanitizedResults[i].Deviation = sanitizeFloat(r.Deviation)
	}

	// Create sanitized struct
	sanitized := struct {
		TotalSamples     int                        `json:"total_samples"`
		Score            float64                    `json:"score"`
		AverageDeviation float64                    `json:"avg_deviation"`
		Failures         int                        `json:"failures"`
		Buckets          map[string]DeviationBucket `json:"buckets"`
		Results          []PredictionResult         `json:"results"`
	}{
		TotalSamples:     dm.TotalSamples,
		Score:            dm.Score,
		AverageDeviation: sanitizeFloat(dm.AverageDeviation),
		Failures:         dm.Failures,
		Buckets:          sanitizedBuckets,
		Results:          sanitizedResults,
	}

	data, err := json.MarshalIndent(sanitized, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %w", err)
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write metrics file: %w", err)
	}

	return nil
}

// sanitizeFloat replaces Inf and NaN with values that can be JSON serialized
func sanitizeFloat(v float64) float64 {
	if math.IsInf(v, 1) {
		return 1e9 // Positive infinity -> very large number
	} else if math.IsInf(v, -1) {
		return -1e9 // Negative infinity -> very large negative number
	} else if math.IsNaN(v) {
		return 0.0
	}
	return v
}

// LoadMetrics loads deviation metrics from a JSON file
func LoadMetrics(filepath string) (*DeviationMetrics, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metrics file: %w", err)
	}

	var metrics DeviationMetrics
	if err := json.Unmarshal(data, &metrics); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metrics: %w", err)
	}

	return &metrics, nil
}

// ============================================================================
// Time-Window Adaptation Metrics (from Test 17/18)
// ============================================================================

// TimeWindow captures metrics for a single time window (typically 1 second)
type TimeWindow struct {
	WindowIndex   int           `json:"window_index"`
	Duration      time.Duration `json:"duration"`
	Outputs       int           `json:"outputs"`  // Number of outputs in this window
	Correct       int           `json:"correct"`  // Number of correct predictions
	Accuracy      float64       `json:"accuracy"` // Accuracy percentage
	OutputsPerSec int           `json:"outputs_per_sec"`
	CurrentTask   string        `json:"current_task"` // Task label for this window
	TaskID        int           `json:"task_id"`      // Numeric task identifier
}

// TaskChange represents a point where the task/goal changes
type TaskChange struct {
	AtTime           time.Duration `json:"at_time"`            // When the change occurs
	FromTask         string        `json:"from_task"`          // Previous task name
	ToTask           string        `json:"to_task"`            // New task name
	PreChangeWindow  int           `json:"pre_change_window"`  // Window index before change
	PostChangeWindow int           `json:"post_change_window"` // Window index after change
	PreAccuracy      float64       `json:"pre_accuracy"`       // Accuracy before change
	PostAccuracy     float64       `json:"post_accuracy"`      // Accuracy after change
	RecoveryWindows  int           `json:"recovery_windows"`   // Windows to recover to 50%+
	RecoveryTime     time.Duration `json:"recovery_time"`      // Time to recover
}

// AdaptationResult captures the full adaptation performance across time windows
type AdaptationResult struct {
	ModelName    string        `json:"model_name"`
	ModeName     string        `json:"mode_name"`
	TotalOutputs int           `json:"total_outputs"`
	AvgAccuracy  float64       `json:"avg_accuracy"`
	Windows      []TimeWindow  `json:"windows"`
	TaskChanges  []TaskChange  `json:"task_changes"`
	Duration     time.Duration `json:"duration"`
}

// AdaptationTracker tracks accuracy over time with task changes
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

	// Task state
	currentTask   string
	currentTaskID int
	taskSchedule  []scheduledTask // Pre-scheduled task changes
}

type scheduledTask struct {
	atTime time.Duration
	taskID int
	name   string
}

// NewAdaptationTracker creates a tracker for measuring adaptation over time
// windowDuration: typically 1 second
// totalDuration: total test duration (determines number of windows)
func NewAdaptationTracker(windowDuration, totalDuration time.Duration) *AdaptationTracker {
	numWindows := int(totalDuration / windowDuration)
	if numWindows < 1 {
		numWindows = 1
	}

	windows := make([]TimeWindow, numWindows)
	for i := range windows {
		windows[i] = TimeWindow{
			WindowIndex: i,
			Duration:    windowDuration,
		}
	}

	return &AdaptationTracker{
		windowDuration: windowDuration,
		numWindows:     numWindows,
		windows:        windows,
		taskChanges:    []TaskChange{},
		taskSchedule:   []scheduledTask{},
	}
}

// SetModelInfo sets the model and mode name for this tracker
func (at *AdaptationTracker) SetModelInfo(modelName, modeName string) {
	at.mu.Lock()
	defer at.mu.Unlock()
	at.modelName = modelName
	at.modeName = modeName
}

// ScheduleTaskChange schedules a task change at a specific time offset
// This should be called before Start()
func (at *AdaptationTracker) ScheduleTaskChange(atOffset time.Duration, taskID int, taskName string) {
	at.mu.Lock()
	defer at.mu.Unlock()
	at.taskSchedule = append(at.taskSchedule, scheduledTask{
		atTime: atOffset,
		taskID: taskID,
		name:   taskName,
	})
}

// Start begins the tracking session
func (at *AdaptationTracker) Start(initialTask string, initialTaskID int) {
	at.mu.Lock()
	defer at.mu.Unlock()
	at.startTime = time.Now()
	at.currentTask = initialTask
	at.currentTaskID = initialTaskID
	at.currentWindow = 0
	at.windows[0].CurrentTask = initialTask
	at.windows[0].TaskID = initialTaskID
}

// RecordOutput records an output (prediction) and whether it was correct
// Returns the current task ID so the caller knows what behavior to expect
func (at *AdaptationTracker) RecordOutput(correct bool) int {
	at.mu.Lock()
	defer at.mu.Unlock()

	elapsed := time.Since(at.startTime)
	newWindow := int(elapsed / at.windowDuration)

	// Check for task changes
	previousTaskID := at.currentTaskID
	for _, scheduled := range at.taskSchedule {
		if elapsed >= scheduled.atTime && at.currentTaskID != scheduled.taskID {
			// Record the task change
			preWindow := newWindow - 1
			if preWindow < 0 {
				preWindow = 0
			}
			at.taskChanges = append(at.taskChanges, TaskChange{
				AtTime:           scheduled.atTime,
				FromTask:         at.currentTask,
				ToTask:           scheduled.name,
				PreChangeWindow:  preWindow,
				PostChangeWindow: newWindow,
			})
			at.currentTask = scheduled.name
			at.currentTaskID = scheduled.taskID
		}
	}

	// Handle window transitions
	if newWindow != at.currentWindow && at.currentWindow < at.numWindows {
		// Finalize previous window
		w := &at.windows[at.currentWindow]
		if w.Outputs > 0 {
			w.Accuracy = float64(w.Correct) / float64(w.Outputs) * 100
			w.OutputsPerSec = w.Outputs
		}
	}

	// Update window index
	if newWindow < at.numWindows {
		at.currentWindow = newWindow
		w := &at.windows[at.currentWindow]
		w.Outputs++
		at.totalOutputs++
		if correct {
			w.Correct++
		}
		w.CurrentTask = at.currentTask
		w.TaskID = at.currentTaskID
	}

	// Return previous task ID (so caller can compare)
	return previousTaskID
}

// GetCurrentTask returns the current task ID
func (at *AdaptationTracker) GetCurrentTask() int {
	at.mu.RLock()
	defer at.mu.RUnlock()
	return at.currentTaskID
}

// Finalize computes final metrics and returns the AdaptationResult
func (at *AdaptationTracker) Finalize() *AdaptationResult {
	at.mu.Lock()
	defer at.mu.Unlock()

	// Finalize last window
	if at.currentWindow < at.numWindows {
		w := &at.windows[at.currentWindow]
		if w.Outputs > 0 {
			w.Accuracy = float64(w.Correct) / float64(w.Outputs) * 100
			w.OutputsPerSec = w.Outputs
		}
	}

	// Calculate average accuracy
	totalAcc := 0.0
	count := 0
	for _, w := range at.windows {
		if w.Outputs > 0 {
			totalAcc += w.Accuracy
			count++
		}
	}
	avgAcc := 0.0
	if count > 0 {
		avgAcc = totalAcc / float64(count)
	}

	// Calculate task change adaptation metrics
	for i := range at.taskChanges {
		tc := &at.taskChanges[i]
		if tc.PreChangeWindow >= 0 && tc.PreChangeWindow < len(at.windows) {
			tc.PreAccuracy = at.windows[tc.PreChangeWindow].Accuracy
		}
		if tc.PostChangeWindow >= 0 && tc.PostChangeWindow < len(at.windows) {
			tc.PostAccuracy = at.windows[tc.PostChangeWindow].Accuracy
		}

		// Calculate recovery time (windows to reach 50%+ accuracy)
		tc.RecoveryWindows = -1
		for j := tc.PostChangeWindow; j < len(at.windows); j++ {
			if at.windows[j].Accuracy >= 50 {
				tc.RecoveryWindows = j - tc.PostChangeWindow
				tc.RecoveryTime = time.Duration(tc.RecoveryWindows) * at.windowDuration
				break
			}
		}
	}

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

// GetWindows returns a copy of the current windows (thread-safe)
func (at *AdaptationTracker) GetWindows() []TimeWindow {
	at.mu.RLock()
	defer at.mu.RUnlock()
	result := make([]TimeWindow, len(at.windows))
	copy(result, at.windows)
	return result
}

// ============================================================================
// Adaptation Result Printing
// ============================================================================

// PrintTimeline prints an ASCII timeline of accuracy over time
func (ar *AdaptationResult) PrintTimeline() {
	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %s — ACCURACY OVER TIME (per window)                                        ║\n", ar.ModelName)

	// Build task phase header
	phases := ""
	if len(ar.TaskChanges) > 0 {
		phases = "║  "
		lastEnd := 0
		for i, tc := range ar.TaskChanges {
			windowsInPhase := tc.PostChangeWindow - lastEnd
			if windowsInPhase < 0 {
				windowsInPhase = 0
			}
			label := fmt.Sprintf("[%s]", tc.FromTask)
			phases += fmt.Sprintf("%-*s", windowsInPhase*5, label)
			lastEnd = tc.PostChangeWindow
			if i == len(ar.TaskChanges)-1 {
				// Last phase
				remaining := len(ar.Windows) - lastEnd
				label2 := fmt.Sprintf("[%s]", tc.ToTask)
				phases += fmt.Sprintf("%-*s", remaining*5, label2)
			}
		}
		phases += " ║"
	}
	if phases != "" {
		fmt.Println(phases)
	}

	// Header
	fmt.Print("╠═══════════════════╦")
	for i := 0; i < len(ar.Windows) && i < 15; i++ {
		fmt.Print("════╦")
	}
	fmt.Println()

	fmt.Printf("║ %-17s ║", "Window")
	for i := 0; i < len(ar.Windows) && i < 15; i++ {
		fmt.Printf(" %2ds ║", i+1)
	}
	fmt.Println()

	fmt.Print("╠═══════════════════╬")
	for i := 0; i < len(ar.Windows) && i < 15; i++ {
		fmt.Print("════╬")
	}
	fmt.Println()

	// Data row
	fmt.Printf("║ %-17s ║", ar.ModeName)
	for i := 0; i < len(ar.Windows) && i < 15; i++ {
		fmt.Printf(" %2.0f%%║", ar.Windows[i].Accuracy)
	}
	fmt.Println()

	fmt.Print("╚═══════════════════╩")
	for i := 0; i < len(ar.Windows) && i < 15; i++ {
		fmt.Print("════╩")
	}
	fmt.Println()

	// Task change markers
	if len(ar.TaskChanges) > 0 {
		markerLine := "                     "
		for _, tc := range ar.TaskChanges {
			pos := tc.PostChangeWindow*5 + 10
			for len(markerLine) < pos {
				markerLine += " "
			}
			markerLine = markerLine[:pos] + "↑ CHANGE"
		}
		fmt.Println(markerLine)
	}
}

// PrintAdaptationSummary prints a summary of adaptation performance
func (ar *AdaptationResult) PrintAdaptationSummary() {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║  %s — ADAPTATION SUMMARY                                                     ║\n", ar.ModelName)
	fmt.Println("╠═══════════════════╦═════════════════╦═══════════════════════════╦══════════════╣")
	fmt.Println("║ Mode              ║ Total Outputs   ║ Task Changes              ║ Avg Acc      ║")
	fmt.Println("╠═══════════════════╬═════════════════╬═══════════════════════════╬══════════════╣")

	changeInfo := ""
	for _, tc := range ar.TaskChanges {
		changeInfo += fmt.Sprintf("%.0f%%→%.0f%% ", tc.PreAccuracy, tc.PostAccuracy)
	}
	if changeInfo == "" {
		changeInfo = "N/A"
	}

	fmt.Printf("║ %-17s ║ %13d   ║ %-25s ║   %5.1f%%    ║\n",
		ar.ModeName,
		ar.TotalOutputs,
		changeInfo,
		ar.AvgAccuracy)

	fmt.Println("╚═══════════════════╩═════════════════╩═══════════════════════════╩══════════════╝")
}

// ============================================================================
// Multi-Model Comparison
// ============================================================================

// AdaptationComparison holds results from multiple models/modes for comparison
type AdaptationComparison struct {
	Results []AdaptationResult `json:"results"`
}

// NewAdaptationComparison creates a new comparison container
func NewAdaptationComparison() *AdaptationComparison {
	return &AdaptationComparison{
		Results: []AdaptationResult{},
	}
}

// AddResult adds a result to the comparison
func (ac *AdaptationComparison) AddResult(result *AdaptationResult) {
	ac.Results = append(ac.Results, *result)
}

// PrintComparisonTimeline prints a side-by-side timeline comparison
func (ac *AdaptationComparison) PrintComparisonTimeline(title string, numWindows int) {
	if numWindows < 1 || numWindows > 20 {
		numWindows = 10
	}

	fmt.Printf("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %-90s  ║\n", title+" — ACCURACY OVER TIME")

	// Header
	fmt.Print("╠═══════════════════╦")
	for i := 0; i < numWindows; i++ {
		fmt.Print("════╦")
	}
	fmt.Println()

	fmt.Printf("║ %-17s ║", "Mode")
	for i := 0; i < numWindows; i++ {
		fmt.Printf(" %2ds ║", i+1)
	}
	fmt.Println()

	fmt.Print("╠═══════════════════╬")
	for i := 0; i < numWindows; i++ {
		fmt.Print("════╬")
	}
	fmt.Println()

	// Data rows
	for _, r := range ac.Results {
		fmt.Printf("║ %-17s ║", r.ModeName)
		for i := 0; i < numWindows; i++ {
			if i < len(r.Windows) {
				fmt.Printf(" %2.0f%%║", r.Windows[i].Accuracy)
			} else {
				fmt.Printf("  - ║")
			}
		}
		fmt.Println()
	}

	fmt.Print("╚═══════════════════╩")
	for i := 0; i < numWindows; i++ {
		fmt.Print("════╩")
	}
	fmt.Println()
}

// PrintComparisonSummary prints a summary table comparing all results
func (ac *AdaptationComparison) PrintComparisonSummary(title string) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║  %-90s  ║\n", title+" — ADAPTATION SUMMARY")
	fmt.Println("╠═══════════════════╦═════════════════╦═══════════════════════════════════════════════╦══════════════╣")
	fmt.Println("║ Mode              ║ Total Outputs   ║ Task Change Adaptations                       ║ Avg Acc      ║")
	fmt.Println("║                   ║                 ║ Before→After (recovery)                       ║              ║")
	fmt.Println("╠═══════════════════╬═════════════════╬═══════════════════════════════════════════════╬══════════════╣")

	for _, r := range ac.Results {
		changeInfo := ""
		for i, tc := range r.TaskChanges {
			recovery := "N/A"
			if tc.RecoveryWindows >= 0 {
				recovery = fmt.Sprintf("%ds", tc.RecoveryWindows)
			}
			changeInfo += fmt.Sprintf("%.0f%%→%.0f%%(%s)", tc.PreAccuracy, tc.PostAccuracy, recovery)
			if i < len(r.TaskChanges)-1 {
				changeInfo += " | "
			}
		}
		if changeInfo == "" {
			changeInfo = "No task changes"
		}
		// Truncate if too long
		if len(changeInfo) > 45 {
			changeInfo = changeInfo[:42] + "..."
		}

		fmt.Printf("║ %-17s ║ %13d   ║ %-45s ║   %5.1f%%    ║\n",
			r.ModeName,
			r.TotalOutputs,
			changeInfo,
			r.AvgAccuracy)
	}

	fmt.Println("╚═══════════════════╩═════════════════╩═══════════════════════════════════════════════╩══════════════╝")
}

// GetBestByAvgAccuracy returns the result with the highest average accuracy
func (ac *AdaptationComparison) GetBestByAvgAccuracy() *AdaptationResult {
	if len(ac.Results) == 0 {
		return nil
	}
	best := &ac.Results[0]
	for i := 1; i < len(ac.Results); i++ {
		if ac.Results[i].AvgAccuracy > best.AvgAccuracy {
			best = &ac.Results[i]
		}
	}
	return best
}

// GetMostStable returns the result with the smallest accuracy variance (most consistent)
func (ac *AdaptationComparison) GetMostStable() *AdaptationResult {
	if len(ac.Results) == 0 {
		return nil
	}

	minVariance := math.MaxFloat64
	var mostStable *AdaptationResult

	for i := range ac.Results {
		r := &ac.Results[i]
		if len(r.Windows) == 0 {
			continue
		}

		// Calculate variance
		sum := 0.0
		for _, w := range r.Windows {
			sum += w.Accuracy
		}
		mean := sum / float64(len(r.Windows))

		variance := 0.0
		for _, w := range r.Windows {
			diff := w.Accuracy - mean
			variance += diff * diff
		}
		variance /= float64(len(r.Windows))

		if variance < minVariance {
			minVariance = variance
			mostStable = r
		}
	}

	return mostStable
}

// SaveToJSON saves the comparison results to a JSON file
func (ac *AdaptationComparison) SaveToJSON(filepath string) error {
	data, err := json.MarshalIndent(ac, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal comparison: %w", err)
	}
	return os.WriteFile(filepath, data, 0644)
}
