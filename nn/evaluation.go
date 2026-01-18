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
	CorrectCount     int                         `json:"correct_count"` // Number of exact matches
	Accuracy         float64                     `json:"accuracy"`      // Percentage of exact matches
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

	// Exact match tracking (for classification accuracy)
	if result.ActualOutput == result.ExpectedOutput {
		dm.CorrectCount++
	}
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

	// Accuracy
	dm.Accuracy = float64(dm.CorrectCount) / float64(dm.TotalSamples)
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
		output, _ := n.Forward(input)

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
		output, _ := n.Forward(cpState)
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

// ============================================================================
// Save/Load Verification (WASM-Compatible, In-Memory)
// ============================================================================

// SaveLoadConsistencyResult captures the results of save/load verification
type SaveLoadConsistencyResult struct {
	Format       string  `json:"format"`        // "json" or "safetensors"
	TestSamples  int     `json:"test_samples"`  // Number of samples tested
	MaxDiff      float32 `json:"max_diff"`      // Maximum difference found
	AvgDiff      float32 `json:"avg_diff"`      // Average difference
	IsConsistent bool    `json:"is_consistent"` // Whether consistency check passed
	Tolerance    float32 `json:"tolerance"`     // Tolerance used
}

// VerifySaveLoadConsistency validates that a model produces identical outputs
// after being serialized and deserialized. Works entirely in-memory (WASM-compatible).
//
// Parameters:
//   - original: The network to test
//   - format: "json" or "safetensors"
//   - testInputs: Sample inputs to test (should be representative of your data)
//   - tolerance: Maximum allowed difference (e.g., 1e-6)
//
// Returns: Detailed results including max/avg differences and pass/fail status
func VerifySaveLoadConsistency(original *Network, format string, testInputs [][]float32, tolerance float32) (*SaveLoadConsistencyResult, error) {
	if len(testInputs) == 0 {
		return nil, fmt.Errorf("must provide at least one test input")
	}

	var serialized string
	var err error

	// Serialize to string (in-memory)
	if format == "json" {
		serialized, err = original.SaveModelToString("verify")
		if err != nil {
			return nil, fmt.Errorf("failed to serialize to JSON: %w", err)
		}
	} else if format == "safetensors" {
		// For safetensors, we need to use the tensor format
		tensors := make(map[string]TensorWithShape)
		for i := range original.Layers {
			l := &original.Layers[i]
			prefix := fmt.Sprintf("layer_%d", i)

			if len(l.Kernel) > 0 {
				tensors[prefix+".kernel"] = TensorWithShape{
					Values: l.Kernel,
					Shape:  []int{len(l.Kernel)},
					DType:  "F32",
				}
			}
			if len(l.Bias) > 0 {
				tensors[prefix+".bias"] = TensorWithShape{
					Values: l.Bias,
					Shape:  []int{len(l.Bias)},
					DType:  "F32",
				}
			}
		}

		// Convert to bytes (still in-memory)
		bytes, err := SerializeSafetensors(tensors)
		if err != nil {
			return nil, fmt.Errorf("failed to serialize to safetensors: %w", err)
		}
		serialized = string(bytes)
	} else {
		return nil, fmt.Errorf("unsupported format: %s (use 'json' or 'safetensors')", format)
	}

	// Deserialize
	var reloaded *Network
	if format == "json" {
		reloaded, err = LoadModelFromString(serialized, "verify")
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize from JSON: %w", err)
		}
	} else {
		// For safetensors, rebuild network and load weights
		reloaded = original.CloneForParallel()
		if reloaded == nil {
			return nil, fmt.Errorf("failed to clone network")
		}

		tensors, err := LoadSafetensorsWithShapes([]byte(serialized))
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize safetensors: %w", err)
		}

		// Load weights back
		for i := range reloaded.Layers {
			l := &reloaded.Layers[i]
			prefix := fmt.Sprintf("layer_%d", i)

			if tensor, ok := tensors[prefix+".kernel"]; ok {
				l.Kernel = tensor.Values
			}
			if tensor, ok := tensors[prefix+".bias"]; ok {
				l.Bias = tensor.Values
			}
		}
	}

	// Compare outputs
	maxDiff := float32(0.0)
	totalDiff := float32(0.0)
	comparisons := 0

	for _, input := range testInputs {
		out1, _ := original.Forward(input)
		out2, _ := reloaded.Forward(input)

		if len(out1) != len(out2) {
			return nil, fmt.Errorf("output size mismatch: %d vs %d", len(out1), len(out2))
		}

		for j := 0; j < len(out1); j++ {
			diff := out1[j] - out2[j]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDiff {
				maxDiff = diff
			}
			totalDiff += diff
			comparisons++
		}
	}

	avgDiff := totalDiff / float32(comparisons)
	isConsistent := maxDiff <= tolerance

	return &SaveLoadConsistencyResult{
		Format:       format,
		TestSamples:  len(testInputs),
		MaxDiff:      maxDiff,
		AvgDiff:      avgDiff,
		IsConsistent: isConsistent,
		Tolerance:    tolerance,
	}, nil
}

// PrintConsistencyResult prints a human-readable summary
func (r *SaveLoadConsistencyResult) PrintConsistencyResult() {
	fmt.Printf("\n=== Save/Load Consistency Verification ===\n")
	fmt.Printf("Format: %s\n", r.Format)
	fmt.Printf("Test Samples: %d\n", r.TestSamples)
	fmt.Printf("Max Difference: %.9f\n", r.MaxDiff)
	fmt.Printf("Avg Difference: %.9f\n", r.AvgDiff)
	fmt.Printf("Tolerance: %.9f\n", r.Tolerance)
	if r.IsConsistent {
		fmt.Println("✓ Consistency PASSED")
	} else {
		fmt.Println("✗ Consistency FAILED")
	}
	fmt.Println()
}

// ============================================================================
// Numerical Type Benchmarking (WASM-Compatible, In-Memory)
// ============================================================================

// NumericalTypeResult captures results for a single numerical type
type NumericalTypeResult struct {
	DType            string  `json:"dtype"`
	QualityScore     float64 `json:"quality_score"`     // 0-100 score
	AverageDeviation float64 `json:"average_deviation"` // Percentage deviation
	MemoryBytes      int64   `json:"memory_bytes"`      // Estimated memory usage
	ScaleFactor      float32 `json:"scale_factor"`      // Scale used for quantization
}

// NumericalTypeBenchmark holds all benchmark results
type NumericalTypeBenchmark struct {
	Results []NumericalTypeResult `json:"results"`
}

// ScaleWeights scales all weights in a network by a factor (for quantization)
// If unsigned=true, shifts values by +1 before scaling (for unsigned types)
func (n *Network) ScaleWeights(scale float32, unsigned bool) {
	for i := range n.Layers {
		l := &n.Layers[i]

		for j := range l.Kernel {
			if unsigned {
				l.Kernel[j] = (l.Kernel[j] + 1.0) * scale
			} else {
				l.Kernel[j] *= scale
			}
		}

		for j := range l.Bias {
			if unsigned {
				l.Bias[j] = (l.Bias[j] + 1.0) * scale
			} else {
				l.Bias[j] *= scale
			}
		}
	}
}

// UnscaleWeights reverses weight scaling (after loading quantized weights)
func (n *Network) UnscaleWeights(scale float32, unsigned bool) {
	invScale := 1.0 / scale
	for i := range n.Layers {
		l := &n.Layers[i]

		for j := range l.Kernel {
			l.Kernel[j] *= invScale
			if unsigned {
				l.Kernel[j] -= 1.0
			}
		}

		for j := range l.Bias {
			l.Bias[j] *= invScale
			if unsigned {
				l.Bias[j] -= 1.0
			}
		}
	}
}

// deepCloneNetwork creates a full copy of a network with deep copied weights.
// Unlike CloneForParallel which shares weight pointers, this function creates
// independent copies of all weight slices so modifications don't affect the original.
func deepCloneNetwork(n *Network) *Network {
	totalLayers := n.TotalLayers()

	clone := &Network{
		GridRows:        n.GridRows,
		GridCols:        n.GridCols,
		LayersPerCell:   n.LayersPerCell,
		InputSize:       n.InputSize,
		BatchSize:       n.BatchSize,
		Layers:          make([]LayerConfig, len(n.Layers)),
		activations:     make([][]float32, totalLayers+1),
		preActivations:  make([][]float32, totalLayers),
		kernelGradients: nil,
		biasGradients:   nil,
		learningRate:    n.learningRate,
	}

	// Deep copy each layer config including all weight slices
	for i, layer := range n.Layers {
		clone.Layers[i] = layer // Shallow copy first

		// Deep copy all weight slices
		l := &clone.Layers[i]

		if len(layer.Kernel) > 0 {
			l.Kernel = make([]float32, len(layer.Kernel))
			copy(l.Kernel, layer.Kernel)
		}
		if len(layer.Bias) > 0 {
			l.Bias = make([]float32, len(layer.Bias))
			copy(l.Bias, layer.Bias)
		}

		// RNN weights
		if len(layer.WeightIH) > 0 {
			l.WeightIH = make([]float32, len(layer.WeightIH))
			copy(l.WeightIH, layer.WeightIH)
		}
		if len(layer.WeightHH) > 0 {
			l.WeightHH = make([]float32, len(layer.WeightHH))
			copy(l.WeightHH, layer.WeightHH)
		}
		if len(layer.BiasH) > 0 {
			l.BiasH = make([]float32, len(layer.BiasH))
			copy(l.BiasH, layer.BiasH)
		}

		// LSTM weights (all 4 gates)
		if len(layer.WeightIH_i) > 0 {
			l.WeightIH_i = make([]float32, len(layer.WeightIH_i))
			copy(l.WeightIH_i, layer.WeightIH_i)
		}
		if len(layer.WeightHH_i) > 0 {
			l.WeightHH_i = make([]float32, len(layer.WeightHH_i))
			copy(l.WeightHH_i, layer.WeightHH_i)
		}
		if len(layer.BiasH_i) > 0 {
			l.BiasH_i = make([]float32, len(layer.BiasH_i))
			copy(l.BiasH_i, layer.BiasH_i)
		}
		if len(layer.WeightIH_f) > 0 {
			l.WeightIH_f = make([]float32, len(layer.WeightIH_f))
			copy(l.WeightIH_f, layer.WeightIH_f)
		}
		if len(layer.WeightHH_f) > 0 {
			l.WeightHH_f = make([]float32, len(layer.WeightHH_f))
			copy(l.WeightHH_f, layer.WeightHH_f)
		}
		if len(layer.BiasH_f) > 0 {
			l.BiasH_f = make([]float32, len(layer.BiasH_f))
			copy(l.BiasH_f, layer.BiasH_f)
		}
		if len(layer.WeightIH_g) > 0 {
			l.WeightIH_g = make([]float32, len(layer.WeightIH_g))
			copy(l.WeightIH_g, layer.WeightIH_g)
		}
		if len(layer.WeightHH_g) > 0 {
			l.WeightHH_g = make([]float32, len(layer.WeightHH_g))
			copy(l.WeightHH_g, layer.WeightHH_g)
		}
		if len(layer.BiasH_g) > 0 {
			l.BiasH_g = make([]float32, len(layer.BiasH_g))
			copy(l.BiasH_g, layer.BiasH_g)
		}
		if len(layer.WeightIH_o) > 0 {
			l.WeightIH_o = make([]float32, len(layer.WeightIH_o))
			copy(l.WeightIH_o, layer.WeightIH_o)
		}
		if len(layer.WeightHH_o) > 0 {
			l.WeightHH_o = make([]float32, len(layer.WeightHH_o))
			copy(l.WeightHH_o, layer.WeightHH_o)
		}
		if len(layer.BiasH_o) > 0 {
			l.BiasH_o = make([]float32, len(layer.BiasH_o))
			copy(l.BiasH_o, layer.BiasH_o)
		}

		// Attention weights
		if len(layer.QWeights) > 0 {
			l.QWeights = make([]float32, len(layer.QWeights))
			copy(l.QWeights, layer.QWeights)
		}
		if len(layer.KWeights) > 0 {
			l.KWeights = make([]float32, len(layer.KWeights))
			copy(l.KWeights, layer.KWeights)
		}
		if len(layer.VWeights) > 0 {
			l.VWeights = make([]float32, len(layer.VWeights))
			copy(l.VWeights, layer.VWeights)
		}
		if len(layer.OutputWeight) > 0 {
			l.OutputWeight = make([]float32, len(layer.OutputWeight))
			copy(l.OutputWeight, layer.OutputWeight)
		}
		if len(layer.QBias) > 0 {
			l.QBias = make([]float32, len(layer.QBias))
			copy(l.QBias, layer.QBias)
		}
		if len(layer.KBias) > 0 {
			l.KBias = make([]float32, len(layer.KBias))
			copy(l.KBias, layer.KBias)
		}
		if len(layer.VBias) > 0 {
			l.VBias = make([]float32, len(layer.VBias))
			copy(l.VBias, layer.VBias)
		}
		if len(layer.OutputBias) > 0 {
			l.OutputBias = make([]float32, len(layer.OutputBias))
			copy(l.OutputBias, layer.OutputBias)
		}

		// SwiGLU weights
		if len(layer.GateWeights) > 0 {
			l.GateWeights = make([]float32, len(layer.GateWeights))
			copy(l.GateWeights, layer.GateWeights)
		}
		if len(layer.UpWeights) > 0 {
			l.UpWeights = make([]float32, len(layer.UpWeights))
			copy(l.UpWeights, layer.UpWeights)
		}
		if len(layer.DownWeights) > 0 {
			l.DownWeights = make([]float32, len(layer.DownWeights))
			copy(l.DownWeights, layer.DownWeights)
		}
		if len(layer.GateBias) > 0 {
			l.GateBias = make([]float32, len(layer.GateBias))
			copy(l.GateBias, layer.GateBias)
		}
		if len(layer.UpBias) > 0 {
			l.UpBias = make([]float32, len(layer.UpBias))
			copy(l.UpBias, layer.UpBias)
		}
		if len(layer.DownBias) > 0 {
			l.DownBias = make([]float32, len(layer.DownBias))
			copy(l.DownBias, layer.DownBias)
		}

		// LayerNorm/RMSNorm
		if len(layer.Gamma) > 0 {
			l.Gamma = make([]float32, len(layer.Gamma))
			copy(l.Gamma, layer.Gamma)
		}
		if len(layer.Beta) > 0 {
			l.Beta = make([]float32, len(layer.Beta))
			copy(l.Beta, layer.Beta)
		}

		// Embedding
		if len(layer.EmbeddingWeights) > 0 {
			l.EmbeddingWeights = make([]float32, len(layer.EmbeddingWeights))
			copy(l.EmbeddingWeights, layer.EmbeddingWeights)
		}

		// Conv1D (uses Kernel and Bias which are already copied above)
	}

	// Initialize activation buffers
	for i := 0; i <= totalLayers; i++ {
		if i < len(n.activations) && n.activations[i] != nil {
			clone.activations[i] = make([]float32, len(n.activations[i]))
		}
	}
	for i := 0; i < totalLayers; i++ {
		if i < len(n.preActivations) && n.preActivations[i] != nil {
			clone.preActivations[i] = make([]float32, len(n.preActivations[i]))
		}
	}

	return clone
}

// BenchmarkNumericalTypes evaluates network performance across different numerical precisions.
// Works entirely in-memory (WASM-compatible). Tests how well the network performs after
// quantization to various data types.
//
// Parameters:
//   - baseNetwork: The network to benchmark (will be cloned, not modified)
//   - dtypes: List of dtype names to test (e.g., ["F32", "F16", "BF16", "I8"])
//   - scales: Corresponding scale factors for each dtype (1.0 for floats, larger for ints)
//   - testInputs: Sample inputs for evaluation
//   - expectedOutputs: Expected output values (for classification: class indices)
//
// Returns: Benchmark results for all dtypes
func BenchmarkNumericalTypes(baseNetwork *Network, dtypes []string, scales []float32, testInputs [][]float32, expectedOutputs []float64) (*NumericalTypeBenchmark, error) {
	if len(dtypes) != len(scales) {
		return nil, fmt.Errorf("dtypes and scales must have same length")
	}
	if len(testInputs) != len(expectedOutputs) {
		return nil, fmt.Errorf("testInputs and expectedOutputs must have same length")
	}

	benchmark := &NumericalTypeBenchmark{
		Results: make([]NumericalTypeResult, 0, len(dtypes)),
	}

	for i, dtype := range dtypes {
		scale := scales[i]
		isUnsigned := len(dtype) > 0 && dtype[0] == 'U'

		// Deep clone network with copied weights (not shared pointers)
		// This is critical because ScaleWeights modifies weights in-place
		testNet := deepCloneNetwork(baseNetwork)
		if testNet == nil {
			return nil, fmt.Errorf("failed to clone network for %s", dtype)
		}

		// Scale weights
		testNet.ScaleWeights(scale, isUnsigned)

		// Convert to safetensors and back (simulates quantization)
		tensors := make(map[string]TensorWithShape)
		for j := range testNet.Layers {
			l := &testNet.Layers[j]
			prefix := fmt.Sprintf("layer_%d", j)

			if len(l.Kernel) > 0 {
				tensors[prefix+".kernel"] = TensorWithShape{
					Values: l.Kernel,
					Shape:  []int{len(l.Kernel)},
					DType:  dtype,
				}
			}
			if len(l.Bias) > 0 {
				tensors[prefix+".bias"] = TensorWithShape{
					Values: l.Bias,
					Shape:  []int{len(l.Bias)},
					DType:  dtype,
				}
			}
		}

		// Serialize to bytes
		serialized, err := SerializeSafetensors(tensors)
		if err != nil {
			log.Printf("Warning: Failed to serialize %s: %v", dtype, err)
			continue
		}

		// Calculate memory usage
		memoryBytes := int64(len(serialized))

		// Deserialize back
		loadedTensors, err := LoadSafetensorsWithShapes(serialized)
		if err != nil {
			log.Printf("Warning: Failed to deserialize %s: %v", dtype, err)
			continue
		}

		// Reload weights
		for j := range testNet.Layers {
			l := &testNet.Layers[j]
			prefix := fmt.Sprintf("layer_%d", j)

			if tensor, ok := loadedTensors[prefix+".kernel"]; ok {
				l.Kernel = tensor.Values
			}
			if tensor, ok := loadedTensors[prefix+".bias"]; ok {
				l.Bias = tensor.Values
			}
		}

		// Unscale weights
		testNet.UnscaleWeights(scale, isUnsigned)

		// Evaluate using deviation metrics
		// Compare the output values at the predicted class index
		// (measures how close the quantized outputs are to the original outputs)
		expected := make([]float64, len(testInputs))
		actual := make([]float64, len(testInputs))

		for k := 0; k < len(testInputs); k++ {
			outOrig, _ := baseNetwork.Forward(testInputs[k])
			outNew, _ := testNet.Forward(testInputs[k])

			// Find which class the original model predicts
			maxIdx := 0
			for j := range outOrig {
				if outOrig[j] > outOrig[maxIdx] {
					maxIdx = j
				}
			}

			// Compare the OUTPUT VALUES at that index (not the class indices)
			expected[k] = float64(outOrig[maxIdx])
			actual[k] = float64(outNew[maxIdx])
		}

		metrics, err := EvaluateModel(expected, actual)
		if err != nil {
			log.Printf("Warning: Evaluation failed for %s: %v", dtype, err)
			continue
		}

		benchmark.Results = append(benchmark.Results, NumericalTypeResult{
			DType:            dtype,
			QualityScore:     metrics.Score,
			AverageDeviation: metrics.AverageDeviation,
			MemoryBytes:      memoryBytes,
			ScaleFactor:      scale,
		})
	}

	return benchmark, nil
}

// PrintNumericalTypeSummary prints a formatted comparison table
func (b *NumericalTypeBenchmark) PrintNumericalTypeSummary() {
	fmt.Println("\n=== NUMERICAL TYPE COMPARISON SUMMARY ===")
	fmt.Println()
	fmt.Printf("%-8s  %-15s  %-15s  %-15s  %-12s\n",
		"DType", "Quality Score", "Avg Deviation", "Memory", "Scale")
	fmt.Println("--------  ---------------  ---------------  ---------------  ------------")

	for _, res := range b.Results {
		fmt.Printf("%-8s  %13.2f%%  %13.4f%%  %15s  %12.1f\n",
			res.DType,
			res.QualityScore,
			res.AverageDeviation,
			formatBytes(res.MemoryBytes),
			res.ScaleFactor,
		)
	}
	fmt.Println()
}

// formatBytes formats a byte count as human-readable string
func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// GetBestByQuality returns the dtype with the highest quality score
func (b *NumericalTypeBenchmark) GetBestByQuality() *NumericalTypeResult {
	if len(b.Results) == 0 {
		return nil
	}
	best := &b.Results[0]
	for i := 1; i < len(b.Results); i++ {
		if b.Results[i].QualityScore > best.QualityScore {
			best = &b.Results[i]
		}
	}
	return best
}

// GetSmallest returns the dtype with the smallest memory footprint
func (b *NumericalTypeBenchmark) GetSmallest() *NumericalTypeResult {
	if len(b.Results) == 0 {
		return nil
	}
	smallest := &b.Results[0]
	for i := 1; i < len(b.Results); i++ {
		if b.Results[i].MemoryBytes < smallest.MemoryBytes {
			smallest = &b.Results[i]
		}
	}
	return smallest
}

// GetBestTradeoff returns the dtype with the best quality-to-size ratio
func (b *NumericalTypeBenchmark) GetBestTradeoff() *NumericalTypeResult {
	if len(b.Results) == 0 {
		return nil
	}

	bestScore := -1.0
	var best *NumericalTypeResult

	for i := range b.Results {
		// Score = Quality / (Memory in MB)
		// Higher is better
		memMB := float64(b.Results[i].MemoryBytes) / (1024 * 1024)
		if memMB < 0.01 {
			memMB = 0.01 // Avoid division by very small numbers
		}
		score := b.Results[i].QualityScore / memMB

		if score > bestScore {
			bestScore = score
			best = &b.Results[i]
		}
	}

	return best
}

// PrintDeviationComparisonTable prints a comparison of two deviation metrics side-by-side
// DeviationComparison holds two sets of metrics for comparison
type DeviationComparison struct {
	Name   string            `json:"name"`
	Before *DeviationMetrics `json:"before"`
	After  *DeviationMetrics `json:"after"`
}

// NewDeviationComparison creates a new comparison object
func NewDeviationComparison(name string, before, after *DeviationMetrics) *DeviationComparison {
	return &DeviationComparison{
		Name:   name,
		Before: before,
		After:  after,
	}
}

// PrintTable prints a formatted comparison table
func (dc *DeviationComparison) PrintTable() {
	if dc.Before == nil || dc.After == nil {
		fmt.Printf("Error: Comparison '%s' has nil metrics\n", dc.Name)
		return
	}

	fmt.Printf("\n╔═══════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %-59s  ║\n", dc.Name)
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Accuracy:      %6.1f%% → %6.1f%%                           ║\n",
		dc.Before.Accuracy*100, dc.After.Accuracy*100)
	fmt.Printf("║  Quality Score: %6.1f   → %6.1f                           ║\n",
		dc.Before.Score, dc.After.Score)
	fmt.Printf("║  Avg Deviation: %6.1f%% → %6.1f%%                          ║\n",
		dc.Before.AverageDeviation, dc.After.AverageDeviation)
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Deviation Distribution:                                      ║\n")
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")

	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucketName := range bucketOrder {
		beforeCount := 0
		if b, ok := dc.Before.Buckets[bucketName]; ok {
			beforeCount = b.Count
		}
		afterCount := 0
		if b, ok := dc.After.Buckets[bucketName]; ok {
			afterCount = b.Count
		}

		beforePct := 0.0
		if dc.Before.TotalSamples > 0 {
			beforePct = float64(beforeCount) / float64(dc.Before.TotalSamples) * 100
		}
		afterPct := 0.0
		if dc.After.TotalSamples > 0 {
			afterPct = float64(afterCount) / float64(dc.After.TotalSamples) * 100
		}

		fmt.Printf("║    %8s: %3d (%5.1f%%) → %3d (%5.1f%%)                    ║\n",
			bucketName, beforeCount, beforePct, afterCount, afterPct)
	}
	fmt.Printf("╚═══════════════════════════════════════════════════════════════╝\n")
}

// PrintDeviationComparisonTable prints a comparison of two deviation metrics side-by-side
// This is a wrapper around DeviationComparison for backward compatibility
func PrintDeviationComparisonTable(name string, before, after *DeviationMetrics) {
	NewDeviationComparison(name, before, after).PrintTable()
}
