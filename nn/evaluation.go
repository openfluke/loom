package nn

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
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
	StepTween      TrainingMetrics `json:"step_tween"`       // Stepping + tween
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
