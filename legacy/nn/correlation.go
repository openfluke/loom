package nn

import (
	"encoding/json"
	"math"
)

// ============================================================================
// Correlation Analysis - WASM-Compatible Data Structures
// ============================================================================

// CorrelationMatrix holds the computed correlation data
type CorrelationMatrix struct {
	Labels  []string    `json:"labels"`  // Feature/column names
	Matrix  [][]float64 `json:"matrix"`  // NxN correlation values (-1 to 1)
	N       int         `json:"n"`       // Number of features
	Samples int         `json:"samples"` // Number of data samples used
}

// CorrelationResult wraps the matrix with additional statistics
type CorrelationResult struct {
	Correlation CorrelationMatrix `json:"correlation"`
	Means       []float64         `json:"means"`       // Mean of each feature
	StdDevs     []float64         `json:"std_devs"`    // Std deviation of each feature
	Mins        []float64         `json:"mins"`        // Min value per feature
	Maxs        []float64         `json:"maxs"`        // Max value per feature
}

// FeaturePair represents a correlation between two specific features
type FeaturePair struct {
	Feature1    string  `json:"feature1"`
	Feature2    string  `json:"feature2"`
	Correlation float64 `json:"correlation"`
	AbsCorr     float64 `json:"abs_correlation"`
}

// ============================================================================
// Correlation Computation Functions
// ============================================================================

// ComputeCorrelationMatrix calculates Pearson correlation matrix for a dataset.
// data: 2D array where rows are samples and columns are features
// labels: optional feature names (if nil, uses "Feature_0", "Feature_1", etc.)
// Returns a CorrelationResult with the matrix and feature statistics.
func ComputeCorrelationMatrix(data [][]float32, labels []string) *CorrelationResult {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}

	numSamples := len(data)
	numFeatures := len(data[0])

	// Generate default labels if not provided
	if labels == nil || len(labels) != numFeatures {
		labels = make([]string, numFeatures)
		for i := 0; i < numFeatures; i++ {
			labels[i] = "F" + itoa(i)
		}
	}

	// Compute means
	means := make([]float64, numFeatures)
	for j := 0; j < numFeatures; j++ {
		sum := 0.0
		for i := 0; i < numSamples; i++ {
			sum += float64(data[i][j])
		}
		means[j] = sum / float64(numSamples)
	}

	// Compute standard deviations, mins, maxs
	stdDevs := make([]float64, numFeatures)
	mins := make([]float64, numFeatures)
	maxs := make([]float64, numFeatures)

	for j := 0; j < numFeatures; j++ {
		mins[j] = float64(data[0][j])
		maxs[j] = float64(data[0][j])
		sumSq := 0.0
		for i := 0; i < numSamples; i++ {
			val := float64(data[i][j])
			diff := val - means[j]
			sumSq += diff * diff
			if val < mins[j] {
				mins[j] = val
			}
			if val > maxs[j] {
				maxs[j] = val
			}
		}
		stdDevs[j] = math.Sqrt(sumSq / float64(numSamples))
	}

	// Compute correlation matrix
	matrix := make([][]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		matrix[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if i == j {
				matrix[i][j] = 1.0 // Self-correlation is always 1
			} else if j < i {
				matrix[i][j] = matrix[j][i] // Symmetric
			} else {
				matrix[i][j] = pearsonCorrelation(data, i, j, means[i], means[j], stdDevs[i], stdDevs[j], numSamples)
			}
		}
	}

	return &CorrelationResult{
		Correlation: CorrelationMatrix{
			Labels:  labels,
			Matrix:  matrix,
			N:       numFeatures,
			Samples: numSamples,
		},
		Means:   means,
		StdDevs: stdDevs,
		Mins:    mins,
		Maxs:    maxs,
	}
}

// pearsonCorrelation computes Pearson correlation between two features
func pearsonCorrelation(data [][]float32, feat1, feat2 int, mean1, mean2, std1, std2 float64, n int) float64 {
	if std1 == 0 || std2 == 0 {
		return 0 // No correlation if no variance
	}

	sum := 0.0
	for i := 0; i < n; i++ {
		d1 := float64(data[i][feat1]) - mean1
		d2 := float64(data[i][feat2]) - mean2
		sum += d1 * d2
	}

	covariance := sum / float64(n)
	return covariance / (std1 * std2)
}

// ComputeCorrelationMatrixFloat64 is the float64 version for higher precision
func ComputeCorrelationMatrixFloat64(data [][]float64, labels []string) *CorrelationResult {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}

	numSamples := len(data)
	numFeatures := len(data[0])

	// Generate default labels if not provided
	if labels == nil || len(labels) != numFeatures {
		labels = make([]string, numFeatures)
		for i := 0; i < numFeatures; i++ {
			labels[i] = "F" + itoa(i)
		}
	}

	// Compute means
	means := make([]float64, numFeatures)
	for j := 0; j < numFeatures; j++ {
		sum := 0.0
		for i := 0; i < numSamples; i++ {
			sum += data[i][j]
		}
		means[j] = sum / float64(numSamples)
	}

	// Compute standard deviations, mins, maxs
	stdDevs := make([]float64, numFeatures)
	mins := make([]float64, numFeatures)
	maxs := make([]float64, numFeatures)

	for j := 0; j < numFeatures; j++ {
		mins[j] = data[0][j]
		maxs[j] = data[0][j]
		sumSq := 0.0
		for i := 0; i < numSamples; i++ {
			val := data[i][j]
			diff := val - means[j]
			sumSq += diff * diff
			if val < mins[j] {
				mins[j] = val
			}
			if val > maxs[j] {
				maxs[j] = val
			}
		}
		stdDevs[j] = math.Sqrt(sumSq / float64(numSamples))
	}

	// Compute correlation matrix
	matrix := make([][]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		matrix[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if i == j {
				matrix[i][j] = 1.0
			} else if j < i {
				matrix[i][j] = matrix[j][i]
			} else {
				matrix[i][j] = pearsonCorrelationF64(data, i, j, means[i], means[j], stdDevs[i], stdDevs[j], numSamples)
			}
		}
	}

	return &CorrelationResult{
		Correlation: CorrelationMatrix{
			Labels:  labels,
			Matrix:  matrix,
			N:       numFeatures,
			Samples: numSamples,
		},
		Means:   means,
		StdDevs: stdDevs,
		Mins:    mins,
		Maxs:    maxs,
	}
}

func pearsonCorrelationF64(data [][]float64, feat1, feat2 int, mean1, mean2, std1, std2 float64, n int) float64 {
	if std1 == 0 || std2 == 0 {
		return 0
	}

	sum := 0.0
	for i := 0; i < n; i++ {
		d1 := data[i][feat1] - mean1
		d2 := data[i][feat2] - mean2
		sum += d1 * d2
	}

	covariance := sum / float64(n)
	return covariance / (std1 * std2)
}

// ============================================================================
// Analysis Helpers
// ============================================================================

// GetStrongCorrelations returns pairs with |correlation| >= threshold, sorted by strength
func (cr *CorrelationResult) GetStrongCorrelations(threshold float64) []FeaturePair {
	var pairs []FeaturePair
	n := cr.Correlation.N

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ { // Upper triangle only (excludes diagonal)
			corr := cr.Correlation.Matrix[i][j]
			absCorr := math.Abs(corr)
			if absCorr >= threshold {
				pairs = append(pairs, FeaturePair{
					Feature1:    cr.Correlation.Labels[i],
					Feature2:    cr.Correlation.Labels[j],
					Correlation: corr,
					AbsCorr:     absCorr,
				})
			}
		}
	}

	// Sort by absolute correlation (descending)
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].AbsCorr > pairs[i].AbsCorr {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	return pairs
}

// GetCorrelationsWithFeature returns correlations for a specific feature, sorted by strength
func (cr *CorrelationResult) GetCorrelationsWithFeature(featureName string) []FeaturePair {
	var pairs []FeaturePair
	n := cr.Correlation.N

	// Find feature index
	featureIdx := -1
	for i, label := range cr.Correlation.Labels {
		if label == featureName {
			featureIdx = i
			break
		}
	}
	if featureIdx == -1 {
		return pairs // Feature not found
	}

	for j := 0; j < n; j++ {
		if j == featureIdx {
			continue // Skip self
		}
		corr := cr.Correlation.Matrix[featureIdx][j]
		pairs = append(pairs, FeaturePair{
			Feature1:    featureName,
			Feature2:    cr.Correlation.Labels[j],
			Correlation: corr,
			AbsCorr:     math.Abs(corr),
		})
	}

	// Sort by absolute correlation (descending)
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].AbsCorr > pairs[i].AbsCorr {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	return pairs
}

// ============================================================================
// JSON Serialization (WASM-Compatible)
// ============================================================================

// ToJSON serializes the correlation result to JSON string
func (cr *CorrelationResult) ToJSON() (string, error) {
	data, err := json.MarshalIndent(cr, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ToJSONCompact serializes without indentation (smaller size)
func (cr *CorrelationResult) ToJSONCompact() (string, error) {
	data, err := json.Marshal(cr)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// CorrelationResultFromJSON deserializes from JSON string
func CorrelationResultFromJSON(jsonStr string) (*CorrelationResult, error) {
	var result CorrelationResult
	err := json.Unmarshal([]byte(jsonStr), &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

// ============================================================================
// Spearman Rank Correlation (Non-Parametric)
// ============================================================================

// ComputeSpearmanMatrix calculates Spearman rank correlation (for non-linear relationships)
func ComputeSpearmanMatrix(data [][]float32, labels []string) *CorrelationResult {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}

	numSamples := len(data)
	numFeatures := len(data[0])

	// Convert to ranks
	rankedData := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		rankedData[i] = make([]float64, numFeatures)
	}

	for j := 0; j < numFeatures; j++ {
		// Get column values with indices
		type valIdx struct {
			val float32
			idx int
		}
		vals := make([]valIdx, numSamples)
		for i := 0; i < numSamples; i++ {
			vals[i] = valIdx{data[i][j], i}
		}

		// Sort by value
		for a := 0; a < len(vals)-1; a++ {
			for b := a + 1; b < len(vals); b++ {
				if vals[b].val < vals[a].val {
					vals[a], vals[b] = vals[b], vals[a]
				}
			}
		}

		// Assign ranks (1-based, handle ties by averaging)
		for i := 0; i < numSamples; {
			// Find end of tie group
			j := i + 1
			for j < numSamples && vals[j].val == vals[i].val {
				j++
			}
			// Average rank for tie group
			avgRank := float64(i+j+1) / 2.0 // (i+1 + j) / 2
			for k := i; k < j; k++ {
				rankedData[vals[k].idx][j] = avgRank
			}
			i = j
		}
	}

	// Now compute Pearson correlation on the ranks
	return ComputeCorrelationMatrixFloat64(rankedData, labels)
}

// ============================================================================
// Utility
// ============================================================================

// itoa converts int to string without importing strconv
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	if neg {
		digits = append([]byte{'-'}, digits...)
	}
	return string(digits)
}
