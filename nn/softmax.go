package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// InitSoftmaxLayer creates a standard softmax layer
func InitSoftmaxLayer() LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxStandard,
		Temperature:    1.0,
	}
}

// InitGridSoftmaxLayer creates a grid softmax layer for multi-agent decisions
// rows: number of agents/groups
// cols: number of actions per agent
func InitGridSoftmaxLayer(rows, cols int) LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxGrid,
		SoftmaxRows:    rows,
		SoftmaxCols:    cols,
		Temperature:    1.0,
	}
}

// InitHierarchicalSoftmaxLayer creates a hierarchical softmax layer
// levels: sizes at each level, e.g., [3, 3, 4] = 3 strategies × 3 units × 4 actions
func InitHierarchicalSoftmaxLayer(levels []int) LayerConfig {
	return LayerConfig{
		Type:            LayerSoftmax,
		SoftmaxVariant:  SoftmaxHierarchical,
		HierarchyLevels: levels,
		Temperature:     1.0,
	}
}

// InitTemperatureSoftmaxLayer creates a temperature-scaled softmax layer
// temperature: controls distribution sharpness (0.1=sharp, 1.0=normal, 5.0=smooth)
func InitTemperatureSoftmaxLayer(temperature float32) LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxTemperature,
		Temperature:    temperature,
	}
}

// InitGumbelSoftmaxLayer creates a Gumbel softmax layer (adds exploration noise)
// temperature: controls noise strength
func InitGumbelSoftmaxLayer(temperature float32) LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxGumbel,
		Temperature:    temperature,
		GumbelNoise:    true,
	}
}

// InitMaskedSoftmaxLayer creates a masked softmax layer
// maskSize: size of the mask array (must match input size)
func InitMaskedSoftmaxLayer(maskSize int) LayerConfig {
	mask := make([]bool, maskSize)
	for i := range mask {
		mask[i] = true // Default: all positions enabled
	}
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxMasked,
		Mask:           mask,
		Temperature:    1.0,
	}
}

// InitSparsemaxLayer creates a sparsemax layer (can output exact zeros)
func InitSparsemaxLayer() LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxSparse,
		Temperature:    1.0,
	}
}

// InitEntmaxLayer creates an entmax layer
// alpha: 1.0=softmax, 1.5=entmax-1.5, 2.0=sparsemax
func InitEntmaxLayer(alpha float32) LayerConfig {
	return LayerConfig{
		Type:           LayerSoftmax,
		SoftmaxVariant: SoftmaxEntmax,
		EntmaxAlpha:    alpha,
		Temperature:    1.0,
	}
}

// ForwardSoftmaxCPU applies the configured softmax variant
func ForwardSoftmaxCPU(input []float32, config *LayerConfig) ([]float32, error) {
	switch config.SoftmaxVariant {
	case SoftmaxStandard:
		return softmaxStandard(input, config.Temperature), nil

	case SoftmaxGrid:
		if config.SoftmaxRows == 0 || config.SoftmaxCols == 0 {
			return nil, fmt.Errorf("grid softmax requires rows and cols to be set")
		}
		return softmaxGrid(input, config.SoftmaxRows, config.SoftmaxCols, config.Temperature), nil

	case SoftmaxHierarchical:
		if len(config.HierarchyLevels) == 0 {
			return nil, fmt.Errorf("hierarchical softmax requires levels to be set")
		}
		return softmaxHierarchical(input, config.HierarchyLevels, config.Temperature), nil

	case SoftmaxTemperature:
		return softmaxStandard(input, config.Temperature), nil

	case SoftmaxGumbel:
		return softmaxGumbel(input, config.Temperature), nil

	case SoftmaxMasked:
		if config.Mask == nil {
			return nil, fmt.Errorf("masked softmax requires mask to be set")
		}
		return softmaxMasked(input, config.Mask, config.Temperature), nil

	case SoftmaxSparse:
		return softmaxSparse(input), nil

	case SoftmaxEntmax:
		return softmaxEntmax(input, config.EntmaxAlpha), nil

	default:
		return softmaxStandard(input, 1.0), nil
	}
}

// softmaxStandard applies standard softmax with optional temperature scaling
func softmaxStandard(logits []float32, temperature float32) []float32 {
	if temperature == 0 {
		temperature = 1.0
	}

	// Scale by temperature
	scaled := make([]float32, len(logits))
	maxLogit := logits[0] / temperature
	for i, v := range logits {
		scaled[i] = v / temperature
		if scaled[i] > maxLogit {
			maxLogit = scaled[i]
		}
	}

	// Numerical stability: subtract max
	exps := make([]float32, len(scaled))
	sum := float32(0.0)
	for i, v := range scaled {
		exps[i] = float32(math.Exp(float64(v - maxLogit)))
		sum += exps[i]
	}

	// Normalize
	probs := make([]float32, len(logits))
	for i := range exps {
		probs[i] = exps[i] / sum
	}

	return probs
}

// softmaxGrid applies independent softmax to each row
func softmaxGrid(logits []float32, rows, cols int, temperature float32) []float32 {
	result := make([]float32, len(logits))

	for r := 0; r < rows; r++ {
		rowStart := r * cols
		rowEnd := rowStart + cols
		rowSlice := logits[rowStart:rowEnd]
		rowProbs := softmaxStandard(rowSlice, temperature)
		copy(result[rowStart:rowEnd], rowProbs)
	}

	return result
}

// softmaxHierarchical applies softmax at each level of hierarchy
// For now, implements as grid softmax with last level
// More sophisticated implementation would do true tree-based softmax
func softmaxHierarchical(logits []float32, levels []int, temperature float32) []float32 {
	// Calculate total size
	totalSize := 1
	for _, size := range levels {
		totalSize *= size
	}

	if len(logits) != totalSize {
		// Fallback to standard softmax if size mismatch
		return softmaxStandard(logits, temperature)
	}

	// For simplicity, treat as grid with last level
	// More sophisticated: implement true hierarchical structure
	rows := totalSize / levels[len(levels)-1]
	cols := levels[len(levels)-1]

	return softmaxGrid(logits, rows, cols, temperature)
}

// softmaxGumbel applies Gumbel softmax (adds noise for exploration)
func softmaxGumbel(logits []float32, temperature float32) []float32 {
	if temperature == 0 {
		temperature = 1.0
	}

	noisy := make([]float32, len(logits))
	for i, v := range logits {
		// Gumbel noise: -log(-log(uniform))
		u := rand.Float32()
		// Avoid log(0)
		if u < 1e-10 {
			u = 1e-10
		}
		gumbel := -float32(math.Log(-math.Log(float64(u))))
		noisy[i] = (v + gumbel) / temperature
	}

	return softmaxStandard(noisy, 1.0)
}

// softmaxMasked applies softmax with masking (ignores masked positions)
func softmaxMasked(logits []float32, mask []bool, temperature float32) []float32 {
	if len(mask) != len(logits) {
		// If mask size mismatch, use standard softmax
		return softmaxStandard(logits, temperature)
	}

	masked := make([]float32, len(logits))
	for i := range logits {
		if mask[i] {
			masked[i] = logits[i]
		} else {
			masked[i] = -1e9 // Effectively -infinity
		}
	}

	return softmaxStandard(masked, temperature)
}

// softmaxSparse implements sparsemax (can output exact zeros)
// This is a simplified version - full sparsemax requires projection onto simplex
func softmaxSparse(logits []float32) []float32 {
	n := len(logits)
	sorted := make([]float32, n)
	copy(sorted, logits)

	// Simple bubble sort (good enough for small arrays)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if sorted[i] < sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Find threshold tau
	cumSum := float32(0)
	k := 0
	for i := 0; i < n; i++ {
		cumSum += sorted[i]
		val := sorted[i] - (cumSum-1.0)/float32(i+1)
		if val > 0 {
			k = i + 1
		} else {
			break
		}
	}

	// Calculate tau
	tau := float32(0)
	if k > 0 {
		cumSum = float32(0)
		for i := 0; i < k; i++ {
			cumSum += sorted[i]
		}
		tau = (cumSum - 1.0) / float32(k)
	}

	// Apply sparsemax: max(0, logit - tau)
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		result[i] = float32(math.Max(0, float64(logits[i]-tau)))
	}

	return result
}

// softmaxEntmax implements entmax (generalization of softmax and sparsemax)
// alpha=1.0 → softmax, alpha=2.0 → sparsemax
// Simplified implementation for alpha=1.5
func softmaxEntmax(logits []float32, alpha float32) []float32 {
	if alpha <= 1.0 {
		return softmaxStandard(logits, 1.0)
	}
	if alpha >= 2.0 {
		return softmaxSparse(logits)
	}

	// For alpha=1.5, use approximate entmax-1.5
	// Full implementation requires iterative root finding
	// This simplified version blends softmax and sparsemax
	weight := (alpha - 1.0) // 0.0 to 1.0 for alpha in [1.0, 2.0]

	softmaxProbs := softmaxStandard(logits, 1.0)
	sparsemaxProbs := softmaxSparse(logits)

	result := make([]float32, len(logits))
	for i := range result {
		result[i] = (1-weight)*softmaxProbs[i] + weight*sparsemaxProbs[i]
	}

	// Renormalize
	sum := float32(0)
	for _, v := range result {
		sum += v
	}
	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}

	return result
}

// SetMask updates the mask for a masked softmax layer
func (config *LayerConfig) SetMask(mask []bool) error {
	if config.Type != LayerSoftmax || config.SoftmaxVariant != SoftmaxMasked {
		return fmt.Errorf("can only set mask on masked softmax layers")
	}
	if len(mask) != len(config.Mask) {
		return fmt.Errorf("mask size mismatch: expected %d, got %d", len(config.Mask), len(mask))
	}
	config.Mask = mask
	return nil
}

// SetTemperature updates the temperature parameter
func (config *LayerConfig) SetTemperature(temp float32) {
	config.Temperature = temp
}
