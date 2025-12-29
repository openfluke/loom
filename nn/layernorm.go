package nn

import (
	"math"
)

// =============================================================================
// Generic LayerNorm Implementation
// =============================================================================

// LayerNormForward performs layer normalization for any numeric type.
// input shape: [batchSize][normSize] (flattened)
// residual: optional residual connection to add before normalization
func LayerNormForward[T Numeric](input, residual, gamma, beta *Tensor[T], normSize, batchSize int, epsilon float64) *Tensor[T] {
	if epsilon == 0 {
		epsilon = 1e-5
	}

	// Add residual if provided
	inputWithResidual := input.Clone()
	if residual != nil && len(residual.Data) == len(input.Data) {
		for i := range inputWithResidual.Data {
			inputWithResidual.Data[i] += residual.Data[i]
		}
	}

	output := NewTensor[T](len(inputWithResidual.Data))

	// Normalize each sample in the batch
	for b := 0; b < batchSize; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual.Data) {
			break
		}

		// Calculate mean
		var sum float64
		for i := start; i < end; i++ {
			sum += float64(inputWithResidual.Data[i])
		}
		mean := sum / float64(normSize)

		// Calculate variance
		var variance float64
		for i := start; i < end; i++ {
			diff := float64(inputWithResidual.Data[i]) - mean
			variance += diff * diff
		}
		variance /= float64(normSize)

		// Normalize and apply affine transformation
		std := math.Sqrt(variance + epsilon)

		for i := 0; i < normSize; i++ {
			idx := start + i
			normalized := (float64(inputWithResidual.Data[idx]) - mean) / std

			// Apply learned scale and shift
			if gamma != nil && len(gamma.Data) > i {
				normalized *= float64(gamma.Data[i])
			}
			if beta != nil && len(beta.Data) > i {
				normalized += float64(beta.Data[i])
			}

			output.Data[idx] = T(normalized)
		}
	}

	return output
}

// =============================================================================
// Backward-compatible float32 function
// =============================================================================

// layerNormForwardCPU performs layer normalization on CPU
func layerNormForwardCPU(input []float32, residual []float32, config *LayerConfig, batchSize int) []float32 {
	inputT := NewTensorFromSlice(input, len(input))
	var residualT *Tensor[float32]
	if len(residual) > 0 {
		residualT = NewTensorFromSlice(residual, len(residual))
	}
	var gammaT, betaT *Tensor[float32]
	if len(config.Gamma) > 0 {
		gammaT = NewTensorFromSlice(config.Gamma, len(config.Gamma))
	}
	if len(config.Beta) > 0 {
		betaT = NewTensorFromSlice(config.Beta, len(config.Beta))
	}

	result := LayerNormForward(inputT, residualT, gammaT, betaT, config.NormSize, batchSize, float64(config.Epsilon))
	return result.Data
}

