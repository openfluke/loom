package nn

import (
	"math"
)

// layerNormForwardCPU performs layer normalization on CPU
// input shape: [batchSize][normSize] (flattened)
// residual: optional residual connection to add before normalization
// Returns: normalized output
func layerNormForwardCPU(input []float32, residual []float32, config *LayerConfig, batchSize int) []float32 {
	normSize := config.NormSize
	gamma := config.Gamma
	beta := config.Beta
	epsilon := config.Epsilon

	if epsilon == 0 {
		epsilon = 1e-5 // Default value
	}

	// Add residual if provided
	inputWithResidual := make([]float32, len(input))
	if len(residual) == len(input) {
		for i := range input {
			inputWithResidual[i] = input[i] + residual[i]
		}
	} else {
		copy(inputWithResidual, input)
	}

	output := make([]float32, len(inputWithResidual))

	// Normalize each sample in the batch
	for b := 0; b < batchSize; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual) {
			break
		}

		// Calculate mean
		var sum float32
		for i := start; i < end; i++ {
			sum += inputWithResidual[i]
		}
		mean := sum / float32(normSize)

		// Calculate variance
		var variance float32
		for i := start; i < end; i++ {
			diff := inputWithResidual[i] - mean
			variance += diff * diff
		}
		variance /= float32(normSize)

		// Normalize and apply affine transformation
		std := float32(math.Sqrt(float64(variance + epsilon)))

		for i := 0; i < normSize; i++ {
			idx := start + i
			normalized := (inputWithResidual[idx] - mean) / std

			// Apply learned scale and shift
			if len(gamma) > i {
				normalized *= gamma[i]
			}
			if len(beta) > i {
				normalized += beta[i]
			}

			output[idx] = normalized
		}
	}

	// Notify observer if present
	if config.Observer != nil {
		notifyObserver(config, "forward", -1, input, output, 0)
	}

	return output
}
