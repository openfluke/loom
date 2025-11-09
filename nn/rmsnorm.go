package nn

import (
	"math"
)

// RmsNormForwardCPU performs RMS normalization on CPU (exported version)
// RMSNorm is simpler than LayerNorm - only uses gamma (no beta)
// Formula: output = input * gamma / sqrt(mean(input^2) + epsilon)
// Used in: Llama, Mistral, and other modern LLMs
//
// input shape: [batchSize][normSize] (flattened)
// residual: optional residual connection to add before normalization
// Returns: normalized output
func RmsNormForwardCPU(input []float32, residual []float32, config *LayerConfig, batchSize int) []float32 {
	return rmsNormForwardCPU(input, residual, config, batchSize)
}

// rmsNormForwardCPU performs RMS normalization on CPU
// RMSNorm is simpler than LayerNorm - only uses gamma (no beta)
// Formula: output = input * gamma / sqrt(mean(input^2) + epsilon)
// Used in: Llama, Mistral, and other modern LLMs
//
// input shape: [batchSize][normSize] (flattened)
// residual: optional residual connection to add before normalization
// Returns: normalized output
func rmsNormForwardCPU(input []float32, residual []float32, config *LayerConfig, batchSize int) []float32 {
	normSize := config.NormSize
	gamma := config.Gamma
	epsilon := config.Epsilon

	if epsilon == 0 {
		epsilon = 1e-6 // Default for RMSNorm (typically 1e-6 vs LayerNorm's 1e-5)
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

	// Calculate actual number of tokens/samples to normalize
	// For transformers: each token is normalized independently
	numSamples := len(inputWithResidual) / normSize

	// Normalize each token independently
	for b := 0; b < numSamples; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual) {
			break
		}

		// Calculate RMS (root mean square)
		var sumSquares float32
		for i := start; i < end; i++ {
			sumSquares += inputWithResidual[i] * inputWithResidual[i]
		}
		rms := float32(math.Sqrt(float64(sumSquares/float32(normSize) + epsilon)))

		// Normalize and apply scale (gamma)
		for i := start; i < end; i++ {
			normalized := inputWithResidual[i] / rms

			// Apply learned scale parameter
			gammaIdx := i - start
			if gammaIdx < len(gamma) {
				output[i] = normalized * gamma[gammaIdx]
			} else {
				output[i] = normalized
			}
		}
	}

	return output
}

// rmsNormBackwardCPU computes gradients for RMS normalization
func rmsNormBackwardCPU(input []float32, residual []float32, gradOutput []float32, config *LayerConfig, batchSize int) []float32 {
	normSize := config.NormSize
	gamma := config.Gamma
	epsilon := config.Epsilon

	if epsilon == 0 {
		epsilon = 1e-6
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

	gradInput := make([]float32, len(inputWithResidual))

	// Compute gradients for each sample
	for b := 0; b < batchSize; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual) {
			break
		}

		// Recompute RMS
		var sumSquares float32
		for i := start; i < end; i++ {
			sumSquares += inputWithResidual[i] * inputWithResidual[i]
		}
		meanSquare := sumSquares / float32(normSize)
		rms := float32(math.Sqrt(float64(meanSquare + epsilon)))

		// Compute gradient
		var gradScale float32
		for i := start; i < end; i++ {
			gammaIdx := i - start
			if gammaIdx < len(gamma) {
				gradScale += gradOutput[i] * gamma[gammaIdx] * inputWithResidual[i]
			}
		}
		gradScale /= float32(normSize) * rms * rms

		// Apply chain rule
		for i := start; i < end; i++ {
			gammaIdx := i - start
			var g float32 = 1.0
			if gammaIdx < len(gamma) {
				g = gamma[gammaIdx]
			}

			gradInput[i] = (gradOutput[i]*g/rms - gradScale*inputWithResidual[i])
		}
	}

	return gradInput
}
