package nn

import (
	"math"
)

// =============================================================================
// Generic RMSNorm Implementation
// =============================================================================

// RMSNormForward performs RMS normalization for any numeric type.
// RMSNorm is simpler than LayerNorm - only uses gamma (no beta)
// Formula: output = input * gamma / sqrt(mean(input^2) + epsilon)
func RMSNormForward[T Numeric](input, residual, gamma *Tensor[T], normSize int, epsilon float64) *Tensor[T] {
	if epsilon == 0 {
		epsilon = 1e-6
	}

	// Add residual if provided
	inputWithResidual := input.Clone()
	if residual != nil && len(residual.Data) == len(input.Data) {
		for i := range inputWithResidual.Data {
			inputWithResidual.Data[i] += residual.Data[i]
		}
	}

	output := NewTensor[T](len(inputWithResidual.Data))
	numSamples := len(inputWithResidual.Data) / normSize

	for b := 0; b < numSamples; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual.Data) {
			break
		}

		// Calculate RMS
		var sumSquares float64
		for i := start; i < end; i++ {
			val := float64(inputWithResidual.Data[i])
			sumSquares += val * val
		}
		rms := math.Sqrt(sumSquares/float64(normSize) + epsilon)

		// Normalize and apply scale
		for i := start; i < end; i++ {
			normalized := float64(inputWithResidual.Data[i]) / rms

			gammaIdx := i - start
			if gamma != nil && gammaIdx < len(gamma.Data) {
				output.Data[i] = T(normalized * float64(gamma.Data[gammaIdx]))
			} else {
				output.Data[i] = T(normalized)
			}
		}
	}

	return output
}

// RMSNormBackward computes gradients for RMS normalization.
func RMSNormBackward[T Numeric](input, residual, gradOutput, gamma *Tensor[T], normSize, batchSize int, epsilon float64) *Tensor[T] {
	if epsilon == 0 {
		epsilon = 1e-6
	}

	inputWithResidual := input.Clone()
	if residual != nil && len(residual.Data) == len(input.Data) {
		for i := range inputWithResidual.Data {
			inputWithResidual.Data[i] += residual.Data[i]
		}
	}

	gradInput := NewTensor[T](len(inputWithResidual.Data))

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual.Data) {
			break
		}

		// Recompute RMS
		var sumSquares float64
		for i := start; i < end; i++ {
			val := float64(inputWithResidual.Data[i])
			sumSquares += val * val
		}
		meanSquare := sumSquares / float64(normSize)
		rms := math.Sqrt(meanSquare + epsilon)

		// Compute gradient
		var gradScale float64
		for i := start; i < end; i++ {
			gammaIdx := i - start
			if gamma != nil && gammaIdx < len(gamma.Data) {
				gradScale += float64(gradOutput.Data[i]) * float64(gamma.Data[gammaIdx]) * float64(inputWithResidual.Data[i])
			}
		}
		gradScale /= float64(normSize) * rms * rms

		// Apply chain rule
		for i := start; i < end; i++ {
			gammaIdx := i - start
			g := 1.0
			if gamma != nil && gammaIdx < len(gamma.Data) {
				g = float64(gamma.Data[gammaIdx])
			}

			gradInput.Data[i] = T(float64(gradOutput.Data[i])*g/rms - gradScale*float64(inputWithResidual.Data[i]))
		}
	}

	return gradInput
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// RmsNormForwardCPU performs RMS normalization on CPU (exported version)
func RmsNormForwardCPU(input []float32, residual []float32, config *LayerConfig, batchSize int) []float32 {
	return rmsNormForwardCPU(input, residual, config, batchSize)
}

// rmsNormForwardCPU performs RMS normalization on CPU
func rmsNormForwardCPU(input []float32, residual []float32, config *LayerConfig, batchSize int) []float32 {
	inputT := NewTensorFromSlice(input, len(input))
	var residualT *Tensor[float32]
	if len(residual) > 0 {
		residualT = NewTensorFromSlice(residual, len(residual))
	}
	var gammaT *Tensor[float32]
	if len(config.Gamma) > 0 {
		gammaT = NewTensorFromSlice(config.Gamma, len(config.Gamma))
	}

	result := RMSNormForward(inputT, residualT, gammaT, config.NormSize, float64(config.Epsilon))
	return result.Data
}

// rmsNormBackwardCPU computes gradients for RMS normalization
func rmsNormBackwardCPU(input []float32, residual []float32, gradOutput []float32, config *LayerConfig, batchSize int) []float32 {
	inputT := NewTensorFromSlice(input, len(input))
	var residualT *Tensor[float32]
	if len(residual) > 0 {
		residualT = NewTensorFromSlice(residual, len(residual))
	}
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))
	var gammaT *Tensor[float32]
	if len(config.Gamma) > 0 {
		gammaT = NewTensorFromSlice(config.Gamma, len(config.Gamma))
	}

	result := RMSNormBackward(inputT, residualT, gradOutputT, gammaT, config.NormSize, batchSize, float64(config.Epsilon))
	return result.Data
}

