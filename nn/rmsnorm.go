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
func RMSNormBackward[T Numeric](input, residual, gradOutput, gamma *Tensor[T], normSize, batchSize int, epsilon float64) (gradInput, gradGamma *Tensor[T]) {
	if epsilon == 0 {
		epsilon = 1e-6
	}

	inputWithResidual := input.Clone()
	if residual != nil && len(residual.Data) == len(input.Data) {
		for i := range inputWithResidual.Data {
			inputWithResidual.Data[i] += residual.Data[i]
		}
	}

	gradInput = NewTensor[T](len(inputWithResidual.Data))
	gradGamma = NewTensor[T](normSize)

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
		invRMS := 1.0 / rms
		invRMS3 := 1.0 / (rms * rms * rms) // for gradient math

		// Compute gradients
		var sum_dxhat_x float64
		
		for i := start; i < end; i++ {
			idx := i
			localIdx := i - start
			
			dL_dy := float64(gradOutput.Data[idx])
			val := float64(inputWithResidual.Data[idx])
			
			// Accumulate Gamma grad
			// y = x_hat * gamma
			x_hat := val * invRMS
			gradGamma.Data[localIdx] += T(dL_dy * x_hat)
			
			// Part of input gradient calc
			g := 1.0
			if gamma != nil && localIdx < len(gamma.Data) {
				g = float64(gamma.Data[localIdx])
			}
			
			sum_dxhat_x += dL_dy * g * val
		}

		// Apply final formula for dL/dx
		// dL/dx_i = (gamma_i * dL/dy_i / rms) - (x_i / rms^3) * (1/N) * sum(dL/dy_k * gamma_k * x_k)
		
		term2 := sum_dxhat_x * invRMS3 / float64(normSize)
		
		for i := start; i < end; i++ {
			localIdx := i - start
			
			dL_dy := float64(gradOutput.Data[i])
			val := float64(inputWithResidual.Data[i])
			
			g := 1.0
			if gamma != nil && localIdx < len(gamma.Data) {
				g = float64(gamma.Data[localIdx])
			}
			
			dx := (dL_dy * g * invRMS) - (val * term2)
			gradInput.Data[i] = T(dx)
		}
	}

	return gradInput, gradGamma
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

	gradInputT, _ := RMSNormBackward(inputT, residualT, gradOutputT, gammaT, config.NormSize, batchSize, float64(config.Epsilon))
	return gradInputT.Data
}

