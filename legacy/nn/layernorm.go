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

// LayerNormBackward computes gradients for Layer normalization.
func LayerNormBackward[T Numeric](input, residual, gradOutput, gamma, beta *Tensor[T], normSize, batchSize int, epsilon float64) (gradInput, gradGamma, gradBeta *Tensor[T]) {
	if epsilon == 0 {
		epsilon = 1e-5
	}

	// Input usually includes residual addition if present
	inputWithResidual := input.Clone()
	if residual != nil && len(residual.Data) == len(input.Data) {
		for i := range inputWithResidual.Data {
			inputWithResidual.Data[i] += residual.Data[i]
		}
	}

	gradInput = NewTensor[T](len(inputWithResidual.Data))
	gradGamma = NewTensor[T](normSize)
	gradBeta = NewTensor[T](normSize)

	for b := 0; b < batchSize; b++ {
		start := b * normSize
		end := start + normSize

		if end > len(inputWithResidual.Data) {
			break
		}

		// Recompute Mean and Std
		var sum float64
		for i := start; i < end; i++ {
			sum += float64(inputWithResidual.Data[i])
		}
		mean := sum / float64(normSize)

		var variance float64
		for i := start; i < end; i++ {
			diff := float64(inputWithResidual.Data[i]) - mean
			variance += diff * diff
		}
		variance /= float64(normSize)
		std := math.Sqrt(variance + epsilon)
		invStd := 1.0 / std

		// Compute gradients for Gamma/Beta and internal terms
		var gradSigma float64
		var gradMu float64

		for i := start; i < end; i++ {
			idx := i
			localIdx := i - start

			dL_dy := float64(gradOutput.Data[idx])

			// Beta gradient
			gradBeta.Data[localIdx] += T(dL_dy)

			// x_hat calculation
			val := float64(inputWithResidual.Data[idx])
			x_hat := (val - mean) * invStd

			// Gamma gradient
			gradGamma.Data[localIdx] += T(dL_dy * x_hat)

			// Propagate through gamma scaling
			g := 1.0
			if gamma != nil && len(gamma.Data) > localIdx {
				g = float64(gamma.Data[localIdx])
			}
			d_xhat := dL_dy * g

			// Accumulate partials for mean/std
			gradSigma += d_xhat * (val - mean)
			gradMu += d_xhat
		}

		gradSigma *= -0.5 * math.Pow(variance+epsilon, -1.5)
		gradMu = gradMu*(-invStd) + gradSigma*(-2.0/float64(normSize))*0 // Wait, gradSigma term for mu is complex

		// Correct derivation for backprop through LayerNorm:
		// dL/dx_i = (1/sigma) * (dL/dxhat_i - mean(dL/dxhat) - xhat_i * mean(dL/dxhat * xhat))
		// Re-loop to apply

		var sum_dxhat float64
		var sum_dxhat_xhat float64

		for i := start; i < end; i++ {
			idx := i
			localIdx := i - start

			dL_dy := float64(gradOutput.Data[idx])
			g := 1.0
			if gamma != nil && len(gamma.Data) > localIdx {
				g = float64(gamma.Data[localIdx])
			}
			d_xhat := dL_dy * g

			val := float64(inputWithResidual.Data[idx])
			x_hat := (val - mean) * invStd

			sum_dxhat += d_xhat
			sum_dxhat_xhat += d_xhat * x_hat
		}

		mean_dxhat := sum_dxhat / float64(normSize)
		mean_dxhat_xhat := sum_dxhat_xhat / float64(normSize)

		for i := start; i < end; i++ {
			idx := i

			val := float64(inputWithResidual.Data[idx])
			x_hat := (val - mean) * invStd

			dL_dy := float64(gradOutput.Data[idx])
			g := 1.0
			localIdx := i - start
			if gamma != nil && len(gamma.Data) > localIdx {
				g = float64(gamma.Data[localIdx])
			}
			d_xhat := dL_dy * g

			dx := invStd * (d_xhat - mean_dxhat - x_hat*mean_dxhat_xhat)
			gradInput.Data[idx] = T(dx)
		}
	}

	return gradInput, gradGamma, gradBeta
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

// layerNormBackwardCPU computes gradients for Layer normalization on CPU
func layerNormBackwardCPU(input, residual, gradOutput []float32, config *LayerConfig, batchSize int) ([]float32, []float32, []float32) {
	inputT := NewTensorFromSlice(input, len(input))
	var residualT *Tensor[float32]
	if len(residual) > 0 {
		residualT = NewTensorFromSlice(residual, len(residual))
	}
	gradOutputT := NewTensorFromSlice(gradOutput, len(gradOutput))

	var gammaT, betaT *Tensor[float32]
	if len(config.Gamma) > 0 {
		gammaT = NewTensorFromSlice(config.Gamma, len(config.Gamma))
	}
	if len(config.Beta) > 0 {
		betaT = NewTensorFromSlice(config.Beta, len(config.Beta))
	}

	gInput, gGamma, gBeta := LayerNormBackward(inputT, residualT, gradOutputT, gammaT, betaT, config.NormSize, batchSize, float64(config.Epsilon))

	// Handle nil returns if gamma/beta missing
	var gGammaData, gBetaData []float32
	if gGamma != nil {
		gGammaData = gGamma.Data
	}
	if gBeta != nil {
		gBetaData = gBeta.Data
	}

	return gInput.Data, gGammaData, gBetaData
}
