package nn

// =============================================================================
// Generic Residual Layer Implementation
// =============================================================================

// ResidualForward adds a residual connection: output = input + previous_input
// In this architecture, "LayerResidual" is strictly a distinct layer that adds
// its input (which is the output of the previous layer) to the input of the previous layer.
// This assumes the previous layer preserved its dimensions.
//
// input: current layer input (output of previous layer)
// skipInput: input to the previous layer (the "skip" connection)
func ResidualForward[T Numeric](input, skipInput *Tensor[T]) *Tensor[T] {
	if skipInput == nil || len(skipInput.Data) != len(input.Data) {
		// If sizes don't match or no skip input, just pass through (identity)
		// This can happen if sizes changed (e.g. Dense layer changed size)
		return input.Clone()
	}

	output := NewTensor[T](len(input.Data))
	for i := range input.Data {
		output.Data[i] = input.Data[i] + skipInput.Data[i]
	}

	return output
}

// ResidualBackward computes gradients for Residual layer.
// Output gradient flows to both the input (previous layer's output) and the skip connection.
// returns:
// gradInput: gradient w.r.t. input (flows to previous layer's output)
// gradSkip: gradient w.r.t. skip input (flows to previous layer's input / skipped layer)
func ResidualBackward[T Numeric](gradOutput *Tensor[T]) (gradInput, gradSkip *Tensor[T]) {
	// Gradient flows equally to both branches: d(x+y)/dx = 1, d(x+y)/dy = 1
	gradInput = gradOutput.Clone()
	gradSkip = gradOutput.Clone()
	return gradInput, gradSkip
}

// =============================================================================
// Backward-compatible float32 functions
// =============================================================================

// ResidualForwardCPU performs residual connection on CPU
func ResidualForwardCPU(input, skipInput []float32) []float32 {
	if len(skipInput) != len(input) {
		return input // Copy logic requires new slice, but here we likely return input if handled by caller
		// Actually, standard pattern returns new slice
		out := make([]float32, len(input))
		copy(out, input)
		return out
	}

	output := make([]float32, len(input))
	for i := range input {
		output[i] = input[i] + skipInput[i]
	}
	return output
}
// ResidualBackwardCPU computes gradients for Residual layer on CPU
// returns: gradInput (for current path), gradSkip (for skip path)
func ResidualBackwardCPU(gradOutput []float32) (gradInput, gradSkip []float32) {
	gradInput = make([]float32, len(gradOutput))
	gradSkip = make([]float32, len(gradOutput))
	copy(gradInput, gradOutput)
	copy(gradSkip, gradOutput)
	return gradInput, gradSkip
}
