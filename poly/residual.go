package poly

// ResidualForwardPolymorphic adds a residual connection: output = input + skip.
func ResidualForwardPolymorphic[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if skip == nil || len(skip.Data) != len(input.Data) {
		// If sizes don't match or no skip input, just pass through (identity)
		return input, input.Clone()
	}

	output := NewTensor[T](input.Shape...)
	for i := range input.Data {
		output.Data[i] = input.Data[i] + skip.Data[i]
	}

	// For residual, pre-activation can just be the skip buffer if we need to store it
	preAct = &Tensor[T]{
		Nested: []*Tensor[T]{skip},
	}

	return preAct, output
}

// ResidualBackwardPolymorphic computes gradients for Residual layer.
func ResidualBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	// Gradient flows equally to both branches: d(x+y)/dx = 1, d(x+y)/dy = 1
	gradInput = gradOutput.Clone()
	
	// gradWeights for Residual is used here to return gradSkip
	var gradSkip *Tensor[T]
	if preAct != nil && len(preAct.Nested) > 0 {
		gradSkip = gradOutput.Clone()
	}

	gradWeights = &Tensor[T]{
		Nested: []*Tensor[T]{gradSkip},
	}
	
	return gradInput, gradWeights
}
