package poly

// ResidualForwardPolymorphic adds a residual connection: output = input + skip.
func ResidualForwardPolymorphic[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if skip == nil || len(skip.Data) != len(input.Data) {
		// If sizes don't match or no skip input, just pass through (identity)
		return input, input.Clone()
	}

	if layer.UseTiling && layer.TileSize > 0 {
		return ResidualForwardTiled(layer, input, skip)
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
	if layer.UseTiling && layer.TileSize > 0 {
		return ResidualBackwardTiled(layer, gradOutput, input, preAct)
	}
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

// ResidualForwardTiled performs a tiled forward pass for Residual.
func ResidualForwardTiled[T Numeric](layer *VolumetricLayer, input, skip *Tensor[T]) (preAct, postAct *Tensor[T]) {
	output := NewTensor[T](input.Shape...)
	tileSize := layer.TileSize
	if tileSize <= 0 { tileSize = 1024 }
	
	for t := 0; t < len(input.Data); t += tileSize {
		end := t + tileSize
		if end > len(input.Data) { end = len(input.Data) }
		for i := t; i < end; i++ {
			output.Data[i] = input.Data[i] + skip.Data[i]
		}
	}

	preAct = &Tensor[T]{
		Nested: []*Tensor[T]{skip},
	}
	return preAct, output
}

// ResidualBackwardTiled performs a tiled backward pass for Residual.
func ResidualBackwardTiled[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
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
