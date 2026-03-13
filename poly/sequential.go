package poly

// SequentialForwardPolymorphic executes multiple sub-layers in sequence.
func SequentialForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if len(layer.SequentialLayers) == 0 {
		return input, input
	}

	current := input
	var lastInput *Tensor[T]
	stepIntermediates := make([]*Tensor[T], len(layer.SequentialLayers))

	for i := range layer.SequentialLayers {
		sub := &layer.SequentialLayers[i]
		
		target := sub
		if layer.UseTiling {
			target.UseTiling = true
			target.TileSize = layer.TileSize
		}
		if sub.IsRemoteLink && layer.Network != nil {
			if remote := layer.Network.GetLayer(sub.TargetZ, sub.TargetY, sub.TargetX, sub.TargetL); remote != nil {
				target = remote
				if layer.UseTiling {
					target.UseTiling = true
					target.TileSize = layer.TileSize
				}
			}
		}

		bPre, bOut := DispatchLayer(target, current, lastInput)
		
		// Step Intermediate carries: [0]: layer-proxied-preAct, [1]: layer-input, [2]: skip-input
		stepContainer := &Tensor[T]{
			Nested: []*Tensor[T]{bPre, current, lastInput},
		}
		stepIntermediates[i] = stepContainer
		lastInput = current
		current = bOut
	}

	preAct = &Tensor[T]{
		Data:   input.Data,
		Shape:  input.Shape,
		DType:  input.DType,
		Nested: stepIntermediates,
	}

	return preAct, current
}

// SequentialBackwardPolymorphic distributes gradients back through the sequence in reverse.
func SequentialBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if len(layer.SequentialLayers) == 0 {
		return gradOutput, nil
	}

	currentGrad := gradOutput
	branchGradWeights := make([]*Tensor[T], len(layer.SequentialLayers))

	// Backward is reverse order
	// Wait, gradient from skip connection flows back to the PREVIOUS layer's input.
	// In forward: y_i = f_i(x_i, x_{i-1})
	// In backward: grad_x_i = dL/dy_i * df_i/dx_i + dL/dy_{i+1} * df_{i+1}/dx_i
	
	// We need to keep track of gradients for skip connections.
	skipGradients := make([]*Tensor[T], len(layer.SequentialLayers)+1)

	for i := len(layer.SequentialLayers) - 1; i >= 0; i-- {
		sub := &layer.SequentialLayers[i]
		
		target := sub
		if layer.UseTiling {
			target.UseTiling = true
			target.TileSize = layer.TileSize
		}
		if sub.IsRemoteLink && layer.Network != nil {
			if remote := layer.Network.GetLayer(sub.TargetZ, sub.TargetY, sub.TargetX, sub.TargetL); remote != nil {
				target = remote
				if layer.UseTiling {
					target.UseTiling = true
					target.TileSize = layer.TileSize
				}
			}
		}

		// Retrieve step container: [0]=bPre, [1]=bInput, [2]=bSkip
		var bPre, bInput, bSkip *Tensor[T]
		if preAct != nil && i < len(preAct.Nested) {
			container := preAct.Nested[i]
			if container != nil && len(container.Nested) >= 3 {
				bPre = container.Nested[0]
				bInput = container.Nested[1]
				bSkip = container.Nested[2]
			}
		}

		// If bInput is missing, fallback to current layer's input (only correct for i=0)
		if bInput == nil {
			bInput = input
		}

		// Total gradient for this step's output is currentGrad + any gradient flowing back from a skip connection
		stepGradOutput := currentGrad
		if skipGradients[i+1] != nil {
			stepGradOutput = stepGradOutput.Clone()
			stepGradOutput.Add(skipGradients[i+1])
		}

		gIn, gW := DispatchLayerBackward(target, stepGradOutput, bInput, bSkip, bPre)
		branchGradWeights[i] = gW
		currentGrad = gIn
		
		// If there's a skip gradient from this layer, it flows to step i-1
		if gW != nil && len(gW.Nested) > 0 && gW.Nested[0] != nil {
			skipGradients[i] = gW.Nested[0]
		}
	}

	return currentGrad, &Tensor[T]{Nested: branchGradWeights}
}
