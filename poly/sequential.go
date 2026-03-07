package poly

// SequentialForwardPolymorphic executes multiple sub-layers in sequence.
func SequentialForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if len(layer.SequentialLayers) == 0 {
		return input, input
	}

	current := input
	stepIntermediates := make([]*Tensor[T], len(layer.SequentialLayers))

	for i := range layer.SequentialLayers {
		sub := &layer.SequentialLayers[i]
		
		target := sub
		if sub.IsRemoteLink && layer.Network != nil {
			if remote := layer.Network.GetLayer(sub.TargetZ, sub.TargetY, sub.TargetX, sub.TargetL); remote != nil {
				target = remote
			}
		}

		bPre, bOut := DispatchLayer(target, current)
		
		// Step Intermediate carries: [0]: layer-proxied-preAct, [1]: layer-input
		stepContainer := &Tensor[T]{
			Nested: []*Tensor[T]{bPre, current},
		}
		stepIntermediates[i] = stepContainer
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
	for i := len(layer.SequentialLayers) - 1; i >= 0; i-- {
		sub := &layer.SequentialLayers[i]
		
		target := sub
		if sub.IsRemoteLink && layer.Network != nil {
			if remote := layer.Network.GetLayer(sub.TargetZ, sub.TargetY, sub.TargetX, sub.TargetL); remote != nil {
				target = remote
			}
		}

		// Retrieve step container: [0]=bPre, [1]=bInput
		var bPre, bInput *Tensor[T]
		if preAct != nil && i < len(preAct.Nested) {
			container := preAct.Nested[i]
			if container != nil && len(container.Nested) >= 2 {
				bPre = container.Nested[0]
				bInput = container.Nested[1]
			}
		}

		// If bInput is missing, fallback to current layer's input (only correct for i=0)
		if bInput == nil {
			bInput = input
		}

		gIn, gW := DispatchLayerBackward(target, currentGrad, bInput, bPre)
		branchGradWeights[i] = gW
		currentGrad = gIn
	}

	return currentGrad, &Tensor[T]{Nested: branchGradWeights}
}
