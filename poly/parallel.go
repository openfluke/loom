package poly

// ParallelForwardPolymorphic executes multiple sub-layers in parallel and combines outputs.
func ParallelForwardPolymorphic[T Numeric](layer *VolumetricLayer, input *Tensor[T]) (preAct, postAct *Tensor[T]) {
	if len(layer.ParallelBranches) == 0 {
		return input, input // Passthrough
	}
	batchSize := 1
	if input != nil && len(input.Shape) > 0 && input.Shape[0] > 0 {
		batchSize = input.Shape[0]
	}

	branchOutputs := make([]*Tensor[T], len(layer.ParallelBranches))
	branchPreActs := make([]*Tensor[T], len(layer.ParallelBranches))
	totalOutputSize := 0

	for i := range layer.ParallelBranches {
		branch := &layer.ParallelBranches[i]

		target := branch
		if branch.IsRemoteLink && layer.Network != nil {
			if remote := layer.Network.GetLayer(branch.TargetZ, branch.TargetY, branch.TargetX, branch.TargetL); remote != nil {
				target = remote
			}
		}

		// Recursive dispatch
		bPre, bOut := DispatchLayer(target, input, nil)
		branchOutputs[i] = bOut
		branchPreActs[i] = bPre

		outPerSample := len(bOut.Data)
		if batchSize > 0 {
			outPerSample = len(bOut.Data) / batchSize
		}
		if layer.CombineMode == "concat" || layer.CombineMode == "" || layer.CombineMode == "grid_scatter" {
			totalOutputSize += outPerSample
		} else {
			if i == 0 {
				totalOutputSize = outPerSample
			}
		}
	}

	// Combine logic
	postAct = NewTensor[T](batchSize, totalOutputSize)
	switch layer.CombineMode {
	case "add":
		for _, out := range branchOutputs {
			for j := range out.Data {
				postAct.Data[j] += out.Data[j]
			}
		}
	case "avg":
		for _, out := range branchOutputs {
			for j := range out.Data {
				postAct.Data[j] += out.Data[j]
			}
		}
		invN := 1.0 / float64(len(branchOutputs))
		for j := range postAct.Data {
			postAct.Data[j] = T(float64(postAct.Data[j]) * invN)
		}
	case "concat", "grid_scatter", "":
		for b := 0; b < batchSize; b++ {
			offset := b * totalOutputSize
			for _, out := range branchOutputs {
				per := len(out.Data)
				if batchSize > 0 {
					per = len(out.Data) / batchSize
				}
				srcStart := b * per
				copy(postAct.Data[offset:offset+per], out.Data[srcStart:srcStart+per])
				offset += per
			}
		}
	case "filter":
		// MoE style gating
		if layer.FilterGateConfig != nil {
			_, gateOut := DispatchLayer(layer.FilterGateConfig, input, nil)
			// Apply Softmax to gate coefficients
			gateLogits := make([]float32, len(layer.ParallelBranches))
			for i := range gateLogits {
				gateLogits[i] = float32(gateOut.Data[i])
			}
			gateWeights := Softmax(gateLogits)

			for i, out := range branchOutputs {
				w := float64(gateWeights[i])
				for j := range out.Data {
					postAct.Data[j] += T(float64(out.Data[j]) * w)
				}
			}
		}
	}

	// Create a proxy preAct that carries the activation tree
	preAct = &Tensor[T]{
		Data:   input.Data,
		Shape:  input.Shape,
		DType:  input.DType,
		Nested: branchPreActs,
	}

	return preAct, postAct
}

// ParallelBackwardPolymorphic distributes gradients back to branches.
func ParallelBackwardPolymorphic[T Numeric](layer *VolumetricLayer, gradOutput, input, preAct *Tensor[T]) (gradInput, gradWeights *Tensor[T]) {
	if len(layer.ParallelBranches) == 0 {
		return gradOutput, nil
	}
	batchSize := 1
	if input != nil && len(input.Shape) > 0 && input.Shape[0] > 0 {
		batchSize = input.Shape[0]
	}

	gradInput = NewTensor[T](input.Shape...)
	branchGradWeights := make([]*Tensor[T], len(layer.ParallelBranches))

	switch layer.CombineMode {
	case "add", "avg", "filter":
		invN := 1.0
		if layer.CombineMode == "avg" {
			invN = 1.0 / float64(len(layer.ParallelBranches))
		}

		for i := range layer.ParallelBranches {
			branch := &layer.ParallelBranches[i]

			target := branch
			if branch.IsRemoteLink && layer.Network != nil {
				if remote := layer.Network.GetLayer(branch.TargetZ, branch.TargetY, branch.TargetX, branch.TargetL); remote != nil {
					target = remote
				}
			}

			scaledGrad := gradOutput
			if invN != 1.0 {
				scaledGrad = NewTensor[T](len(gradOutput.Data))
				for j := range gradOutput.Data {
					scaledGrad.Data[j] = T(float64(gradOutput.Data[j]) * invN)
				}
			}

			// Recursive backward with nested preAct
			var bPre *Tensor[T]
			if preAct != nil && i < len(preAct.Nested) {
				bPre = preAct.Nested[i]
			}

			gIn, gW := DispatchLayerBackward(target, scaledGrad, input, nil, bPre)
			branchGradWeights[i] = gW

			if gIn != nil {
				limit := len(gradInput.Data)
				if len(gIn.Data) < limit {
					limit = len(gIn.Data)
				}
				for j := 0; j < limit; j++ {
					gradInput.Data[j] += gIn.Data[j]
				}
			}
		}

	case "concat", "grid_scatter", "":
		offset := 0
		for i := range layer.ParallelBranches {
			branch := &layer.ParallelBranches[i]

			target := branch
			if branch.IsRemoteLink && layer.Network != nil {
				if remote := layer.Network.GetLayer(branch.TargetZ, branch.TargetY, branch.TargetX, branch.TargetL); remote != nil {
					target = remote
				}
			}

			// Determine size for slicing
			size := 0
			_, out := DispatchLayer(target, input, nil)
			size = len(out.Data)

			perSample := size
			if batchSize > 0 {
				perSample = size / batchSize
			}
			branchGrad := NewTensorFromSlice(gradOutput.Data[offset:offset+size], batchSize, perSample)

			var bPre *Tensor[T]
			if preAct != nil && i < len(preAct.Nested) {
				bPre = preAct.Nested[i]
			}

			gIn, gW := DispatchLayerBackward(target, branchGrad, input, nil, bPre)
			branchGradWeights[i] = gW

			if gIn != nil {
				limit := len(gradInput.Data)
				if len(gIn.Data) < limit {
					limit = len(gIn.Data)
				}
				for j := 0; j < limit; j++ {
					gradInput.Data[j] += gIn.Data[j]
				}
			}
			offset += size
		}
	}

	gradWeights = &Tensor[T]{
		Nested: branchGradWeights,
	}

	return gradInput, gradWeights
}
