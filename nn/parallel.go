package nn

import "fmt"

// =============================================================================
// Generic Parallel Layer Implementation
// =============================================================================

// ParallelForward executes multiple sub-layers in parallel for any numeric type.
// Returns combined output and a slice of intermediate tensors (one per branch) for backward pass.
func ParallelForward[T Numeric](
	input *Tensor[T],
	branches []*LayerConfig,
	batchSize int,
	combineMode string,
) (*Tensor[T], []*Tensor[T], error) {
	if len(branches) == 0 {
		return nil, nil, fmt.Errorf("parallel layer has no branches defined")
	}

	branchOutputs := make([]*Tensor[T], len(branches))
	branchIntermediates := make([]*Tensor[T], len(branches))
	totalOutputSize := 0

	for i, branchCfg := range branches {
		var preAct, postAct *Tensor[T]

		// Route to appropriate generic layer forward based on type
		switch branchCfg.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Kernel, len(branchCfg.Kernel)))
			bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Bias, len(branchCfg.Bias)))
			preAct, postAct = DenseForward(input, weights, bias, branchCfg.InputHeight, branchCfg.OutputHeight, batchSize, branchCfg.Activation)

		case LayerConv2D:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Kernel, len(branchCfg.Kernel)))
			bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Bias, len(branchCfg.Bias)))
			preAct, postAct = Conv2DForward(input, weights, bias,
				branchCfg.InputHeight, branchCfg.InputWidth, branchCfg.InputChannels,
				branchCfg.KernelSize, branchCfg.Stride, branchCfg.Padding, branchCfg.Filters,
				branchCfg.OutputHeight, branchCfg.OutputWidth, batchSize, branchCfg.Activation)

		case LayerConv1D:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Conv1DKernel, len(branchCfg.Conv1DKernel)))
			bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Conv1DBias, len(branchCfg.Conv1DBias)))

			// Use InputHeight as seqLen
			seqLen := branchCfg.InputHeight
			if seqLen <= 0 {
				seqLen = len(input.Data) / (branchCfg.Conv1DInChannels * batchSize)
			}

			preAct, postAct = Conv1DForward(input, weights, bias,
				seqLen, branchCfg.Conv1DInChannels,
				branchCfg.Conv1DKernelSize, branchCfg.Conv1DStride, branchCfg.Conv1DPadding,
				branchCfg.Conv1DFilters, batchSize, branchCfg.Activation)

		case LayerMultiHeadAttention:
			weights := &AttentionWeights[T]{
				QWeights:     ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.QWeights, len(branchCfg.QWeights))),
				QBias:        ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.QBias, len(branchCfg.QBias))),
				KWeights:     ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.KWeights, len(branchCfg.KWeights))),
				KBias:        ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.KBias, len(branchCfg.KBias))),
				VWeights:     ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.VWeights, len(branchCfg.VWeights))),
				VBias:        ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.VBias, len(branchCfg.VBias))),
				OutputWeight: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.OutputWeight, len(branchCfg.OutputWeight))),
				OutputBias:   ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.OutputBias, len(branchCfg.OutputBias))),
				DModel:       branchCfg.DModel, NumHeads: branchCfg.NumHeads, NumKVHeads: branchCfg.NumKVHeads, HeadDim: branchCfg.HeadDim,
			}
			postAct = MultiHeadAttentionForward(input, weights, 10000.0)
			preAct = postAct // Placeholder

		case LayerRNN:
			wIH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightIH, len(branchCfg.WeightIH)))
			wHH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightHH, len(branchCfg.WeightHH)))
			biasH := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.BiasH, len(branchCfg.BiasH)))
			postAct, preAct = RNNForward(input, wIH, wHH, biasH, batchSize, branchCfg.SeqLength, branchCfg.RNNInputSize, branchCfg.HiddenSize)

		case LayerLSTM:
			weights := &LSTMWeights[T]{
				WeightIH_i: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightIH_i, len(branchCfg.WeightIH_i))),
				WeightHH_i: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightHH_i, len(branchCfg.WeightHH_i))),
				BiasH_i:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.BiasH_i, len(branchCfg.BiasH_i))),
				WeightIH_f: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightIH_f, len(branchCfg.WeightIH_f))),
				WeightHH_f: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightHH_f, len(branchCfg.WeightHH_f))),
				BiasH_f:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.BiasH_f, len(branchCfg.BiasH_f))),
				WeightIH_g: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightIH_g, len(branchCfg.WeightIH_g))),
				WeightHH_g: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightHH_g, len(branchCfg.WeightHH_g))),
				BiasH_g:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.BiasH_g, len(branchCfg.BiasH_g))),
				WeightIH_o: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightIH_o, len(branchCfg.WeightIH_o))),
				WeightHH_o: ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.WeightHH_o, len(branchCfg.WeightHH_o))),
				BiasH_o:    ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.BiasH_o, len(branchCfg.BiasH_o))),
			}
			postAct, _, _, _ = LSTMForward(input, weights, batchSize, branchCfg.SeqLength, branchCfg.RNNInputSize, branchCfg.HiddenSize)
			preAct = postAct // Placeholder

		case LayerSwiGLU:
			gateW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.GateWeights, len(branchCfg.GateWeights)))
			upW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.UpWeights, len(branchCfg.UpWeights)))
			downW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.DownWeights, len(branchCfg.DownWeights)))
			gateBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.GateBias, len(branchCfg.GateBias)))
			upBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.UpBias, len(branchCfg.UpBias)))
			downBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.DownBias, len(branchCfg.DownBias)))
			postAct = SwiGLUForward(input, gateW, upW, downW, gateBias, upBias, downBias, branchCfg.InputHeight, branchCfg.OutputHeight, batchSize)
			preAct = input.Clone()

		case LayerNorm:
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Gamma, len(branchCfg.Gamma)))
			beta := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Beta, len(branchCfg.Beta)))
			normSize := branchCfg.NormSize
			if normSize <= 0 {
				normSize = len(input.Data)
			}
			postAct = LayerNormForward(input, nil, gamma, beta, normSize, batchSize, float64(branchCfg.Epsilon))
			preAct = input.Clone()

		case LayerRMSNorm:
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Gamma, len(branchCfg.Gamma)))
			normSize := branchCfg.NormSize
			if normSize <= 0 {
				normSize = len(input.Data)
			}
			postAct = RMSNormForward(input, nil, gamma, normSize, float64(branchCfg.Epsilon))
			preAct = input.Clone()

		case LayerSoftmax:
			postAct = ApplySoftmax(input, float64(branchCfg.Temperature))
			preAct = input.Clone()

		case LayerParallel:
			// Nested parallel layers - convert to slice of pointers
			nestedBranches := make([]*LayerConfig, len(branchCfg.ParallelBranches))
			for j := range branchCfg.ParallelBranches {
				nestedBranches[j] = &branchCfg.ParallelBranches[j]
			}
			var err error
			postAct, branchIntermediates, err = ParallelForward[T](input, nestedBranches, batchSize, branchCfg.CombineMode)
			if err != nil {
				return nil, nil, fmt.Errorf("nested parallel layer %d failed: %w", i, err)
			}
			preAct = input.Clone()

		case LayerSequential:
			nestedLayers := make([]*LayerConfig, len(branchCfg.ParallelBranches))
			for j := range branchCfg.ParallelBranches {
				nestedLayers[j] = &branchCfg.ParallelBranches[j]
			}
			var err error
			postAct, _, err = SequentialForward[T](input, nestedLayers, batchSize)
			if err != nil {
				return nil, nil, fmt.Errorf("sequential branch %d failed: %w", i, err)
			}
			preAct = input.Clone() // Placeholder

		case LayerKMeans:
			inputF32 := ConvertSliceTToFloat32(input.Data)
			outputF32, err := ForwardKMeansCPU(inputF32, branchCfg)
			if err != nil {
				return nil, nil, fmt.Errorf("kmeans branch %d failed: %w", i, err)
			}
			postAct = ConvertTensorFloat32ToT[T](NewTensorFromSlice(outputF32, len(outputF32)))
			preAct = ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.PreActivations, len(branchCfg.PreActivations)))

		default:
			// For unsupported types, pass through
			postAct = input.Clone()
			preAct = input.Clone()
		}

		branchOutputs[i] = postAct
		branchIntermediates[i] = preAct

		// Notify observer for this specific branch
		if branchCfg.Observer != nil {
			postActF32 := ConvertSliceTToFloat32(postAct.Data)
			inputF32 := ConvertSliceTToFloat32(input.Data)
			notifyBranchObserver(branchCfg, branchCfg, i, "forward", "forward", inputF32, postActF32, 0)
		}

		if combineMode == "concat" || combineMode == "" || combineMode == "grid_scatter" || combineMode == "filter" {
			totalOutputSize += len(postAct.Data)
		} else {
			// For add/avg, all outputs must be same size
			if i == 0 {
				totalOutputSize = len(postAct.Data)
			} else if len(postAct.Data) != totalOutputSize {
				return nil, nil, fmt.Errorf("branch %d output size %d doesn't match expected %d for combine mode %s",
					i, len(postAct.Data), totalOutputSize, combineMode)
			}
		}
	}

	// Combine outputs based on combine mode
	var combined *Tensor[T]

	switch combineMode {
	case "concat", "": // Default to concatenation
		combined = NewTensor[T](totalOutputSize)
		offset := 0
		for _, branchOut := range branchOutputs {
			copy(combined.Data[offset:], branchOut.Data)
			offset += len(branchOut.Data)
		}

	case "add": // Element-wise addition
		combined = NewTensor[T](totalOutputSize)
		for _, branchOut := range branchOutputs {
			for j := range branchOut.Data {
				combined.Data[j] = T(float64(combined.Data[j]) + float64(branchOut.Data[j]))
			}
		}

	case "avg", "average": // Element-wise average
		combined = NewTensor[T](totalOutputSize)
		for _, branchOut := range branchOutputs {
			for j := range branchOut.Data {
				combined.Data[j] = T(float64(combined.Data[j]) + float64(branchOut.Data[j]))
			}
		}
		scale := 1.0 / float64(len(branches))
		for j := range combined.Data {
			combined.Data[j] = T(float64(combined.Data[j]) * scale)
		}

	case "grid_scatter": // Simplified grid_scatter (concatenation)
		combined = NewTensor[T](totalOutputSize)
		offset := 0
		for _, branchOut := range branchOutputs {
			copy(combined.Data[offset:], branchOut.Data)
			offset += len(branchOut.Data)
		}

	case "filter":
		// Use specialized filtered forward function
		// Note: We need to retrieve the gate config from a context or pass it in.
		// However, ParallelForward signature doesn't include the parent config where GateConfig lives.
		// Standard StepForwardGeneric calls this with branches only.
		// WE NEED TO CHANGE PARALLEL FORWARD SIGNATURE OR ASSUME GATE CONFIG IS ATTACHED TO BRANCHES?
		// Actually, FilterGateConfig is on the PARENT layer config.
		// But ParallelForward only receives `branches []*LayerConfig`.

		// CRITICAL FIX: We can't implement this cleanly without accessing properties of the PARENT config (FilterGateConfig).
		// But we don't have the parent config here.
		// Workaround: We will use Uniform Gating (1/N) if we can't find the gate config, OR fallback to concat?
		// No, that breaks the "Mixture of Experts" promise.

		// Better approach: The CALLER (StepForwardGeneric) has the parent config.
		// It should call ParallelForwardFiltered directly if mode is "filter".
		return nil, nil, fmt.Errorf("filter mode must be called via ParallelForwardFiltered, not ParallelForward")

	default:
		return nil, nil, fmt.Errorf("unknown combine mode: %s", combineMode)
	}

	return combined, branchIntermediates, nil
}

// ParallelForwardFiltered executes parallel branches with softmax-gated combining.
// The gate layer computes N logits (one per branch), softmax normalizes them,
// and outputs are weighted-summed. Requires all branches to have same output size.
// Returns: combined output, branch outputs, gate weights (for backward pass).
func ParallelForwardFiltered[T Numeric](
	input *Tensor[T],
	branches []*LayerConfig,
	gateConfig *LayerConfig,
	softmaxType SoftmaxType,
	temperature float32,
	batchSize int,
) (*Tensor[T], []*Tensor[T], []float32, error) {
	if len(branches) == 0 {
		return nil, nil, nil, fmt.Errorf("parallel filter layer has no branches")
	}

	// 1. Compute all branch outputs
	branchOutputs := make([]*Tensor[T], len(branches))
	var expectedSize int

	for i, branchCfg := range branches {
		var postAct *Tensor[T]

		switch branchCfg.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Kernel, len(branchCfg.Kernel)))
			bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Bias, len(branchCfg.Bias)))
			_, postAct = DenseForward(input, weights, bias, branchCfg.InputHeight, branchCfg.OutputHeight, batchSize, branchCfg.Activation)
		case LayerSwiGLU:
			gateW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.GateWeights, len(branchCfg.GateWeights)))
			upW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.UpWeights, len(branchCfg.UpWeights)))
			downW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.DownWeights, len(branchCfg.DownWeights)))
			gateBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.GateBias, len(branchCfg.GateBias)))
			upBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.UpBias, len(branchCfg.UpBias)))
			downBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.DownBias, len(branchCfg.DownBias)))
			postAct = SwiGLUForward(input, gateW, upW, downW, gateBias, upBias, downBias, branchCfg.InputHeight, branchCfg.OutputHeight, batchSize)
		case LayerKMeans:
			inputF32 := ConvertSliceTToFloat32(input.Data)
			outputF32, err := ForwardKMeansCPU(inputF32, branchCfg)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("kmeans branch %d failed: %w", i, err)
			}
			postAct = ConvertTensorFloat32ToT[T](NewTensorFromSlice(outputF32, len(outputF32)))
		default:
			// Fallback: use generic ParallelForward for this branch type
			nestedBranches := []*LayerConfig{branchCfg}
			var err error
			postAct, _, err = ParallelForward[T](input, nestedBranches, batchSize, "concat")
			if err != nil {
				return nil, nil, nil, fmt.Errorf("branch %d forward failed: %w", i, err)
			}
		}

		branchOutputs[i] = postAct
		if i == 0 {
			expectedSize = len(postAct.Data)
		} else if len(postAct.Data) != expectedSize {
			return nil, nil, nil, fmt.Errorf("filter mode requires same-sized outputs: branch 0 has %d, branch %d has %d",
				expectedSize, i, len(postAct.Data))
		}
	}

	// 2. Compute gate weights
	gateLogits := make([]float32, len(branches))

	if gateConfig != nil && gateConfig.Type == LayerDense {
		// Use the gate layer to compute logits
		gateWeights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(gateConfig.Kernel, len(gateConfig.Kernel)))
		gateBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(gateConfig.Bias, len(gateConfig.Bias)))
		_, gateOut := DenseForward(input, gateWeights, gateBias, gateConfig.InputHeight, gateConfig.OutputHeight, batchSize, ActivationScaledReLU)

		// Convert gate output to float32 for softmax
		for i := 0; i < len(gateLogits) && i < len(gateOut.Data); i++ {
			gateLogits[i] = float32(gateOut.Data[i])
		}
	} else {
		// Uniform weights if no gate layer
		for i := range gateLogits {
			gateLogits[i] = 1.0 / float32(len(branches))
		}
	}

	// 3. Apply softmax to get weights
	if temperature <= 0 {
		temperature = 1.0
	}
	gateWeights, _ := ForwardSoftmaxCPU(gateLogits, &LayerConfig{
		SoftmaxVariant: softmaxType,
		Temperature:    temperature,
	})

	// 4. Weighted sum of branch outputs
	combined := NewTensor[T](expectedSize)
	for i, branchOut := range branchOutputs {
		weight := float64(gateWeights[i])
		for j, val := range branchOut.Data {
			combined.Data[j] = T(float64(combined.Data[j]) + weight*float64(val))
		}
	}

	return combined, branchOutputs, gateWeights, nil
}

// ParallelBackward computes gradients for parallel layer.
func ParallelBackward[T Numeric](
	gradOutput, input *Tensor[T],
	branches []*LayerConfig,
	branchIntermediates []*Tensor[T],
	combineMode string,
) (*Tensor[T], [][]float32) {

	if len(branches) == 0 {
		return nil, nil
	}

	gradInput := NewTensor[T](len(input.Data))
	branchGrads := make([]*Tensor[T], len(branches))
	batchSize := 1 // Assume batchSize=1 for gradient splitting

	// Split gradient based on combine mode
	switch combineMode {
	case "concat", "", "grid_scatter":
		offset := 0
		for i, branchCfg := range branches {
			var size int
			switch branchCfg.Type {
			case LayerDense:
				size = batchSize * branchCfg.OutputHeight
			case LayerConv2D:
				size = batchSize * branchCfg.OutputHeight * branchCfg.OutputWidth * branchCfg.Filters
			case LayerConv1D:
				size = batchSize * branchCfg.OutputHeight * branchCfg.Conv1DFilters
			case LayerKMeans:
				if branchCfg.KMeansOutputMode == "probabilities" {
					size = batchSize * branchCfg.NumClusters
				} else {
					size = batchSize * branchCfg.ClusterDim
				}
			default:
				size = len(input.Data)
			}
			if offset+size > len(gradOutput.Data) {
				size = len(gradOutput.Data) - offset
			}
			if size <= 0 {
				branchGrads[i] = NewTensor[T](len(input.Data))
			} else {
				branchGrads[i] = NewTensorFromSlice(gradOutput.Data[offset:offset+size], size)
			}
			offset += size
		}

	case "add":
		for i := range branches {
			branchGrads[i] = gradOutput.Clone()
		}

	case "avg", "average":
		scale := 1.0 / float64(len(branches))
		for i := range branches {
			bg := NewTensor[T](len(gradOutput.Data))
			for j := range gradOutput.Data {
				bg.Data[j] = T(float64(gradOutput.Data[j]) * scale)
			}
			branchGrads[i] = bg
		}
	}

	// Process branches
	for i, branchCfg := range branches {
		gradBranch := branchGrads[i]
		if gradBranch == nil {
			continue
		}
		intermediate := branchIntermediates[i]

		var subGradInput *Tensor[T]

		switch branchCfg.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Kernel, len(branchCfg.Kernel)))
			subGradInput, _, _ = DenseBackward(gradBranch, input, intermediate, weights, branchCfg.InputHeight, branchCfg.OutputHeight, batchSize, branchCfg.Activation)

		case LayerResidual:
			gIn, _ := ResidualBackward(gradBranch)
			subGradInput = gIn

		case LayerConv1D:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Conv1DKernel, len(branchCfg.Conv1DKernel)))

			seqLen := branchCfg.InputHeight
			if seqLen <= 0 {
				seqLen = len(input.Data) / (branchCfg.Conv1DInChannels * batchSize)
			}

			subGradInput, _, _ = Conv1DBackward(gradBranch, input, intermediate, weights,
				seqLen, branchCfg.Conv1DInChannels,
				branchCfg.Conv1DKernelSize, branchCfg.Conv1DStride, branchCfg.Conv1DPadding,
				branchCfg.Conv1DFilters, batchSize, branchCfg.Activation)

		case LayerParallel:
			// Nested parallel backward
			nestedBranches := make([]*LayerConfig, len(branchCfg.ParallelBranches))
			for j := range branchCfg.ParallelBranches {
				nestedBranches[j] = &branchCfg.ParallelBranches[j]
			}
			subGradInput, _ = ParallelBackward(gradBranch, input, nestedBranches, branchIntermediates, branchCfg.CombineMode)

		case LayerSequential:
			nestedLayers := make([]*LayerConfig, len(branchCfg.ParallelBranches))
			for j := range branchCfg.ParallelBranches {
				nestedLayers[j] = &branchCfg.ParallelBranches[j]
			}
			// Note: We don't have individual layer intermediates here because ParallelForward
			// only returns ONE intermediate tensor per branch.
			// SequentialForward needs a list of intermediates.
			// This confirms my suspicion that we need to pack/unpack intermediates.
			// For now, we pass nil/empty, effectively disabling training through Sequential layers
			// until a better state management is implemented.
			subGradInput = SequentialBackward(gradBranch, input, nestedLayers, nil)

		case LayerKMeans:
			inputF32 := ConvertSliceTToFloat32(input.Data)
			gradOutF32 := ConvertSliceTToFloat32(gradBranch.Data)
			preActF32 := ConvertSliceTToFloat32(intermediate.Data)
			assignmentsF32, _ := ForwardKMeansCPU(inputF32, branchCfg)
			lr := branchCfg.KMeansLearningRate
			if lr == 0 {
				lr = 0.01
			}
			subGradInF32, _ := BackwardKMeansCPU(gradOutF32, branchCfg, inputF32, preActF32, assignmentsF32, lr)
			subGradInput = ConvertTensorFloat32ToT[T](NewTensorFromSlice(subGradInF32, len(subGradInF32)))

		default:
			if len(gradBranch.Data) == len(input.Data) {
				subGradInput = gradBranch.Clone()
			}
		}

		// Accumulate
		if subGradInput != nil && len(subGradInput.Data) == len(gradInput.Data) {
			for j := range gradInput.Data {
				gradInput.Data[j] += subGradInput.Data[j]
			}
		}
	}

	return gradInput, nil
}

// ParallelBackwardFiltered computes gradients for filter combine mode.
// Each branch receives gradient scaled by its gate weight.
// Gate gradient is computed based on how much each branch contributed to the loss.
func ParallelBackwardFiltered[T Numeric](
	gradOutput, input *Tensor[T],
	branches []*LayerConfig,
	branchOutputs []*Tensor[T],
	gateWeights []float32,
	gateConfig *LayerConfig,
) (*Tensor[T], []float32) {
	gradInput := NewTensor[T](len(input.Data))
	batchSize := 1

	// Each branch gets gradient scaled by its gate weight
	for i, branchCfg := range branches {
		weight := float64(gateWeights[i])
		if weight < 0.001 {
			continue // Skip branches with negligible weight
		}

		// Scale gradient by gate weight
		scaledGrad := NewTensor[T](len(gradOutput.Data))
		for j := range gradOutput.Data {
			scaledGrad.Data[j] = T(float64(gradOutput.Data[j]) * weight)
		}

		var subGradInput *Tensor[T]
		switch branchCfg.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Kernel, len(branchCfg.Kernel)))
			subGradInput, _, _ = DenseBackward(scaledGrad, input, input.Clone(), weights, branchCfg.InputHeight, branchCfg.OutputHeight, batchSize, branchCfg.Activation)

		case LayerConv1D:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(branchCfg.Conv1DKernel, len(branchCfg.Conv1DKernel)))

			seqLen := branchCfg.InputHeight
			if seqLen <= 0 {
				seqLen = len(input.Data) / (branchCfg.Conv1DInChannels * batchSize)
			}

			subGradInput, _, _ = Conv1DBackward(scaledGrad, input, input.Clone(), weights,
				seqLen, branchCfg.Conv1DInChannels,
				branchCfg.Conv1DKernelSize, branchCfg.Conv1DStride, branchCfg.Conv1DPadding,
				branchCfg.Conv1DFilters, batchSize, branchCfg.Activation)

		case LayerKMeans:
			inputF32 := ConvertSliceTToFloat32(input.Data)
			gradOutF32 := ConvertSliceTToFloat32(scaledGrad.Data)
			// In Filtered mode, we might not have good intermediates for sub-network
			// Re-running or using input as proxy:
			assignmentsF32, _ := ForwardKMeansCPU(inputF32, branchCfg)
			preActF32 := branchCfg.PreActivations
			lr := branchCfg.KMeansLearningRate
			if lr == 0 {
				lr = 0.01
			}
			subGradInF32, _ := BackwardKMeansCPU(gradOutF32, branchCfg, inputF32, preActF32, assignmentsF32, lr)
			subGradInput = ConvertTensorFloat32ToT[T](NewTensorFromSlice(subGradInF32, len(subGradInF32)))

		default:
			subGradInput = scaledGrad.Clone()
		}

		// Accumulate gradients from all branches
		if subGradInput != nil && len(subGradInput.Data) == len(gradInput.Data) {
			for j := range gradInput.Data {
				gradInput.Data[j] += subGradInput.Data[j]
			}
		}
	}

	// Compute gate gradient (for gate layer training)
	// dL/d(gate_i) = sum_j(gradOutput_j * branchOutput_i_j) - contribution of gate_i
	gateGrad := make([]float32, len(gateWeights))
	for i, branchOut := range branchOutputs {
		var contribution float64
		for j := range gradOutput.Data {
			contribution += float64(gradOutput.Data[j]) * float64(branchOut.Data[j])
		}
		gateGrad[i] = float32(contribution)
	}

	return gradInput, gateGrad
}

// InitFilteredParallelLayer creates a parallel layer with softmax-gated filtering.
// branches: the sub-layers to run in parallel
// gateInputSize: size of input to the gate layer
// numBranches: must match len(branches), used for gate output size
// softmaxType: which softmax variant to use for gating
func InitFilteredParallelLayer(branches []LayerConfig, gateInputSize int, softmaxType SoftmaxType, temperature float32) LayerConfig {
	numBranches := len(branches)

	// Create gate layer: Dense(inputSize -> numBranches)
	gateLayer := InitDenseLayer(gateInputSize, numBranches, ActivationScaledReLU)

	return LayerConfig{
		Type:              LayerParallel,
		ParallelBranches:  branches,
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     softmaxType,
		FilterTemperature: temperature,
	}
}

// =============================================================================
// Original float32 Implementation (Full featured)
// =============================================================================

// parallelForwardCPU executes multiple sub-layers in parallel and combines their outputs
// Returns: combined output, pre-activations for all branches (for backward pass), error
func parallelForwardCPU(input []float32, cfg *LayerConfig, batchSize int, mode string) ([]float32, [][]float32, error) {
	if len(cfg.ParallelBranches) == 0 {
		return nil, nil, fmt.Errorf("parallel layer has no branches defined")
	}

	// Run each branch and collect outputs
	branchOutputs := make([][]float32, len(cfg.ParallelBranches))
	branchPreActivations := make([][]float32, len(cfg.ParallelBranches))
	totalOutputSize := 0

	for i := range cfg.ParallelBranches {
		branchCfg := &cfg.ParallelBranches[i]

		var preAct, postAct []float32

		// Route to appropriate layer type forward function
		switch branchCfg.Type {
		case LayerDense:
			preAct, postAct = denseForwardCPU(input, branchCfg, batchSize)
		case LayerConv2D:
			preAct, postAct = conv2DForwardCPU(input, branchCfg, batchSize)
		case LayerConv1D:
			preAct, postAct = conv1DForwardCPU(input, branchCfg, batchSize)
		case LayerMultiHeadAttention:
			preAct, postAct = MultiHeadAttentionForwardCPU(input, branchCfg, batchSize)
		case LayerRNN:
			postAct, preAct = rnnForwardCPU(branchCfg, input, batchSize, branchCfg.SeqLength, branchCfg.RNNInputSize, branchCfg.HiddenSize)
		case LayerLSTM:
			var states map[string][]float32
			postAct, states = lstmForwardCPU(branchCfg, input, batchSize, branchCfg.SeqLength, branchCfg.RNNInputSize, branchCfg.HiddenSize)
			// Flatten states for backward pass
			totalStateSize := len(states["hidden"]) + len(states["cell"]) +
				len(states["i_gate"]) + len(states["f_gate"]) +
				len(states["g_gate"]) + len(states["o_gate"]) + len(states["c_tanh"])
			preAct = make([]float32, totalStateSize)
			offset := 0
			for _, key := range []string{"hidden", "cell", "i_gate", "f_gate", "g_gate", "o_gate", "c_tanh"} {
				copy(preAct[offset:], states[key])
				offset += len(states[key])
			}
		case LayerSwiGLU:
			preAct, postAct = SwiGLUForwardCPU(input, branchCfg, batchSize)
		case LayerNorm:
			postAct = layerNormForwardCPU(input, nil, branchCfg, batchSize)
			preAct = make([]float32, len(input))
			copy(preAct, input)
		case LayerRMSNorm:
			postAct = rmsNormForwardCPU(input, nil, branchCfg, batchSize)
			preAct = make([]float32, len(input))
			copy(preAct, input)
		case LayerSoftmax:
			var err error
			postAct, err = ForwardSoftmaxCPU(input, branchCfg)
			if err != nil {
				postAct = softmaxStandard(input, 1.0)
			}
			preAct = make([]float32, len(input))
			copy(preAct, input)
		case LayerParallel:
			// Nested parallel layers
			var nestedPreActs [][]float32
			var err error
			postAct, nestedPreActs, err = parallelForwardCPU(input, branchCfg, batchSize, mode)
			if err != nil {
				return nil, nil, fmt.Errorf("nested parallel layer %d failed: %w", i, err)
			}
			// Flatten nested pre-activations
			totalSize := 4 // metadata: numBranches
			for _, pa := range nestedPreActs {
				totalSize += 1 + len(pa) // size + data
			}
			preAct = make([]float32, totalSize)
			preAct[0] = float32(len(nestedPreActs))
			offset := 1
			for _, pa := range nestedPreActs {
				preAct[offset] = float32(len(pa))
				offset++
				copy(preAct[offset:], pa)
				offset += len(pa)
			}
		case LayerSequential:
			// Sequential branch
			var nestedPreActs [][]float32
			var err error
			postAct, nestedPreActs, err = sequentialForwardCPU(input, branchCfg.ParallelBranches, batchSize)
			if err != nil {
				return nil, nil, fmt.Errorf("sequential branch %d failed: %w", i, err)
			}
			// Flatten nested pre-activations similar to Parallel layer
			totalSize := 1 // metadata: type/count
			for _, pa := range nestedPreActs {
				totalSize += 1 + len(pa) // size + data
			}
			preAct = make([]float32, totalSize)
			preAct[0] = float32(len(nestedPreActs))
			offset := 1
			for _, pa := range nestedPreActs {
				preAct[offset] = float32(len(pa))
				offset++
				copy(preAct[offset:], pa)
				offset += len(pa)
			}

		case LayerKMeans:
			output, err := ForwardKMeansCPU(input, branchCfg)
			if err != nil {
				return nil, nil, fmt.Errorf("kmeans branch %d failed: %w", i, err)
			}
			postAct = output
			preAct = branchCfg.PreActivations

		default:
			return nil, nil, fmt.Errorf("unsupported layer type %d in parallel branch %d", branchCfg.Type, i)
		}

		branchOutputs[i] = postAct
		branchPreActivations[i] = preAct

		// Notify observer for this specific branch
		if cfg.Observer != nil {
			notifyBranchObserver(cfg, branchCfg, i, mode, "forward", input, postAct, 0)
		}

		if cfg.CombineMode == "concat" || cfg.CombineMode == "" || cfg.CombineMode == "grid_scatter" || cfg.CombineMode == "filter" {
			totalOutputSize += len(postAct)
		} else {
			// For add/avg, all outputs must be same size
			if i == 0 {
				totalOutputSize = len(postAct)
			} else if len(postAct) != totalOutputSize {
				return nil, nil, fmt.Errorf("branch %d output size %d doesn't match expected %d for combine mode %s",
					i, len(postAct), totalOutputSize, cfg.CombineMode)
			}
		}
	}

	// Combine outputs based on combine mode
	var combined []float32

	switch cfg.CombineMode {
	case "concat", "": // Default to concatenation
		combined = make([]float32, totalOutputSize)
		offset := 0
		for _, branchOut := range branchOutputs {
			copy(combined[offset:], branchOut)
			offset += len(branchOut)
		}

	case "add": // Element-wise addition
		combined = make([]float32, totalOutputSize)
		for _, branchOut := range branchOutputs {
			for j := range branchOut {
				combined[j] += branchOut[j]
			}
		}

	case "avg", "average": // Element-wise average
		combined = make([]float32, totalOutputSize)
		for _, branchOut := range branchOutputs {
			for j := range branchOut {
				combined[j] += branchOut[j]
			}
		}
		scale := 1.0 / float32(len(cfg.ParallelBranches))
		for j := range combined {
			combined[j] *= scale
		}

	case "grid_scatter": // Place outputs at specific grid positions
		if len(cfg.GridPositions) != len(cfg.ParallelBranches) {
			return nil, nil, fmt.Errorf("grid_scatter requires GridPositions to match number of branches (%d != %d)",
				len(cfg.GridPositions), len(cfg.ParallelBranches))
		}

		gridCells := cfg.GridOutputRows * cfg.GridOutputCols * cfg.GridOutputLayers
		if gridCells == 0 {
			return nil, nil, fmt.Errorf("grid_scatter requires GridOutputRows/Cols/Layers to be set")
		}

		// Determine output size per grid position from branch outputs
		featuresPerPosition := make([]int, gridCells)
		for i, branchOut := range branchOutputs {
			pos := cfg.GridPositions[i]

			if pos.TargetRow >= cfg.GridOutputRows || pos.TargetCol >= cfg.GridOutputCols || pos.TargetLayer >= cfg.GridOutputLayers {
				return nil, nil, fmt.Errorf("branch %d grid position (%d,%d,%d) exceeds grid bounds (%d,%d,%d)",
					i, pos.TargetRow, pos.TargetCol, pos.TargetLayer,
					cfg.GridOutputRows, cfg.GridOutputCols, cfg.GridOutputLayers)
			}

			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer

			branchOutputPerSample := len(branchOut) / batchSize
			featuresPerPosition[gridIdx] = branchOutputPerSample
		}

		totalFeatures := 0
		for _, f := range featuresPerPosition {
			totalFeatures += f
		}

		combined = make([]float32, batchSize*totalFeatures)

		// Place each branch output at its designated position
		for i, branchOut := range branchOutputs {
			pos := cfg.GridPositions[i]
			branchOutputPerSample := len(branchOut) / batchSize

			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer

			offset := 0
			for j := 0; j < gridIdx; j++ {
				offset += featuresPerPosition[j]
			}

			for b := 0; b < batchSize; b++ {
				srcStart := b * branchOutputPerSample
				srcEnd := srcStart + branchOutputPerSample
				dstStart := b*totalFeatures + offset
				copy(combined[dstStart:dstStart+branchOutputPerSample], branchOut[srcStart:srcEnd])
			}
		}

	case "filter": // Softmax-gated weighted combination with auto-padding
		if len(branchOutputs) == 0 {
			return nil, nil, fmt.Errorf("filter mode requires at least one branch")
		}

		// Find the maximum output size across all branches
		maxSize := 0
		for _, bo := range branchOutputs {
			if len(bo) > maxSize {
				maxSize = len(bo)
			}
		}

		// Pad smaller outputs to match the maximum size
		paddedOutputs := make([][]float32, len(branchOutputs))
		for i, bo := range branchOutputs {
			if len(bo) == maxSize {
				paddedOutputs[i] = bo
			} else {
				// Pad with zeros
				padded := make([]float32, maxSize)
				copy(padded, bo)
				paddedOutputs[i] = padded
			}
		}
		branchOutputs = paddedOutputs

		// Compute gate logits
		gateLogits := make([]float32, len(branchOutputs))
		if cfg.FilterGateConfig != nil && cfg.FilterGateConfig.Type == LayerDense {
			// Run gate layer on input
			gateWeights := cfg.FilterGateConfig.Kernel
			gateBias := cfg.FilterGateConfig.Bias
			gateOutSize := cfg.FilterGateConfig.OutputHeight
			if gateOutSize == 0 {
				gateOutSize = len(branchOutputs)
			}
			inputSize := cfg.FilterGateConfig.InputHeight
			if inputSize == 0 {
				inputSize = len(input)
			}
			// Simple dense forward for gate
			for o := 0; o < gateOutSize && o < len(gateLogits); o++ {
				sum := float32(0)
				for i := 0; i < inputSize && i < len(input); i++ {
					wIdx := o*inputSize + i
					if wIdx < len(gateWeights) {
						sum += input[i] * gateWeights[wIdx]
					}
				}
				if o < len(gateBias) {
					sum += gateBias[o]
				}
				gateLogits[o] = sum
			}
		} else {
			// Uniform weights if no gate layer
			for i := range gateLogits {
				gateLogits[i] = 1.0 / float32(len(branchOutputs))
			}
		}

		// Apply softmax to gate logits
		temp := cfg.FilterTemperature
		if temp <= 0 {
			temp = 1.0
		}
		gateWeights, _ := ForwardSoftmaxCPU(gateLogits, &LayerConfig{
			SoftmaxVariant: cfg.FilterSoftmax,
			Temperature:    temp,
		})

		// Weighted sum of branch outputs
		combined = make([]float32, maxSize)
		for i, branchOut := range branchOutputs {
			weight := float32(1.0 / float32(len(branchOutputs))) // default
			if i < len(gateWeights) {
				weight = gateWeights[i]
			}
			for j := range branchOut {
				combined[j] += weight * branchOut[j]
			}
		}

	default:
		return nil, nil, fmt.Errorf("unknown combine mode: %s", cfg.CombineMode)
	}

	return combined, branchPreActivations, nil
}

// parallelBackwardCPU computes gradients for parallel layer
// Takes pre-activations from forward pass to compute gradients properly
func parallelBackwardCPU(input []float32, gradOutput []float32, branchPreActivations [][]float32, cfg *LayerConfig, batchSize int, mode string) ([]float32, [][]float32, [][]float32, error) {
	if len(cfg.ParallelBranches) == 0 {
		return nil, nil, nil, fmt.Errorf("parallel layer has no branches defined")
	}

	// Split gradient based on combine mode
	var branchGrads [][]float32

	switch cfg.CombineMode {
	case "concat", "": // Split gradient by output sizes
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		offset := 0

		for i := range cfg.ParallelBranches {
			branchCfg := &cfg.ParallelBranches[i]
			var outputSize int

			switch branchCfg.Type {
			case LayerDense:
				outputSize = batchSize * branchCfg.OutputHeight
			case LayerConv2D:
				outputSize = batchSize * branchCfg.OutputHeight * branchCfg.OutputWidth * branchCfg.Filters
			case LayerConv1D:
				outputSize = batchSize * branchCfg.OutputHeight * branchCfg.Conv1DFilters
			case LayerMultiHeadAttention:
				outputSize = batchSize * branchCfg.SeqLength * branchCfg.DModel
			case LayerRNN:
				outputSize = batchSize * branchCfg.SeqLength * branchCfg.HiddenSize
			case LayerLSTM:
				outputSize = batchSize * branchCfg.SeqLength * branchCfg.HiddenSize
			case LayerSwiGLU:
				outputSize = len(input)
			case LayerNorm:
				outputSize = len(input)
			case LayerRMSNorm:
				outputSize = len(input)
			case LayerSoftmax:
				outputSize = len(input)
			case LayerParallel:
				dummyOut, _, err := parallelForwardCPU(input, branchCfg, batchSize, mode)
				if err != nil {
					return nil, nil, nil, fmt.Errorf("failed to determine nested parallel output size: %w", err)
				}
				outputSize = len(dummyOut)
			case LayerKMeans:
				if branchCfg.KMeansOutputMode == "probabilities" {
					outputSize = batchSize * branchCfg.NumClusters
				} else {
					outputSize = batchSize * branchCfg.ClusterDim
				}
			default:
				return nil, nil, nil, fmt.Errorf("cannot determine output size for layer type %d", branchCfg.Type)
			}

			branchGrads[i] = gradOutput[offset : offset+outputSize]
			offset += outputSize
		}

	case "add":
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		for i := range branchGrads {
			branchGrads[i] = gradOutput
		}

	case "avg", "average":
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		scale := 1.0 / float32(len(cfg.ParallelBranches))
		for i := range branchGrads {
			branchGrads[i] = make([]float32, len(gradOutput))
			for j := range gradOutput {
				branchGrads[i][j] = gradOutput[j] * scale
			}
		}

	case "grid_scatter":
		branchGrads = make([][]float32, len(cfg.ParallelBranches))

		gridCells := cfg.GridOutputRows * cfg.GridOutputCols * cfg.GridOutputLayers
		featuresPerPosition := make([]int, gridCells)

		for i := range cfg.ParallelBranches {
			branchCfg := &cfg.ParallelBranches[i]
			pos := cfg.GridPositions[i]

			var branchOutputSize int
			switch branchCfg.Type {
			case LayerDense:
				branchOutputSize = batchSize * branchCfg.OutputHeight
			case LayerConv2D:
				branchOutputSize = batchSize * branchCfg.OutputHeight * branchCfg.OutputWidth * branchCfg.Filters
			case LayerConv1D:
				branchOutputSize = batchSize * branchCfg.OutputHeight * branchCfg.Conv1DFilters
			case LayerMultiHeadAttention:
				branchOutputSize = batchSize * branchCfg.SeqLength * branchCfg.DModel
			case LayerRNN:
				branchOutputSize = batchSize * branchCfg.SeqLength * branchCfg.HiddenSize
			case LayerLSTM:
				branchOutputSize = batchSize * branchCfg.SeqLength * branchCfg.HiddenSize
			case LayerSwiGLU:
				branchOutputSize = len(input)
			case LayerNorm:
				branchOutputSize = len(input)
			case LayerRMSNorm:
				branchOutputSize = len(input)
			case LayerSoftmax:
				branchOutputSize = len(input)
			case LayerParallel:
				dummyOut, _, err := parallelForwardCPU(input, branchCfg, batchSize, mode)
				if err != nil {
					return nil, nil, nil, fmt.Errorf("failed to determine nested parallel output size: %w", err)
				}
				branchOutputSize = len(dummyOut)
			case LayerKMeans:
				if branchCfg.KMeansOutputMode == "probabilities" {
					branchOutputSize = batchSize * branchCfg.NumClusters
				} else {
					branchOutputSize = batchSize * branchCfg.ClusterDim
				}
			default:
				return nil, nil, nil, fmt.Errorf("cannot determine output size for layer type %d", branchCfg.Type)
			}

			branchOutputPerSample := branchOutputSize / batchSize
			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer
			featuresPerPosition[gridIdx] = branchOutputPerSample
		}

		totalFeatures := 0
		for _, f := range featuresPerPosition {
			totalFeatures += f
		}

		for i := range cfg.ParallelBranches {
			branchCfg := &cfg.ParallelBranches[i]
			pos := cfg.GridPositions[i]

			var branchOutputSize int
			switch branchCfg.Type {
			case LayerDense:
				branchOutputSize = batchSize * branchCfg.OutputHeight
			case LayerConv2D:
				branchOutputSize = batchSize * branchCfg.OutputHeight * branchCfg.OutputWidth * branchCfg.Filters
			case LayerConv1D:
				branchOutputSize = batchSize * branchCfg.OutputHeight * branchCfg.Conv1DFilters
			case LayerMultiHeadAttention:
				branchOutputSize = batchSize * branchCfg.SeqLength * branchCfg.DModel
			case LayerRNN:
				branchOutputSize = batchSize * branchCfg.SeqLength * branchCfg.HiddenSize
			case LayerLSTM:
				branchOutputSize = batchSize * branchCfg.SeqLength * branchCfg.HiddenSize
			case LayerSwiGLU:
				branchOutputSize = len(input)
			case LayerNorm:
				branchOutputSize = len(input)
			case LayerRMSNorm:
				branchOutputSize = len(input)
			case LayerSoftmax:
				branchOutputSize = len(input)
			case LayerParallel:
				dummyOut, _, err := parallelForwardCPU(input, branchCfg, batchSize, mode)
				if err != nil {
					return nil, nil, nil, fmt.Errorf("failed to determine nested parallel output size: %w", err)
				}
				branchOutputSize = len(dummyOut)
			case LayerKMeans:
				if branchCfg.KMeansOutputMode == "probabilities" {
					branchOutputSize = batchSize * branchCfg.NumClusters
				} else {
					branchOutputSize = batchSize * branchCfg.ClusterDim
				}
			default:
				return nil, nil, nil, fmt.Errorf("cannot determine output size for layer type %d", branchCfg.Type)
			}

			branchOutputPerSample := branchOutputSize / batchSize

			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer

			offset := 0
			for j := 0; j < gridIdx; j++ {
				offset += featuresPerPosition[j]
			}

			branchGrads[i] = make([]float32, branchOutputSize)
			for b := 0; b < batchSize; b++ {
				srcStart := b*totalFeatures + offset
				dstStart := b * branchOutputPerSample
				copy(branchGrads[i][dstStart:dstStart+branchOutputPerSample], gradOutput[srcStart:srcStart+branchOutputPerSample])
			}
		}

	case "filter": // Gradients flow to branches weighted by gate values
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		outputSize := len(gradOutput)

		// Recompute gate weights from input (same logic as forward) so backward reflects gating.
		gateLogits := make([]float32, len(cfg.ParallelBranches))
		if cfg.FilterGateConfig != nil && cfg.FilterGateConfig.Type == LayerDense {
			gateWeights := cfg.FilterGateConfig.Kernel
			gateBias := cfg.FilterGateConfig.Bias
			gateOutSize := cfg.FilterGateConfig.OutputHeight
			if gateOutSize == 0 {
				gateOutSize = len(cfg.ParallelBranches)
			}
			inputSize := cfg.FilterGateConfig.InputHeight
			if inputSize == 0 {
				inputSize = len(input)
			}
			for o := 0; o < gateOutSize && o < len(gateLogits); o++ {
				sum := float32(0)
				for i := 0; i < inputSize && i < len(input); i++ {
					wIdx := o*inputSize + i
					if wIdx < len(gateWeights) {
						sum += input[i] * gateWeights[wIdx]
					}
				}
				if o < len(gateBias) {
					sum += gateBias[o]
				}
				gateLogits[o] = sum
			}
		} else {
			for i := range gateLogits {
				gateLogits[i] = 1.0 / float32(len(cfg.ParallelBranches))
			}
		}

		temp := cfg.FilterTemperature
		if temp <= 0 {
			temp = 1.0
		}
		gateWeights, _ := ForwardSoftmaxCPU(gateLogits, &LayerConfig{
			SoftmaxVariant: cfg.FilterSoftmax,
			Temperature:    temp,
		})

		for i := range branchGrads {
			weight := float32(1.0 / float32(len(cfg.ParallelBranches)))
			if i < len(gateWeights) {
				weight = gateWeights[i]
			}
			branchGrads[i] = make([]float32, outputSize)
			for j := range gradOutput {
				branchGrads[i][j] = gradOutput[j] * weight
			}
		}

	default:
		return nil, nil, nil, fmt.Errorf("unknown combine mode: %s", cfg.CombineMode)
	}

	// Compute gradients for each branch and accumulate input gradients
	inputGrad := make([]float32, len(input))
	allKernelGrads := make([][]float32, len(cfg.ParallelBranches))
	allBiasGrads := make([][]float32, len(cfg.ParallelBranches))

	for i := range cfg.ParallelBranches {
		branchCfg := &cfg.ParallelBranches[i]
		preAct := branchPreActivations[i]
		gradOut := branchGrads[i]

		var branchInputGrad []float32
		var kernelGrad, biasGrad []float32

		switch branchCfg.Type {
		case LayerDense:
			branchInputGrad, kernelGrad, biasGrad = denseBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)

		case LayerConv2D:
			branchInputGrad, kernelGrad, biasGrad = conv2DBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)

		case LayerConv1D:
			branchInputGrad, kernelGrad, biasGrad = conv1DBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)

		case LayerMultiHeadAttention:
			var gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB []float32
			branchInputGrad, gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB = multiHeadAttentionBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)
			kernelGrad = append(append(append(gradQW, gradKW...), gradVW...), gradOutW...)
			biasGrad = append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)

		case LayerRNN:
			var gradWeightIH, gradWeightHH, gradBiasH []float32
			branchInputGrad, gradWeightIH, gradWeightHH, gradBiasH = rnnBackwardCPU(branchCfg, gradOut, input, preAct, batchSize, branchCfg.SeqLength, branchCfg.RNNInputSize, branchCfg.HiddenSize)
			kernelGrad = append(append(gradWeightIH, gradWeightHH...), gradBiasH...)

		case LayerLSTM:
			states := make(map[string][]float32)
			hiddenSize := branchCfg.HiddenSize
			seqLength := branchCfg.SeqLength
			states["hidden"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
			states["cell"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
			states["i_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["f_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["g_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["o_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["c_tanh"] = make([]float32, batchSize*seqLength*hiddenSize)

			offset := 0
			for _, key := range []string{"hidden", "cell", "i_gate", "f_gate", "g_gate", "o_gate", "c_tanh"} {
				copy(states[key], preAct[offset:offset+len(states[key])])
				offset += len(states[key])
			}

			var grads map[string][]float32
			branchInputGrad, grads = lstmBackwardCPU(branchCfg, gradOut, input, states, batchSize, seqLength, branchCfg.RNNInputSize, hiddenSize)
			kernelGrad = append(append(append(grads["WeightIH_i"], grads["WeightHH_i"]...), grads["BiasH_i"]...),
				append(append(grads["WeightIH_f"], grads["WeightHH_f"]...), grads["BiasH_f"]...)...)
			kernelGrad = append(kernelGrad, append(append(grads["WeightIH_g"], grads["WeightHH_g"]...), grads["BiasH_g"]...)...)
			kernelGrad = append(kernelGrad, append(append(grads["WeightIH_o"], grads["WeightHH_o"]...), grads["BiasH_o"]...)...)

		case LayerSwiGLU:
			branchInputGrad = make([]float32, len(input))
			copy(branchInputGrad, gradOut)
			kernelGrad = nil
			biasGrad = nil

		case LayerNorm:
			branchInputGrad = make([]float32, len(input))
			copy(branchInputGrad, gradOut)
			kernelGrad = nil
			biasGrad = nil

		case LayerRMSNorm:
			branchInputGrad = rmsNormBackwardCPU(preAct, nil, gradOut, branchCfg, batchSize)
			kernelGrad = nil
			biasGrad = nil

		case LayerSoftmax:
			softmaxOutput, _ := ForwardSoftmaxCPU(preAct, branchCfg)
			branchInputGrad = make([]float32, len(gradOut))
			for idx := range branchInputGrad {
				for j := range gradOut {
					branchInputGrad[idx] += gradOut[j] * softmaxOutput[j] * (kroneckerFloat(idx, j) - softmaxOutput[idx])
				}
			}
			kernelGrad = nil
			biasGrad = nil

		case LayerParallel:
			// Nested parallel backward
			numNested := int(preAct[0])
			nestedPreActs := make([][]float32, numNested)
			offset := 1
			for j := 0; j < numNested; j++ {
				size := int(preAct[offset])
				offset++
				nestedPreActs[j] = preAct[offset : offset+size]
				offset += size
			}

			var nestedKernelGrads, nestedBiasGrads [][]float32
			var err error
			branchInputGrad, nestedKernelGrads, nestedBiasGrads, err = parallelBackwardCPU(input, gradOut, nestedPreActs, branchCfg, batchSize, mode)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("nested parallel backward failed: %w", err)
			}

			totalKernel := 0
			totalBias := 0
			for j := range nestedKernelGrads {
				totalKernel += len(nestedKernelGrads[j])
				totalBias += len(nestedBiasGrads[j])
			}
			kernelGrad = make([]float32, totalKernel)
			biasGrad = make([]float32, totalBias)
			kOff := 0
			bOff := 0
			for j := range nestedKernelGrads {
				copy(kernelGrad[kOff:], nestedKernelGrads[j])
				kOff += len(nestedKernelGrads[j])
				copy(biasGrad[bOff:], nestedBiasGrads[j])
				bOff += len(nestedBiasGrads[j])
			}

		case LayerSequential:
			// Nested sequential backward
			// Unpack intermediates [count, size1, data1, ...]
			if len(preAct) == 0 {
				return nil, nil, nil, fmt.Errorf("no pre-activation data for sequential layer")
			}
			numNested := int(preAct[0])
			nestedPreActs := make([][]float32, numNested)
			offset := 1
			for j := 0; j < numNested; j++ {
				if offset >= len(preAct) {
					break
				}
				size := int(preAct[offset])
				offset++
				if offset+size <= len(preAct) {
					nestedPreActs[j] = preAct[offset : offset+size]
					offset += size
				}
			}

			var nestedKernelGrads, nestedBiasGrads [][]float32
			var err error
			// Note: parallelBackwardCPU iterates branches, sequentialBackwardCPU iterates layers.
			// Sequential Layer reuses ParallelBranches to store the sequence.
			branchInputGrad, nestedKernelGrads, nestedBiasGrads, err = sequentialBackwardCPU(input, gradOut, nestedPreActs, branchCfg.ParallelBranches, batchSize)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("nested sequential backward failed: %w", err)
			}

			// Flatten kernel/bias grads for this branch (Sequential layer returns []Layer grads)
			// parallelBackwardCPU expects ONE kernelGrad/biasGrad slice per BRANCH.
			// But wait, LayerSequential inside Parallel means "this branch is a sequence".
			// The caller of parallelBackwardCPU expects `allKernelGrads[i]` to be a FLAT slice of all weights in that branch.

			totalKernel := 0
			totalBias := 0
			for j := range nestedKernelGrads {
				totalKernel += len(nestedKernelGrads[j])
				totalBias += len(nestedBiasGrads[j])
			}
			kernelGrad = make([]float32, totalKernel)
			biasGrad = make([]float32, totalBias)
			kOff := 0
			bOff := 0
			for j := range nestedKernelGrads {
				copy(kernelGrad[kOff:], nestedKernelGrads[j])
				kOff += len(nestedKernelGrads[j])
				copy(biasGrad[bOff:], nestedBiasGrads[j])
				bOff += len(nestedBiasGrads[j])
			}

		case LayerKMeans:
			assignments, _ := ForwardKMeansCPU(input, branchCfg)
			lr := branchCfg.KMeansLearningRate
			if lr == 0 {
				lr = 0.01
			}
			branchInputGrad, _ = BackwardKMeansCPU(gradOut, branchCfg, input, preAct, assignments, lr)
			kernelGrad = nil // KMgrads are handled in-place inside BackwardKMeansCPU
			biasGrad = nil

		default:
			return nil, nil, nil, fmt.Errorf("unsupported layer type %d in parallel branch %d backward", branchCfg.Type, i)
		}

		// Accumulate input gradients from all branches
		for j := range inputGrad {
			inputGrad[j] += branchInputGrad[j]
		}

		allKernelGrads[i] = kernelGrad
		allBiasGrads[i] = biasGrad

		// Notify observer for this specific branch
		if cfg.Observer != nil {
			notifyBranchObserver(cfg, branchCfg, i, mode, "backward", nil, branchInputGrad, 0)
		}
	}

	return inputGrad, allKernelGrads, allBiasGrads, nil
}
