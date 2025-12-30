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

		if combineMode == "concat" || combineMode == "" || combineMode == "grid_scatter" {
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

	default:
		return nil, nil, fmt.Errorf("unknown combine mode: %s", combineMode)
	}

	return combined, branchIntermediates, nil
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

		case LayerParallel:
			// Nested parallel backward
			nestedBranches := make([]*LayerConfig, len(branchCfg.ParallelBranches))
			for j := range branchCfg.ParallelBranches {
				nestedBranches[j] = &branchCfg.ParallelBranches[j]
			}
			subGradInput, _ = ParallelBackward(gradBranch, input, nestedBranches, branchIntermediates, branchCfg.CombineMode)

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
		default:
			return nil, nil, fmt.Errorf("unsupported layer type %d in parallel branch %d", branchCfg.Type, i)
		}

		branchOutputs[i] = postAct
		branchPreActivations[i] = preAct

		// Notify observer for this specific branch
		if cfg.Observer != nil {
			notifyBranchObserver(cfg, branchCfg, i, mode, "forward", input, postAct, 0)
		}

		if cfg.CombineMode == "concat" || cfg.CombineMode == "" || cfg.CombineMode == "grid_scatter" {
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
