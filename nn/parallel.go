package nn

import "fmt"

// parallelForwardCPU executes multiple sub-layers in parallel and combines their outputs
// Returns: combined output, pre-activations for all branches (for backward pass), error
func parallelForwardCPU(input []float32, cfg *LayerConfig, batchSize int) ([]float32, [][]float32, error) {
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
			// LayerNorm doesn't use residual in parallel branches
			postAct = layerNormForwardCPU(input, nil, branchCfg, batchSize)
			preAct = make([]float32, len(input))
			copy(preAct, input) // Store input for backward
		case LayerRMSNorm:
			// RMSNorm doesn't use residual in parallel branches
			postAct = rmsNormForwardCPU(input, nil, branchCfg, batchSize)
			preAct = make([]float32, len(input))
			copy(preAct, input) // Store input for backward
		case LayerSoftmax:
			// Softmax forward
			var err error
			postAct, err = ForwardSoftmaxCPU(input, branchCfg)
			if err != nil {
				// Fallback to standard softmax
				postAct = softmaxStandard(input, 1.0)
			}
			preAct = make([]float32, len(input))
			copy(preAct, input) // Store logits for backward
		case LayerParallel:
			// Nested parallel layers
			var nestedPreActs [][]float32
			var err error
			postAct, nestedPreActs, err = parallelForwardCPU(input, branchCfg, batchSize)
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
			notifyBranchObserver(cfg, branchCfg, i, "forward", input, postAct, 0)
		}

		if cfg.CombineMode == "concat" || cfg.CombineMode == "" {
			totalOutputSize += len(postAct)
		} else if cfg.CombineMode == "grid_scatter" {
			// For grid_scatter, branches can have different sizes (accumulated later)
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
		// Divide by number of branches
		scale := 1.0 / float32(len(cfg.ParallelBranches))
		for j := range combined {
			combined[j] *= scale
		}

	case "grid_scatter": // Place outputs at specific grid positions
		if len(cfg.GridPositions) != len(cfg.ParallelBranches) {
			return nil, nil, fmt.Errorf("grid_scatter requires GridPositions to match number of branches (%d != %d)",
				len(cfg.GridPositions), len(cfg.ParallelBranches))
		}

		// Calculate total features per grid position
		// First pass: determine max features needed per grid cell
		gridCells := cfg.GridOutputRows * cfg.GridOutputCols * cfg.GridOutputLayers
		if gridCells == 0 {
			return nil, nil, fmt.Errorf("grid_scatter requires GridOutputRows/Cols/Layers to be set")
		}

		// Determine output size per grid position from branch outputs
		featuresPerPosition := make([]int, gridCells)
		for i, branchOut := range branchOutputs {
			pos := cfg.GridPositions[i]

			// Validate position
			if pos.TargetRow >= cfg.GridOutputRows || pos.TargetCol >= cfg.GridOutputCols || pos.TargetLayer >= cfg.GridOutputLayers {
				return nil, nil, fmt.Errorf("branch %d grid position (%d,%d,%d) exceeds grid bounds (%d,%d,%d)",
					i, pos.TargetRow, pos.TargetCol, pos.TargetLayer,
					cfg.GridOutputRows, cfg.GridOutputCols, cfg.GridOutputLayers)
			}

			// Calculate flat index for this grid position
			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer

			// Each branch contributes features to its grid position
			branchOutputPerSample := len(branchOut) / batchSize
			featuresPerPosition[gridIdx] = branchOutputPerSample
		}

		// Calculate total output size
		totalFeatures := 0
		for _, f := range featuresPerPosition {
			totalFeatures += f
		}

		// Initialize output
		combined = make([]float32, batchSize*totalFeatures)

		// Place each branch output at its designated position
		for i, branchOut := range branchOutputs {
			pos := cfg.GridPositions[i]
			branchOutputPerSample := len(branchOut) / batchSize

			// Calculate offset: sum of features from all previous grid positions
			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer

			offset := 0
			for j := 0; j < gridIdx; j++ {
				offset += featuresPerPosition[j]
			}

			// Copy branch output for each sample in batch
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
func parallelBackwardCPU(input []float32, gradOutput []float32, branchPreActivations [][]float32, cfg *LayerConfig, batchSize int) ([]float32, [][]float32, [][]float32, error) {
	if len(cfg.ParallelBranches) == 0 {
		return nil, nil, nil, fmt.Errorf("parallel layer has no branches defined")
	}

	// Split gradient based on combine mode
	var branchGrads [][]float32

	switch cfg.CombineMode {
	case "concat", "": // Split gradient by output sizes
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		offset := 0

		// We need to know output sizes - recompute from forward (TODO: cache this)
		for i := range cfg.ParallelBranches {
			branchCfg := &cfg.ParallelBranches[i]
			var outputSize int

			// Quick size calculation based on layer type
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
				outputSize = len(input) // SwiGLU outputs same size as input
			case LayerNorm:
				outputSize = len(input) // LayerNorm preserves size
			case LayerRMSNorm:
				outputSize = len(input) // RMSNorm preserves size
			case LayerSoftmax:
				outputSize = len(input) // Softmax preserves size
			case LayerParallel:
				// For nested parallel, need to compute from config
				// This is complex, so run a dummy forward to get size
				dummyOut, _, err := parallelForwardCPU(input, branchCfg, batchSize)
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

	case "add": // Each branch gets full gradient
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		for i := range branchGrads {
			branchGrads[i] = gradOutput
		}

	case "avg", "average": // Each branch gets scaled gradient
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		scale := 1.0 / float32(len(cfg.ParallelBranches))
		for i := range branchGrads {
			branchGrads[i] = make([]float32, len(gradOutput))
			for j := range gradOutput {
				branchGrads[i][j] = gradOutput[j] * scale
			}
		}

	case "grid_scatter": // Extract gradients from specific grid positions
		branchGrads = make([][]float32, len(cfg.ParallelBranches))

		// Reconstruct features per position (same logic as forward)
		gridCells := cfg.GridOutputRows * cfg.GridOutputCols * cfg.GridOutputLayers
		featuresPerPosition := make([]int, gridCells)

		for i := range cfg.ParallelBranches {
			branchCfg := &cfg.ParallelBranches[i]
			pos := cfg.GridPositions[i]

			// Calculate output size for this branch
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
				dummyOut, _, err := parallelForwardCPU(input, branchCfg, batchSize)
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

		// Calculate total features
		totalFeatures := 0
		for _, f := range featuresPerPosition {
			totalFeatures += f
		}

		// Extract gradient for each branch from its grid position
		for i := range cfg.ParallelBranches {
			branchCfg := &cfg.ParallelBranches[i]
			pos := cfg.GridPositions[i]

			// Calculate branch output size
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
				dummyOut, _, err := parallelForwardCPU(input, branchCfg, batchSize)
				if err != nil {
					return nil, nil, nil, fmt.Errorf("failed to determine nested parallel output size: %w", err)
				}
				branchOutputSize = len(dummyOut)
			default:
				return nil, nil, nil, fmt.Errorf("cannot determine output size for layer type %d", branchCfg.Type)
			}

			branchOutputPerSample := branchOutputSize / batchSize

			// Calculate offset: sum of features from all previous grid positions
			gridIdx := pos.TargetRow*cfg.GridOutputCols*cfg.GridOutputLayers +
				pos.TargetCol*cfg.GridOutputLayers +
				pos.TargetLayer

			offset := 0
			for j := 0; j < gridIdx; j++ {
				offset += featuresPerPosition[j]
			}

			// Extract gradient from grid position
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

		// Route to appropriate layer type backward function
		switch branchCfg.Type {
		case LayerDense:
			branchInputGrad, kernelGrad, biasGrad = denseBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)

		case LayerConv2D:
			branchInputGrad, kernelGrad, biasGrad = conv2DBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)

		case LayerMultiHeadAttention:
			var gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB []float32
			branchInputGrad, gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB = multiHeadAttentionBackwardCPU(gradOut, input, preAct, branchCfg, batchSize)
			// Concatenate all weight gradients
			kernelGrad = append(append(append(gradQW, gradKW...), gradVW...), gradOutW...)
			biasGrad = append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)

		case LayerRNN:
			var gradWeightIH, gradWeightHH, gradBiasH []float32
			branchInputGrad, gradWeightIH, gradWeightHH, gradBiasH = rnnBackwardCPU(branchCfg, gradOut, input, preAct, batchSize, branchCfg.SeqLength, branchCfg.RNNInputSize, branchCfg.HiddenSize)
			kernelGrad = append(append(gradWeightIH, gradWeightHH...), gradBiasH...)

		case LayerLSTM:
			// Unflatten states from preActivations
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
			// Concatenate all LSTM gradients
			kernelGrad = append(append(append(grads["WeightIH_i"], grads["WeightHH_i"]...), grads["BiasH_i"]...),
				append(append(grads["WeightIH_f"], grads["WeightHH_f"]...), grads["BiasH_f"]...)...)
			kernelGrad = append(kernelGrad, append(append(grads["WeightIH_g"], grads["WeightHH_g"]...), grads["BiasH_g"]...)...)
			kernelGrad = append(kernelGrad, append(append(grads["WeightIH_o"], grads["WeightHH_o"]...), grads["BiasH_o"]...)...)

		case LayerSwiGLU:
			// SwiGLU backward - TODO: implement proper backward pass
			// For now, just pass gradient through
			branchInputGrad = make([]float32, len(input))
			copy(branchInputGrad, gradOut)
			kernelGrad = nil
			biasGrad = nil

		case LayerNorm:
			// LayerNorm backward - TODO: implement proper backward pass
			// For now, just pass gradient through
			branchInputGrad = make([]float32, len(input))
			copy(branchInputGrad, gradOut)
			kernelGrad = nil
			biasGrad = nil

		case LayerRMSNorm:
			// RMSNorm has backward function
			branchInputGrad = rmsNormBackwardCPU(preAct, nil, gradOut, branchCfg, batchSize)
			kernelGrad = nil // Gamma gradients not computed yet
			biasGrad = nil

		case LayerSoftmax:
			// Softmax backward
			// Get softmax output from forward (need to recompute for now)
			softmaxOutput, _ := ForwardSoftmaxCPU(preAct, branchCfg)

			if branchCfg.SoftmaxRows > 0 {
				// Grid Softmax
				rows := branchCfg.SoftmaxRows
				cols := branchCfg.SoftmaxCols
				branchInputGrad = make([]float32, len(gradOut))

				for row := 0; row < rows; row++ {
					start := row * cols
					end := start + cols
					for i := start; i < end; i++ {
						var gradSum float32
						for j := start; j < end; j++ {
							jacobian := softmaxOutput[j] * (kroneckerFloat(i, j) - softmaxOutput[i])
							gradSum += gradOut[j] * jacobian
						}
						branchInputGrad[i] = gradSum
					}
				}
			} else {
				// Standard softmax
				branchInputGrad = make([]float32, len(gradOut))
				for i := range branchInputGrad {
					var gradSum float32
					for j := range gradOut {
						jacobian := softmaxOutput[j] * (kroneckerFloat(i, j) - softmaxOutput[i])
						gradSum += gradOut[j] * jacobian
					}
					branchInputGrad[i] = gradSum
				}
			}
			kernelGrad = nil
			biasGrad = nil

		case LayerParallel:
			// Nested parallel backward
			// Unflatten nested pre-activations
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
			branchInputGrad, nestedKernelGrads, nestedBiasGrads, err = parallelBackwardCPU(input, gradOut, nestedPreActs, branchCfg, batchSize)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("nested parallel backward failed: %w", err)
			}

			// Flatten nested gradients
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

		// Store gradients for this branch
		allKernelGrads[i] = kernelGrad
		allBiasGrads[i] = biasGrad
	}

	return inputGrad, allKernelGrads, allBiasGrads, nil
}
