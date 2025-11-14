package nn

import "fmt"

// parallelForwardCPU executes multiple sub-layers in parallel and combines their outputs
func parallelForwardCPU(input []float32, cfg *LayerConfig, batchSize int) ([]float32, error) {
	if len(cfg.ParallelBranches) == 0 {
		return nil, fmt.Errorf("parallel layer has no branches defined")
	}

	// Run each branch and collect outputs
	branchOutputs := make([][]float32, len(cfg.ParallelBranches))
	totalOutputSize := 0

	for i, branchCfg := range cfg.ParallelBranches {
		// Create a mini 1x1x1 network for this branch
		branchNet := &Network{
			GridRows:      1,
			GridCols:      1,
			LayersPerCell: 1,
			BatchSize:     batchSize,
			Layers:        []LayerConfig{branchCfg},
		}

		// Initialize activation storage
		branchNet.activations = make([][]float32, 2)
		branchNet.preActivations = make([][]float32, 1)
		branchNet.kernelGradients = make([][]float32, 1)
		branchNet.biasGradients = make([][]float32, 1)

		// Run forward pass on this branch
		output, _ := branchNet.ForwardCPU(input)
		branchOutputs[i] = output // Track output size for combine mode
		if cfg.CombineMode == "concat" || cfg.CombineMode == "" {
			totalOutputSize += len(output)
		} else {
			// For add/avg, all outputs must be same size
			if i == 0 {
				totalOutputSize = len(output)
			} else if len(output) != totalOutputSize {
				return nil, fmt.Errorf("branch %d output size %d doesn't match expected %d for combine mode %s",
					i, len(output), totalOutputSize, cfg.CombineMode)
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

	default:
		return nil, fmt.Errorf("unknown combine mode: %s", cfg.CombineMode)
	}

	return combined, nil
}

// parallelBackwardCPU computes gradients for parallel layer
func parallelBackwardCPU(input []float32, gradOutput []float32, cfg *LayerConfig, batchSize int) ([]float32, error) {
	if len(cfg.ParallelBranches) == 0 {
		return nil, fmt.Errorf("parallel layer has no branches defined")
	}

	// Split gradient based on combine mode
	var branchGrads [][]float32

	switch cfg.CombineMode {
	case "concat", "": // Split gradient by output sizes
		branchGrads = make([][]float32, len(cfg.ParallelBranches))
		offset := 0

		// First, run forward to get output sizes (needed for gradient split)
		for i, branchCfg := range cfg.ParallelBranches {
			branchNet := &Network{
				GridRows:      1,
				GridCols:      1,
				LayersPerCell: 1,
				BatchSize:     batchSize,
				Layers:        []LayerConfig{branchCfg},
			}
			branchNet.activations = make([][]float32, 2)
			branchNet.preActivations = make([][]float32, 1)

			output, _ := branchNet.ForwardCPU(input)
			outputSize := len(output)
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

	default:
		return nil, fmt.Errorf("unknown combine mode: %s", cfg.CombineMode)
	}

	// Compute gradients for each branch and accumulate input gradients
	inputGrad := make([]float32, len(input))

	for i, branchCfg := range cfg.ParallelBranches {
		// Create mini network for this branch
		branchNet := &Network{
			GridRows:      1,
			GridCols:      1,
			LayersPerCell: 1,
			BatchSize:     batchSize,
			Layers:        []LayerConfig{branchCfg},
		}
		branchNet.activations = make([][]float32, 2)
		branchNet.preActivations = make([][]float32, 1)
		branchNet.kernelGradients = make([][]float32, 1)
		branchNet.biasGradients = make([][]float32, 1)

		// Forward pass to fill activations
		branchNet.ForwardCPU(input)

		// Backward pass
		branchInputGrad, _ := branchNet.BackwardCPU(branchGrads[i])

		// Accumulate input gradients from all branches
		for j := range inputGrad {
			inputGrad[j] += branchInputGrad[j]
		}

		// Update the branch config with computed gradients
		cfg.ParallelBranches[i] = branchNet.Layers[0]
	}

	return inputGrad, nil
}
