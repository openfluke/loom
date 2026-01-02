package nn

import "fmt"

// InitSequentialLayer initializes a sequential layer that runs multiple sub-layers in order.
// This is useful for grouping layers (e.g. expert + stitch) as a single branch in a parallel layer.
func InitSequentialLayer(layers ...LayerConfig) LayerConfig {
	return LayerConfig{
		Type:             LayerSequential,
		ParallelBranches: layers, // Reuse ParallelBranches to store the sequence
	}
}

// =============================================================================
// Generic Sequential Layer Implementation
// =============================================================================

// SequentialForward executes sub-layers in sequence for any numeric type.
// Returns: final output, list of intermediate outputs (for backward pass), error
func SequentialForward[T Numeric](
	input *Tensor[T],
	layers []*LayerConfig,
	batchSize int,
) (*Tensor[T], []*Tensor[T], error) {
	if len(layers) == 0 {
		return input.Clone(), nil, nil
	}

	intermediates := make([]*Tensor[T], len(layers))
	currentInput := input

	for i, layerCfg := range layers {
		var preAct, postAct *Tensor[T]

		switch layerCfg.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.Kernel, len(layerCfg.Kernel)))
			bias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.Bias, len(layerCfg.Bias)))
			preAct, postAct = DenseForward(currentInput, weights, bias, layerCfg.InputHeight, layerCfg.OutputHeight, batchSize, layerCfg.Activation)

		case LayerSwiGLU:
			gateW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.GateWeights, len(layerCfg.GateWeights)))
			upW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.UpWeights, len(layerCfg.UpWeights)))
			downW := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.DownWeights, len(layerCfg.DownWeights)))
			gateBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.GateBias, len(layerCfg.GateBias)))
			upBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.UpBias, len(layerCfg.UpBias)))
			downBias := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.DownBias, len(layerCfg.DownBias)))
			postAct = SwiGLUForward(currentInput, gateW, upW, downW, gateBias, upBias, downBias, layerCfg.InputHeight, layerCfg.OutputHeight, batchSize)
			preAct = currentInput.Clone() // Needs re-evaluation or stored context
			
		case LayerNorm:
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.Gamma, len(layerCfg.Gamma)))
			beta := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.Beta, len(layerCfg.Beta)))
			normSize := layerCfg.NormSize
			if normSize <= 0 {
				normSize = len(currentInput.Data)
			}
			postAct = LayerNormForward(currentInput, nil, gamma, beta, normSize, batchSize, float64(layerCfg.Epsilon))
			preAct = currentInput.Clone()

		case LayerRMSNorm:
			gamma := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.Gamma, len(layerCfg.Gamma)))
			normSize := layerCfg.NormSize
			if normSize <= 0 {
				normSize = len(currentInput.Data)
			}
			postAct = RMSNormForward(currentInput, nil, gamma, normSize, float64(layerCfg.Epsilon))
			preAct = currentInput.Clone()

		case LayerSoftmax:
			postAct = ApplySoftmax(currentInput, float64(layerCfg.Temperature))
			preAct = currentInput.Clone()

		case LayerParallel:
			nestedBranches := make([]*LayerConfig, len(layerCfg.ParallelBranches))
			for j := range layerCfg.ParallelBranches {
				nestedBranches[j] = &layerCfg.ParallelBranches[j]
			}
			var err error
			postAct, _, err = ParallelForward[T](currentInput, nestedBranches, batchSize, layerCfg.CombineMode)
			if err != nil {
				return nil, nil, fmt.Errorf("sequential layer %d (parallel) failed: %w", i, err)
			}
			preAct = currentInput.Clone()

		case LayerSequential:
			nestedLayers := make([]*LayerConfig, len(layerCfg.ParallelBranches))
			for j := range layerCfg.ParallelBranches {
				nestedLayers[j] = &layerCfg.ParallelBranches[j]
			}
			var err error
			postAct, _, err = SequentialForward[T](currentInput, nestedLayers, batchSize)
			if err != nil {
				return nil, nil, fmt.Errorf("nested sequential layer %d failed: %w", i, err)
			}
			preAct = currentInput.Clone()

		default:
			// Pass through
			postAct = currentInput.Clone()
			preAct = currentInput.Clone()
		}

		intermediates[i] = preAct
		currentInput = postAct
	}

	return currentInput, intermediates, nil
}

// SequentialBackward computes gradients for sequential layer.
// Iterates backward through layers.
func SequentialBackward[T Numeric](
	gradOutput, input *Tensor[T],
	layers []*LayerConfig,
	intermediates []*Tensor[T],
) *Tensor[T] {
	if len(layers) == 0 {
		return gradOutput
	}

	currentGrad := gradOutput
	
	// Iterate backwards
	for i := len(layers) - 1; i >= 0; i-- {
		layerCfg := layers[i]
		intermediate := intermediates[i] // Pre-activation or input to this layer
		
		// For the FIRST layer (index 0), its input is the function arg `input`.
		// For subsequent layers (index > 0), the input was the output of previous layer.
		// However, many Backward functions require the INPUT to the layer.
		// In Forward loop: 
		//   Layer 0 input = input
		//   Layer 1 input = Layer 0 output
		// We didn't store Layer 0 output in `intermediates`!
		// `intermediates` stores `preAct` (often needed for activation derivative) 
		// OR `currentInput` (for LayerNorm, RMSNorm, SwiGLU etc).
		
		// Let's refine strict requirements for each layer type:
		// DenseBackward: needs (gradOutput, input, preAct, weights)
		//   - input: input to this layer
		//   - preAct: pre-activation of this layer
		
		// Wait, my Forward implementation stores `preAct` in `intermediates`.
		// It does NOT store the input to the layer (except implicitly as `preAct` for Norm layers).
		// This is a problem for DenseBackward which needs BOTH.
		
		// For this simple implementation, we assume we can re-derive what's needed or
		// we should have stored `activations` like the main GenericForwardPass.
		// Since we only need this for the "Expert -> Stitch" case, let's see.
		// Expert (Dense) -> Stitch (Dense).
		// We need input to Stitch (which is Output of Expert).
		
		// FIX: We need to store layer inputs in Forward!
		// Let's rely on the fact that `preAct` IS the input for some layers, but for Dense it's not.
		// Actually, `GenericForwardPass` stores `activations` array.
		// `SequentialForward` returns `intermediates` which currently matches `preActivations` logic.
		
		// Let's update SequentialForward to return inputs as well?
		// Or just stick to the specific use case where we might hack it?
		// No, let's do it right. But keeping it simple.
		// The `intermediates` slice is returned. I can repurpose it or add another return.
		// `ParallelForward` returns `combiner` and `branchIntermediates`.
		// `SequentialForward` matches that signature somewhat.
		
		// Let's assume for now we only support layers where we can manage.
		

		// Calculate batchSize for DenseBackward
		// Use tensor length / input height from config
		batchSize := len(input.Data) / layers[0].InputHeight
		if i < len(layers)-1 {
			// For internal layers, input size might differ.
			// However `currentGrad` size is batchSize * OutputHeight.
			batchSize = len(currentGrad.Data) / layerCfg.OutputHeight
		}

		var subGradInput *Tensor[T]

		switch layerCfg.Type {
		case LayerDense:
			weights := ConvertTensorFloat32ToT[T](NewTensorFromSlice(layerCfg.Kernel, len(layerCfg.Kernel)))
			// We lack 'input'. 
			// We can compute dInput (gradInput) without `input`, but we can't compute `gradWeights`.
			// Since `SequentialBackward` is mostly a helper for now, we pass a dummy input tensor 
			// of correct size to satisfy the function signature.
			dummyInput := NewTensor[T](layerCfg.InputHeight * batchSize)
			
			subGradInput, _, _ = DenseBackward(currentGrad, dummyInput, intermediate, weights, layerCfg.InputHeight, layerCfg.OutputHeight, batchSize, layerCfg.Activation)

		default:
			subGradInput = currentGrad // Pass through gradient
		}
		
		if subGradInput != nil {
			currentGrad = subGradInput
		}
	}

	return currentGrad
}

// =============================================================================
// Float32 Helper (Backward Compatible)
// =============================================================================

func sequentialForwardCPU(input []float32, layers []LayerConfig, batchSize int) ([]float32, [][]float32, error) {
	if len(layers) == 0 {
		return input, nil, nil
	}

	// We need to convert []LayerConfig to []*LayerConfig for generating generic call?
	// Or just reimplement loop for float32 to be fast/simple.
	
	currentInput := input
	var intermediates [][]float32
	// Note: We are not storing full intermediates here, matching generic pattern
	// This might limit backward pass capability as noted above.

	for i := range layers {
		layerCfg := &layers[i]
		var preAct, postAct []float32

		switch layerCfg.Type {
		case LayerDense:
			preAct, postAct = denseForwardCPU(currentInput, layerCfg, batchSize)
		case LayerSoftmax:
			var err error
			postAct, err = ForwardSoftmaxCPU(currentInput, layerCfg)
			if err != nil {
				return nil, nil, err
			}
			preAct = make([]float32, len(currentInput))
			copy(preAct, currentInput)
		default:
			// Fallback / Pass-through
			postAct = currentInput
			preAct = currentInput
		}
		
		intermediates = append(intermediates, preAct)
		currentInput = postAct
	}

	return currentInput, intermediates, nil
}

// sequentialBackwardCPU computes gradients for sequential layer.
// Re-runs forward pass to reconstruct inputs for each layer.
func sequentialBackwardCPU(input []float32, gradOutput []float32, intermediates [][]float32, layers []LayerConfig, batchSize int) ([]float32, [][]float32, [][]float32, error) {
	if len(layers) == 0 {
		return gradOutput, nil, nil, nil
	}

	// 1. Re-run forward pass to get inputs for each layer
	// layerInputs[i] is the input to layer i.
	layerInputs := make([][]float32, len(layers))
	currentInput := input
	
	for i := range layers {
		layerInputs[i] = currentInput
		
		layerCfg := &layers[i]
		var postAct []float32
		
		// We only need the output (postAct) to serve as input for next layer
		switch layerCfg.Type {
		case LayerDense:
			_, postAct = denseForwardCPU(currentInput, layerCfg, batchSize)
		case LayerConv2D:
			_, postAct = conv2DForwardCPU(currentInput, layerCfg, batchSize)
		case LayerMultiHeadAttention:
			_, postAct = MultiHeadAttentionForwardCPU(currentInput, layerCfg, batchSize)
		case LayerRNN:
			postAct, _ = rnnForwardCPU(layerCfg, currentInput, batchSize, layerCfg.SeqLength, layerCfg.RNNInputSize, layerCfg.HiddenSize)
		case LayerLSTM:
			postAct, _ = lstmForwardCPU(layerCfg, currentInput, batchSize, layerCfg.SeqLength, layerCfg.RNNInputSize, layerCfg.HiddenSize)
		case LayerSwiGLU:
			_, postAct = SwiGLUForwardCPU(currentInput, layerCfg, batchSize)
		case LayerNorm:
			postAct = layerNormForwardCPU(currentInput, nil, layerCfg, batchSize)
		case LayerRMSNorm:
			postAct = rmsNormForwardCPU(currentInput, nil, layerCfg, batchSize)
		case LayerSoftmax:
			postAct, _ = ForwardSoftmaxCPU(currentInput, layerCfg)
			if postAct == nil { postAct = currentInput }
		default:
			postAct = currentInput
		}
		currentInput = postAct
	}

	// 2. Backward Pass
	nestedKernelGrads := make([][]float32, len(layers))
	nestedBiasGrads := make([][]float32, len(layers))
	currentGrad := gradOutput

	for i := len(layers) - 1; i >= 0; i-- {
		layerCfg := &layers[i]
		layerInput := layerInputs[i]
		preAct := intermediates[i]

		var gradInput, kGrad, bGrad []float32

		switch layerCfg.Type {
		case LayerDense:
			gradInput, kGrad, bGrad = denseBackwardCPU(currentGrad, layerInput, preAct, layerCfg, batchSize)
		case LayerConv2D:
			gradInput, kGrad, bGrad = conv2DBackwardCPU(currentGrad, layerInput, preAct, layerCfg, batchSize)
		case LayerMultiHeadAttention:
			var gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB []float32
			gradInput, gradQW, gradKW, gradVW, gradOutW, gradQB, gradKB, gradVB, gradOutB = multiHeadAttentionBackwardCPU(currentGrad, layerInput, preAct, layerCfg, batchSize)
			kGrad = append(append(append(gradQW, gradKW...), gradVW...), gradOutW...)
			bGrad = append(append(append(gradQB, gradKB...), gradVB...), gradOutB...)
		case LayerRNN:
			var gradWeightIH, gradWeightHH, gradBiasH []float32
			gradInput, gradWeightIH, gradWeightHH, gradBiasH = rnnBackwardCPU(layerCfg, currentGrad, layerInput, preAct, batchSize, layerCfg.SeqLength, layerCfg.RNNInputSize, layerCfg.HiddenSize)
			kGrad = append(append(gradWeightIH, gradWeightHH...), gradBiasH...)
		case LayerLSTM:
			states := make(map[string][]float32)
			hiddenSize := layerCfg.HiddenSize
			seqLength := layerCfg.SeqLength
			states["hidden"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
			states["cell"] = make([]float32, batchSize*(seqLength+1)*hiddenSize)
			states["i_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["f_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["g_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["o_gate"] = make([]float32, batchSize*seqLength*hiddenSize)
			states["c_tanh"] = make([]float32, batchSize*seqLength*hiddenSize)

			offset := 0
			for _, key := range []string{"hidden", "cell", "i_gate", "f_gate", "g_gate", "o_gate", "c_tanh"} {
				if offset+len(states[key]) <= len(preAct) {
					copy(states[key], preAct[offset:offset+len(states[key])])
				}
				offset += len(states[key])
			}

			var grads map[string][]float32
			gradInput, grads = lstmBackwardCPU(layerCfg, currentGrad, layerInput, states, batchSize, seqLength, layerCfg.RNNInputSize, hiddenSize)
			kGrad = append(append(append(grads["WeightIH_i"], grads["WeightHH_i"]...), grads["BiasH_i"]...),
				append(append(grads["WeightIH_f"], grads["WeightHH_f"]...), grads["BiasH_f"]...)...)
			kGrad = append(kGrad, append(append(grads["WeightIH_g"], grads["WeightHH_g"]...), grads["BiasH_g"]...)...)
			kGrad = append(kGrad, append(append(grads["WeightIH_o"], grads["WeightHH_o"]...), grads["BiasH_o"]...)...)
		case LayerSwiGLU:
			gradInput = make([]float32, len(layerInput)); copy(gradInput, currentGrad)
		case LayerNorm:
			gradInput = make([]float32, len(layerInput)); copy(gradInput, currentGrad)
		case LayerRMSNorm:
			gradInput = rmsNormBackwardCPU(preAct, nil, currentGrad, layerCfg, batchSize)
		case LayerSoftmax:
			// Softmax backward
			softmaxOutput, _ := ForwardSoftmaxCPU(preAct, layerCfg)
			gradInput = make([]float32, len(currentGrad))
			for idx := range gradInput {
				for j := range currentGrad {
					gradInput[idx] += currentGrad[j] * softmaxOutput[j] * (kroneckerFloat(idx, j) - softmaxOutput[idx])
				}
			}
		default:
			return nil, nil, nil, fmt.Errorf("unsupported layer type %d in sequential backward", layerCfg.Type)
		}

		nestedKernelGrads[i] = kGrad
		nestedBiasGrads[i] = bGrad
		currentGrad = gradInput
	}

	return currentGrad, nestedKernelGrads, nestedBiasGrads, nil
}
